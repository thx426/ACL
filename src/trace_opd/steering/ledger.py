"""steering 记账（规范 §4 的稀疏实现口径）→ rollout extra_fields。

记账自洽原则：log_b 必须是**实际行为分布**（保尾核）对最终执行 token 的
真值——训练端 W_beh 的无偏性只建立在这一条上。

稀疏口径（审阅 IF-07/IF-15 定案，spec §4 的实现注记）：只有「门参与」的
位置（base_open / probe_flipped / η>0）落逐 token 行；其余位置 log_b==log_pi
（ratio 0），训练端展开为权重 1，无需行。分布/率类审计由三个全量通道补齐：
① D 观测直方图（每个候选位置一次 observe_d）；② 计数器组（候选/开门/探针
翻转/预算阻断/旁路）；③ 轨迹级汇总（budget_spent/remaining、turns_capped）。
「门关且未翻转」位置的 propensity 恒为 1−ε，可由计数器重建——审计意图
（重建行为状态与率）在此口径下完整达成，而 32k 轨迹不必携带数万行全量账。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

__all__ = ["TokenSteerRecord", "SteeringLedger"]


@dataclass(frozen=True)
class TokenSteerRecord:
    position: int          # response 内位置（与 turn_spans 同坐标）
    log_pi: float
    log_b: float
    eta: float
    delta: float           # 本位置预算扣费（=2ηc；η=0 ⇒ 0）
    d_value: float
    base_open: bool
    probe_flipped: bool
    propensity: float
    rejected: bool
    cap_active: bool       # 本行写入时 cap 是否已激活(第 R_max 次拒绝行=True)
    budget_after: float = -1.0   # 本位置结算后的预算余额(审阅 IF-07)
    unlocked_at: float = -1.0    # 本位置时刻的累计解锁额度(pacing 审计曲线)
    draft_token: int = -1        # draft 提案 token(观察:教师把什么改成了什么)
    final_token: int = -1        # 实际执行 token(= draft 或残差重采结果)
    final_token_pi_rank: int = -1  # 重采 token 在 π 观测窗内的 rank(仅拒绝位;
                                 # 冷起边界观察:贴近 obs_topk = 教师踩保尾核边界)
    # viz 下钻数据(v3.2 起仅**拒绝位**携带,防 always_on 全开 dump 膨胀;
    # [[token_id, logprob]×16],两侧分布对比)
    pi_top16: list | None = None
    pt_top16: list | None = None
    # v3.3 M4 软强度门观察面
    intensity: float = 1.0       # w(η = η_max·w;硬门档恒 1/0)
    entropy: float = -1.0        # 该位置的学生截断熵(-1 = 未计算,硬门档)


#: D 直方图口径:D = Var_{π|S}[ℓ̃] ∈ [0, c²]。bins 数固定;上界随 run 的 c
#: 派生(d_hist_max = c²,审阅 IF-12——硬编码 6.25 会在 c>2.5 时把尾部
#: 全压末桶还输出伪精确分位数)。上界随 finalize 落盘,标定聚合时校验一致。
D_HIST_BINS = 2048   # 2026-08-04 加密 64→256:实测 D 的 95% 质量 < 0.1,而上界
                    # = c² = 6.25 —— 64 桶时主质量挤在前 2 桶(0.098/桶),P85 标定
                    # 3 步实测 P85≈0.02 全挤在最前桶,2048 桶(0.003/桶)才够分辨。
                    # dump 体积 ~16KB/轨迹,可接受

#: ℓ 观测直方图(未截断的 log p_T − log π,S 内 p_T 观测窗内逐 token 等权):
#: c 的目标语境标定数据面(候选 c 的截断饱和率 P(|ℓ|>c) 直接可读)+ 教师
#: 异议方向观察(正半区=抬升/负半区=压制)。固定对称范围,双侧溢出计入端桶。
ELL_HIST_BINS = 96
ELL_HIST_RANGE = 12.0   # 覆盖 [-12, 12];分辨率 0.25/桶,c∈[2,3] 标定绰绰有余

#: 学生截断熵直方图(v3.3 M4 G-E 门的阈值追踪数据源;h = −Σ_{topK} p·logp,
#: 观测窗内截断口径——标定与运行同口径即自洽)。上界取 8.0:ln(256)≈5.55,
#: 双侧留量;溢出计末桶。
ENT_HIST_BINS = 1024
ENT_HIST_MAX = 8.0


@dataclass
class SteeringLedger:
    """一条轨迹的记账。finalize() 输出进 extra_fields 的紧凑结构。"""

    records: list[TokenSteerRecord] = field(default_factory=list)
    budget_spent: float = 0.0
    turns_capped: int = 0
    # D 观测直方图:每个候选位置(无论门开关)一次 observe_d —— θ_D 标定与
    # 门率漂移监控的数据源(逐值存储在 32k 轨迹上不可承受,直方图分位数够用)。
    d_hist: list[int] = field(default_factory=lambda: [0] * D_HIST_BINS)
    d_hist_max: float = 6.25          # = c²,由构造方(hook state)按 run 的 c 传入
    d_count: int = 0
    d_sum: float = 0.0
    # 计数器组(IF-07:率类审计的全量通道;稀疏行之外的位置由此重建)
    n_candidates: int = 0             # 进入判定的位置数(= observe_d 次数)
    n_base_open: int = 0
    n_probe_flipped: int = 0
    n_budget_blocked: int = 0         # 门开但预算不允许(η 缩到 0)
    n_bypassed: int = 0               # cap/预算耗尽后未判定直通的 token 数
    n_pacing_wait: int = 0            # pacing 额度暂未解锁而直通的 token 数(非耗尽)
    n_gate_bypassed: int = 0          # v3.3 M4:强度不足(段内全部 w<w_min)未
                                      # verify 直通的 token 数(G-E 工程红利面)
    n_pt_window_miss: int = 0         # S 内 p_T 窗外单侧鲁棒代入次数(max-IF-13)
    # ℓ 观测直方图(c 标定 + 方向观察;窗内逐 token 等权)
    ell_hist: list[int] = field(default_factory=lambda: [0] * ELL_HIST_BINS)
    ell_count: int = 0
    # 逐候选位置的 (position, D) 轻量全量通道(viz:「为什么没触发」可视;
    # ~15KB/轨迹。直方图管分布,这里管逐位置定位。)
    cand_positions: list[int] = field(default_factory=list)
    d_all: list[float] = field(default_factory=list)
    # 学生截断熵直方图(v3.3 M4:G-E 阈值 H_q 的 tracker 数据源;每个生成
    # 位置一次 observe_entropy——含 bypass 段,熵在 draft 侧免费)
    ent_hist: list[int] = field(default_factory=lambda: [0] * ENT_HIST_BINS)
    ent_count: int = 0
    ent_sum: float = 0.0

    def observe_d(self, d_value: float, position: int = -1) -> None:
        b = int(d_value / self.d_hist_max * D_HIST_BINS)
        self.d_hist[min(max(b, 0), D_HIST_BINS - 1)] += 1
        self.d_count += 1
        self.d_sum += float(d_value)
        self.n_candidates += 1
        if position >= 0:
            self.cand_positions.append(int(position))
            self.d_all.append(round(float(d_value), 4))

    def observe_entropy(self, h: float) -> None:
        """每个生成位置一次(bypass 段也观察——熵来自 draft,免费)。"""
        b = int(h / ENT_HIST_MAX * ENT_HIST_BINS)
        self.ent_hist[min(max(b, 0), ENT_HIST_BINS - 1)] += 1
        self.ent_count += 1
        self.ent_sum += float(h)

    def observe_ell(self, raw_ell_values) -> None:
        """S 内 p_T 观测窗内 token 的未截断 ℓ,逐 token 等权入直方图。"""
        half = ELL_HIST_BINS / 2.0
        for v in raw_ell_values:
            b = int(half + float(v) / ELL_HIST_RANGE * half)
            self.ell_hist[min(max(b, 0), ELL_HIST_BINS - 1)] += 1
            self.ell_count += 1

    def append(self, rec: TokenSteerRecord) -> None:
        if self.records and rec.position <= self.records[-1].position:
            raise ValueError("ledger positions must be strictly increasing")
        self.records.append(rec)

    def finalize(self) -> dict[str, Any]:
        """紧凑数组形式（extra_fields 可序列化；训练端按 position 对齐）。

        约定：只存 steer 相关位置的稀疏记录 + 全轨迹汇总；未记录位置训练端
        默认 log_b == log_pi_old（重算路径），W_beh = 1。
        """
        return {
            "steer_positions": [r.position for r in self.records],
            "steer_log_pi": [r.log_pi for r in self.records],
            "steer_log_b": [r.log_b for r in self.records],
            "steer_eta": [r.eta for r in self.records],
            "steer_delta": [r.delta for r in self.records],
            "steer_d": [r.d_value for r in self.records],
            "steer_base_open": [int(r.base_open) for r in self.records],
            "steer_probe": [int(r.probe_flipped) for r in self.records],
            "steer_propensity": [r.propensity for r in self.records],
            "steer_rejected": [int(r.rejected) for r in self.records],
            "steer_cap_active": [int(r.cap_active) for r in self.records],
            # b_max=inf(maxopen)下余额/解锁额无语义,写 −1 哨兵防 Infinity
            # 进 JSON/曲线(math.isfinite 判空,inf 与 −1 缺省同归)。
            "steer_budget_after": [
                round(r.budget_after, 4) if math.isfinite(r.budget_after) else -1.0
                for r in self.records
            ],
            "steer_unlocked_at": [
                round(r.unlocked_at, 4) if math.isfinite(r.unlocked_at) else -1.0
                for r in self.records
            ],
            "steer_budget_spent": self.budget_spent,
            "steer_turns_capped": self.turns_capped,
            # steer_count = 实际倾斜位置数(η>0),不是记录行数——记录行还含
            # 「门开但预算阻断」「探针翻转」的零 η 行(审阅 IF-07/max-IF-03)
            "steer_count": sum(1 for r in self.records if r.eta > 0),
            "steer_record_rows": len(self.records),
            "steer_reject_count": sum(1 for r in self.records if r.rejected),
            "steer_d_hist": list(self.d_hist),
            "steer_d_hist_max": self.d_hist_max,
            "steer_d_hist_bins": len(self.d_hist),
            "steer_d_count": self.d_count,
            "steer_d_mean": (self.d_sum / self.d_count) if self.d_count else 0.0,
            "steer_n_candidates": self.n_candidates,
            "steer_n_base_open": self.n_base_open,
            "steer_n_probe_flipped": self.n_probe_flipped,
            "steer_n_budget_blocked": self.n_budget_blocked,
            "steer_n_bypassed": self.n_bypassed,
            "steer_n_pacing_wait": self.n_pacing_wait,
            "steer_n_gate_bypassed": self.n_gate_bypassed,
            "steer_n_pt_window_miss": self.n_pt_window_miss,
            "steer_draft_token": [r.draft_token for r in self.records],
            "steer_final_token": [r.final_token for r in self.records],
            "steer_final_pi_rank": [r.final_token_pi_rank for r in self.records],
            "steer_ell_hist": list(self.ell_hist),
            "steer_ell_count": self.ell_count,
            # viz 全量通道与下钻数据(2026-08-04 实验侧规格)
            "steer_cand_positions": list(self.cand_positions),
            "steer_d_all": list(self.d_all),
            "steer_pi_top16": [r.pi_top16 or [] for r in self.records],
            "steer_pt_top16": [r.pt_top16 or [] for r in self.records],
            # v3.3 M4 软强度观察面(行级 w/熵 + 全量熵直方图 + 曲线标量:
            # 标量键自动进 rollout_extra 曲线——倾斜位平均强度与全位置平均熵)
            "steer_intensity": [round(r.intensity, 4) for r in self.records],
            "steer_entropy": [round(r.entropy, 4) for r in self.records],
            "steer_ent_hist": list(self.ent_hist),
            "steer_ent_count": self.ent_count,
            "steer_ent_hist_bins": len(self.ent_hist),
            "steer_ent_hist_max": ENT_HIST_MAX,
            "steer_intensity_mean": (
                sum(r.intensity for r in self.records) / len(self.records)
                if self.records else 0.0
            ),
            "steer_entropy_mean": (
                self.ent_sum / self.ent_count if self.ent_count else 0.0
            ),
            # 轨迹级实测门率(swanlab 单曲线;分子分母另有计数器可复核)
            "steer_gate_open_rate": (
                self.n_base_open / self.n_candidates if self.n_candidates else 0.0
            ),
        }
