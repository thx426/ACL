"""保尾核（tail-preserving steering kernel）—— PGS 定稿规范 §2.1/§2.2。

行为分布的唯一定义对象（复审 NF01 修法）：位置 j 在看到 draft token 之前
固定候选集 S，S 内按截断文档增量做几何倾斜重分配，S 外恒等于学生分布 π::

    b(a) = M · π(a)·exp(η·ℓ̃(a)) / A      (a ∈ S)
    b(a) = π(a)                            (a ∉ S)
    M = Σ_{S} π(a)，  A = Σ_{S} π(a)·exp(η·ℓ̃(a))，  ℓ̃ = clip(log p_T − log π, ±c)

性质（构造保证，均有断言/测试）：
  * 全词表正规（S 内重分配保质量 M，S 外原样）；
  * 密度比 e^{−2ηc} ≤ b/π ≤ e^{2ηc} 全词表成立（S 外恒 1）；
  * η=0 ⇒ b ≡ π；
  * speculative 拒绝只可能发生在 S 内（S 外接受率恒 1），残差支撑 ⊆ S。

所有概率入口均为 log 域（fp 下溢安全）；S 上的量以 float64 计算。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["SteeringKernel", "truncated_increment"]


def truncated_increment(log_p_teacher: np.ndarray, log_p_student: np.ndarray, c: float) -> np.ndarray:
    """ℓ̃ = clip(log p_T − log π, −c, +c)。规范 §2.2（单点话语权限幅）。"""
    if c <= 0:
        raise ValueError(f"truncation c must be positive, got {c}")
    return np.clip(
        np.asarray(log_p_teacher, dtype=np.float64) - np.asarray(log_p_student, dtype=np.float64),
        -c,
        c,
    )


@dataclass
class SteeringKernel:
    """一个位置的保尾核。构造后不可变（speculative 定理要求固定目标对象）。

    Args:
        token_ids: 候选集 S 的 token id（去重；顺序即内部索引）。
        log_pi:    π 在 S 上的 logprob（全词表归一化下的真值，来自 verify 前向）。
        ell_tilde: 截断增量 ℓ̃（与 token_ids 对齐；由 :func:`truncated_increment` 产出）。
        eta:       倾斜强度 η ∈ [0, 1)。
        c:         截断参数（用于界断言与 δ 计费）。
    """

    token_ids: np.ndarray
    log_pi: np.ndarray
    ell_tilde: np.ndarray
    eta: float
    c: float

    # 派生量（__post_init__ 填充）
    _index: dict[int, int] = field(init=False, repr=False)
    _pi_S: np.ndarray = field(init=False, repr=False)          # π(a), a∈S
    _b_S: np.ndarray = field(init=False, repr=False)           # b(a), a∈S
    _log_ratio_S: np.ndarray = field(init=False, repr=False)   # log(b/π), a∈S
    mass_M: float = field(init=False)

    def __post_init__(self) -> None:
        self.token_ids = np.asarray(self.token_ids, dtype=np.int64)
        self.log_pi = np.asarray(self.log_pi, dtype=np.float64)
        self.ell_tilde = np.asarray(self.ell_tilde, dtype=np.float64)
        n = len(self.token_ids)
        if not (len(self.log_pi) == len(self.ell_tilde) == n and n > 0):
            raise ValueError("token_ids/log_pi/ell_tilde must be equal-length and non-empty")
        if len(set(self.token_ids.tolist())) != n:
            raise ValueError("token_ids in S must be unique")
        if not (0.0 <= self.eta < 1.0):
            raise ValueError(f"eta must be in [0,1), got {self.eta}")
        if np.any(np.abs(self.ell_tilde) > self.c + 1e-9):
            raise ValueError("ell_tilde exceeds truncation bound c — clip upstream")

        self._index = {int(t): i for i, t in enumerate(self.token_ids)}
        self._pi_S = np.exp(self.log_pi)
        self.mass_M = float(self._pi_S.sum())
        if not (0.0 < self.mass_M <= 1.0 + 1e-6):
            raise ValueError(f"π mass on S must lie in (0,1], got {self.mass_M}")

        # A = Σ π·e^{ηℓ̃}，log 域稳定计算
        log_terms = self.log_pi + self.eta * self.ell_tilde
        log_A = _logsumexp(log_terms)
        # log(b/π) on S = log M + η·ℓ̃ − log A（与 a 无关的常数部分 + 逐点项）
        self._log_ratio_S = np.log(self.mass_M) + self.eta * self.ell_tilde - log_A
        self._b_S = self._pi_S * np.exp(self._log_ratio_S)

        # 不变量（规范 §2.1 性质；构造即校验）
        bound = 2.0 * self.eta * self.c + 1e-9
        assert np.all(np.abs(self._log_ratio_S) <= bound), "density-ratio bound violated"
        assert abs(float(self._b_S.sum()) - self.mass_M) < 1e-9, "b must preserve S mass"

    # ------------------------------------------------------------- queries

    def in_support(self, token_id: int) -> bool:
        return int(token_id) in self._index

    def log_ratio(self, token_id: int) -> float:
        """log(b/π)(token)。S 外恒 0（保尾）。"""
        i = self._index.get(int(token_id))
        return 0.0 if i is None else float(self._log_ratio_S[i])

    def accept_prob(self, token_id: int) -> float:
        """speculative 接受概率 min(1, b/π)。S 外恒 1。"""
        return min(1.0, float(np.exp(self.log_ratio(token_id))))

    def log_b(self, token_id: int, log_pi_token: float) -> float:
        """log b(token) = log π(token) + log_ratio(token)。

        S 外的 token 由调用方提供其 log π（verify 前向的实现 token logprob），
        本核只贡献 ratio=0 —— 保尾性质使 S 外记账无需知道全词表。
        """
        return float(log_pi_token) + self.log_ratio(token_id)

    # ------------------------------------------------------- residual / gate

    def residual_distribution(self) -> tuple[np.ndarray, np.ndarray]:
        """残差 norm(max(0, b−π))，支撑 ⊆ S（规范 §2.5）。

        Returns:
            (token_ids, probs)。残差总质量为 0 时（η=0）抛错——调用方在
            accept_prob≡1 时不应走到残差。
        """
        raw = np.maximum(0.0, self._b_S - self._pi_S)
        z = float(raw.sum())
        if z <= 0.0:
            raise RuntimeError("empty residual — resample must not be reached when b≡π")
        return self.token_ids.copy(), raw / z

    def d_variance(self) -> float:
        """D = Var_{a~π|S}[ℓ̃(a)]（门控信号；规范 §2.3，保尾核对象上计算）。"""
        w = self._pi_S / self.mass_M
        mean = float(np.dot(w, self.ell_tilde))
        return float(np.dot(w, (self.ell_tilde - mean) ** 2))

    def delta_charge(self) -> float:
        """预算计费 δ = 2ηc（全词表上界保守记账；规范 §2.4，封 V′-max 缺口）。"""
        return 2.0 * self.eta * self.c


def _logsumexp(x: np.ndarray) -> float:
    m = float(np.max(x))
    return m + float(np.log(np.sum(np.exp(x - m))))
