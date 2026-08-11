"""Formula recalculation before grading — the official pipeline's step that
the first-generation port dropped.

官方评估(RUCKBReasoning/SpreadsheetBench ``evaluation/open_spreadsheet.py``)
在比对前会把每个产出 workbook 打开重算:Linux/macOS 走 LibreOffice headless
``--convert-to "xlsx:Calc MS Excel 2007 XML"``,Windows 走 Excel COM——公式
在打开时被计算、值写入缓存,然后才用 openpyxl ``data_only=True`` 比对。
skills_evolve 的移植漏掉了这一步 ⇒ 公式解法结构性必挂(48 探针台账是该
缺陷下的 workaround 考古)。第二代(experience_internalization_claude,
用户裁定 2026-07-16「对齐官方;公式解法必须可判分」)补齐,本文件自其
``recalc.py`` 移植适配。**写公式与写静态值都是合法解法。**

纪律(继承第二代实测经验):
- 重算只作用于评分用的临时字节,episode 沙箱的权威文件不动;
- 便宜预检(zip 扫 worksheet XML 的 ``<f`` 标签)先行,无公式的 workbook
  不付 LibreOffice 往返;
- 每次 soffice 调用独立 ``-env:UserInstallation`` profile(并发评估不抢
  默认 profile 锁);
- ``SAL_DISABLE_OPENCL=1`` + ``SAL_DISABLESKIA=1`` 强制纯 CPU:LibreOffice
  默认会抓 ~412MiB 显存(第二代在线实测),在推理引擎吃满 GPU 的训练机上
  是潜在 OOM 因素;
- 重算失败降级为按原字节评分并响亮记日志(官方脚本对失败同样跳过),
  绝不炸 episode;soffice 完全缺失则在装配期 fail loud(见 factory)。
"""
from __future__ import annotations

import io
import logging
import os
import shutil
import subprocess
import tempfile
import urllib.error
import urllib.request
import zipfile
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger(__name__)


class RecalcError(RuntimeError):
    pass


def recalc_remote_url() -> str | None:
    """Windows Excel 重算服务地址(部署事实走 env,SSB_RECALC_URL)。

    设置时优先于本地 soffice——用户裁定 2026-08-11:golden 是官方在真实
    Excel 里执行保存的产物,LibreOffice 重算与 Excel 求值语义差(pivot
    重建/函数兼容/#VALUE 传播/bool 存储)造成 SRL 全量 6.9%、SSB 7.0% 的
    unreliable;Excel 重算与 golden 同引擎对齐是根治。服务端见
    experiments/bench_env/win_recalc/excel_recalc_server.py。"""
    url = os.environ.get("SSB_RECALC_URL", "").strip()
    return url or None


def recalculate_remote(data: bytes, *, url: str, timeout_s: float) -> bytes:
    """POST xlsx 字节到 Windows Excel 重算服务,返回重算后的字节。

    协议:POST {url}/recalc,body=xlsx bytes,200=重算后 xlsx bytes,
    非 200 = 失败(body 为错误文本)。"""
    req = urllib.request.Request(
        url.rstrip("/") + "/recalc",
        data=data,
        headers={"Content-Type": "application/octet-stream"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            out = resp.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read()[:300].decode("utf-8", "replace")
        raise RecalcError(f"remote recalc HTTP {exc.code}: {detail}") from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise RecalcError(f"remote recalc unreachable: {exc}") from exc
    if len(out) < 100:
        raise RecalcError(f"remote recalc returned {len(out)} bytes (not a workbook)")
    return out


@lru_cache(maxsize=1)
def find_soffice() -> str | None:
    """``SSB_SOFFICE`` 覆盖(部署事实走 env);否则 PATH 找官方候选名。"""
    override = os.environ.get("SSB_SOFFICE")
    if override:
        return override if Path(override).is_file() else None
    for candidate in ("libreoffice", "soffice"):
        path = shutil.which(candidate)
        if path:
            return path
    return None


def workbook_has_formulas(data: bytes) -> bool:
    """xlsx 的公式存为 worksheet XML 的 ``<f>`` 元素;扫 zip 成员比
    LibreOffice 往返便宜几个数量级(少数 false positive 只是多做一次
    重算,无害)。"""
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            for name in archive.namelist():
                if name.startswith("xl/worksheets/") and name.endswith(".xml"):
                    if b"<f" in archive.read(name):
                        return True
    except (zipfile.BadZipFile, KeyError, OSError):
        return False
    return False


def recalculate(data: bytes, *, soffice: str, timeout_s: float) -> bytes:
    """官方语义:LibreOffice Calc headless 打开(即重算)并另存 xlsx,
    返回新字节。"""
    env = {
        **os.environ,
        "SAL_DISABLE_OPENCL": "1",
        "SAL_DISABLESKIA": "1",
    }
    with tempfile.TemporaryDirectory(prefix="ssb_recalc_") as tmp:
        tmpdir = Path(tmp)
        source = tmpdir / "workbook.xlsx"
        source.write_bytes(data)
        outdir = tmpdir / "out"
        outdir.mkdir()
        profile = tmpdir / "profile"
        try:
            result = subprocess.run(
                [
                    soffice,
                    "--headless",
                    "--calc",
                    f"-env:UserInstallation=file://{profile}",
                    "--convert-to",
                    "xlsx:Calc MS Excel 2007 XML",
                    "--outdir",
                    str(outdir),
                    str(source),
                ],
                capture_output=True,
                text=True,
                timeout=timeout_s,
                env=env,
            )
        except subprocess.TimeoutExpired as exc:
            raise RecalcError(f"soffice timed out after {timeout_s:.0f}s") from exc
        if result.returncode != 0:
            raise RecalcError(
                f"soffice exited {result.returncode}: {result.stderr.strip()[:500]}"
            )
        converted = outdir / "workbook.xlsx"
        if not converted.is_file():
            raise RecalcError(
                f"soffice produced no output ({result.stdout.strip()[:300]})"
            )
        return converted.read_bytes()


def recalculated_for_grading(
    data: bytes, *, soffice: str | None, timeout_s: float, task_id: str
) -> bytes:
    """评分用字节:有公式且有后端 ⇒ 重算;否则原字节(公式 workbook 无法
    重算时响亮记日志——那意味着公式解法将被判 0,是口径事故信号)。

    后端优先级:SSB_RECALC_URL(Windows Excel 服务,与 golden 同引擎)>
    本地 soffice(LibreOffice,实测 SRL 全量 6.9% / SSB 7.0% 任务求值
    漂移)。remote 失败**不**静默回落 soffice——两引擎结果不同,混用会让
    同一任务的 reward 语义随故障漂移;失败按原字节评分并响亮记日志
    (与 soffice 失败同语义)。"""
    if not workbook_has_formulas(data):
        return data
    remote = recalc_remote_url()
    if remote is not None:
        try:
            out = recalculate_remote(data, url=remote, timeout_s=timeout_s)
        except RecalcError as exc:
            logger.error("ssb remote recalc FAILED for task %s: %s", task_id, exc)
            return data
        logger.info(
            "ssb remote recalc ok for task %s (%d -> %d bytes)",
            task_id, len(data), len(out),
        )
        return out
    if soffice is None:
        logger.error(
            "ssb recalc UNAVAILABLE for task %s: workbook has formulas but no "
            "recalc backend (neither SSB_RECALC_URL nor LibreOffice) — "
            "formula answers will grade as misses",
            task_id,
        )
        return data
    try:
        out = recalculate(data, soffice=soffice, timeout_s=timeout_s)
    except RecalcError as exc:
        logger.error("ssb recalc FAILED for task %s: %s", task_id, exc)
        return data
    logger.info(
        "ssb recalc ok for task %s (%d -> %d bytes)", task_id, len(data), len(out)
    )
    return out
