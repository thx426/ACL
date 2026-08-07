#!/usr/bin/env python3
"""162 dynamic-researcher max-open run launcher.

Credentials are read only from stdin as a two-line envelope and inherited by the
child process. They never appear in argv, files, manifests, or this script.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = Path(
    os.environ.get(
        "PGS_RESEARCHER_MAXOPEN_BUNDLE",
        str(ROOT.parent if ROOT.name == "trace_opd" else ROOT),
    )
)
PYTHON = Path("/data4/ws/PGS/env_sglang_train_1p7b/bin/python")
RUNS_ROOT = Path("/data4/ws/PGS")
LOG_ROOT = BUNDLE / "logs"
CALIB_CONFIG = ROOT / "configs/experiments/162-pgs-s2-sglang-1p7b-researcher-maxopen-calibration.yaml"
FORMAL_CONFIG = ROOT / "configs/experiments/162-pgs-s2-sglang-1p7b-researcher-maxopen.yaml"
CALIB_RUN = "pgs_s2_sgl_1p7b_162_researcher_maxopen_calib_p80"
SMOKE_RUN = "pgs_s2_sgl_1p7b_162_researcher_maxopen_smoke"
FORMAL_RUN = "pgs_s2_sgl_1p7b_162_researcher_maxopen"
ALLOWED_MODES = ("dry-run-calibrate", "calibrate", "materialize", "dry-run", "smoke", "formal", "resume")


def _read_credentials() -> tuple[str, str]:
    researcher = sys.stdin.readline().rstrip("\n")
    swanlab = sys.stdin.readline().rstrip("\n")
    if not researcher or not swanlab:
        raise SystemExit("FATAL: credential envelope needs non-empty researcher and SwanLab lines")
    return researcher, swanlab


def _base_env(*, dry_run: bool) -> dict[str, str]:
    env = dict(os.environ)
    env.update(
        {
            "PYTHONUNBUFFERED": "1",
            "PGS_REPO": str(BUNDLE),
            "PGS_ROOT": str(ROOT),
            "VENV_TRAIN": str(PYTHON.parent.parent),
            "VENV_INFER": str(PYTHON.parent.parent),
            "VENV_ALFWORLD": "/data3/wushang/workspace/TRACE-OPD/venv_alfworld",
            "PYTHONPATH": f"{ROOT / 'src'}:{ROOT / 'verl'}",
            "CUDA_HOME": str(PYTHON.parent.parent),
            "CUDACXX": str(PYTHON.parent / "nvcc"),
            "CC": str(PYTHON.parent / "x86_64-conda-linux-gnu-gcc"),
            "CXX": str(PYTHON.parent / "x86_64-conda-linux-gnu-g++"),
            "CUDAHOSTCXX": str(PYTHON.parent / "x86_64-conda-linux-gnu-g++"),
            "TRITON_PTXAS_PATH": str(PYTHON.parent / "ptxas"),
            "CPATH": str(PYTHON.parent.parent / "targets/x86_64-linux/include"),
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "CUDA_VISIBLE_DEVICES": env.get("CUDA_VISIBLE_DEVICES", "0,1,2,3"),
            "TORCH_NCCL_BLOCKING_WAIT": "1",
            "ALFWORLD_DATA": "/data3/wushang/data/alfworld_data",
            "ALFWORLD_WORKER_PYTHON": "/data3/wushang/workspace/TRACE-OPD/venv_alfworld/bin/python",
            "TRACE_MODEL_PATH": "/data3/wushang/model/Qwen/Qwen3-1.7B",
            "TRACE_TRAIN_FILE": "/data3/wushang/workspace/TRACE-OPD/trace_opd-pgc/data/rl_train_v2.parquet",
            "TRACE_VAL_SEEN_FILE": "/data3/wushang/workspace/TRACE-OPD/trace_opd/data/out/valid_seen.parquet",
            "TRACE_VAL_UNSEEN_FILE": "/data3/wushang/workspace/TRACE-OPD/trace_opd/data/out/valid_unseen.parquet",
        }
    )
    prefix = PYTHON.parent.parent
    libraries = [prefix / "lib", prefix / "targets/x86_64-linux/lib"]
    env["PATH"] = f"{PYTHON.parent}:{env.get('PATH', '')}"
    env["LD_LIBRARY_PATH"] = ":".join(str(path) for path in libraries) + (
        f":{env['LD_LIBRARY_PATH']}" if env.get("LD_LIBRARY_PATH") else ""
    )
    env["LIBRARY_PATH"] = ":".join(
        str(path)
        for path in libraries + [prefix / "targets/x86_64-linux/lib/stubs"]
    ) + (f":{env['LIBRARY_PATH']}" if env.get("LIBRARY_PATH") else "")
    env["NVCC_PREPEND_FLAGS"] = (
        f"-ccbin={env['CXX']}" +
        (f" {env['NVCC_PREPEND_FLAGS']}" if env.get("NVCC_PREPEND_FLAGS") else "")
    )
    env.pop("LD_PRELOAD", None)
    if dry_run:
        env["TRACE_RESEARCHER_API_KEY"] = "dry-run-placeholder"
        env["SWANLAB_API_KEY"] = "dry-run-placeholder"
        env.pop("SWANLAB_MODE", None)
    else:
        researcher, swanlab = _read_credentials()
        env["TRACE_RESEARCHER_API_KEY"] = researcher
        env["SWANLAB_API_KEY"] = swanlab
        mode = env.get("SWANLAB_MODE", "").strip().lower()
        if mode in {"offline", "disabled", "disable"}:
            raise SystemExit("FATAL: formal-grade modes require online SwanLab")
        env.pop("SWANLAB_MODE", None)
    return env


def _config_contract(path: Path, *, calibration: bool) -> dict:
    if not path.is_file():
        raise SystemExit(f"FATAL: config does not exist: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    st = data["steering"]
    train = data["train"]
    role = data["roles"]["researcher"]
    expected = {
        "doc_source": "researcher",
        "b_max": 81920.0,
        "budget_pacing": False,
        "r_max": 32769,
        "epsilon": 0.0,
        "gate_target_rate": 0.20,
        "eta_max": 0.5,
        "c": 2.5,
    }
    for key, value in expected.items():
        if st.get(key) != value:
            raise SystemExit(f"FATAL: config contract steering.{key}={st.get(key)!r}, expected {value!r}")
    if train.get("response_length") != 32768 or data["eval"].get("response_budget") != 32768:
        raise SystemExit("FATAL: response horizon is not 32768")
    if st["b_max"] < train["response_length"] * 2 * st["eta_max"] * st["c"]:
        raise SystemExit("FATAL: b_max can bind before the response horizon")
    if st["r_max"] <= train["response_length"]:
        raise SystemExit("FATAL: r_max can bind before the response horizon")
    if data["research"].get("n_exp") != 16:
        raise SystemExit("FATAL: researcher n_exp is not 16")
    if role.get("extra_body", {}).get("chat_template_kwargs", {}).get("enable_thinking") is not False:
        raise SystemExit("FATAL: researcher thinking is not disabled")
    if calibration:
        if st.get("theta_d") != 1_000_000_000.0 or st.get("gate_adaptive") is not False:
            raise SystemExit("FATAL: calibration config is not a closed-gate sentinel")
    else:
        if not (0 < float(st.get("theta_d", 0)) < 1_000_000_000.0) or st.get("gate_adaptive") is not True:
            raise SystemExit("FATAL: formal config has no materialized adaptive P80")
    return data


def _target_absent(run_name: str) -> Path:
    target = RUNS_ROOT / run_name
    if target.exists():
        raise SystemExit(f"FATAL: target exists; refusing overwrite: {target}")
    return target


def _assert_resume_ready() -> Path:
    target = RUNS_ROOT / FORMAL_RUN
    marker = target / "checkpoints/latest_checkpointed_iteration.txt"
    fingerprint = target / "objective_fingerprint.txt"
    if not marker.is_file() or not fingerprint.is_file():
        raise SystemExit(f"FATAL: no auditable resume point in {target}")
    try:
        step = int(marker.read_text(encoding="utf-8").strip())
    except ValueError as exc:
        raise SystemExit(f"FATAL: invalid checkpoint marker {marker}") from exc
    checkpoint = target / f"checkpoints/global_step_{step}"
    if not (checkpoint / "data.pt").is_file():
        raise SystemExit(f"FATAL: incomplete checkpoint {checkpoint}")
    return target


def _launch_command(config: Path, run_name: str, extra: list[str], *, dry_run: bool) -> list[str]:
    command = [
        str(PYTHON),
        str(ROOT / "scripts/launch_train.py"),
        "--config",
        str(config),
        "--run-name",
        run_name,
    ]
    if dry_run:
        command.append("--dry-run")
    return command + extra


def _start_background(command: list[str], env: dict[str, str], run_name: str, *, append: bool = False) -> int:
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    log_path = LOG_ROOT / f"{run_name}.log"
    pid_path = LOG_ROOT / f"{run_name}.pid"
    if pid_path.exists():
        try:
            old_pid = int(pid_path.read_text(encoding="utf-8").strip())
        except ValueError as exc:
            raise SystemExit(f"FATAL: invalid pid file: {pid_path}") from exc
        cmdline = Path(f"/proc/{old_pid}/cmdline")
        if cmdline.is_file():
            command_text = cmdline.read_bytes().replace(b"\0", b" ").decode(errors="replace")
            if str(ROOT / "scripts/launch_train.py") in command_text and run_name in command_text:
                raise SystemExit(f"FATAL: exact run launcher is already alive as PID {old_pid}")
            raise SystemExit(
                f"FATAL: pid file points to an unrelated live process {old_pid}; refusing overwrite"
            )
        if not append:
            raise SystemExit(f"FATAL: stale pid file exists; refusing ambiguous fresh start: {pid_path}")
        pid_path.unlink()
    with log_path.open("ab" if append else "xb", buffering=0) as stream:
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=stream,
            stderr=subprocess.STDOUT,
            env=env,
            cwd=BUNDLE,
            start_new_session=True,
            close_fds=True,
        )
    pid_path.write_text(f"{process.pid}\n", encoding="utf-8")
    print(f"MODE=start RUN={run_name} PID={process.pid} LOG={log_path}")
    return process.pid


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=ALLOWED_MODES)
    parser.add_argument("extra", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    if os.environ.get("PGS_MAXOPEN_ENTRY_SOURCED") != "1":
        raise SystemExit(
            "FATAL: source scripts/env_162_researcher_maxopen.sh before using this launcher"
        )
    if not PYTHON.is_file() or not (ROOT / "scripts/launch_train.py").is_file():
        raise SystemExit("FATAL: bundle or Python environment is incomplete")

    if args.mode == "materialize":
        if args.extra:
            raise SystemExit("FATAL: materialize takes no extra arguments")
        _config_contract(CALIB_CONFIG, calibration=True)
        audit = BUNDLE / "logs/maxopen_p80_calibration_audit.json"
        command = [
            str(PYTHON),
            str(ROOT / "scripts/materialize_162_researcher_maxopen.py"),
            "--calibration-run", str(RUNS_ROOT / CALIB_RUN),
            "--calibration-config", str(CALIB_CONFIG),
            "--output", str(FORMAL_CONFIG),
            "--audit-output", str(audit),
        ]
        return subprocess.run(command, env=_base_env(dry_run=True), cwd=BUNDLE).returncode

    calibration = args.mode in {"dry-run-calibrate", "calibrate"}
    config = CALIB_CONFIG if calibration else FORMAL_CONFIG
    _config_contract(config, calibration=calibration)
    dry_run = args.mode in {"dry-run-calibrate", "dry-run"}
    env = _base_env(dry_run=dry_run)

    if args.mode == "dry-run-calibrate":
        command = _launch_command(
            config, CALIB_RUN, ["--set", "train.loggers=[console]", *args.extra], dry_run=True
        )
        return subprocess.run(command, env=env, cwd=BUNDLE).returncode
    if args.mode == "dry-run":
        command = _launch_command(config, FORMAL_RUN, args.extra, dry_run=True)
        return subprocess.run(command, env=env, cwd=BUNDLE).returncode
    if args.mode == "calibrate":
        _target_absent(CALIB_RUN)
        command = _launch_command(
            config,
            CALIB_RUN,
            [
                "--set", "train.total_epochs=1",
                "--set", "train.val_before_train=false",
                "--set", "train.test_freq=-1",
                "--set", "train.save_freq=-1",
                "--set", "train.resume_mode=disable",
                "--set", "train.loggers=[console]",
                "trainer.total_training_steps=1",
                *args.extra,
            ],
            dry_run=False,
        )
        _start_background(command, env, CALIB_RUN)
        return 0
    if args.mode == "smoke":
        _target_absent(SMOKE_RUN)
        command = _launch_command(
            config,
            SMOKE_RUN,
            [
                "--set", "train.total_epochs=1",
                "--set", "train.val_before_train=false",
                "--set", "train.test_freq=-1",
                "--set", "train.save_freq=1",
                "--set", "train.resume_mode=disable",
                "--set", "train.loggers=[console]",
                "trainer.total_training_steps=1",
                *args.extra,
            ],
            dry_run=False,
        )
        _start_background(command, env, SMOKE_RUN)
        return 0
    if args.mode == "formal":
        _target_absent(FORMAL_RUN)
        command = _launch_command(config, FORMAL_RUN, args.extra, dry_run=False)
        _start_background(command, env, FORMAL_RUN)
        return 0
    if args.mode == "resume":
        _assert_resume_ready()
        command = _launch_command(config, FORMAL_RUN, args.extra, dry_run=False)
        _start_background(command, env, FORMAL_RUN, append=True)
        return 0
    raise AssertionError(args.mode)


if __name__ == "__main__":
    raise SystemExit(main())
