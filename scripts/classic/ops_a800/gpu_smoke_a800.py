#!/usr/bin/env python
"""A800 硬件修复后八卡冒烟(2026-08-27):真实张量运算,不以 is_available 为判据。
单卡(并行子进程+超时,楔死卡不拖死全局):fp16 matmul 对拍 fp32 参考 / 8192² 吞吐 / HBM 大块写读校验;
多卡:NCCL all_reduce 256MB×5 带 120s 超时 + 结果校验。
用法:python gpu_smoke_a800.py --all            (八卡单卡 + NCCL 0-3)
      python gpu_smoke_a800.py --single 3       (内部用)
      python gpu_smoke_a800.py --nccl           (内部用,按 CUDA_VISIBLE_DEVICES)
"""
import os, sys, json, time, argparse, subprocess

def single(card):
    import torch, signal
    signal.alarm(280)
    t0 = time.time(); dev = torch.device("cuda:0"); torch.cuda.set_device(dev)
    name = torch.cuda.get_device_name(0)
    a = torch.randn(512, 512); b = torch.randn(512, 512); ref = a @ b
    got = (a.to(dev).half() @ b.to(dev).half()).float().cpu()
    rel = ((got - ref).abs().max() / (ref.abs().max() + 1e-6)).item()
    x = torch.randn(8192, 8192, device=dev, dtype=torch.float16); y = torch.randn(8192, 8192, device=dev, dtype=torch.float16)
    z = x @ y; torch.cuda.synchronize(); t = time.time()
    for _ in range(20): z = x @ y
    torch.cuda.synchronize(); dt = time.time() - t
    tflops = 20 * 2 * 8192 ** 3 / dt / 1e12; finite = bool(torch.isfinite(z).all().item())
    del x, y, z; torch.cuda.empty_cache()
    free, total = torch.cuda.mem_get_info(); n = int((free - 3 * (1 << 30)) // (4 << 30)); bufs = []
    for i in range(n):
        t_ = torch.empty(4 << 30, dtype=torch.uint8, device=dev); t_.fill_(i % 251 + 1); bufs.append(t_)
    torch.cuda.synchronize(); bad = 0   # ★ aminmax 在 uint8 上原地归约,不上转(ne().sum() 会把 4GiB bool 转 32GiB int64 → 假 OOM)
    for i, t_ in enumerate(bufs):
        mn, mx = torch.aminmax(t_); bad += int(mn.item() != i % 251 + 1 or mx.item() != i % 251 + 1)
    print(json.dumps(dict(card=card, name=name, rel_err=round(rel, 4), tflops=round(tflops, 1), finite=finite,
                          hbm_swept_gb=4 * n, hbm_bad_chunks=bad, sec=round(time.time() - t0, 1))), flush=True)

def nccl_worker(rank, world):
    import torch, torch.distributed as dist, signal, datetime
    signal.alarm(200)
    os.environ["MASTER_ADDR"] = "127.0.0.1"; os.environ["MASTER_PORT"] = "29571"
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world, timeout=datetime.timedelta(seconds=120))
    x = torch.ones(64 * 1024 * 1024, device="cuda") * (rank + 1)
    dist.all_reduce(x); torch.cuda.synchronize(); t = time.time()
    for _ in range(4): dist.all_reduce(x)
    torch.cuda.synchronize(); dt = time.time() - t
    exp = world * (world + 1) / 2 * world ** 4
    ok = bool(torch.allclose(x, torch.full_like(x, exp)))
    if rank == 0:
        print(json.dumps(dict(nccl_world=world, ok=ok, algbw_GBps=round(4 * 256 / 1024 * 2 * (world - 1) / world / dt, 1), sec=round(dt, 2))), flush=True)
    dist.destroy_process_group()

def nccl_main():
    import torch, torch.multiprocessing as mp
    world = torch.cuda.device_count()
    mp.spawn(nccl_worker, args=(world,), nprocs=world, join=True)

def run_all(cards, nccl_group):
    here = os.path.abspath(__file__); py = sys.executable; procs = {}
    for c in cards:
        env = dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES=str(c))
        procs[c] = subprocess.Popen([py, here, "--single", str(c)], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    fail = 0
    for c, p in procs.items():
        try:
            out, err = p.communicate(timeout=300)
        except subprocess.TimeoutExpired:
            p.kill(); print(f"card {c}: TIMEOUT(楔死?)"); fail += 1; continue
        line = [l for l in out.splitlines() if l.startswith("{")]
        if p.returncode != 0 or not line:
            print(f"card {c}: FAIL rc={p.returncode} {err.strip().splitlines()[-1][:160] if err.strip() else ''}"); fail += 1; continue
        r = json.loads(line[-1]); bad = (r["rel_err"] > 0.02) or (not r["finite"]) or (r["hbm_bad_chunks"] != 0) or (r["tflops"] < 100)
        print(("BAD  " if bad else "ok   ") + json.dumps(r)); fail += int(bad)
    env = dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES=nccl_group)
    try:
        out = subprocess.run([py, here, "--nccl"], env=env, capture_output=True, text=True, timeout=300)
        line = [l for l in out.stdout.splitlines() if l.startswith("{")]
        if out.returncode != 0 or not line:
            print(f"NCCL[{nccl_group}]: FAIL rc={out.returncode} {(out.stderr.strip().splitlines() or [''])[-1][:200]}"); fail += 1
        else:
            r = json.loads(line[-1]); print(("ok   " if r["ok"] else "BAD  ") + f"NCCL[{nccl_group}] " + json.dumps(r)); fail += int(not r["ok"])
    except subprocess.TimeoutExpired:
        print(f"NCCL[{nccl_group}]: TIMEOUT"); fail += 1
    print("SMOKE-RESULT:", "ALL-PASS" if fail == 0 else f"FAIL x{fail}"); sys.exit(1 if fail else 0)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--single", type=int); ap.add_argument("--nccl", action="store_true")
    ap.add_argument("--all", action="store_true"); ap.add_argument("--cards", default="0,1,2,3,4,5,6,7"); ap.add_argument("--nccl-group", default="0,1,2,3")
    a = ap.parse_args()
    if a.single is not None: single(a.single)
    elif a.nccl: nccl_main()
    else: run_all([int(c) for c in a.cards.split(",")], a.nccl_group)
