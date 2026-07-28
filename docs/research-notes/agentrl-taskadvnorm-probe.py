import torch, math
torch.manual_seed(0)

def grpo_adv(scores, lens, eps=1e-6):
    s = torch.tensor(scores, dtype=torch.float)
    mean, std = s.mean(), s.std()
    a = (s - mean) / (std + eps)
    return [torch.full((L,), a[i].item()) for i, L in enumerate(lens)]

def task_norm_factor(p_success, n_groups=32, n=8, L_win=200, L_lose=800, seed=0):
    """simulate one task: n_groups groups of n rollouts, bernoulli(p) success.
       win traj short, lose traj long (agentic typical)."""
    g = torch.Generator().manual_seed(seed)
    all_adv, eff = [], 0
    for _ in range(n_groups):
        sc = (torch.rand(n, generator=g) < p_success).float().tolist()
        if len(set(sc)) > 1: eff += 1
        lens = [L_win if s == 1 else L_lose for s in sc]
        for t in grpo_adv(sc, lens): all_adv.append(t)
    A = torch.cat(all_adv)
    return A.mean().item(), A.std().item(), eff / n_groups

print(f"{'p_succ':>7} {'q(eff grp)':>11} {'tok-mean mu':>12} {'tok-std sig':>12} {'1/sig (gain)':>13} {'1/sqrt(q)':>10}")
for p in [0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99]:
    mu, sg, q = task_norm_factor(p)
    print(f"{p:7.2f} {q:11.3f} {mu:12.4f} {sg:12.4f} {1/(sg+1e-6):13.3f} {(1/math.sqrt(q) if q>0 else float('nan')):10.3f}")

print("\n--- equal-length control (L_win == L_lose) => mu should be ~0 ---")
for p in [0.3, 0.9]:
    mu, sg, q = task_norm_factor(p, L_win=500, L_lose=500)
    print(f"p={p}: tok-mean mu={mu:+.4f} sig={sg:.4f} q={q:.3f}")
