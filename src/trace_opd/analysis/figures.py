"""The six paper figures of the distillation microscope (+ fork appendix).

Pure functions: schema rows in, matplotlib Figure out — saving/stamping is the
caller's job (style.save / style.mark_synthetic). Encodings are pinned by
analysis-design.md §3; change them there first.
"""
from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.gridspec import GridSpec

from .schema import ActionFlow, FingerprintCase, ForkStats, LedgerPoint, ProbePoint, TokenRecord, TurnProfile
from .style import DIVERGING_CMAP, GOLDEN, OKABE_ITO, SEQUENTIAL_CMAP, SINGLE, style

TOKEN_TYPES = ("action-verb", "action-arg", "reasoning-connective", "format")
_TYPE_COLORS = {
    "action-verb": OKABE_ITO["blue"],
    "action-arg": OKABE_ITO["green"],
    "reasoning-connective": OKABE_ITO["orange"],
    "format": OKABE_ITO["purple"],
}


# ------------------------------------------------- Fig A: internalization ledger


def fig_ledger(points: list[LedgerPoint], events: dict[int, str] | None = None):
    points = sorted(points, key=lambda p: p.step)
    steps = [p.step for p in points]
    with style():
        fig, (ax1, ax2) = plt.subplots(
            2, 1, sharex=True, figsize=(SINGLE, SINGLE * 1.1),
            height_ratios=[1, 1], constrained_layout=True,
        )
        priv = [p.priv_success for p in points]
        score = [p.score for p in points]
        ax1.plot(steps, priv, color=OKABE_ITO["vermillion"], ls="--",
                 label="doc-privileged")
        ax1.plot(steps, score, color=OKABE_ITO["blue"], label="bare policy")
        ax1.fill_between(steps, score, priv, color="0.85", zorder=0)
        ax1.set_ylabel("success rate")
        ax1.set_ylim(0, 1.02)
        ax1.legend(loc="upper left")

        ax2.plot(steps, [p.g for p in points], color=OKABE_ITO["black"],
                 label=r"$G_k$ (current doc)")
        ax2.plot(steps, [p.g_frozen for p in points], color=OKABE_ITO["skyblue"],
                 label=r"$G_k^{\mathrm{frozen}}$")
        ax2.set_ylabel("residual privileged\ninfo (nats/token)")
        ax2.set_xlabel("training step")
        ax2.set_ylim(bottom=0)
        ax2.legend(loc="upper right")

        for ax in (ax1, ax2):
            for s, label in (events or {}).items():
                ax.axvline(s, color="0.6", lw=0.7, ls=":")
        for s, label in (events or {}).items():
            ax1.annotate(label, (s, ax1.get_ylim()[1]), fontsize=6,
                         ha="center", va="bottom", color="0.4")
    return fig


# ---------------------------------------- Fig B: internalization trajectory


def fig_trajectory(arms: dict[str, list[ProbePoint]], y_attr: str = "d_frozen"):
    """x = KL(pi_k || pi_0), y = residual info; time as color. Healthy runs
    move lower-right; exhaustion/drift hooks back up-right."""
    markers = ("o", "s", "^", "D")
    all_steps = [p.step for pts in arms.values() for p in pts]
    norm = Normalize(min(all_steps), max(all_steps))
    with style():
        fig, ax = plt.subplots(figsize=(SINGLE, SINGLE * 0.85), constrained_layout=True)
        for (name, pts), marker in zip(arms.items(), markers):
            pts = sorted(pts, key=lambda p: p.step)
            xs = [p.d_init for p in pts]
            ys = [getattr(p, y_attr) for p in pts]
            ax.plot(xs, ys, color="0.75", lw=0.8, zorder=1)
            sc = ax.scatter(xs, ys, c=[p.step for p in pts], cmap=SEQUENTIAL_CMAP,
                            norm=norm, s=14, marker=marker, zorder=2,
                            label=name, edgecolors="none")
        cbar = fig.colorbar(sc, ax=ax, pad=0.02)
        cbar.set_label("training step")
        ax.set_xlabel(r"$d_{\mathrm{init}} = \mathrm{KL}(\pi_k\,\|\,\pi_0)$ (nats)")
        ax.set_ylabel(
            r"$G^{\mathrm{frozen}}$ (nats/token)" if y_attr == "d_frozen"
            else r"$d_{\mathrm{gain}}$ (nats/token)"
        )
        ax.set_ylim(bottom=0)
        ax.legend(loc="upper right")
        ax.annotate("internalization", xy=(0.60, 0.04), xycoords="axes fraction",
                    xytext=(0.16, 0.16), textcoords="axes fraction",
                    fontsize=6, color="0.45",
                    arrowprops={"arrowstyle": "->", "color": "0.45", "lw": 0.8})
    return fig


# ------------------------------------------------- Fig C: depth terrain


def _terrain_matrix(profiles: list[TurnProfile], gated: bool):
    steps = sorted({p.step for p in profiles})
    turns = sorted({p.turn for p in profiles})
    acc: dict[tuple[int, int], list[float]] = {}
    for p in profiles:
        acc.setdefault((p.step, p.turn), []).append(p.g_gated if gated else p.g_mean)
    mat = np.full((len(steps), len(turns)), np.nan)
    for (s, t), vals in acc.items():
        mat[steps.index(s), turns.index(t)] = float(np.mean(vals))
    return np.array(steps), np.array(turns), np.ma.masked_invalid(mat)


def fig_depth_terrain(profiles: list[TurnProfile], gated: bool = False,
                      title: str | None = None):
    steps, turns, mat = _terrain_matrix(profiles, gated)
    with style():
        fig = plt.figure(figsize=(SINGLE * 1.35, SINGLE * 0.8), constrained_layout=True)
        gs = GridSpec(1, 2, figure=fig, width_ratios=[3.0, 1.35])
        ax = fig.add_subplot(gs[0])
        im = ax.imshow(
            mat, origin="lower", aspect="auto", cmap=SEQUENTIAL_CMAP,
            extent=[turns.min() - 0.5, turns.max() + 0.5, steps.min(), steps.max()],
        )
        ax.set_xlabel(r"turn depth $\tau$")
        ax.set_ylabel("training step")
        if title:
            ax.set_title(title)
        cbar = fig.colorbar(im, ax=ax, pad=0.02, fraction=0.08)
        cbar.set_label(
            r"$G_\tau$ gated (nats/token)" if gated else r"$G_\tau$ (nats/token)"
        )

        axm = fig.add_subplot(gs[1])
        n = max(1, len(steps) // 5)
        early = np.nanmean(mat[:n].filled(np.nan), axis=0)
        late = np.nanmean(mat[-n:].filled(np.nan), axis=0)
        axm.plot(turns, early, color=OKABE_ITO["skyblue"],
                 label=f"early (first {n})")
        axm.plot(turns, late, color=OKABE_ITO["vermillion"],
                 label=f"late (last {n})")
        axm.set_xlabel(r"turn depth $\tau$")
        axm.set_ylabel(r"$G_\tau$")
        axm.set_ylim(bottom=0)
        axm.legend(loc="upper right")
    return fig


# ---------------------------------------------- Fig D: token microscope


def fig_token_microscope(track: list[TokenRecord], steps: list[int],
                         type_shares: dict[str, list[float]]):
    track = sorted(track, key=lambda r: r.t)
    ratios = np.array([[r.logratio for r in track]])
    vmax = max(1e-6, float(np.abs(ratios).max()))
    with style():
        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(SINGLE * 1.35, SINGLE * 0.95),
            height_ratios=[1, 1.6], constrained_layout=True,
        )
        im = ax1.imshow(ratios, aspect="auto", cmap=DIVERGING_CMAP,
                        norm=TwoSlopeNorm(0, -vmax, vmax),
                        extent=[track[0].t, track[-1].t + 1, 0, 1])
        for a, b in zip(track, track[1:]):
            if b.turn != a.turn:
                ax1.axvline(b.t, color="0.25", lw=0.5)
        action_ts = [r.t + 0.5 for r in track
                     if r.token_type in ("action-verb", "action-arg")]
        ax1.scatter(action_ts, [1.06] * len(action_ts), marker="v", s=6,
                    color=OKABE_ITO["black"], clip_on=False)
        ax1.set_yticks([])
        ax1.set_xlabel("token position (turn boundaries marked)")
        cbar = fig.colorbar(im, ax=ax1, pad=0.02, fraction=0.06)
        cbar.set_label(r"$\log p_T - \log p_S$")

        order = [t for t in TOKEN_TYPES if t in type_shares]
        ax2.stackplot(steps, [type_shares[t] for t in order],
                      labels=order, colors=[_TYPE_COLORS[t] for t in order],
                      alpha=0.9)
        ax2.set_xlabel("training step")
        ax2.set_ylabel("share of KL mass")
        ax2.set_ylim(0, 1)
        ax2.legend(loc="upper right", ncols=2)
    return fig


# ---------------------------------------------- Fig E: action flow


def fig_action_flow(flow: ActionFlow):
    groups = list(flow.student_probs)
    colors = [list(OKABE_ITO.values())[i % 8] for i in range(len(groups))]
    with style():
        fig, (ax1, ax2) = plt.subplots(
            2, 1, sharex=True, figsize=(SINGLE, SINGLE * 1.1),
            constrained_layout=True,
        )
        ax1.stackplot(flow.steps, [flow.student_probs[g] for g in groups],
                      labels=groups, colors=colors, alpha=0.9)
        ax1.set_ylabel("bare policy\naction prob.")
        ax1.set_ylim(0, 1)
        ax1.legend(loc="upper right", ncols=2)
        for g, c in zip(groups, colors):
            if g in flow.doc_uplift:
                ax2.plot(flow.steps, flow.doc_uplift[g], color=c, label=g)
        ax2.axhline(0, color="0.7", lw=0.6)
        ax2.set_ylabel("doc uplift\n" + r"$\Delta$prob.")
        ax2.set_xlabel("training step")
    return fig


# ---------------------------------------------- Fig F: alignment fingerprint


def fig_fingerprint(sweep: list[ProbePoint], case: FingerprintCase):
    pts = sorted((p for p in sweep if not np.isnan(p.align)), key=lambda p: p.step)
    with style():
        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(SINGLE, SINGLE * 1.15),
            height_ratios=[1, 1.3], constrained_layout=True,
        )
        steps = [p.step for p in pts]
        ax1.plot(steps, [p.align for p in pts], color=OKABE_ITO["blue"],
                 label="cosine fingerprint")
        ax1.plot(steps, [p.align_robust for p in pts], color=OKABE_ITO["green"],
                 ls="--", label="uplift-set overlap")
        for key, color in (("align", OKABE_ITO["blue"]),):
            lo_hi = [p.ci.get(key) for p in pts]
            if all(lo_hi):
                ax1.fill_between(steps, [c[0] for c in lo_hi],
                                 [c[1] for c in lo_hi], color=color, alpha=0.15,
                                 lw=0)
        ax1.axhline(0, color="0.7", lw=0.6)
        ax1.set_ylabel(r"align($\Delta_{doc}, \Delta_{train}$)")
        ax1.set_xlabel("training step")
        ax1.legend(loc="lower right")

        pos = np.arange(len(case.labels))
        ax2.barh(pos + 0.19, case.delta_doc, height=0.36,
                 color=OKABE_ITO["vermillion"], label=r"$\Delta_{doc}$")
        ax2.barh(pos - 0.19, case.delta_train, height=0.36,
                 color=OKABE_ITO["blue"], label=r"$\Delta_{train}$")
        ax2.set_yticks(pos, case.labels)
        ax2.invert_yaxis()
        ax2.axvline(0, color="0.7", lw=0.6)
        ax2.set_xlabel(r"$\Delta$ probability")
        if case.context:
            ax2.set_title(case.context, fontsize=7)
        ax2.legend(loc="lower right")
    return fig


# ---------------------------------------------- appendix: fork analysis


def fig_fork(stats: ForkStats):
    with style():
        fig, (ax1, ax2) = plt.subplots(
            2, 1, sharex=True, figsize=(SINGLE, SINGLE * 1.0),
            constrained_layout=True,
        )
        ax1.plot(stats.steps, stats.fork_rate, color=OKABE_ITO["vermillion"],
                 label="fork rate (top-1 < 0.25)")
        ax1.plot(stats.steps, stats.lock_rate, color=OKABE_ITO["blue"],
                 label="lock rate (top-1 > 0.65)")
        ax1.set_ylabel("share of positions")
        ax1.legend(loc="center right")

        for i, (g, series) in enumerate(stats.fork_uplift.items()):
            ax2.plot(stats.steps, series, color=list(OKABE_ITO.values())[i % 8],
                     label=g)
        ax2.axhline(0, color="0.7", lw=0.6)
        ax2.set_ylabel("doc uplift at forks\n" + r"$\Delta$prob.")
        ax2.set_xlabel("training step")
        ax2.legend(loc="upper right", ncols=2)
    return fig
