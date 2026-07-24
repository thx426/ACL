"""Academic plotting standard: ICLR column geometry, colorblind-safe palette,
spare axes, vector-first output. Every figure in the suite goes through
`style()` + `save()` so the paper stays typographically uniform.
"""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

# ICLR/NeurIPS column widths (inches).
SINGLE = 3.35
DOUBLE = 6.75
GOLDEN = 0.618

# Okabe-Ito colorblind-safe palette (order = draw priority).
OKABE_ITO = {
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "green": "#009E73",
    "orange": "#E69F00",
    "skyblue": "#56B4E9",
    "purple": "#CC79A7",
    "yellow": "#F0E442",
    "black": "#000000",
}
CYCLE = list(OKABE_ITO.values())

# Sequential quantities (G terrain) use viridis; diverging (log-ratio,
# centered at 0) use RdBu_r — pinned here so figures never disagree.
SEQUENTIAL_CMAP = "viridis"
DIVERGING_CMAP = "RdBu_r"

_RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 8,
    "axes.titlesize": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "legend.frameon": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "lines.linewidth": 1.4,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "axes.prop_cycle": matplotlib.cycler(color=CYCLE),
    "figure.dpi": 110,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42,  # embed TrueType — camera-ready requirement
    "ps.fonttype": 42,
}


@contextmanager
def style():
    with matplotlib.rc_context(_RC):
        yield


def figsize(width: float = SINGLE, aspect: float = GOLDEN) -> tuple[float, float]:
    return (width, width * aspect)


def save(fig, stem: str | Path, formats: tuple[str, ...] = ("pdf", "png")) -> list[Path]:
    """Vector PDF for the paper + 300dpi PNG for quick viewing."""
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    out = []
    for fmt in formats:
        path = stem.with_suffix(f".{fmt}")
        # bbox must be explicit: save() usually runs outside the rc context.
        fig.savefig(path, dpi=300 if fmt == "png" else None,
                    bbox_inches="tight", pad_inches=0.02)
        out.append(path)
    plt.close(fig)
    return out


def mark_synthetic(fig) -> None:
    """Stamp demo figures — synthetic data must never pass as real (repo rule)."""
    fig.text(
        0.995, 0.995, "synthetic demo", ha="right", va="top",
        fontsize=6, color="#999999", style="italic",
    )
