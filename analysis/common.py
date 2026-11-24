"""Shared loading helpers and paths for the NexaGlow analysis."""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FIG = ROOT / "outputs" / "figures"
TAB = ROOT / "outputs" / "tables"
FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)

DATES = {
    "customers": ["signup_date"],
    "campaign_catalog": ["start_date", "end_date"],
    "campaign_exposures": ["exposure_date"],
    "orders": ["order_date"],
    "weekly_channel_performance": ["week_start_date"],
}
FILES = ["customers", "campaign_catalog", "campaign_exposures", "orders",
         "order_items", "product_catalog", "weekly_channel_performance",
         "brand_perception_survey"]


def load(name):
    return pd.read_csv(DATA / f"{name}.csv", parse_dates=DATES.get(name, []))


def load_all():
    return {n: load(n) for n in FILES}


# ---------------------------------------------------------------- plotting
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SEQ_CMAP = LinearSegmentedColormap.from_list("seq", SEQ)
DIV_CMAP = LinearSegmentedColormap.from_list("div", ["#184f95", "#6da7ec", "#f0efec", "#ec8a89", "#b52f2e"])
# colour follows the entity, fixed everywhere in the report
CHANNEL_COLORS = dict(zip(["Email", "SMS", "Social", "Search", "Display", "WhatsApp"], SERIES))
CATEGORY_COLORS = dict(zip(["Skincare", "Haircare", "Bodycare", "Bundles"], SERIES))

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "DejaVu Sans", "font.size": 9, "text.color": INK,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "axes.titlecolor": INK, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "axes.prop_cycle": matplotlib.cycler(color=SERIES), "lines.linewidth": 2,
    "legend.frameon": False, "legend.fontsize": 8, "figure.dpi": 110, "savefig.dpi": 160,
    "patch.linewidth": 0,
})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / f"{name}.png", bbox_inches="tight")
    plt.close(fig)
