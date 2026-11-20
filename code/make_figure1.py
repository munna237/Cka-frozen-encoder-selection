"""Figure 1 of the paper: (a) the primary competence-matched pair of every pool with
its fold-level 95% interval and the pooled registered estimate; (b) cross-fitted
selection regret of each rule by domain.

Reads only files in this repository:
    results/battery.json                            registered estimand (A), primary pairs
    results/reviewer/reviewer2_analyses.json        S7: per-pool primary pairs with fold CIs
    results/reviewer/reviewer_analyses.json         R5: cross-fitted regret by rule and domain

    python code/make_figure1.py            -> figures/figure1_overview.pdf and .png

The figure is drawn at its printed width (the full text width of the IEEE Access
template, 7.0 in), so the font sizes below are the sizes on the page.
"""

from __future__ import annotations

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "figures")

LABEL = {"bracs": "BRACS", "sicapv2": "SICAPv2", "mhist": "MHIST", "crc_val_7k": "CRC-VAL-7K",
         "nct_crc_100k": "NCT-CRC", "pcam": "PCam", "dtd": "DTD", "aircraft": "Aircraft", "eurosat": "EuroSAT",
         "pets": "Pets", "food101": "Food-101", "cifar100": "CIFAR-100", "gtsrb": "GTSRB",
         "country211": "Country211", "emotion": "Emotion", "trec_coarse": "TREC", "newsgroups20": "20NG",
         "sst5": "SST-5"}
DOMAIN_COLOR = {"pathology": "#b03a2e", "vision": "#1f5f8b", "text": "#1e8449"}
SUBSET_COLOR = {"all": "#222222", **DOMAIN_COLOR}
RULES = [("lowest_cka", "Lowest CKA"), ("disagreement", "Disagreement"), ("random", "Random"),
         ("competence", "Competence"), ("unique", "Unique contr."), ("gated_diversity", "Gated CKA")]


def load():
    bat = json.load(open(os.path.join(REPO, "results", "battery.json")))["all"]
    r1 = json.load(open(os.path.join(REPO, "results", "reviewer", "reviewer_analyses.json")))
    r2 = json.load(open(os.path.join(REPO, "results", "reviewer", "reviewer2_analyses.json")))
    domain = {p["pool"]: p["domain"] for p in r1["R8_metrics"]["per_pool_primary"]}
    pairs = [dict(p, domain=domain[p["pool"]]) for p in r2["S7_primary_pairs"]]
    pooled = bat["A_matched_null_primary_only"]["cluster_robust"]
    return pairs, pooled, r1["R5_regret_crossfit"]


def regret_intervals(r5, subsets, n_boot=10000, seed=0):
    """Percentile bootstrap over pools of each rule's mean cross-fitted regret (descriptive).
    Pools are resampled with replacement; regret is non-negative, so the interval is too."""
    key = {"lowest_cka": "lowest_cka_regret_crossfit", "disagreement": "disagreement_regret_crossfit",
           "random": "random_regret", "competence": "competence_regret_crossfit",
           "unique": "unique_regret_crossfit", "gated_diversity": "gated_diversity_regret_crossfit"}
    doms = {"all": ("pathology", "vision", "text"), "pathology": ("pathology",), "vision": ("vision",),
            "text": ("text",)}
    rng = np.random.default_rng(seed)
    out = {}
    for s in subsets:
        rows = [p for p in r5["per_pool"] if p["domain"] in doms[s]]
        idx = rng.integers(0, len(rows), size=(n_boot, len(rows)))
        out[s] = {}
        for k, f in key.items():
            v = np.array([p[f] for p in rows])
            stored = r5[s]["random_regret"] if k == "random" else r5[s][f"{k}_crossfit"]
            assert abs(v.mean() - stored) < 1e-9, (s, k)
            out[s][k] = tuple(np.percentile(v[idx].mean(1), [2.5, 97.5]))
    return out


def main():
    pairs, pooled, r5 = load()
    plt.rcParams.update({"font.size": 9.5, "axes.titlesize": 10, "axes.labelsize": 9.5, "xtick.labelsize": 9,
                         "ytick.labelsize": 9, "legend.fontsize": 8.5, "font.family": "DejaVu Sans",
                         "pdf.fonttype": 42})
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.0, 4.3), gridspec_kw={"width_ratios": [1.05, 1]})

    # (a) per-pool primary pairs
    n = len(pairs)
    y = np.arange(n)[::-1] + 1.5
    ax.axvspan(-0.5, 0.5, color="#dcdcdc", zorder=0, label=r"$\pm$0.5 pp margin (all matched pairs)")
    ax.axvline(0, color="black", lw=0.8, zorder=1)
    for yi, p in zip(y, pairs):
        c = DOMAIN_COLOR[p["domain"]]
        ax.errorbar(p["D"], yi, xerr=[[p["D"] - p["fold_ci_low"]], [p["fold_ci_high"] - p["D"]]], fmt="o",
                    color=c, ms=6, lw=1.5, capsize=2.5, zorder=3)
    ax.errorbar(pooled["estimate_pp"], 0.4,
                xerr=[[pooled["estimate_pp"] - pooled["ci_low"]], [pooled["ci_high"] - pooled["estimate_pp"]]],
                fmt="D", color="black", ms=7.5, lw=2.0, capsize=3, zorder=4)
    ax.text(-4.55, 0.4,
            f"{pooled['estimate_pp']:+.2f} [{pooled['ci_low']:+.2f}, {pooled['ci_high']:+.2f}], G = {pooled['G']}",
            va="center", ha="left", fontsize=8)
    ax.set_yticks(list(y) + [0.4])
    ax.set_yticklabels([LABEL[p["pool"]] for p in pairs] + ["Pooled"])
    ax.get_yticklabels()[-1].set_fontweight("bold")
    ax.set_ylim(-0.4, y[0] + 0.7)
    ax.set_xlim(-4.6, 1.7)
    ax.set_xlabel("D = utility(higher-CKA) − utility(lower-CKA), pp\n(fold-level 95% CI per pool)")
    ax.set_title("(a) Primary matched pair per pool", loc="left")
    for d, c in DOMAIN_COLOR.items():
        ax.plot([], [], "o", color=c, ms=6, label=d)
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.23), ncol=2, frameon=False, handletextpad=0.4, columnspacing=1.0)
    ax.grid(axis="x", color="#eeeeee", lw=0.6, zorder=0)

    # (b) cross-fitted regret by rule and domain
    subsets = ["all", "pathology", "vision", "text"]
    ci = regret_intervals(r5, subsets)
    w = 0.2
    x = np.arange(len(RULES))
    for i, s in enumerate(subsets):
        v = r5[s]
        vals = [v["random_regret"] if k == "random" else v[f"{k}_crossfit"] for k, _ in RULES]
        lo, hi = zip(*[ci[s][k] for k, _ in RULES])
        name = "All" if s == "all" else s.capitalize()
        bx.bar(x + (i - 1.5) * w, vals, w, color=SUBSET_COLOR[s], label=f"{name} ({v['pools']})", zorder=2,
               yerr=[np.array(vals) - np.array(lo), np.array(hi) - np.array(vals)],
               error_kw={"elinewidth": 0.8, "capsize": 1.5, "ecolor": "#555555"})
    bx.set_xticks(x)
    bx.set_xticklabels([lab for _, lab in RULES], fontsize=8.5, rotation=35, ha="right", rotation_mode="anchor")
    bx.set_ylabel("Mean regret vs. best candidate (pp)")
    bx.set_title("(b) Selection regret, cross-fitted", loc="left")
    bx.legend(loc="upper right", frameon=True, framealpha=0.95, fontsize=8)
    bx.grid(axis="y", color="#eeeeee", lw=0.6, zorder=0)
    for a in (ax, bx):
        a.spines[["top", "right"]].set_visible(False)

    fig.tight_layout(w_pad=1.2)
    os.makedirs(OUT, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"figure1_overview.{ext}"), dpi=300, bbox_inches="tight")
    print("wrote", os.path.join(OUT, "figure1_overview.pdf"))


if __name__ == "__main__":
    main()
