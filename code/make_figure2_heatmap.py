"""Figure 2 of the paper with panel (a) as a binned heatmap, from pilot/results/*.csv.

(a) plots g_A, the gain of the fused pair over the anchor alone (criteria_results.csv: gain_over_anchor);
(c) plots g*, the gain over the better of the two teachers (dissoc_results.csv: gain).

Usage (from the repository root):
    python code/make_figure2_heatmap.py pilot/results/ figures/
"""
import sys, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import spearmanr
R, O = sys.argv[1], sys.argv[2]
c = pd.read_csv(R + "criteria_results.csv"); d = pd.read_csv(R + "dissoc_results.csv")
c["util"] = 100 * c.gain_over_anchor
pilot = [("BRACS","Phikon-v2",0.828,.7526),("BRACS","CTransPath",0.569,.7463),
         ("SICAPv2","Phikon-v2",0.690,.8919),("SICAPv2","CTransPath",0.739,.8987),
         ("MHIST","Phikon-v2",0.691,.8195),("MHIST","CTransPath",0.679,.8331)]
dcol = {"BRACS":"#d62728","SICAPv2":"#ff7f0e","MHIST":"#e377c2"}
XW, YW, NMIN = 0.1, 0.075, 3
xe = np.arange(0, 1 + 1e-9, XW); ye = np.arange(0.40, 1 + 1e-9, YW)
xi = np.clip(np.digitize(c.cka, xe) - 1, 0, len(xe) - 2); yi = np.clip(np.digitize(c.acc_cand, ye) - 1, 0, len(ye) - 2)
M = np.full((len(ye) - 1, len(xe) - 1), np.nan); shown = 0
for a in range(M.shape[0]):
    for b in range(M.shape[1]):
        m = (yi == a) & (xi == b)
        if m.sum() >= NMIN: M[a, b] = c.util[m].mean(); shown += m.sum()
# drawn at its printed width (0.95 of the 7.0 in text width), so these sizes are the sizes on the page
plt.rcParams.update({"font.size": 8.5, "axes.titlesize": 8.5, "axes.labelsize": 8.5, "xtick.labelsize": 8, "ytick.labelsize": 8})
fig, ax = plt.subplots(1, 3, figsize=(6.65, 3.15), gridspec_kw={"width_ratios": [1.35, 1, 1]})
a = ax[0]; cmap = matplotlib.colormaps["viridis"].copy(); cmap.set_bad("#f2f2f2")
im = a.pcolormesh(xe, ye, np.ma.masked_invalid(M), cmap=cmap, vmin=0, vmax=14, edgecolors="white", linewidth=0.5)
for ds, enc, k, acc in pilot:
    a.scatter(k, acc, marker="o" if enc == "Phikon-v2" else "s", s=40, facecolor=dcol[ds], edgecolor="black", linewidth=0.8, zorder=3)
a.set_xlim(0, 1); a.set_ylim(0.40, 1.0); a.set_xlabel("CKA with anchor"); a.set_ylabel("candidate accuracy")
a.set_title(f"(a) Gain follows competence, not CKA\n(bin means; {shown} of {len(c)} configs)")
cb = fig.colorbar(im, ax=a, fraction=0.05, pad=0.02); cb.set_label("mean gain over anchor, $g_A$ (pp)")
a.legend(handles=[Line2D([], [], marker="o", ls="", mfc="white", mec="black", label="Phikon-v2"),
                  Line2D([], [], marker="s", ls="", mfc="white", mec="black", label="CTransPath")]
                 + [Line2D([], [], marker="s", ls="", mfc=col, mec=col, label=ds) for ds, col in dcol.items()],
         loc="upper center", bbox_to_anchor=(0.5, -0.2), fontsize=7.5, handletextpad=0.2, borderpad=0.3,
         labelspacing=0.25, ncol=3, columnspacing=0.8, frameon=False)
r = spearmanr(d.cka, d.disagree)[0]
ax[1].scatter(d.cka, 100 * d.disagree, s=9, c="#c0392b"); ax[1].set_xlabel("CKA"); ax[1].set_ylabel("teacher disagreement (%)")
ax[1].set_title(f"(b) CKA does not track functional\ndiversity ($\\rho$={r:.2f})")
r2 = spearmanr(d.disagree, d.gain)[0]
ax[2].scatter(100 * d.disagree, 100 * d.gain, s=9, c="#1f77b4"); ax[2].set_xlabel("teacher disagreement (%)"); ax[2].set_ylabel("gain over best teacher, $g^{\\star}$ (pp)")
ax[2].set_title(f"(c) Functional diversity does\n($\\rho$={r2:.2f})")
fig.tight_layout(w_pad=0.8); fig.savefig(O + "figure3_synthetic_heatmap.png", dpi=300, bbox_inches="tight"); fig.savefig(O + "figure3_synthetic_heatmap.pdf", bbox_inches="tight")
print("shown", shown, "of", len(c))
