"""Interactive 3D version of Figure 2a (rotatable), from pilot/results/criteria_results.csv.

Usage (from the repository root):
    python code/make_figure2_3d.py pilot/results/ figures/
"""
import sys, pandas as pd, plotly.graph_objects as go
from scipy.stats import spearmanr
R, O = sys.argv[1], sys.argv[2]
c = pd.read_csv(R + "criteria_results.csv"); c["util"] = 100 * c.gain_over_anchor
pilot = [("BRACS","Phikon-v2",0.828,75.26,2.19),("BRACS","CTransPath",0.569,74.63,1.41),
         ("SICAPv2","Phikon-v2",0.690,89.19,1.85),("SICAPv2","CTransPath",0.739,89.87,2.04),
         ("MHIST","Phikon-v2",0.691,81.95,1.30),("MHIST","CTransPath",0.679,83.31,1.52)]
dcol = {"BRACS":"#d62728","SICAPv2":"#ff7f0e","MHIST":"#e377c2"}
ra = spearmanr(c.acc_cand, c.util)[0]; rk = spearmanr(c.cka, c.util)[0]
fig = go.Figure()
fig.add_trace(go.Scatter3d(
    x=c.cka, y=100 * c.acc_cand, z=c.util, mode="markers", name="Controlled configurations (405)",
    marker=dict(size=3.2, color=c.util, colorscale="Viridis", cmin=0, cmax=14, opacity=0.85,
                colorbar=dict(title="marginal<br>fusion<br>utility (pp)", thickness=14, len=0.6)),
    customdata=c[["d_shared", "s_priv2", "seed"]].values,
    hovertemplate="CKA %{x:.3f}<br>accuracy %{y:.1f}%<br>utility %{z:.2f} pp"
                  "<br>shared width %{customdata[0]}, signal %{customdata[1]}, seed %{customdata[2]}<extra></extra>"))
for ds, col in dcol.items():
    rows = [p for p in pilot if p[0] == ds]
    fig.add_trace(go.Scatter3d(
        x=[p[2] for p in rows], y=[p[3] for p in rows], z=[p[4] for p in rows], mode="markers",
        name=f"Pilot, {ds} (real encoders; utility over UNI+ResNet-50, not the same base)",
        marker=dict(size=7, color=col, symbol=["circle" if p[1] == "Phikon-v2" else "square" for p in rows],
                    line=dict(color="black", width=1.5)),
        text=[p[1] for p in rows],
        hovertemplate="%{text} on " + ds + "<br>CKA %{x:.3f}<br>accuracy %{y:.2f}%<br>"
                      "marginal utility %{z:.2f} pp (real pilot base)<extra></extra>"))
cams = {"Rotated": dict(eye=dict(x=1.6, y=-1.6, z=0.9)),
        "Look along CKA (utility vs accuracy)": dict(eye=dict(x=2.4, y=0.01, z=0.15)),
        "Look along accuracy (utility vs CKA)": dict(eye=dict(x=0.01, y=-2.4, z=0.15)),
        "Top (as in the paper)": dict(eye=dict(x=0.01, y=0.01, z=2.5))}
fig.update_layout(
    title=dict(text=f"Figure 2(a), interactive: utility follows competence (ρ = {ra:.2f}), not CKA (ρ = {rk:.2f})"
                    "<br><sup>Controlled study, 9 × 9 grid × 5 seeds. Drag to rotate, scroll to zoom; "
                    "the buttons show each axis edge-on. Illustration of how CKA and utility can come apart, "
                    "not evidence about real encoders.</sup>", x=0.02),
    scene=dict(xaxis_title="CKA with anchor", yaxis_title="candidate accuracy (%)", zaxis_title="marginal fusion utility (pp)",
               xaxis=dict(range=[0, 1]), yaxis=dict(range=[40, 100]), camera=cams["Rotated"], aspectmode="cube"),
    updatemenus=[dict(type="buttons", direction="right", x=0.02, y=1.0, xanchor="left", yanchor="bottom",
                      buttons=[dict(label=k, method="relayout", args=[{"scene.camera": v}]) for k, v in cams.items()])],
    legend=dict(x=0.0, y=0.0, bgcolor="rgba(255,255,255,0.8)", font=dict(size=11)),
    margin=dict(l=0, r=0, t=110, b=0), template="plotly_white", height=760)
fig.write_html(O + "figure2a_interactive_3d.html", include_plotlyjs=True, full_html=True,
               config={"displaylogo": False, "toImageButtonOptions": {"format": "png", "scale": 3}})
print("ok", round(ra, 3), round(rk, 3))
