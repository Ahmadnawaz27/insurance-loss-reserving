"""
Plotly chart builders for the Streamlit app. Each function takes plain DataFrames and returns a go.Figure, so charts can be tested without Streamlit.

Design rules: one color per method in every chart; chain-ladder is gray; color is never the only cue (legend, direct labels or a table view); one axis per chart; blue-to-red diverging scales through gray; hover tooltips, no zoom or drag.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go

import scoring
from scoring import SHORT

# ------------------------------------------------------------------- palette
SURFACE = "#fcfcfb"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
FONT = '"Source Sans 3", "Source Sans Pro", system-ui, -apple-system, "Segoe UI", sans-serif'

COLOR = {                                    # color follows the method, in every chart
    "CL": "#8a8984", "ML": "#2a78d6", "ML_HIST": "#eb6834", "CORR": "#1baf7a", "BF": "#4a3aa7",
    "ML_HIST_NO_N": "#898781", "CL_BIAS": "#898781",
}
CHART_ORDER = ["CL", "ML", "ML_HIST", "CORR", "BF"]       # BF last: it is the reference-only method
MAIN4 = ["CL", "ML", "ML_HIST", "CORR"]

BLUE_SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
GRAY_SEQ = ["#f1f0ec", "#dedcd5", "#c9c7be", "#b3b1a7", "#9d9b91"]
DIVERGING = [[0.0, "#1c5cab"], [0.17, "#3987e5"], [0.34, "#9ec5f4"], [0.5, "#f0efec"],
             [0.66, "#f4a9a8"], [0.83, "#e34948"], [1.0, "#a82a2a"]]


def _scale(colors):
    n = len(colors) - 1
    return [[i / n, c] for i, c in enumerate(colors)]


def rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    return f"rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{alpha})"


AGE_TICKS = {"old": "Old years<br>1999-2002", "middle": "Middle years<br>2003-2004",
             "young": "Young years<br>2005-2007"}
HEAT_TICKS = {"CL": "Chain-<br>ladder", "ML": "ML,<br>base", "ML_HIST": "ML +<br>history",
              "CORR": "CL + ML<br>correction", "BF": "Born.-<br>Ferguson"}


# ---------------------------------------------------------------- shared style
def _theme(fig, height, legend=False, hover="closest", top=8, right=10):
    fig.update_layout(
        height=height, margin=dict(l=6, r=right, t=top, b=6),
        paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
        font=dict(family=FONT, size=13, color=INK2),
        hoverlabel=dict(bgcolor="#ffffff", bordercolor=AXIS, font=dict(family=FONT, size=13, color=INK)),
        hovermode=hover, dragmode=False, showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0,
                    font=dict(size=12.5, color=INK2), bgcolor="rgba(0,0,0,0)", itemdoubleclick=False),
    )
    fig.update_xaxes(showline=True, linecolor=AXIS, gridcolor=GRID, zeroline=False, ticks="",
                     tickfont=dict(color=INK2), title_font=dict(color=INK2, size=12.5), fixedrange=True,
                     automargin=True)
    fig.update_yaxes(showline=False, gridcolor=GRID, zeroline=False, ticks="",
                     tickfont=dict(color=INK2), title_font=dict(color=INK2, size=12.5), fixedrange=True,
                     automargin=True)
    return fig


def _label_gap(fig, px=8):
    """Air between the category labels and the bars: invisible outside ticks push the labels out."""
    fig.update_yaxes(ticks="outside", ticklen=px, tickcolor=SURFACE)
    return fig


def _bar_labels(fig, xs, texts, size=13, shift=6):
    """Value labels at the ends of horizontal bars, drawn as annotations with a surface-colored box,
    so a reference line behind them is interrupted instead of struck through the text.
    Bars are addressed by position (0, 1, 2 ...), which works on a category axis."""
    for i, (x, t) in enumerate(zip(xs, texts)):
        left = x < 0
        fig.add_annotation(x=x, y=i, xref="x", yref="y", text=t, showarrow=False,
                           xanchor="right" if left else "left", xshift=-shift if left else shift,
                           font=dict(size=size, color=INK), bgcolor=SURFACE, borderpad=1)
    return fig


def _round_bars(fig, radius=4):
    """Rounded data ends where the installed plotly supports it; plain bars otherwise."""
    try:
        fig.update_traces(marker_cornerradius=radius, selector=dict(type="bar"))
    except Exception:
        pass
    return fig


def _short(m):
    return SHORT.get(m, m)


# --------------------------------------------------------------- overview charts
def fig_method_bars(values: dict, methods, fmt="{:.2f}%", xtitle=None, ref="CL"):
    """Horizontal bars, one per method, with a dotted line at the chain-ladder value."""
    methods = list(methods)
    vals = [values[m] for m in methods]
    names = [_short(m) for m in methods]
    top = max(vals)
    fig = go.Figure(go.Bar(
        x=vals, y=names, orientation="h", width=0.5,
        marker=dict(color=[COLOR[m] for m in methods]),
        customdata=[[v - values[ref]] for v in vals],
        hovertemplate="<b>%{y}</b><br>%{x:.2f}%<br>vs chain-ladder: %{customdata[0]:+.2f} pts<extra></extra>",
    ))
    if ref in values:
        fig.add_vline(x=values[ref], line_color=INK2, line_width=1, line_dash="dot")
    _theme(fig, height=44 * len(methods) + 54, right=34)
    fig.update_xaxes(range=[0, top * 1.28], ticksuffix="%", title_text=xtitle, showgrid=True)
    fig.update_yaxes(autorange="reversed", showgrid=False, tickfont=dict(color=INK2, size=13))
    _bar_labels(fig, vals, [fmt.format(v) for v in vals])
    return _round_bars(_label_gap(fig))


def fig_forest(sc: pd.DataFrame, methods):
    """Difference in mean company miss vs chain-ladder, with the 95% interval."""
    methods = list(methods)
    labels = {}
    for m in methods:
        r = sc.loc[m]
        labels[m] = (f"{_short(m)}<br><span style='font-size:11.5px;color:{INK2}'>"
                     f"{r['vs_chain_ladder_pts']:+.2f}  [{r['vs_cl_ci_low']:+.2f}, {r['vs_cl_ci_high']:+.2f}]</span>")
    fig = go.Figure()
    for m in methods:
        r = sc.loc[m]; d, lo, hi = r["vs_chain_ladder_pts"], r["vs_cl_ci_low"], r["vs_cl_ci_high"]
        clear = "better" if hi < 0 else ("worse" if lo > 0 else "no clear difference")
        fig.add_trace(go.Scatter(
            x=[d], y=[labels[m]], mode="markers", showlegend=False,
            marker=dict(size=12, color=COLOR[m], line=dict(color=SURFACE, width=2)),
            error_x=dict(type="data", symmetric=False, array=[hi - d], arrayminus=[d - lo],
                         color=COLOR[m], thickness=2.5, width=0),
            hovertemplate=(f"<b>{_short(m)}</b><br>difference: {d:+.2f} pts<br>95% interval: "
                           f"{lo:+.2f} to {hi:+.2f}<br>verdict: {clear}<extra></extra>")))
    fig.add_vline(x=0, line_color=INK2, line_width=1.3)
    _theme(fig, height=66 * len(methods) + 70, top=26)
    lo_all = min(sc.loc[m, "vs_cl_ci_low"] for m in methods); hi_all = max(sc.loc[m, "vs_cl_ci_high"] for m in methods)
    pad = (hi_all - lo_all) * 0.06
    fig.update_xaxes(range=[lo_all - pad, hi_all + pad], showgrid=True,
                     title_text="Difference in average miss vs chain-ladder (points)")
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.add_annotation(xref="paper", yref="paper", x=0, y=1.0, yshift=14, showarrow=False, xanchor="left",
                       text="← better", font=dict(size=12, color=INK2))
    fig.add_annotation(xref="paper", yref="paper", x=1, y=1.0, yshift=14, showarrow=False, xanchor="right",
                       text="worse →", font=dict(size=12, color=INK2))
    return fig


def fig_reserve_error(sc: pd.DataFrame, still_thousands: float, methods):
    """Whole-book net reserve error, % of the money still to be paid."""
    methods = list(methods)
    vals = [sc.loc[m, "net_reserve_error_pct"] for m in methods]
    bn = [v / 100 * still_thousands / 1e6 for v in vals]
    fig = go.Figure(go.Bar(
        x=vals, y=[_short(m) for m in methods], orientation="h", width=0.56,
        marker=dict(color=[COLOR[m] for m in methods]),
        text=[f"{v:+.1f}%" for v in vals], textposition="outside", cliponaxis=False,
        textfont=dict(color=INK, size=13),
        customdata=[[b, "too much" if v > 0 else "too little"] for v, b in zip(vals, bn)],
        hovertemplate=("<b>%{y}</b><br>%{x:+.2f}% of the money still to be paid<br>"
                       "about $%{customdata[0]:+.2f}bn (%{customdata[1]} set aside)<extra></extra>")))
    fig.add_vline(x=0, line_color=INK2, line_width=1.3)
    _theme(fig, height=44 * len(methods) + 70, top=26, right=40)
    m = max(abs(v) for v in vals)
    fig.update_xaxes(range=[-m * 1.6, m * 1.6], ticksuffix="%", showgrid=True,
                     title_text="Net reserve error (% still to pay)")
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.add_annotation(xref="paper", yref="paper", x=0, y=1.0, yshift=14, showarrow=False, xanchor="left",
                       text="← too little", font=dict(size=12, color=INK2))
    fig.add_annotation(xref="paper", yref="paper", x=1, y=1.0, yshift=14, showarrow=False, xanchor="right",
                       text="too much →", font=dict(size=12, color=INK2))
    return _round_bars(_label_gap(fig))


# ------------------------------------------------------------ comparison charts
def fig_by_age(sc: pd.DataFrame, methods, ages=None):
    """Median miss by accident-year age group, grouped bars."""
    ages = list(ages) if ages else list(scoring.AGE_ORDER)
    fig = go.Figure()
    for m in methods:
        vals = [sc.loc[m, f"{a}_years_median_miss_pct"] for a in ages]
        fig.add_trace(go.Bar(
            x=[AGE_TICKS[a] for a in ages], y=vals, name=_short(m), marker=dict(color=COLOR[m]),
            text=[f"{v:.2f}" if v < 1 else f"{v:.1f}" for v in vals], textposition="outside",
            textfont=dict(color=INK2, size=12), cliponaxis=False,
            hovertemplate=f"<b>{_short(m)}</b><br>%{{x}}<br>median miss %{{y:.2f}}%<extra></extra>"))
    _theme(fig, height=360, legend=True, top=40)
    fig.update_layout(barmode="group", bargap=0.28, bargroupgap=0.06)
    fig.update_yaxes(ticksuffix="%", title_text="Median miss", rangemode="tozero")
    fig.update_xaxes(showgrid=False)
    return _round_bars(fig, 3)


def _cell_scale_color(scale, t):
    """Color at position t (0-1) of a plotly colorscale, as (r, g, b)."""
    t = min(max(t, 0.0), 1.0)
    for (p0, c0), (p1, c1) in zip(scale[:-1], scale[1:]):
        if p0 <= t <= p1:
            u = 0 if p1 == p0 else (t - p0) / (p1 - p0)
            a = [int(c0.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4)]
            b = [int(c1.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4)]
            return tuple(a[i] + (b[i] - a[i]) * u for i in range(3))
    return (0, 0, 0)


def _text_on(scale, t):
    """Ink or white, whichever reads better on the cell color."""
    r, g, b = _cell_scale_color(scale, t)
    lum = (0.299 * r + 0.587 * g + 0.114 * b) / 255
    return "#ffffff" if lum < 0.52 else INK


def fig_error_heatmap(df: pd.DataFrame, methods, stat="bias"):
    """Accident year x method. stat 'bias' = median signed error (blue under, red over);
    stat 'miss' = median absolute error (one-hue blue)."""
    methods = [m for m in methods if m in df.columns]
    z = df[methods].to_numpy(dtype=float)
    ays = [int(a) for a in df.index]
    if stat == "bias":
        lim = max(1.0, float(np.nanmax(np.abs(z))))
        scale, zmin, zmax = DIVERGING, -lim, lim
        fmt = "{:+.1f}"
        hov = "accident year <b>%{y}</b>, <b>%{customdata}</b><br>median signed error %{z:+.2f}%<extra></extra>"
    else:
        scale, zmin, zmax = _scale(BLUE_SEQ), 0.0, max(1.0, float(np.nanmax(z)))
        fmt = "{:.1f}"
        hov = "accident year <b>%{y}</b>, <b>%{customdata}</b><br>median miss %{z:.2f}%<extra></extra>"
    xs = list(range(len(methods)))
    fig = go.Figure(go.Heatmap(
        z=z, x=xs, y=ays, colorscale=scale, zmin=zmin, zmax=zmax,
        xgap=2, ygap=2, hovertemplate=hov, showscale=False,
        customdata=np.array([[_short(m) for m in methods]] * len(ays))))
    for i, a in enumerate(ays):
        for j in xs:
            v = z[i, j]
            t = (v - zmin) / (zmax - zmin)
            shown = "0.0" if abs(v) < 0.05 else fmt.format(v)
            fig.add_annotation(x=j, y=a, text=shown, showarrow=False,
                               font=dict(size=12.5, color=_text_on(scale, t)))
    _theme(fig, height=34 * len(ays) + 74, top=6)
    fig.update_xaxes(side="top", showline=False, showgrid=False, tickvals=xs, tickangle=0,
                     tickfont=dict(size=11, color=INK2),
                     ticktext=[HEAT_TICKS.get(m, _short(m)) for m in methods], range=[-0.5, len(methods) - 0.5])
    fig.update_yaxes(range=[ays[-1] + 0.5, ays[0] - 0.5], tickvals=ays, showgrid=False,
                     title_text="Accident year", title_standoff=12)
    return fig


def fig_error_spread(res: pd.DataFrame, methods):
    """Where the bulk of the errors fall: box = middle half, whiskers = 10th to 90th percentile."""
    sp = scoring.error_spread(res, methods)
    labels = [AGE_TICKS[a] for a in scoring.AGE_ORDER]
    fig = go.Figure()
    for m in methods:
        s = sp[sp["method"] == m].set_index("age").reindex(scoring.AGE_ORDER)
        fig.add_trace(go.Box(
            x=labels, name=_short(m), q1=s["q25"], median=s["q50"], q3=s["q75"],
            lowerfence=s["q10"], upperfence=s["q90"], boxpoints=False,
            line=dict(color=COLOR[m], width=1.6), fillcolor=rgba(COLOR[m], 0.30), whiskerwidth=0.5,
            hoverinfo="y+name"))
    fig.add_hline(y=0, line_color=INK2, line_width=1.2)
    _theme(fig, height=400, legend=True, top=40)
    fig.update_layout(boxmode="group", boxgap=0.22, boxgroupgap=0.08)
    fig.update_yaxes(ticksuffix="%", title_text="Signed error", showgrid=True)
    fig.update_xaxes(showgrid=False)
    return fig


def fig_scatter(table: pd.DataFrame, method: str, lo=0.15, hi=60):
    """One dot per company: chain-ladder miss (x) against the method's miss (y), log axes."""
    t = table.copy()
    better = t[method] < t["CL"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", showlegend=False, hoverinfo="skip",
                             line=dict(color=INK2, width=1.2, dash="dash")))
    for mask, name, color in [(~better, "Chain-ladder better", COLOR["CL"]),
                              (better, f"{_short(method)} better", COLOR[method])]:
        s = t[mask]
        fig.add_trace(go.Scatter(
            x=s["CL"], y=s[method], mode="markers", name=f"{name} ({len(s)})",
            marker=dict(size=9, color=color, opacity=0.92, line=dict(color=SURFACE, width=1)),
            customdata=np.c_[s["name"], s["size_tier"], s["CL"], s[method], s["CL"] - s[method]],
            hovertemplate=("<b>%{customdata[0]}</b> (%{customdata[1]})<br>chain-ladder miss %{customdata[2]:.2f}%<br>"
                           + _short(method) + " miss %{customdata[3]:.2f}%<br>difference %{customdata[4]:+.2f} pts"
                           "<extra></extra>")))
    ticks = [0.2, 0.5, 1, 2, 5, 10, 30]
    _theme(fig, height=470, legend=True, top=34)
    fig.update_xaxes(type="log", range=[np.log10(lo), np.log10(hi)], tickvals=ticks, tickangle=0,
                     ticktext=[f"{x:g}%" for x in ticks], showgrid=True,
                     title_text="Chain-ladder: average miss")
    fig.update_yaxes(type="log", range=[np.log10(lo), np.log10(hi)], tickvals=ticks,
                     ticktext=[f"{x:g}%" for x in ticks], showgrid=True,
                     title_text=f"{_short(method)}: average miss")
    fig.add_annotation(xref="paper", yref="paper", x=0.98, y=0.03, showarrow=False, xanchor="right",
                       yanchor="bottom", align="right", font=dict(size=12, color=INK2),
                       text="below the line:<br>method misses by less")
    return fig


def fig_ecdf(per: pd.DataFrame, methods, lo=0.15, hi=60):
    """Share of companies whose average miss is below each level, one step line per method."""
    fig = go.Figure()
    n = len(per)
    for m in methods:
        x = np.sort(per[m].to_numpy())
        y = np.arange(1, n + 1) / n * 100
        fig.add_trace(go.Scatter(
            x=x, y=y, mode="lines", name=_short(m), line=dict(color=COLOR[m], width=2.2, shape="hv"),
            hovertemplate=f"<b>{_short(m)}</b><br>%{{y:.0f}}% of companies miss by less than %{{x:.2f}}%<extra></extra>"))
    ticks = [0.2, 0.5, 1, 2, 5, 10, 30]
    _theme(fig, height=380, legend=True, top=40)
    fig.update_xaxes(type="log", range=[np.log10(lo), np.log10(hi)], tickvals=ticks, tickangle=0,
                     ticktext=[f"{x:g}%" for x in ticks], showgrid=True,
                     title_text="Average miss of the company")
    fig.update_yaxes(range=[0, 101], ticksuffix="%", showgrid=True, title_text="Share of companies")
    return fig


def fig_win_loss(table: pd.DataFrame, method: str, k=8):
    """Companies where the method helps most and hurts most, in points of average miss."""
    t = table.assign(gain=table["CL"] - table[method])
    top = pd.concat([t.nlargest(k, "gain"), t.nsmallest(k, "gain")]).drop_duplicates("GRCODE")
    top = top.sort_values("gain", ascending=False)
    names = [n if len(n) <= 22 else n[:21] + "…" for n in top["name"]]
    colors = [COLOR[method] if g > 0 else COLOR["CL"] for g in top["gain"]]
    fig = go.Figure(go.Bar(
        x=top["gain"], y=names, orientation="h", width=0.62, marker=dict(color=colors),
        text=[f"{g:+.1f}" for g in top["gain"]], textposition="outside", cliponaxis=False,
        textfont=dict(color=INK, size=12),
        customdata=np.c_[top["name"], top["CL"], top[method], top["size_tier"]],
        hovertemplate=("<b>%{customdata[0]}</b> (%{customdata[3]})<br>chain-ladder miss %{customdata[1]:.2f}%<br>"
                       + _short(method) + " miss %{customdata[2]:.2f}%<br>gain %{x:+.2f} pts<extra></extra>")))
    fig.add_vline(x=0, line_color=INK2, line_width=1.3)
    _theme(fig, height=26 * len(top) + 70, top=26, right=36)
    m = max(abs(top["gain"]))
    fig.update_xaxes(range=[-m * 1.4, m * 1.4], showgrid=True,
                     title_text="Points gained vs chain-ladder")
    fig.update_yaxes(autorange="reversed", showgrid=False, tickfont=dict(size=12, color=INK2))
    fig.add_annotation(xref="paper", yref="paper", x=0, y=1.0, yshift=14, showarrow=False, xanchor="left",
                       text="← worse", font=dict(size=12, color=INK2))
    fig.add_annotation(xref="paper", yref="paper", x=1, y=1.0, yshift=14, showarrow=False, xanchor="right",
                       text="better →", font=dict(size=12, color=INK2))
    return _round_bars(_label_gap(fig), 3)


def fig_size_bars(size_tbl: pd.DataFrame, methods):
    """Mean company miss by size tier, grouped bars."""
    fig = go.Figure()
    xl = [f"{t.capitalize()}<br>{int(n)} companies" for t, n in zip(size_tbl.index, size_tbl["companies"])]
    for m in methods:
        v = size_tbl[m]
        fig.add_trace(go.Bar(
            x=xl, y=v, name=_short(m), marker=dict(color=COLOR[m]),
            text=[f"{x:.1f}" for x in v], textposition="outside", textfont=dict(color=INK2, size=12),
            cliponaxis=False,
            hovertemplate=f"<b>{_short(m)}</b><br>%{{x}}<br>average company miss %{{y:.2f}}%<extra></extra>"))
    _theme(fig, height=340, legend=True, top=40)
    fig.update_layout(barmode="group", bargap=0.28, bargroupgap=0.06)
    fig.update_yaxes(ticksuffix="%", title_text="Average miss", rangemode="tozero")
    fig.update_xaxes(showgrid=False)
    return _round_bars(fig, 3)


# ---------------------------------------------------------------- company charts
def money_unit(max_thousands: float):
    """Amounts are in $000. Show $ billion for the very largest, $ million for most, $ thousand for the tiny."""
    if max_thousands >= 1_000_000:
        return 1_000_000.0, "$ billion"
    return (1000.0, "$ million") if max_thousands >= 1000 else (1.0, "$ thousand")


def _pivot(df, value, ays, lags):
    return df.pivot(index="AccidentYear", columns="DevelopmentLag", values=value).reindex(index=ays, columns=lags)


def fig_triangle(tri: pd.DataFrame, fill="none", paths=None, unit="money", fill_name=""):
    """The claims triangle for one company. Blue cells were known at the end of 2007.
    fill = 'none' leaves the corner empty, 'actual' shows what was really paid later,
    'path' fills it with `paths` (one method's projections)."""
    ays = sorted(int(a) for a in tri["AccidentYear"].unique()); lags = list(range(1, 11))
    cum = _pivot(tri, "CumPaidLoss", ays, lags).astype(float)
    known = _pivot(tri, "is_known", ays, lags).astype(float).eq(1.0)
    prem = tri.groupby("AccidentYear")["EarnedPremNet"].first().reindex(ays)
    if unit == "premium":
        scale = lambda m: m.div(prem, axis=0) * 100
        txt = lambda v: f"{v:.0f}%"; unit_label = "% of that year's premium"
    else:
        div, unit_label = money_unit(float(np.nanmax(cum.to_numpy())))
        scale = lambda m: m / div
        mx = np.nanmax(cum.to_numpy()) / div
        txt = (lambda v: f"{v:,.0f}") if mx >= 100 else ((lambda v: f"{v:,.1f}") if mx >= 10 else (lambda v: f"{v:,.2f}"))
    known_z = scale(cum.where(known))
    future_z = None
    if fill == "actual":
        future_z = scale(cum.where(~known))
    elif fill == "path" and paths is not None:
        pm = _pivot(paths, "paid", ays, lags).astype(float)
        future_z = scale(pm.where(~known))
    zmax = float(np.nanmax(np.concatenate([known_z.to_numpy().ravel()] +
                                          ([future_z.to_numpy().ravel()] if future_z is not None else []))))
    cal = pd.DataFrame({l: [a + l - 1 for a in ays] for l in lags}, index=ays)

    value_line = "%{text}" if unit == "premium" else "%{text} " + unit_label

    def traces(z, scale_colors, status):
        text = z.apply(lambda col: col.map(lambda v: "" if pd.isna(v) else txt(v)))      
        return go.Heatmap(
            z=z.to_numpy(), x=lags, y=ays, colorscale=_scale(scale_colors), zmin=0, zmax=zmax, showscale=False,
            xgap=2, ygap=2, text=text.to_numpy(), texttemplate="%{text}", hoverongaps=False,
            textfont=dict(size=11), customdata=cal.to_numpy(),
            hovertemplate=("accident year %{y}, lag %{x}<br>" + value_line + "<br>" + status
                           + " (calendar year %{customdata})<extra></extra>"))

    fig = go.Figure()
    fig.add_trace(traces(known_z, BLUE_SEQ, "known at the end of 2007"))
    if future_z is not None:
        fig.add_trace(traces(future_z, GRAY_SEQ, f"not known in 2007: {fill_name}"))
    # the staircase between known and unknown
    pts = [(10.5, ays[0] - 0.5)]
    for a in ays:
        k = int(known.loc[a].sum())
        pts += [(k + 0.5, a - 0.5), (k + 0.5, a + 0.5)]
    fig.add_trace(go.Scatter(x=[p[0] for p in pts], y=[p[1] for p in pts], mode="lines", hoverinfo="skip",
                             showlegend=False, line=dict(color=INK, width=2.2)))
    _theme(fig, height=36 * len(ays) + 70, top=6)
    fig.update_xaxes(side="top", showline=False, showgrid=False, tickvals=lags, range=[0.5, 10.5],
                     title_text="Development lag (year-end 1 to 10)", title_standoff=6)
    fig.update_yaxes(autorange=False, range=[ays[-1] + 0.5, ays[0] - 0.5], tickvals=ays, showgrid=False,
                     title_text="Accident year", title_standoff=12)
    if future_z is None:
        fig.add_annotation(x=8.1, y=ays[0] + 0.62 * (len(ays) - 1), showarrow=False, align="center",
                           text="not yet known<br>at the end of 2007", font=dict(size=12.5, color=MUTED))
    fig.update_layout(meta={"unit_label": unit_label})
    return fig


def fig_curve(tri: pd.DataFrame, ay: int, paths: pd.DataFrame, res_row: pd.Series, methods=MAIN4,
              show_bf=True, zoom=False):
    """Cumulative paid by lag for one accident year: what was known, what actually happened,
    and where each method projected it to go."""
    t = tri[tri["AccidentYear"] == ay].sort_values("DevelopmentLag")
    div, unit_label = money_unit(float(tri["CumPaidLoss"].max()))
    kn = t[t["is_known"] == 1]; fu = t[t["is_known"] == 0]
    last = int(kn["DevelopmentLag"].max())
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=kn["DevelopmentLag"], y=kn["CumPaidLoss"] / div, mode="lines+markers", name="Known at end of 2007",
        line=dict(color=INK, width=2.6), marker=dict(size=8, color=INK),
        hovertemplate="%{y:,.2f}<extra>Known at end of 2007</extra>"))
    fig.add_trace(go.Scatter(
        x=[last] + fu["DevelopmentLag"].tolist(),
        y=[kn["CumPaidLoss"].iloc[-1] / div] + (fu["CumPaidLoss"] / div).tolist(), mode="lines+markers",
        name="What actually happened", line=dict(color=INK2, width=2, dash="dash"),
        marker=dict(size=8, color=SURFACE, line=dict(color=INK2, width=1.8)),
        hovertemplate="%{y:,.2f}<extra>What actually happened</extra>"))
    for m in methods:
        p = paths[(paths["method"] == m) & (paths["AccidentYear"] == ay)].sort_values("DevelopmentLag")
        fig.add_trace(go.Scatter(
            x=p["DevelopmentLag"], y=p["paid"] / div, mode="lines+markers", name=_short(m),
            line=dict(color=COLOR[m], width=2.2, dash="dot"), marker=dict(size=6, color=COLOR[m]),
            hovertemplate="%{y:,.2f}<extra>" + _short(m) + "</extra>"))
    if show_bf:
        fig.add_trace(go.Scatter(
            x=[10], y=[res_row["BF"] / div], mode="markers", name="Born.-Ferguson (final only)",
            marker=dict(size=11, color=COLOR["BF"], symbol="diamond", line=dict(color=SURFACE, width=1.5)),
            hovertemplate="%{y:,.2f}<extra>Bornhuetter-Ferguson, final</extra>"))
    _theme(fig, height=420, legend=True, hover="x unified", top=66)
    fig.update_xaxes(tickvals=list(range(1, 11)), range=[0.7, 10.3], showgrid=False,
                     title_text="Development lag (year-end 1 to 10)")
    fig.update_yaxes(rangemode="tozero", title_text=f"Cumulative paid, {unit_label}", showgrid=True)
    if zoom:                                    # frame the part of the chart where the methods differ
        vals = [kn["CumPaidLoss"].iloc[-1] / div] + list(fu["CumPaidLoss"] / div) \
            + list(paths.loc[(paths["AccidentYear"] == ay) & paths["method"].isin(methods), "paid"] / div)
        if show_bf:
            vals.append(res_row["BF"] / div)
        lo, hi = min(vals), max(vals)
        pad = (hi - lo) * 0.12 or hi * 0.05
        fig.update_yaxes(rangemode="normal", range=[lo - pad, hi + pad])
        fig.update_xaxes(range=[max(0.7, last - 0.3), 10.3])
    fig.update_layout(meta={"unit_label": unit_label})
    return fig


def fig_company_errors(res_c: pd.DataFrame, methods=CHART_ORDER):
    """Signed error of each method's projected final amount, by accident year, for one company."""
    r = scoring.scored(res_c).sort_values("AccidentYear")
    fig = go.Figure()
    for m in methods:
        fig.add_trace(go.Scatter(
            x=r["AccidentYear"], y=r[f"{m}_err"], mode="lines+markers", name=_short(m),
            visible="legendonly" if m == "BF" else True,
            line=dict(color=COLOR[m], width=2), marker=dict(size=8, color=COLOR[m], line=dict(color=SURFACE, width=1.2)),
            hovertemplate="%{y:+.2f}%<extra>" + _short(m) + "</extra>"))
    fig.add_hline(y=0, line_color=INK2, line_width=1.2)
    _theme(fig, height=360, legend=True, hover="x unified", top=40)
    yrs = sorted(int(a) for a in r["AccidentYear"].unique())
    fig.update_xaxes(tickvals=yrs[::2], showgrid=False, title_text="Accident year", tickangle=0)
    fig.update_yaxes(ticksuffix="%", showgrid=True, title_text="Signed error (above 0 = too high)")
    return fig


def fig_company_reserve(res_c: pd.DataFrame, methods=CHART_ORDER):
    """The reserve each method would have held for one company (projected final minus paid so far,
    summed over its scored accident years) next to the money that was really still to be paid."""
    r = scoring.scored(res_c)
    need = float((r["actual"] - r["paid_to_date"]).sum())
    vals = {m: float((r[m] - r["paid_to_date"]).sum()) for m in methods}
    div, unit_label = money_unit(max([need] + list(vals.values())))
    top = max([need] + list(vals.values())) / div
    dec = 0 if top >= 100 else (1 if top >= 10 else 2)
    fmt = f"{{:,.{dec}f}}"
    names = ["What actually<br>happened"] + [_short(m) for m in methods]
    x = [need / div] + [vals[m] / div for m in methods]
    colors = [INK] + [COLOR[m] for m in methods]
    def gap_text(v):
        diff = (v - need) / div
        pct = f"{(v - need) / abs(need) * 100:+.1f}% = " if abs(need) > 1e-9 else ""
        return f"{pct}{diff:+,.2f} {unit_label} against what was actually paid"
    hover_gap = ["the money really still to be paid"] + [gap_text(vals[m]) for m in methods]
    fig = go.Figure(go.Bar(
        x=x, y=names, orientation="h", width=0.56, marker=dict(color=colors),
        customdata=hover_gap,
        hovertemplate="<b>%{y}</b><br>%{x:,.2f} " + unit_label + "<br>%{customdata}<extra></extra>"))
    fig.add_vline(x=need / div, line_color=INK2, line_width=1, line_dash="dot")
    _theme(fig, height=44 * len(names) + 60, top=6, right=44)
    low = min(0.0, min(x))
    fig.update_xaxes(range=[low * 1.3 if low < 0 else 0, top * 1.22], showgrid=True, title_text=f"Reserve, {unit_label}")
    fig.update_yaxes(autorange="reversed", showgrid=False, tickfont=dict(color=INK2, size=13))
    _bar_labels(fig, x, [fmt.format(v) for v in x])
    fig.update_layout(meta={"unit_label": unit_label})
    return _round_bars(_label_gap(fig))


# --------------------------------------------------------------- under the hood
VARIANT_NAMES = {
    "CL one-step (baseline)": "Chain-ladder factor (baseline)",
    "ML base features": "ML, base features",
    "ML + history: hist+n": "ML + company history",
    "ML + history: hist only": "ML + history, no hist_n",
    "Stack S3: + CL deviations": "ML + chain-ladder deviations",
    "Residual R1: + CL factor": "Correction R1: + CL factor",
    "Residual R3: + CL deviations": "Correction R3: + CL deviations",
    "CL + lag bias": "Chain-ladder + lag bias",
}
VARIANT_METHOD = {"CL one-step (baseline)": "CL", "ML base features": "ML",
                  "ML + history: hist+n": "ML_HIST", "Residual R3: + CL deviations": "CORR"}


def fig_onestep(df: pd.DataFrame, metric="all (mean)"):
    """One-step test: mean miss of next payment for every variant, three test years plus the average."""
    avg = df.groupby("variant")[metric].mean().sort_values()
    fig = go.Figure()
    order = list(avg.index)
    for v in order:
        c = COLOR[VARIANT_METHOD[v]] if v in VARIANT_METHOD else MUTED
        s = df[df["variant"] == v].sort_values("test_year")
        name = VARIANT_NAMES.get(v, v)
        fig.add_trace(go.Scatter(
            x=s[metric], y=[name] * len(s), mode="markers", showlegend=False,
            marker=dict(size=9, color=rgba(c, 0.55), line=dict(color=SURFACE, width=1.5)),
            customdata=s["test_year"], hovertemplate=f"<b>{name}</b><br>test year %{{customdata}}: %{{x:.2f}}<extra></extra>"))
        fig.add_trace(go.Scatter(
            x=[avg[v]], y=[name], mode="markers+text", showlegend=False, text=[f"{avg[v]:.2f}"],
            textposition="middle right", textfont=dict(color=INK, size=12.5),
            marker=dict(size=13, color=c, symbol="diamond", line=dict(color=SURFACE, width=1.5)),
            hovertemplate=f"<b>{name}</b><br>average of the three test years: %{{x:.2f}}<extra></extra>"))
    _theme(fig, height=36 * len(order) + 70, top=6, right=34)
    fig.update_xaxes(showgrid=True, title_text="Mean one-step miss (%)")
    fig.update_yaxes(autorange="reversed", showgrid=True, gridcolor=GRID)
    return fig


FEATURE_NAMES = {
    "DevelopmentLag": "Development lag", "paid_to_premium": "Paid so far / premium",
    "log_prior_growth": "Growth over the previous lag", "log_premium": "Company size (log premium)",
    "hist_growth": "Company's own average growth", "hist_n": "Years behind that average",
    "log_cl": "Chain-ladder factor (log)", "cl_disp": "Spread of past CL factors",
    "resid_prev": "Last step vs chain-ladder", "cum_dev": "Total deviation so far",
}


def fig_importance(df: pd.DataFrame, model: str):
    """Permutation importance: how much the one-step miss worsens when one feature is shuffled."""
    s = df[df["model"] == model].sort_values("importance", ascending=False)
    c = COLOR[VARIANT_METHOD.get(model, "ML")]
    names = [FEATURE_NAMES.get(f, f) for f in s["feature"]]
    fig = go.Figure(go.Bar(
        x=s["importance"], y=names, orientation="h", width=0.55, marker=dict(color=c),
        text=[f"{v:.2f}" for v in s["importance"]], textposition="outside", cliponaxis=False,
        textfont=dict(color=INK, size=12.5), customdata=s["std"],
        hovertemplate=("<b>%{y}</b><br>miss rises by %{x:.2f} points when shuffled "
                       "(spread over 20 shuffles: %{customdata:.2f})<extra></extra>")))
    _theme(fig, height=38 * len(s) + 60, right=36)
    fig.update_xaxes(showgrid=True, range=[0, float(s["importance"].max()) * 1.2],
                     title_text="Rise in miss when shuffled (points)")
    fig.update_yaxes(autorange="reversed", showgrid=False)
    return _round_bars(_label_gap(fig), 3)


def fig_by_lag(df: pd.DataFrame, band_method="CORR"):
    """How far each ML method moves the growth factor away from chain-ladder's, by development step."""
    fig = go.Figure()
    b = df[df["method"] == band_method].sort_values("from_lag")
    x = b["from_lag"].tolist()
    fig.add_trace(go.Scatter(x=x, y=b["p90_pct"], mode="lines", line=dict(width=0, color=COLOR[band_method]),
                             hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(x=x, y=b["p10_pct"], mode="lines", line=dict(width=0, color=COLOR[band_method]),
                             fill="tonexty", fillcolor=rgba(COLOR[band_method], 0.14), hoverinfo="skip",
                             name=f"{_short(band_method)}: middle 80% of companies"))
    for m in ["ML", "ML_HIST", "CORR"]:
        s = df[df["method"] == m].sort_values("from_lag")
        fig.add_trace(go.Scatter(
            x=s["from_lag"], y=s["median_pct"], mode="lines+markers", name=f"{_short(m)} (median)",
            line=dict(color=COLOR[m], width=2.4), marker=dict(size=7, color=COLOR[m]),
            customdata=np.c_[s["mean_pct"], s["n"]],
            hovertemplate="median %{y:+.2f}%<br>mean %{customdata[0]:+.2f}% over %{customdata[1]} steps<extra>"
                          + _short(m) + "</extra>"))
    fig.add_hline(y=0, line_color=INK2, line_width=1.2)
    _theme(fig, height=400, legend=True, hover="x unified", top=62)
    fig.update_xaxes(tickvals=list(range(1, 10)), ticktext=[f"{j}→{j + 1}" for j in range(1, 10)],
                     showgrid=False, title_text="Development step", tickangle=0, tickfont=dict(size=11))
    fig.update_yaxes(ticksuffix="%", showgrid=True, title_text="Growth factor vs chain-ladder's")
    return fig


def fig_payout(q: pd.DataFrame):
    """How fast claims get paid: the share of the final amount (cumulative paid at lag 10) already paid at
    each development lag. q has the columns lag, p10, p50, p90 (in percent)."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=q["lag"], y=q["p90"], mode="lines", line=dict(width=0), hoverinfo="skip",
                             showlegend=False))
    fig.add_trace(go.Scatter(x=q["lag"], y=q["p10"], mode="lines", line=dict(width=0), fill="tonexty",
                             fillcolor=rgba(INK, 0.10), hoverinfo="skip", name="Middle 80% of accident years"))
    labels = [f"{v:.0f}%" if int(l) <= 4 else "" for l, v in zip(q["lag"], q["p50"])]
    fig.add_trace(go.Scatter(
        x=q["lag"], y=q["p50"], mode="lines+markers+text", name="Median", text=labels, textposition="bottom right",
        textfont=dict(color=INK, size=12.5), line=dict(color=INK, width=2.6), marker=dict(size=8, color=INK),
        customdata=np.c_[q["p10"], q["p90"]],
        hovertemplate=("by year-end %{x}: median %{y:.0f}% paid<br>middle 80%: %{customdata[0]:.0f}% to "
                       "%{customdata[1]:.0f}%<extra></extra>")))
    _theme(fig, height=370, legend=True, top=40)
    fig.update_xaxes(tickvals=list(range(1, 11)), range=[0.7, 10.3], showgrid=False,
                     title_text="Development lag (year-end 1 to 10)")
    fig.update_yaxes(ticksuffix="%", range=[0, 104], showgrid=True, title_text="Share of the final amount paid")
    return fig
