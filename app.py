"""
Streamlit app: can machine learning beat the chain-ladder on loss reserving?

Run with: streamlit run app.py
Reads only app_data/ (written by pipeline steps 02 to 06), so it needs no database and no scikit-learn. Scoring comes from scoring.py and charts from viz.py.
"""

import inspect
import json
import math
import random
from pathlib import Path

import pandas as pd
import streamlit as st

import scoring
import viz

# --------------------------------------------------------------------- settings
AUTHOR = "Ahmad Nawaz"
LINKEDIN_URL = ""      # for example "https://www.linkedin.com/in/your-name"; shown only when set
GITHUB_URL = ""        # for example "https://github.com/your-name/insurance-loss-reserve"; shown only when set
DEMO_COMPANY = 965     # Secura Insurance, the company used in the single-company walk-through
DATA_DIR = Path(__file__).parent / "app_data"

st.set_page_config(page_title="Chain-ladder vs machine learning", page_icon="📈", layout="wide",
                   initial_sidebar_state="collapsed")

CSS = """
<style>
.block-container {max-width: 1180px; padding-top: 4.2rem; padding-bottom: 3rem;}
footer {visibility: hidden;}
h1 {font-weight: 750; letter-spacing: -0.02em; line-height: 1.12; margin-bottom: 0.3rem;}
h2, h3 {letter-spacing: -0.01em;}
.eyebrow {color: __INK2__; font-size: 0.82rem; letter-spacing: 0.08em; text-transform: uppercase; font-weight: 600;}
.lead {font-size: 1.12rem; line-height: 1.55; color: __INK2__; max-width: 60rem; margin: 0.2rem 0 0.4rem 0;}
.byline {color: __MUTED__; font-size: 0.88rem; margin-bottom: 0.6rem;}
.byline a {color: __INK2__;}
.tile {background: #ffffff; border: 1px solid rgba(11,11,11,0.10); border-radius: 12px; padding: 14px 16px 12px 16px;
       box-sizing: border-box;}
.tile .tlabel {font-size: 0.86rem; color: __INK2__; font-weight: 600; display: flex; align-items: center; gap: 8px;}
.tile .chip {display: inline-block; width: 11px; height: 11px; border-radius: 3px;}
.tile .tvalue {font-size: 2.05rem; font-weight: 750; letter-spacing: -0.02em; line-height: 1.15; margin-top: 4px; color: __INK__;}
.tile.small .tvalue {font-size: 1.32rem; line-height: 1.3; margin-top: 7px; margin-bottom: 3px;}
.tile .tsub {font-size: 0.82rem; color: __INK2__; margin-top: 1px;}
.tile .tfoot {font-size: 0.84rem; color: __INK__; margin-top: 8px; padding-top: 7px; border-top: 1px solid rgba(11,11,11,0.08);
              line-height: 1.4;}
.tile .tfoot span {color: __INK2__;}
.card {background: #ffffff; border: 1px solid rgba(11,11,11,0.10); border-radius: 12px; padding: 14px 16px;
       box-sizing: border-box;}
.card .ctitle {font-weight: 700; font-size: 0.98rem; margin-bottom: 4px; color: __INK__;}
.card .cbody {font-size: 0.92rem; line-height: 1.5; color: __INK2__;}
.card .cnum {font-size: 0.78rem; font-weight: 700; color: __MUTED__; letter-spacing: 0.06em;}
.grid {display: grid; gap: 10px; align-items: stretch;}
.g3 {grid-template-columns: repeat(3, minmax(0, 1fr));}
.g4 {grid-template-columns: repeat(4, minmax(0, 1fr));}
@media (max-width: 900px) {.g3 {grid-template-columns: 1fr;} .g4 {grid-template-columns: repeat(2, minmax(0, 1fr));}}
@media (max-width: 520px) {.g4 {grid-template-columns: 1fr;}}
.answer {background: #f1f0ec; border-radius: 12px; padding: 13px 18px; font-size: 1.04rem; line-height: 1.55; color: __INK__;}
@media (max-width: 640px) {
  div[data-testid="stColumn"]:has([class*="st-key-pick"]) {flex: 1 1 calc(50% - 12px) !important; min-width: calc(50% - 12px) !important;}
  .st-key-cmp_age .bartext, .st-key-cmp_size .bartext {display: none;}
}
.stTabs [data-baseweb="tab-list"] {gap: 6px;}
.stTabs [data-baseweb="tab"] {font-weight: 600; padding-left: 10px; padding-right: 10px;}
div[data-testid="stCaptionContainer"], div[data-testid="stCaptionContainer"] p {color: __INK2__ !important; opacity: 1 !important;}
</style>
"""
CSS = CSS.replace("__INK2__", viz.INK2).replace("__INK__", viz.INK).replace("__MUTED__", viz.MUTED)
st.markdown(CSS, unsafe_allow_html=True)


# ----------------------------------------------------------- small compat helpers
def _accepts(fn, name):
    try:
        return name in inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False


def show(fig, key):
    """Draw a plotly chart full width, with the toolbar off, on old and new Streamlit alike."""
    kw = dict(config={"displayModeBar": False, "responsive": True, "scrollZoom": False}, key=key)
    if _accepts(st.plotly_chart, "theme"):
        kw["theme"] = None
    if _accepts(st.plotly_chart, "width"):
        st.plotly_chart(fig, width="stretch", **kw)
    else:
        st.plotly_chart(fig, use_container_width=True, **kw)


def table(df, height=None):
    kw = {"hide_index": True}
    if height:
        kw["height"] = height
    if _accepts(st.dataframe, "width"):
        kw["width"] = "stretch"
    else:
        kw["use_container_width"] = True
    st.dataframe(df, **kw)


def card():
    """A bordered box where the installed Streamlit supports it."""
    try:
        return st.container(border=True)
    except TypeError:
        return st.container()


def button(label, **kw):
    if _accepts(st.button, "width"):
        return st.button(label, width="stretch", **kw)
    return st.button(label, use_container_width=True, **kw)


def html(s):
    """Raw HTML block. A dollar sign is written as an entity so that two of them never turn into a math formula."""
    st.markdown(s.replace("$", "&#36;"), unsafe_allow_html=True)


def md(text):
    """st.markdown with dollar signs escaped (a pair of bare ones would be drawn as a math formula)."""
    st.markdown(text.replace("$", "\\$"))


def cap(text):
    st.caption(text.replace("$", "\\$"))


# ------------------------------------------------------------------------- data
@st.cache_data(show_spinner=False)
def load():
    need = ["ultimate_results.csv", "companies.csv", "triangles.csv", "projection_paths.csv", "one_step_results.csv",
            "feature_importance.csv", "method_vs_cl_by_lag.csv", "run_info.json"]
    missing = [f for f in need if not (DATA_DIR / f).exists()]
    if missing:
        return {"missing": missing}
    out = {
        "res": pd.read_csv(DATA_DIR / "ultimate_results.csv"),
        "comp": pd.read_csv(DATA_DIR / "companies.csv"),
        "tri": pd.read_csv(DATA_DIR / "triangles.csv"),
        "paths": pd.read_csv(DATA_DIR / "projection_paths.csv"),
        "one": pd.read_csv(DATA_DIR / "one_step_results.csv"),
        "imp": pd.read_csv(DATA_DIR / "feature_importance.csv"),
        "lag": pd.read_csv(DATA_DIR / "method_vs_cl_by_lag.csv"),
        "info": json.loads((DATA_DIR / "run_info.json").read_text()),
    }
    out["comp"]["size_tier"] = pd.Categorical(out["comp"]["size_tier"], ["small", "medium", "large"], ordered=True)
    return out


@st.cache_data(show_spinner=False)
def full_scores():
    d = load()
    return scoring.scorecard(d["res"], scoring.METHODS + scoring.ROBUSTNESS)


@st.cache_data(show_spinner=False)
def filtered_scores(sizes: tuple, ages: tuple):
    """Scores for a slice of the data: chosen company sizes and accident-year age groups."""
    d = load()
    res = scoring.with_age(d["res"])
    tier = d["comp"].set_index("GRCODE")["size_tier"].astype(str)
    res = res[res["GRCODE"].map(tier).isin(sizes) & res["age"].isin(ages)].drop(columns="age")
    if res["GRCODE"].nunique() < 3:
        return None
    return res


@st.cache_data(show_spinner=False)
def payout_quantiles():
    """Share of the final amount already paid at each lag: 10th, 50th and 90th percentile over all accident years."""
    pat = load()["tri"].pivot_table(index=["GRCODE", "AccidentYear"], columns="DevelopmentLag", values="CumPaidLoss")
    pat = pat[pat[10] > 0]
    q = (pat.div(pat[10], axis=0) * 100).quantile([0.1, 0.5, 0.9]).T
    q.columns = ["p10", "p50", "p90"]
    return q.rename_axis("lag").reset_index()


D = load()
if "missing" in D:
    st.error("The app data is missing: " + ", ".join(D["missing"]) + ". "
             "Run the pipeline (steps 02 to 06) so that app_data/ is filled, then reload.")
    st.stop()

RES, COMP, TRI, PATHS = D["res"], D["comp"], D["tri"], D["paths"]
INFO = D["info"]
SC, PER = full_scores()
NAMES_BY_CODE = COMP.set_index("GRCODE")["name"].to_dict()
TIER_BY_CODE = COMP.set_index("GRCODE")["size_tier"].astype(str).to_dict()
SCOPE = scoring.scope(RES)
STILL = INFO["still_to_pay_thousands"]
CHAIN4 = viz.MAIN4
ALL5 = viz.CHART_ORDER
LONG = scoring.NAMES
SHORT = scoring.SHORT


def tile(label, value, sub, color=None, foot="", small=False):
    chip = f"<span class='chip' style='background:{color}'></span>" if color else ""
    foot_html = f"<div class='tfoot'>{foot}</div>" if foot else ""
    return (f"<div class='tile{' small' if small else ''}'><div class='tlabel'>{chip}{label}</div>"
            f"<div class='tvalue'>{value}</div><div class='tsub'>{sub}</div>{foot_html}</div>")


def card_html(num, title, body):
    return (f"<div class='card'><div class='cnum'>{num}</div><div class='ctitle'>{title}</div>"
            f"<div class='cbody'>{body}</div></div>")


def grid(items, cols):
    """Equal-height boxes in one row (3 or 4 columns); they stack on narrow screens."""
    return f"<div class='grid g{cols}'>" + "".join(items) + "</div>"


def money(thousands, digits=1):
    """Amounts in the file are $000. Print them as $k, $m or $bn."""
    v = float(thousands)
    sign = "-" if v < 0 else ""
    v = abs(v)
    if v >= 1_000_000:
        return f"{sign}${v / 1_000_000:,.2f}bn"
    if v >= 1000:
        return f"{sign}${v / 1000:,.{digits}f}m"
    return f"{sign}${v:,.0f}k"


def pts(v, d=2):
    return f"{v:+.{d}f}"


def verdict_word(lo, hi):
    return "beats" if hi < 0 else ("loses to" if lo > 0 else "is not clearly different from")


AGE_SHORT = {"old": "Old (1999-2002)", "middle": "Middle (2003-2004)", "young": "Young (2005-2007)"}


# ----------------------------------------------------------------------- header
html("<div class='eyebrow'>Insurance loss reserving</div>")
st.title("Can machine learning beat the chain-ladder?")
html(f"<div class='lead'>A test on {SCOPE['companies']} US auto insurers. Every method sees only what was known at the "
     f"end of {INFO['valuation_year']}, projects what each accident year will finally cost, and is scored against "
     f"what was actually paid.</div>")
links = []
if LINKEDIN_URL:
    links.append(f"<a href='{LINKEDIN_URL}' target='_blank'>LinkedIn</a>")
if GITHUB_URL:
    links.append(f"<a href='{GITHUB_URL}' target='_blank'>Code on GitHub</a>")
html(f"<div class='byline'>By {AUTHOR} &nbsp;·&nbsp; Data: NAIC Schedule P, private passenger auto"
     + ("".join(f" &nbsp;·&nbsp; {x}" for x in links)) + "</div>")

tab_over, tab_co, tab_cmp, tab_hood, tab_data = st.tabs(
    ["Overview", "Explore a company", "Compare methods", "Under the hood", "Data"])


# ============================================================================ OVERVIEW
with tab_over:
    cl_r = SC.loc["CL"]
    h, c, mb, bf = SC.loc["ML_HIST"], SC.loc["CORR"], SC.loc["ML"], SC.loc["BF"]
    best_ml, worst_ml = sorted([h["mean_company_miss_pct"], c["mean_company_miss_pct"]])
    html("<div style='height:14px'></div>")
    html(f"<div class='answer'><b>Short answer: partly.</b> Given each company's own history, or used to correct the "
         f"chain-ladder, machine learning misses the final amount by {best_ml:.2f}% to {worst_ml:.2f}% for the average company, "
         f"against {cl_r['mean_company_miss_pct']:.2f}% for the chain-ladder. For the typical company and in dollars the "
         f"gap is small, so the case rests on the average and on the younger accident years, not on a sweeping win.</div>")

    html("<div style='height:14px'></div>")
    foots = {}
    for m in CHAIN4:
        r = SC.loc[m]
        if m == "CL":
            foots[m] = "<span>The baseline every other method is compared with</span>"
        else:
            word = verdict_word(r["vs_cl_ci_low"], r["vs_cl_ci_high"])
            sign = "lower" if r["vs_chain_ladder_pts"] < 0 else "higher"
            foots[m] = (f"{abs(r['vs_chain_ladder_pts']):.2f} points {sign} than chain-ladder<br>"
                        f"<span>95% interval {pts(r['vs_cl_ci_low'])} to {pts(r['vs_cl_ci_high'])}. "
                        f"{'Beats it' if word == 'beats' else ('Worse' if word == 'loses to' else 'No clear difference')}"
                        f"</span>")
    html(grid([tile(LONG[m], f"{SC.loc[m, 'mean_company_miss_pct']:.2f}%", "average company miss", viz.COLOR[m], foots[m])
               for m in CHAIN4], 4))
    html("<div style='height:14px'></div>")
    html(grid([
        card_html("THE JOB", "Set money aside for claims not yet paid",
                  "An insurer must reserve today for claims it will still pay in the years ahead. "
                  "The standard tool, the chain-ladder, measures how fast claims were paid in the past and assumes the "
                  "future repeats it."),
        card_html("THE TEST", f"Go back to the end of {INFO['valuation_year']}",
                  "Each method sees only the data available then and projects the final amount paid for "
                  "every accident year. The data runs to 2016, so every projection can be checked against what really happened."),
        card_html("THE SCORE", "How far off was the projection?",
                  "The miss is the error as a percentage of the true final amount. It is shown three ways: "
                  "for the average company, for the typical company, and weighted by dollars."),
    ], 3))

    st.markdown("")
    st.subheader("How far each method misses the final paid amount")
    cap(f"Lower is better. {SCOPE['projections']} projections from {SCOPE['companies']} companies. "
        "The dotted line marks the chain-ladder.")
    b1, b2, b3 = st.columns(3)
    with b1:
        md("**Average company**")
        cap("Each company's miss averaged over its accident years, then averaged across companies.")
        show(viz.fig_method_bars(SC["mean_company_miss_pct"].to_dict(), CHAIN4), "ov_mean")
    with b2:
        md("**Typical (median) company**")
        cap("The middle company, so a few very bad ones cannot move it.")
        show(viz.fig_method_bars(SC["median_company_miss_pct"].to_dict(), CHAIN4), "ov_median")
    with b3:
        md("**Dollar-weighted, whole book**")
        cap("Total dollars missed over total dollars paid, so large insurers count more.")
        show(viz.fig_method_bars(SC["dollar_weighted_miss_pct"].to_dict(), CHAIN4), "ov_dollar")

    # ---- what the data says (all numbers come from the scorecard)
    d_pair, lo_pair, hi_pair = scoring.bootstrap_pair(PER, "CORR", "ML_HIST")
    old_max = SC.loc[scoring.METHODS, "old_years_median_miss_pct"].max()
    mid_max = SC.loc[scoring.METHODS, "middle_years_median_miss_pct"].max()
    gap_med = max(abs(h["median_company_miss_pct"] - cl_r["median_company_miss_pct"]),
                  abs(c["median_company_miss_pct"] - cl_r["median_company_miss_pct"]))
    if lo_pair < 0 < hi_pair:
        tie_line = (f"**The two ML ideas that work are a tie.** The correction model's average miss is {abs(d_pair):.2f} points "
                    f"{'higher' if d_pair > 0 else 'lower'} than ML + history's, but the 95% interval for that difference "
                    f"({pts(lo_pair)} to {pts(hi_pair)}) includes zero.")
    else:
        ahead = "ML + history" if d_pair > 0 else "The correction model"
        tie_line = (f"**{ahead} is ahead of the other ML idea** by {abs(d_pair):.2f} points of average miss "
                    f"(95% interval {pts(lo_pair)} to {pts(hi_pair)} for correction model minus ML + history).")
    with card():
        md("#### What the data says")
        lines = [
            f"**Company history is what helps.** ML on the base features alone {verdict_word(mb['vs_cl_ci_low'], mb['vs_cl_ci_high'])} "
            f"chain-ladder ({mb['mean_company_miss_pct']:.2f}% vs {cl_r['mean_company_miss_pct']:.2f}%). Adding each company's own "
            f"history, or letting ML correct the chain-ladder factors, "
            f"{verdict_word(h['vs_cl_ci_low'], h['vs_cl_ci_high'])} it for the average company: "
            f"{h['mean_company_miss_pct']:.2f}% and {c['mean_company_miss_pct']:.2f}% vs {cl_r['mean_company_miss_pct']:.2f}%, "
            f"better in {h['beats_chain_ladder_pct']:.0f}% and {c['beats_chain_ladder_pct']:.0f}% of companies.",
            f"**The gain is modest and shows up in the average, not everywhere.** For the typical company the three are within "
            f"{gap_med:.2f} points of each other ({cl_r['median_company_miss_pct']:.2f}% chain-ladder), and in dollars "
            f"chain-ladder is as good as any ({cl_r['dollar_weighted_miss_pct']:.2f}% vs {c['dollar_weighted_miss_pct']:.2f}% "
            f"and {h['dollar_weighted_miss_pct']:.2f}%).",
            tie_line,
            f"**Young accident years carry the differences.** For 2005 to 2007 the median miss is "
            f"{cl_r['young_years_median_miss_pct']:.1f}% for chain-ladder against {c['young_years_median_miss_pct']:.1f}% for the correction model, "
            f"and chain-ladder tends to project too high ({cl_r['young_years_bias_pct']:+.1f}% median error). "
            f"For 1999 to 2002 every method's median miss is {old_max:.2f}% or less, and for 2003 and 2004 it is {mid_max:.2f}% or less.",
            f"**Bornhuetter-Ferguson is the weakest here** ({bf['mean_company_miss_pct']:.2f}% average miss), partly because its "
            f"expected loss ratio is estimated from the chain-ladder ultimates and earned premiums of the five oldest accident years.",
        ]
        for ln in lines:
            md("- " + ln)

    st.markdown("")
    r1, r2 = st.columns([1, 1])
    with r1:
        st.subheader("If each method had set the reserve for the whole book")
        md(f"Add up every projection and compare it with the money that was actually still to be paid at the end of "
           f"{INFO['valuation_year']} ({money(STILL)} across the {SCOPE['companies']} companies). Negative means the method would "
           f"have set aside too little. Overs and unders across companies cancel out here, so this is the view of the "
           f"whole portfolio rather than of any one insurer.")
        cap("Net reserve error, % of the money still to be paid. Hover a bar for dollars.")
    with r2:
        show(viz.fig_reserve_error(SC, STILL, ALL5), "ov_reserve")

    st.markdown("")
    f1, f2 = st.columns([1, 1])
    with f1:
        st.subheader("How sure are we?")
        md("We redrew the companies at random 5,000 times and recomputed each difference. The dot is the best estimate and "
           "the line covers 95% of the redraws. When a line crosses zero, the data cannot rule out that the method is no "
           "better than chain-ladder. Only 95 companies and one valuation date are involved, so treat the intervals as "
           "the honest size of the evidence.")
        cap("Difference in each company's average miss, method minus chain-ladder.")
    with f2:
        show(viz.fig_forest(SC, ["ML", "ML_HIST", "CORR", "BF"]), "ov_forest")

    with st.expander("The five methods in plain English"):
        md(
            "- **Chain-ladder.** Measures how much the cumulative paid amount grows from each development lag to the next, "
            "using the company's older accident years, then multiplies those growth factors forward to lag 10.\n"
            "- **Bornhuetter-Ferguson.** Takes what has been paid and adds an expected share of an a-priori loss "
            "(premium times an expected loss ratio) for the part still unpaid. The expected loss ratio here comes from the five oldest accident years.\n"
            "- **ML, base features.** A gradient-boosted tree model predicts the growth to the next lag from four features: "
            "development lag, paid-to-premium, growth over the previous lag and company size. It projects step by step to lag 10.\n"
            "- **ML + company history.** The same model, given the company's own average growth at that lag from older accident years.\n"
            "- **Chain-ladder + ML correction.** Starts from the chain-ladder growth factor and lets ML predict the correction to it, "
            "using how far earlier steps of the same accident year landed from chain-ladder.")
    with st.expander("Glossary"):
        md(
            "- **Accident year:** the year the accidents happened.\n"
            "- **Development lag:** how many year-ends after the accident year start we look. Lag 1 is the end of the accident year, lag 10 is nine years later.\n"
            "- **Cumulative paid:** all claim payments for an accident year up to that lag.\n"
            f"- **Valuation date:** the end of {INFO['valuation_year']}. Cells after it are the future the methods try to predict.\n"
            "- **Final amount:** cumulative paid at lag 10, the last point in the data. It is the target every method is scored on.\n"
            "- **Miss:** absolute error of a projected final amount, as a percentage of the true final amount.\n"
            "- **Reserve:** projected final amount minus what has been paid so far.\n"
            f"- **Accident year 1998** is already fully paid at the valuation date, so it is shown but not scored.")
    with st.expander("Full scorecard (every number behind these charts)"):
        view = SC.copy()
        view.insert(0, "Method", [LONG[m] for m in view.index])
        view = view.rename(columns={
            "mean_company_miss_pct": "Average company miss %", "median_company_miss_pct": "Median company miss %",
            "worst_company_miss_pct": "Worst company miss %", "beats_chain_ladder_pct": "Beats chain-ladder, % of companies",
            "dollar_weighted_miss_pct": "Dollar-weighted miss %", "total_error_pct": "Total error %",
            "net_reserve_error_pct": "Net reserve error %", "vs_chain_ladder_pts": "vs chain-ladder, points",
            "vs_cl_ci_low": "Interval low", "vs_cl_ci_high": "Interval high",
            "old_years_median_miss_pct": "Old years median miss %", "middle_years_median_miss_pct": "Middle years median miss %",
            "young_years_median_miss_pct": "Young years median miss %", "young_years_bias_pct": "Young years bias %"})
        table(view.round(2).reset_index(drop=True))
        cap("The last two rows are robustness checks, not part of the headline: ML + history without the hist_n feature, "
            "and chain-ladder plus a simple average correction per development lag.")


# ======================================================================= EXPLORE A COMPANY
def _pick(code):
    st.session_state["company"] = int(code)


def _pick_random():
    st.session_state["company"] = int(random.choice(sorted(NAMES_BY_CODE)))


with tab_co:
    md("Pick any of the "
       f"{SCOPE['companies']} companies. The triangle is the data an actuary would have had at the end of "
       f"{INFO['valuation_year']}. Below it you can see what each method projected and what really happened.")
    TBL = scoring.company_table(RES, COMP, scoring.METHODS)
    codes_sorted = COMP.sort_values("name")["GRCODE"].tolist()
    if "company" not in st.session_state or st.session_state["company"] not in NAMES_BY_CODE:
        st.session_state["company"] = DEMO_COMPANY if DEMO_COMPANY in NAMES_BY_CODE else codes_sorted[0]

    st.selectbox("Company (type to search)", codes_sorted, key="company",
                 format_func=lambda c: f"{NAMES_BY_CODE[c]} ({c})")
    gain = TBL.assign(g=TBL["CL"] - TBL["ML_HIST"])
    picks = [
        ("Secura (the demo)", DEMO_COMPANY),
        ("Chain-ladder's worst", int(TBL.loc[TBL["CL"].idxmax(), "GRCODE"])),
        ("Biggest ML gain", int(gain.loc[gain["g"].idxmax(), "GRCODE"])),
        ("Biggest ML loss", int(gain.loc[gain["g"].idxmin(), "GRCODE"])),
        ("A typical company", int(TBL.loc[(TBL["CL"] - TBL["CL"].median()).abs().idxmin(), "GRCODE"])),
    ]
    pc = st.columns(len(picks) + 1)
    for i, (label, code) in enumerate(picks):
        with pc[i]:
            button(label, key=f"pick{i}", on_click=_pick, args=(code,))
    with pc[-1]:
        button("Surprise me", key="pick_rand", on_click=_pick_random)

    code = int(st.session_state["company"])
    name = NAMES_BY_CODE[code]
    t, p, r = TRI[TRI["GRCODE"] == code], PATHS[PATHS["GRCODE"] == code], RES[RES["GRCODE"] == code]
    rs = scoring.scored(r)
    row = TBL[TBL["GRCODE"] == code].iloc[0]
    tier = TIER_BY_CODE[code]
    to_pay = float((rs["actual"] - rs["paid_to_date"]).sum())
    md(f"### {name}")
    cap(f"Group code {code}. {tier.capitalize()} insurer in this set (average yearly premium {money(row['avg_premium'])}). "
        f"Still to be paid at the end of {INFO['valuation_year']}: {money(to_pay)}.")

    best = row["best_method"]
    tiles = [tile("Chain-ladder", f"{row['CL']:.2f}%", "average miss for this company", viz.COLOR["CL"],
                  "<span>The baseline</span>")]
    for m in ["ML_HIST", "CORR"]:
        d = row[m] - row["CL"]
        tiles.append(tile(LONG[m], f"{row[m]:.2f}%", "average miss for this company", viz.COLOR[m],
                          f"{abs(d):.2f} points {'lower' if d < 0 else 'higher'} than chain-ladder"))
    tiles.append(tile("Closest of the five here", LONG[best], f"average miss {row[best]:.2f}%", viz.COLOR[best],
                      "<span>Ten accident years is a small sample, so read this as one story, not a ranking.</span>",
                      small=True))
    html(grid(tiles, 4))
    html("<div style='height:8px'></div>")

    # ---- the triangle
    md("#### The claims triangle")
    cap("Each row is an accident year. Each column is how many year-ends later we look at the cumulative amount paid. "
        "Blue cells were known at the end of 2007; the bold line marks the edge of what was known.")
    t1, t2 = st.columns([3, 2])
    with t1:
        fill_opts = {"Nothing": ("none", ""), "What happened": ("actual", "what actually happened"),
                     "Chain-ladder": ("CL", "chain-ladder projection"),
                     "ML + history": ("ML_HIST", "ML + history projection"),
                     "CL + ML correction": ("CORR", "chain-ladder + ML correction projection")}
        fill_label = st.radio("Fill the empty corner with", list(fill_opts), horizontal=True, key="fill")
    with t2:
        unit_label = st.radio("Show amounts as", ["Dollars", "% of premium"], horizontal=True, key="unit")
    mode, fname = fill_opts[fill_label]
    unit = "premium" if unit_label.startswith("%") else "money"
    if mode in ("none", "actual"):
        fig = viz.fig_triangle(t, fill=mode, unit=unit, fill_name=fname)
    else:
        fig = viz.fig_triangle(t, fill="path", paths=p[p["method"] == mode], unit=unit, fill_name=fname)
    unit_text = fig.layout.meta["unit_label"] if unit == "money" else "% of that year's premium"
    show(fig, "co_triangle")
    cap(f"Amounts are cumulative paid, in {unit_text}. "
        "Gray cells were not known at the end of 2007 and are shown only for comparison.")

    # ---- one accident year
    md("#### One accident year in detail")
    ays = sorted(int(a) for a in rs["AccidentYear"].unique())
    s1, s2 = st.columns([3, 2])
    with s1:
        ay = st.select_slider("Accident year", options=ays, value=ays[-1], key="ay")
    with s2:
        zoom = st.checkbox("Zoom in on the projection", value=True, key="zoom")
    row_ay = r[r["AccidentYear"] == ay].iloc[0]
    show(viz.fig_curve(t, ay, p, row_ay, zoom=zoom), "co_curve")
    cap("Solid black: paid so far. Dashed: what was actually paid later. Dotted colors: each method's projection from the "
        "last known point. The diamond is the Bornhuetter-Ferguson final amount. Hover for exact values.")
    rows = []
    for m in ["CL", "BF", "ML", "ML_HIST", "CORR"]:
        rows.append({"Method": LONG[m], "Projected final": money(row_ay[m]), "Error": f"{row_ay[f'{m}_err']:+.2f}%",
                     "Projected still to pay": money(row_ay[m] - row_ay["paid_to_date"])})
    rows.append({"Method": "What actually happened", "Projected final": money(row_ay["actual"]), "Error": "",
                 "Projected still to pay": money(row_ay["actual"] - row_ay["paid_to_date"])})
    table(pd.DataFrame(rows))

    # ---- errors by accident year
    md("#### How each method did, year by year")
    cap("Signed error of the projected final amount. Above zero means the method projected more than was finally paid. "
        "Click a name in the legend to show or hide it (Bornhuetter-Ferguson starts hidden).")
    show(viz.fig_company_errors(r), "co_errors")

    # ---- the reserve this company would have held
    md("#### The reserve each method would have set")
    rc1, rc2 = st.columns([3, 2])
    with rc1:
        show(viz.fig_company_reserve(r), "co_reserve")
    with rc2:
        md("The black bar is the money this company really still had to pay at the end of "
           f"{INFO['valuation_year']}. Each colored bar is the reserve a method would have held for the same accident years "
           "(projected final amount minus paid so far, added up).")
        md("Within one company, overs in some accident years cancel unders in others. A method can look close here and still "
           "miss single years by more, as the chart above shows.")
        cap("Hover a bar for the dollars and the gap to the actual.")

    with st.expander("Reserve table for all accident years"):
        rv = rs.sort_values("AccidentYear")
        view = pd.DataFrame({
            "Accident year": rv["AccidentYear"].to_numpy(),
            "Paid so far": [money(x) for x in rv["paid_to_date"]],
            "Actually still to pay": [money(a - b) for a, b in zip(rv["actual"], rv["paid_to_date"])],
            "Chain-ladder": [money(a - b) for a, b in zip(rv["CL"], rv["paid_to_date"])],
            "ML + history": [money(a - b) for a, b in zip(rv["ML_HIST"], rv["paid_to_date"])],
            "CL + ML correction": [money(a - b) for a, b in zip(rv["CORR"], rv["paid_to_date"])],
        })
        total = pd.DataFrame([{
            "Accident year": "Total", "Paid so far": money(rv["paid_to_date"].sum()),
            "Actually still to pay": money((rv["actual"] - rv["paid_to_date"]).sum()),
            "Chain-ladder": money((rv["CL"] - rv["paid_to_date"]).sum()),
            "ML + history": money((rv["ML_HIST"] - rv["paid_to_date"]).sum()),
            "CL + ML correction": money((rv["CORR"] - rv["paid_to_date"]).sum())}])
        view["Accident year"] = view["Accident year"].astype(str)
        table(pd.concat([view, total], ignore_index=True))
        cap("Reserve = projected final amount minus paid so far. Accident year 1998 is fully paid and not listed.")
    st.download_button(f"Download {name} results (CSV)", r.to_csv(index=False).encode(), f"company_{code}_results.csv",
                       "text/csv", key="dl_company")


# ============================================================================ COMPARE
with tab_cmp:
    md("Slice the results by company size and by how old the accident years are. These slices are exploratory: "
       "cutting 95 companies into groups leaves few in each, so small differences between groups are noise.")
    f_a, f_b = st.columns(2)
    with f_a:
        sizes = st.multiselect("Company size", ["small", "medium", "large"], default=["small", "medium", "large"],
                               format_func=str.capitalize, key="f_size")
    with f_b:
        ages = st.multiselect("Accident years", scoring.AGE_ORDER, default=scoring.AGE_ORDER,
                              format_func=lambda a: AGE_SHORT[a], key="f_age")
    sub = filtered_scores(tuple(sizes), tuple(ages)) if sizes and ages else None
    if sub is None:
        st.warning("Pick at least one size and one age group, with at least three companies in the slice.")
    else:
        n_co = sub["GRCODE"].nunique()
        if n_co < 15:
            st.warning(f"Only {n_co} companies in this slice. Differences here are very uncertain.")
        sc_f, per_f = scoring.scorecard(sub, scoring.METHODS, n_boot=0)
        tbl_f = scoring.company_table(sub, COMP, scoring.METHODS)
        cap(f"In view: {n_co} companies, {len(scoring.scored(sub))} projections.")

        md("#### Average and typical miss for this slice")
        a1, a2 = st.columns(2)
        with a1:
            md("**Average company**")
            show(viz.fig_method_bars(sc_f["mean_company_miss_pct"].to_dict(), CHAIN4), "cmp_mean")
        with a2:
            md("**Typical (median) company**")
            show(viz.fig_method_bars(sc_f["median_company_miss_pct"].to_dict(), CHAIN4), "cmp_median")

        md("#### By age of the accident year")
        cap("Median miss of the final amount. Young years have little paid so far, so there is far more to project.")
        sub_size_only = filtered_scores(tuple(sizes), tuple(scoring.AGE_ORDER))
        sc_age, _ = scoring.scorecard(sub_size_only, scoring.METHODS, n_boot=0)
        show(viz.fig_by_age(sc_age, ALL5), "cmp_age")

        md("#### Year by year")
        hm = st.radio("Show", ["Median signed error (bias)", "Median miss (absolute)"], horizontal=True, key="hm_stat")
        stat = "bias" if hm.startswith("Median signed") else "miss"
        eby = scoring.error_by_accident_year(sub, ALL5, stat)
        if stat == "bias":
            cap("Blue: the method projected too low. Red: too high. Gray: close to right. Young accident years sit at the bottom.")
        else:
            cap("Darker means a bigger typical miss.")
        show(viz.fig_error_heatmap(eby, ALL5, stat), "cmp_heat")

        md("#### How wide are the errors?")
        cap("Signed error of the projected final amount: above 0 means too high. Each box covers the middle half of the errors "
            "and the whiskers run from the 10th to the 90th percentile. Taller means less predictable.")
        show(viz.fig_error_spread(sub, ALL5), "cmp_spread")

        md("#### Company by company")
        mc = st.radio("Method to compare with chain-ladder", ["ML", "ML_HIST", "CORR"], index=2, horizontal=True,
                      format_func=lambda m: LONG[m], key="cmp_method")
        g1, g2 = st.columns([1, 1])
        with g1:
            md("**One dot per company**")
            cap("Each dot is a company's average miss: chain-ladder across, the chosen method up. "
                "Below the dashed line, the method did better. Hover for the name.")
            show(viz.fig_scatter(tbl_f, mc), "cmp_scatter")
        with g2:
            md("**Where it helps most and hurts most**")
            cap("The eight biggest gains and the eight biggest losses against chain-ladder, in points of average miss.")
            show(viz.fig_win_loss(tbl_f, mc), "cmp_winloss")

        md("**How the misses are spread across companies**")
        cap("Pick a miss level along the bottom. The height of a line is the share of companies that miss by less than that. "
            "A line that sits higher or further left is better.")
        show(viz.fig_ecdf(per_f, ALL5), "cmp_ecdf")

        md("#### By company size")
        cap("Average company miss by size tier (the smallest, middle and largest third by premium).")
        sub_age_only = filtered_scores(("small", "medium", "large"), tuple(ages))
        show(viz.fig_size_bars(scoring.size_table(sub_age_only, scoring.METHODS, COMP), ALL5), "cmp_size")

        with st.expander("Table view: the numbers behind these charts"):
            md("**Every company in this slice**, sorted by chain-ladder miss. Click a column heading to sort.")
            ct = tbl_f.sort_values("CL", ascending=False).copy()
            ct["size_tier"] = ct["size_tier"].astype(str)
            ct["best_method"] = ct["best_method"].map(LONG)
            ct = ct.rename(columns={"GRCODE": "Group code", "name": "Company", "size_tier": "Size",
                                    "avg_premium": "Average yearly premium ($000)", "best_method": "Closest method",
                                    **{m: f"{SHORT[m]} miss %" for m in scoring.METHODS}})
            table(ct.round(2))
            st.download_button("Download this table (CSV)", ct.round(4).to_csv(index=False).encode(),
                               "companies_in_slice.csv", "text/csv", key="dl_slice")
            md("**By age of the accident year**, median miss % (all selected sizes)")
            ta = sc_age[[f"{a}_years_median_miss_pct" for a in scoring.AGE_ORDER]].copy()
            ta.columns = [AGE_SHORT[a] for a in scoring.AGE_ORDER]
            ta.insert(0, "Method", [LONG[m] for m in ta.index])
            table(ta.round(2).reset_index(drop=True))
            md("**How wide are the errors**, signed error % by age of the accident year")
            sp = scoring.error_spread(sub, ALL5)
            sp["age"] = sp["age"].map(AGE_SHORT)
            sp["method"] = sp["method"].map(LONG)
            sp = sp.rename(columns={"age": "Accident years", "method": "Method", "n": "Projections", "q10": "10th pct",
                                    "q25": "25th pct", "q50": "Median", "q75": "75th pct", "q90": "90th pct"})
            table(sp.round(2))

        with st.expander("Robustness checks (full data)"):
            rb = SC.loc[["CL", "ML_HIST", "ML_HIST_NO_N", "CORR", "CL_BIAS"],
                        ["mean_company_miss_pct", "median_company_miss_pct", "dollar_weighted_miss_pct",
                         "vs_chain_ladder_pts", "vs_cl_ci_low", "vs_cl_ci_high"]].copy()
            rb.insert(0, "Method", [LONG[m] for m in rb.index])
            rb.columns = ["Method", "Average company miss %", "Median company miss %", "Dollar-weighted miss %",
                          "vs chain-ladder, points", "Interval low", "Interval high"]
            table(rb.round(2).reset_index(drop=True))
            cap("ML + history without its hist_n feature still beats chain-ladder, so the result does not hinge on that one feature. "
                "Chain-ladder plus a plain average correction per development lag barely moves, so the gain comes from what "
                "the model learns, not from a blanket adjustment.")


# ========================================================================== UNDER THE HOOD
with tab_hood:
    st.subheader("How the pipeline fits together")
    steps = [
        ("01", "Load and check", "The NAIC file goes into SQLite with quality checks and an is_known flag: was this cell known at the end of 2007?"),
        ("02", "Benchmark", "Chain-ladder and Bornhuetter-Ferguson on every company with a full 10 by 10 history."),
        ("03", "One-step tests", "Which ML variant predicts the next lag best, on three calendar splits. The winner is picked by a rule fixed in advance."),
        ("04", "Projection", "Roll every accident year forward to lag 10, one step at a time, with each step's features rebuilt from older years."),
        ("05", "Scorecard", "Every headline number and chart, defined once and shared with this app."),
        ("06", "Export", "Freeze the data this app reads, so it runs without the database."),
    ]
    html(grid([card_html(f"STEP {n}", t_, b) for n, t_, b in steps], 3))

    st.markdown("")
    w1, w2 = st.columns(2)
    with w1:
        st.subheader("Why predict growth, not dollars")
        md(
            "- A growth factor is the same kind of quantity a chain-ladder estimates, so the comparison is like for like.\n"
            "- The target is the log of next lag's paid over this lag's paid, so an error reads roughly as a percent of the next payment.\n"
            "- Dollar amounts differ by a factor of 100,000 between small and large insurers. Growth rates do not.\n"
            "- Predicting one step and rolling forward lets each prediction feed the next, exactly as chain-ladder does.")
    with w2:
        st.subheader("Guards against peeking at the future")
        md(
            "- Only cells known at the end of 2007 reach the models. The file holds results to 2016; the is_known flag removes them first.\n"
            "- Window functions run after that filter, so a LEAD() cannot reach a future cell.\n"
            "- A company's history features use only older accident years at the same lag, whose step happened earlier.\n"
            "- The model is chosen by a rule fixed in advance, on one-step tests over test years 2005, 2006 and 2007.\n"
            "- Final scores are computed once, after the choice was fixed.\n"
            "- Fed chain-ladder factors, the projector reproduces the step 2 chain-ladder results to 1e-16.")

    st.subheader("What the chain-ladder relies on: a steady payout pattern")
    pq = payout_quantiles().set_index("lag")
    cap("Share of the final amount already paid at each year-end, over every company and accident year in the file "
        "(cells after 2007 included, to show the pattern as it turned out). Line: median. Band: middle 80%.")
    show(viz.fig_payout(pq.reset_index()), "hood_payout")
    md(f"By the end of the first year the median accident year has paid {pq.loc[1, 'p50']:.0f}% of its final amount, and by "
       f"the end of year 3 it has paid {pq.loc[3, 'p50']:.0f}%. The chain-ladder assumes each company's own pattern repeats. "
       f"At year 1 the middle 80% run from {pq.loc[1, 'p10']:.0f}% to {pq.loc[1, 'p90']:.0f}%, so that assumption carries the "
       "most risk for the youngest accident years, which is where the methods differ most.")

    st.subheader("Which model variant predicts the next step best")
    cap("Mean miss of the next payment, in % of that payment, for each variant. Small dots are the three test years "
        "(2005, 2006, 2007); the diamond is their average. Lower is better. The green diamond is the variant the "
        "pre-declared rule selected.")
    show(viz.fig_onestep(D["one"]), "hood_onestep")
    with st.expander("Table view"):
        piv = D["one"].pivot(index="variant", columns="test_year", values="all (mean)")
        piv["Average"] = piv.mean(axis=1)
        piv = piv.sort_values("Average").round(2).reset_index().rename(columns={"variant": "Variant"})
        piv["Variant"] = piv["Variant"].map(lambda v: viz.VARIANT_NAMES.get(v, v))
        piv.columns = [str(c) for c in piv.columns]
        table(piv)

    st.subheader("What the models look at")
    cap("Permutation importance on the 2007 test year: shuffle one feature, see how much worse the one-step miss gets. "
        "Bigger means the model leans on it more.")
    imp_models = {"ML + company history": "ML + history: hist+n", "ML, base features": "ML base features",
                  "CL + ML correction": "Residual R3: + CL deviations"}
    imp_label = st.radio("Model", list(imp_models), horizontal=True, key="imp_model")
    show(viz.fig_importance(D["imp"], imp_models[imp_label]), "hood_imp")

    st.subheader("Where the ML disagrees with chain-ladder")
    cap("For every projected step, how much the method's growth factor differs from the chain-ladder factor for the same "
        "company and step. The lines are medians; the shaded band holds the middle 80% of steps for the chosen method.")
    band = st.radio("Show the spread for", ["ML", "ML_HIST", "CORR"], index=2, horizontal=True,
                    format_func=lambda m: LONG[m], key="band_method")
    show(viz.fig_by_lag(D["lag"], band), "hood_lag")
    late = D["lag"][D["lag"]["from_lag"] >= 6]
    late_med = math.ceil(late["median_pct"].abs().max() * 10) / 10          # rounded up to one decimal
    late_wide = math.ceil(max(late["p10_pct"].abs().max(), late["p90_pct"].abs().max()) * 10) / 10
    md("The methods mostly part ways in the first few steps, while payments are still growing fast. "
       f"From step 6→7 onward the median difference from chain-ladder is under {late_med:.1f}% and the middle 80% of "
       f"steps sit within {late_wide:.1f}%.")

    st.subheader("What this does not show")
    md(
        "- One line of business, one valuation date (end of 2007) and 95 insurers. Other dates or lines may behave differently.\n"
        "- Paid losses only. Case reserves and claim counts, which actuaries also use, are not in the models.\n"
        "- The benchmarks are textbook chain-ladder and Bornhuetter-Ferguson with simple settings, not an actuary's judgment, tail factors, smoothing or outlier handling.\n"
        "- The final amount is cumulative paid at lag 10, not a true ultimate: payments after year 10 are not in the data. All methods are scored on the same target, so the comparison stands.\n"
        f"- The models learn from {INFO['training_steps']:,} growth steps. More data or other model families might shift the result.\n"
        "- The whole-book reserve error nets overs against unders across companies.")
    st.subheader("Run it yourself")
    st.code("pip install -r requirements-pipeline.txt\n"
            "python 01_data_setup.py && python 02_benchmark.py && python 03_one_step_tests.py\n"
            "python 04_ultimate_projection.py && python 05_scorecard.py && python 06_export_app_data.py\n"
            "streamlit run app.py", language="bash")


# ================================================================================== DATA
with tab_data:
    md("Every number in the app comes from these files in `app_data/`. Amounts are in thousands of US dollars, as in the NAIC file.")
    html(grid([tile(label, val, sub) for label, val, sub in [
        ("Companies in the file", f"{INFO['companies_in_file']}", "insurers with any data"),
        ("With a full history", f"{INFO['companies_full_history']}", "all 10 by 10 cells"),
        ("Modelled", f"{INFO['companies_modelled']}", "usable premiums and paid amounts"),
        ("Projections scored", f"{INFO['projections_scored']}", f"{INFO['settled_rows_left_out']} already-paid rows left out")]], 4))
    st.markdown("")
    st.subheader("Results by company and accident year")
    flt = st.multiselect("Show only these companies (leave empty for all)", sorted(NAMES_BY_CODE, key=lambda c: NAMES_BY_CODE[c]),
                         format_func=lambda c: f"{NAMES_BY_CODE[c]} ({c})", key="data_filter")
    view = RES.copy()
    if flt:
        view = view[view["GRCODE"].isin(flt)]
    view.insert(1, "Company", view["GRCODE"].map(NAMES_BY_CODE))
    shown = view.round(3)
    table(shown, height=420)
    cap("actual is the cumulative paid at lag 10. The columns named CL, BF, ML, ML_HIST and CORR are each method's projection "
        "of it; a column ending in _err is that projection's signed error in %. latest_lag is the last development lag known "
        "at the end of 2007 (10 means the year is already fully paid).")
    dl1, dl2, dl3 = st.columns(3)
    with dl1:
        st.download_button("Download results (CSV)", RES.to_csv(index=False).encode(), "ultimate_results.csv", "text/csv")
    with dl2:
        st.download_button("Download scorecard (CSV)", SC.round(4).reset_index().to_csv(index=False).encode(),
                           "scorecard.csv", "text/csv")
    with dl3:
        st.download_button("Download company list (CSV)", COMP.to_csv(index=False).encode(), "companies.csv", "text/csv")
    with st.expander("Data source and notes"):
        md(
            "- **Source:** NAIC Schedule P, private passenger auto, from the public Casualty Actuarial Society loss-reserving data set. "
            "Accident years 1998 to 2007, each followed for ten development years.\n"
            f"- **Companies:** {INFO['companies_in_file']} in the file, {INFO['companies_full_history']} with a full 10 by 10 history, "
            f"{INFO['companies_modelled']} modelled after removing those with non-positive premiums, paid amounts or growth factors.\n"
            "- **Size tiers:** the smallest, middle and largest third of the modelled companies by average yearly net earned premium.\n"
            "- **Posted reserves** in the source file are deliberately not used as a benchmark: they include years before 1998 and a tail beyond "
            "development year 10, so they are not comparable with a lag-10 target.")

html(f"<div class='byline' style='margin-top:2rem'>Educational portfolio project by {AUTHOR}. Not actuarial advice. "
     "Built with pandas, scikit-learn, Streamlit and Plotly.</div>")