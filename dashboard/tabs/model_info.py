import streamlit as st
import pandas as pd
import altair as alt
from dashboard.client import get, safe

SPLITS = [("valid", "Validation (2018)"), ("oot", "Out-of-time (2019)"), ("stress", "Stress (2020 to 2021)")]


def _totals(split_report: dict):
    deciles = split_report.get("deciles") or []
    loans = sum(d.get("loans", 0) for d in deciles)
    defaults = sum(d.get("defaults", 0) for d in deciles)
    mean_pred = (sum(d.get("mean_predicted_probability", 0.0) * d.get("loans", 0) for d in deciles) / loans
                 if loans else None)
    obs = defaults / loans if loans else None
    return loans, defaults, obs, mean_pred


def _f(x, d=4):
    return "N/A" if x is None else f"{x:.{d}f}"


def render(params: dict, meta: dict):
    st.header("Model & Data")

    report, _ = safe(get, "/model-report")
    if not report:
        return

    st.caption(f"Model version **{report.get('model_version', 'n/a')}** | "
               f"Freddie Mac data release **{report.get('data_release', 'n/a')}**")

    # ---- Metrics by split
    st.subheader("Discrimination and calibration metrics by split")
    rows = []
    for key, label in SPLITS:
        s = report.get(key)
        if not s:
            continue
        loans, defaults, obs, mean_pred = _totals(s)
        rows.append({
            "Split": label,
            "Loans": f"{loans:,}",
            "Defaults": f"{defaults:,}",
            "Observed default rate": "N/A" if obs is None else f"{obs * 100:.2f}%",
            "Mean predicted probability": "N/A" if mean_pred is None else f"{mean_pred * 100:.2f}%",
            "ROC-AUC": _f(s.get("auc")),
            "Gini": _f(s.get("gini")),
            "KS": _f(s.get("ks_statistic")),
            "PR-AUC": _f(s.get("pr_auc")),
            "Brier score": _f(s.get("brier_score")),
            "Log loss": _f(s.get("log_loss")),
        })
    if rows:
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
        st.caption("Gini = 2 x AUC - 1. Brier score and log loss are lower-is-better. "
                   "The out-of-time and stress splits were not used for training or tuning.")

    st.divider()

    # ---- Per-split detail
    labels = {k: l for k, l in SPLITS if report.get(k)}
    if not labels:
        st.warning("The model report contains no split results.")
        return
    sel = st.selectbox("Split to inspect", options=list(labels), format_func=lambda k: labels[k],
                       index=list(labels).index("oot") if "oot" in labels else 0, key="model_info_split")
    s = report[sel]
    loans, defaults, obs, mean_pred = _totals(s)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("ROC-AUC", _f(s.get("auc"), 3))
    c2.metric("KS statistic", _f(s.get("ks_statistic"), 3))
    c3.metric("Observed default rate", "N/A" if obs is None else f"{obs * 100:.2f}%")
    c4.metric("Mean predicted probability", "N/A" if mean_pred is None else f"{mean_pred * 100:.2f}%")

    if obs and mean_pred:
        ratio = obs / mean_pred
        if ratio < 0.8 or ratio > 1.25:
            st.warning(
                f"Calibration check: the observed default rate is {ratio:.1f} times the mean predicted "
                "probability on this split. The model ranks loans, but its probabilities are not "
                "calibrated to the observed default rate, which also affects the optimiser threshold."
            )

    st.subheader("Score deciles (decile 1 = lowest predicted risk)")
    dec = pd.DataFrame(s.get("deciles") or [])
    if not dec.empty:
        dec_chart_df = pd.concat([
            pd.DataFrame({"Decile": dec["decile"].astype(int), "Series": "Observed default rate",
                          "Rate (%)": dec["observed_default_rate"] * 100.0}),
            pd.DataFrame({"Decile": dec["decile"].astype(int), "Series": "Mean predicted probability",
                          "Rate (%)": dec["mean_predicted_probability"] * 100.0}),
        ])
        dec_chart = alt.Chart(dec_chart_df).mark_line(point=True).encode(
            x=alt.X("Decile:O", title="Score decile"),
            y=alt.Y("Rate (%):Q", title="Rate (%)"),
            color=alt.Color("Series:N", title=""),
            tooltip=[alt.Tooltip("Decile:O"), alt.Tooltip("Series:N"),
                     alt.Tooltip("Rate (%):Q", format=".2f")],
        ).properties(height=300, title=f"Observed versus predicted by decile ({labels[sel]})")
        st.altair_chart(dec_chart, width="stretch")

        table = pd.DataFrame({
            "Decile": dec["decile"].astype(int),
            "Loans": dec["loans"].astype(int),
            "Defaults": dec["defaults"].astype(int),
            "Observed default rate": (dec["observed_default_rate"] * 100).map("{:.2f}%".format),
            "Mean predicted probability": (dec["mean_predicted_probability"] * 100).map("{:.2f}%".format),
        })
        st.dataframe(table, width="stretch", hide_index=True)

    st.subheader("Calibration table")
    cal = pd.DataFrame(s.get("calibration") or [])
    if not cal.empty:
        cal_table = pd.DataFrame({
            "Predicted range": cal.apply(lambda r: f"{r['bin_lower']:.1f} to {r['bin_upper']:.1f}", axis=1),
            "Loans": cal["loans"].astype(int),
            "Defaults": cal["defaults"].astype(int),
            "Mean predicted probability": (cal["mean_predicted_probability"] * 100).map("{:.2f}%".format),
            "Observed default rate": (cal["observed_default_rate"] * 100).map("{:.2f}%".format),
        })
        st.dataframe(cal_table, width="stretch", hide_index=True)
        st.caption("Bins are fixed-width probability ranges, so most loans fall in the lowest bins.")

    st.divider()

    # ---- Global importance
    st.subheader("Global feature importance")
    imp, _ = safe(get, "/explain/global", top=20)
    if imp and imp.get("features"):
        df_imp = pd.DataFrame(imp["features"])
        df_imp["feature"] = df_imp["feature"].astype(str)
        imp_chart = alt.Chart(df_imp).mark_bar(color="#4E79A7").encode(
            x=alt.X("mean_abs_contribution:Q", title="Mean absolute contribution (log-odds)"),
            y=alt.Y("feature:N", sort="-x", title="Feature"),
            tooltip=[alt.Tooltip("feature:N", title="Feature"),
                     alt.Tooltip("mean_abs_contribution:Q", title="Mean |contribution|", format=".4f")],
        ).properties(height=480, title="Top features by mean absolute TreeSHAP contribution")
        st.altair_chart(imp_chart, width="stretch")
        st.caption("Contributions are in log-odds, not probability. "
                   "Computed offline on a random sample of 5,000 validation loans.")
        st.dataframe(df_imp.rename(columns={"feature": "Feature", "mean_abs_contribution": "Mean |contribution|"}),
                     width="stretch", hide_index=True)

    st.divider()

    # ---- Data governance
    st.subheader("Data, label and known limitations")
    st.markdown(
        "**Data:** Freddie Mac Single-Family Loan-Level Dataset, sample files of up to 50,000 loans per vintage. "
        "Training 2015 to 2017, validation 2018, out-of-time test 2019, stress 2020 to 2021.\n\n"
        "**Label (`default_24m`):** 1 if, within 24 months of the first payment date, the loan reaches 90 or more "
        "days delinquent or REO acquisition, or has a credit-event zero balance code (02, 03, 09, 15, 16). "
        "Otherwise 0.\n\n"
        "**Features:** origination-time fields only. Performance-file data is used only to build the label."
    )
    st.markdown(
        "**Known limitations**\n"
        "- Only loans Freddie Mac accepted are in the data, so rejected applicants are invisible (selection bias).\n"
        "- Fairness is audited with proxy groups, not protected attributes.\n"
        "- The 2020 to 2021 stress set reflects an unusual interest-rate environment and pandemic forbearance.\n"
        "- Cost assumptions in the optimiser are illustrative.\n"
        "- This is an academic demonstration and is not intended for real lending decisions."
    )