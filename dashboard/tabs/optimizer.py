import streamlit as st
import pandas as pd
import altair as alt
from dashboard.client import get, safe


def render(params: dict, meta: dict):
    st.header("Business Threshold Optimizer")
    st.markdown("Tune the two cost sliders in the sidebar to see the net financial benefit of "
                "different approval thresholds.")

    opt_res, _ = safe(get, "/optimize", **params)
    if not opt_res:
        return

    if opt_res.get("reject_none"):
        st.info("Rejecting nothing is optimal at these costs, so the recommended policy approves every loan.")

    if opt_res.get("calibration_warning"):
        st.warning(
            f"**Calibration warning:** the optimal threshold ({opt_res['threshold']:.4f}) differs by more than "
            f"a factor of two from the analytic threshold g / (g + L) = {opt_res['analytic_threshold']:.4f}. "
            "The predicted probabilities may not be calibrated to the observed default rate on this split."
        )

    m = opt_res["metrics"]
    t_val = opt_res["threshold"]
    inputs = opt_res.get("inputs", {}) or {}

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric("Effective threshold", f"{t_val * 100:.2f}%",
                  help="Probability at or above which a loan is rejected")
    with c2:
        st.metric("Net benefit", f"${m['net_benefit']:,.0f}",
                  help="Money saved compared with approving every loan")
    with c3:
        st.metric("Profit", f"${m['profit']:,.0f}",
                  help="Portfolio profit under this threshold; approve-all profit is "
                       f"${m['approve_all_profit']:,.0f}")
    with c4:
        st.metric("Default capture rate", f"{m['recall'] * 100:.1f}%",
                  help="Share of defaulting loans that are rejected")
    with c5:
        st.metric("Rejection rate", f"{m['rejection_rate'] * 100:.1f}%",
                  help="Share of loans rejected")

    st.divider()

    st.subheader("Net benefit versus decision threshold")
    curve_data = pd.DataFrame(opt_res.get("curve", []))

    if not curve_data.empty:
        x_max = max(float(curve_data["threshold"].max()), float(t_val)) * 1.05
        base_line = alt.Chart(curve_data).mark_line(color="#4E79A7", strokeWidth=2.5).encode(
            x=alt.X("threshold:Q", title="Decision threshold", scale=alt.Scale(domain=[0, x_max])),
            y=alt.Y("net_benefit:Q", title="Net benefit (USD)", axis=alt.Axis(format="$,.0f")),
            tooltip=[alt.Tooltip("threshold:Q", format=".4f", title="Threshold"),
                     alt.Tooltip("net_benefit:Q", format="$,.0f", title="Net benefit")],
        )
        point_df = pd.DataFrame([{"threshold": float(t_val), "net_benefit": float(m["net_benefit"])}])
        opt_point = alt.Chart(point_df).mark_point(color="#E15759", size=150, filled=True).encode(
            x="threshold:Q", y="net_benefit:Q",
            tooltip=[alt.Tooltip("threshold:Q", format=".4f", title="Selected threshold"),
                     alt.Tooltip("net_benefit:Q", format="$,.0f", title="Selected net benefit")],
        )
        rule = alt.Chart(point_df).mark_rule(color="#E15759", strokeDash=[4, 4]).encode(x="threshold:Q")

        st.altair_chart(
            (base_line + opt_point + rule).properties(
                title="Net benefit curve (split: " + str(opt_res.get("split", "")).upper() + ")", height=380),
            width="stretch",
        )

    st.subheader("Performance across validation, out-of-time and stress splits")
    st.caption("The same threshold applied to each split (OP-11). The optimum is computed on the selected split only.")

    rows = []
    for s_entry in opt_res.get("splits", []):
        sm = s_entry["metrics"]
        prec = sm.get("precision")
        rows.append({
            "Split": str(s_entry["split"]).upper(),
            "Threshold": f"{t_val * 100:.2f}%",
            "Rejection rate": f"{sm['rejection_rate'] * 100:.1f}%",
            "Default capture rate": f"{sm['recall'] * 100:.1f}%",
            "Precision": "N/A" if prec is None else f"{prec * 100:.1f}%",
            "Approved default rate": f"{sm['approved_default_rate'] * 100:.2f}%",
            "Net benefit": f"${sm['net_benefit']:,.0f}",
            "Net benefit per loan": f"${sm['net_benefit_per_loan']:,.2f}",
            "Profit": f"${sm['profit']:,.0f}",
        })
    if rows:
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    st.divider()
    st.subheader("Assumptions and caveats (OP-03)")
    fp = inputs.get("fp_cost_pct")
    fn = inputs.get("fn_cost_pct")
    cost_line = (f"A false approval costs {fn}% of original balance (L) and a false rejection costs {fp}% (g)."
                 if fp is not None and fn is not None else
                 "A false approval costs L and a false rejection costs g, both as a percentage of original balance.")
    st.markdown(
        f"- **Cost model:** {cost_line} The costs are flat percentages of balance, with no time value of money, "
        "prepayment or servicing costs.\n"
        "- **Illustrative inputs:** the default false-positive cost (2%) is a placeholder, not estimated from the data.\n"
        "- **Sample, not the whole book:** results describe the evaluated sample (up to 50,000 loans per vintage "
        "year), not Freddie Mac's whole book. Currency is USD.\n"
        "- **Out-of-sample check:** a threshold tuned on the 2018 validation split is also evaluated on 2019 "
        "(out-of-time) and 2020 to 2021 (stress)."
    )