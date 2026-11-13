import streamlit as st
import pandas as pd
import altair as alt
from dashboard.client import get, safe


def _as_text(v) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "N/A"
    if isinstance(v, float):
        return f"{v:,.4f}".rstrip("0").rstrip(".")
    return str(v)


def render(params: dict, meta: dict):
    st.header("Loan Lookup & Local Explanation")
    st.markdown("Search for loans in the active evaluation split, inspect the risk prediction, "
                "and review the local TreeSHAP explanation.")

    split = params.get("split", "valid")
    col_search, col_pick = st.columns([1, 2])

    with col_search:
        search_query = st.text_input("Search loan ID prefix", value="", placeholder="e.g. F118Q1...",
                                     key="loan_search_query")

    loans_resp, _ = safe(get, "/loans", split=split, limit=50, search=search_query if search_query else None)
    if not loans_resp or not loans_resp.get("items"):
        st.info("No loans found matching the criteria in split: " + split)
        return

    items = loans_resp["items"]
    loan_options = [it["loan_id"] for it in items]

    with col_pick:
        selected_loan_id = st.selectbox(
            f"Select loan ({loans_resp['total']} total found in {split})",
            options=loan_options,
            key="selected_loan_id",
        )

    if not selected_loan_id:
        return

    threshold_arg = params.get("threshold")
    detail, _ = safe(get, f"/loans/{selected_loan_id}")
    pred, _ = safe(get, f"/predict/{selected_loan_id}", threshold=threshold_arg)
    exp, _ = safe(get, f"/explain/{selected_loan_id}")

    if not detail or not pred or not exp:
        return

    st.divider()

    # Summary metrics
    m1, m2, m3, m4 = st.columns(4)
    prob_pct = pred["probability"] * 100.0
    thresh_pct = pred["threshold"] * 100.0
    is_rejected = (pred["decision"] == "reject")

    outcome = detail.get("actual_outcome")
    if outcome is None:
        actual_str = "Unknown"
    elif outcome == 1:
        actual_str = "Defaulted within 24 months"
    else:
        actual_str = "No default within 24 months"

    with m1:
        st.metric("Default probability", f"{prob_pct:.2f}%")
    with m2:
        st.metric(f"Decision ({str(pred['threshold_source']).title()} threshold: {thresh_pct:.2f}%)",
                  "REJECT" if is_rejected else "APPROVE")
    with m3:
        st.metric("Actual 24-month outcome", actual_str)
    with m4:
        st.metric("Original balance (UPB)", f"${detail['original_upb']:,.0f}")

    if is_rejected:
        st.error(f"Decision: **REJECT**. The predicted default probability ({prob_pct:.2f}%) is at or above "
                 f"the decision threshold ({thresh_pct:.2f}%).")
    else:
        st.success(f"Decision: **APPROVE**. The predicted default probability ({prob_pct:.2f}%) is below "
                   f"the decision threshold ({thresh_pct:.2f}%).")

    tab_exp, tab_feats = st.tabs(["Local explanation (TreeSHAP)", "Loan attributes"])

    with tab_exp:
        st.subheader("Feature contributions (log-odds)")
        st.caption(f"Base value (log-odds): {exp['base_value']:.4f} | "
                   f"Reconciled probability: {exp['probability'] * 100:.2f}%")

        contrib_list = list(exp["contributions"])
        contrib_list.append({
            "feature": "Other features (combined)",
            "value": None,
            "contribution_log_odds": exp["other_contribution_log_odds"],
        })

        df_contrib = pd.DataFrame(contrib_list)
        df_contrib["feature"] = df_contrib["feature"].astype(str)
        df_contrib["feature_value"] = df_contrib["value"].map(_as_text)
        df_contrib["direction"] = df_contrib["contribution_log_odds"].apply(
            lambda v: "Increases risk" if v > 0 else "Decreases risk")
        # The raw 'value' column mixes numbers and strings (e.g. 'KY'), which Arrow cannot serialize.
        # Everything downstream uses the string version in 'feature_value'.
        df_contrib = df_contrib.drop(columns=["value"])

        chart = alt.Chart(df_contrib).mark_bar().encode(
            x=alt.X("contribution_log_odds:Q", title="Contribution (log-odds)"),
            y=alt.Y("feature:N", sort=None, title="Feature"),
            color=alt.Color("direction:N",
                            scale=alt.Scale(domain=["Increases risk", "Decreases risk"],
                                            range=["#E15759", "#59A14F"]),
                            title="Impact"),
            tooltip=[
                alt.Tooltip("feature:N", title="Feature"),
                alt.Tooltip("feature_value:N", title="Feature value"),
                alt.Tooltip("contribution_log_odds:Q", format="+.4f", title="Log-odds contribution"),
                alt.Tooltip("direction:N", title="Impact"),
            ],
        ).properties(title="Top 10 contributions plus the combined remainder", height=380)
        st.altair_chart(chart, width="stretch")

        st.markdown("**Contribution table**")
        disp_df = pd.DataFrame({
            "Feature": df_contrib["feature"],
            "Feature value": df_contrib["feature_value"],
            "Contribution (log-odds)": df_contrib["contribution_log_odds"].map(lambda x: f"{x:+.4f}"),
            "Impact": df_contrib["direction"],
        })
        st.dataframe(disp_df, width="stretch", hide_index=True)

    with tab_feats:
        st.subheader("Origination characteristics")
        feat_dict = detail.get("features", {})
        f_df = pd.DataFrame([{"Feature": str(k), "Value": _as_text(v)} for k, v in sorted(feat_dict.items())])
        st.dataframe(f_df, width="stretch", height=400, hide_index=True)