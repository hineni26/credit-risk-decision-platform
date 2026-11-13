import streamlit as st
import pandas as pd
import altair as alt
from dashboard.client import get, safe

GREEN, AMBER, RED, GREY = "#59A14F", "#EDC949", "#E15759", "#BAB0AC"


def _band_color(band: str) -> str:
    b = str(band).lower()
    if "stable" in b:
        return GREEN
    if "warn" in b or "moderate" in b:
        return AMBER
    if "retrain" in b or "significant" in b:
        return RED
    return GREY


def _band_from_psi(p) -> str:
    if p is None or pd.isna(p):
        return "unknown"
    if p >= 0.20:
        return "retrain"
    if p >= 0.10:
        return "warning"
    return "stable"


def _tier(status, score_psi) -> str:
    """'red', 'amber' or 'green' from the API status, falling back to score PSI."""
    s = str(status or "").lower()
    if "retrain" in s or "significant" in s:
        return "red"
    if "warn" in s or "moderate" in s:
        return "amber"
    if "stable" in s:
        return "green"
    return {"retrain": "red", "warning": "amber"}.get(_band_from_psi(score_psi), "green")


def _yes_no(v) -> str:
    return "Yes" if v else "No"


def _score_bin_rows(score_bins, window_title: str):
    """Turn the API's score_bins into long-format rows for the overlay chart."""
    ref, cur, labels = [], [], []
    if isinstance(score_bins, dict):
        ref = score_bins.get("reference") or []
        cur = score_bins.get("current") or []
        edges = score_bins.get("edges") or []
        n = min(len(ref), len(cur))
        if edges and n == len(edges) + 2:
            labels = [f"D{i + 1}" for i in range(n - 1)] + ["Missing"]
        else:
            labels = [f"Bin {i + 1}" for i in range(n)]
        ref, cur = ref[:n], cur[:n]
    elif isinstance(score_bins, list):
        for i, b in enumerate(score_bins):
            if isinstance(b, dict):
                labels.append(str(b.get("bin", f"Bin {i + 1}")))
                ref.append(b.get("reference_share", 0.0))
                cur.append(b.get("current_share", 0.0))
    rows = []
    for lab, r, c in zip(labels, ref, cur):
        if lab == "Missing" and not r and not c:
            continue
        rows.append({"Bin": lab, "Distribution": "Reference (validation)", "Share (%)": (r or 0.0) * 100.0})
        rows.append({"Bin": lab, "Distribution": f"Current ({window_title})", "Share (%)": (c or 0.0) * 100.0})
    return rows


def render(params: dict, meta: dict):
    st.header("Drift Monitor")
    st.markdown(
        "Track distribution drift of the model score and of the input features relative to the "
        "reference distributions, using the Population Stability Index (DR-01 to DR-15). "
        "Alerts are informational only: nothing retrains automatically."
    )

    v_col1, v_col2 = st.columns([1, 2])
    with v_col1:
        view_type = st.radio("Monitoring view", ["Real Vintages", "Simulated Shock"],
                             horizontal=True, key="drift_view_type")
    is_simulated = (view_type == "Simulated Shock")

    if not is_simulated:
        with v_col2:
            real_window = st.selectbox("Evaluation window", ["oot", "stress", "valid"], index=0,
                                       key="drift_real_window")
        drift_data, _ = safe(get, "/drift", source="real", window=real_window)
        window_title = real_window.upper()
    else:
        with v_col2:
            scen_col, sev_col = st.columns(2)
            with scen_col:
                scenario = st.selectbox(
                    "Simulated scenario",
                    ["rate_shock", "debt_stress", "credit_deterioration", "leverage_squeeze"],
                    format_func=lambda s: s.replace("_", " ").title(),
                    key="drift_sim_scen",
                )
            with sev_col:
                severity = st.radio("Severity", ["mild", "moderate", "severe"],
                                    horizontal=True, key="drift_sim_sev")
        drift_data, _ = safe(get, "/drift", source="simulated", scenario=scenario, severity=severity)
        window_title = f"SIMULATED: {scenario.replace('_', ' ').upper()} ({severity.upper()})"

    if not drift_data:
        return

    st.divider()

    # Status banner (UI-08, DR-15). Status is always stated in words, not only by colour.
    score_psi = drift_data.get("score_psi")
    triggers = drift_data.get("triggers", []) or []
    triggers_text = ", ".join(str(t) for t in triggers) if triggers else "None"
    psi_text = "n/a" if score_psi is None else f"{score_psi:.4f}"
    tier = _tier(drift_data.get("status"), score_psi)

    if tier == "red":
        st.error(
            f"**STATUS: RETRAIN RECOMMENDED** | Score PSI = {psi_text}\n\n"
            f"Triggering items: {triggers_text}\n\n"
            "Next step (DR-15): add newer data and rerun the full pipeline (`make all`), "
            "then compare the new reports."
        )
    elif tier == "amber":
        st.warning(
            f"**STATUS: WARNING** | Score PSI = {psi_text}\n\n"
            f"Triggering items: {triggers_text}\n\n"
            "Next step: monitor more closely and inspect the borrower mix and feature drift."
        )
    else:
        st.success(
            f"**STATUS: STABLE** | Score PSI = {psi_text}\n\n"
            f"Triggering items: {triggers_text}\n\n"
            "No action needed. Continue regular monitoring."
        )

    if is_simulated:
        st.info("**Simulated.** This view shows a counterfactual shock applied to the out-of-time split. "
                "Simulated loans have no true outcomes, so only PSI and score shifts are reported (DR-14).")

    st.divider()

    # Feature PSI chart and table
    st.subheader(f"Feature stability ({window_title})")
    feat_data = drift_data.get("features", [])

    if feat_data:
        df_feats = pd.DataFrame(feat_data)
        if "feature" not in df_feats.columns and "name" in df_feats.columns:
            df_feats = df_feats.rename(columns={"name": "feature"})
        if "psi" not in df_feats.columns or "feature" not in df_feats.columns:
            st.warning("The drift response does not contain feature PSI values.")
        else:
            if "band" not in df_feats.columns:
                df_feats["band"] = df_feats["psi"].apply(_band_from_psi)
            if "structural" not in df_feats.columns:
                df_feats["structural"] = 0
            df_feats["feature"] = df_feats["feature"].astype(str)
            df_feats["band"] = df_feats["band"].astype(str)
            df_feats["structural"] = df_feats["structural"].map(_yes_no)
            if "constant" in df_feats.columns:
                df_feats["constant"] = df_feats["constant"].map(_yes_no)
            df_feats = df_feats.sort_values("psi", ascending=False)

            bands = list(dict.fromkeys(df_feats["band"]))
            band_scale = alt.Scale(domain=bands, range=[_band_color(b) for b in bands])

            bar_chart = alt.Chart(df_feats).mark_bar().encode(
                x=alt.X("psi:Q", title="Population Stability Index (PSI)"),
                y=alt.Y("feature:N", sort="-x", title="Feature"),
                color=alt.Color("band:N", scale=band_scale, title="Band"),
                tooltip=[
                    alt.Tooltip("feature:N", title="Feature"),
                    alt.Tooltip("psi:Q", title="PSI", format=".4f"),
                    alt.Tooltip("band:N", title="Band"),
                    alt.Tooltip("structural:N", title="Structural"),
                ],
            )
            rule10 = alt.Chart(pd.DataFrame({"psi": [0.10]})).mark_rule(
                color=AMBER, strokeDash=[4, 4], strokeWidth=2).encode(x="psi:Q")
            rule20 = alt.Chart(pd.DataFrame({"psi": [0.20]})).mark_rule(
                color=RED, strokeDash=[4, 4], strokeWidth=2).encode(x="psi:Q")

            st.altair_chart(
                (bar_chart + rule10 + rule20).properties(
                    title=f"Feature PSI with 0.10 and 0.20 reference lines ({window_title})", height=520),
                width="stretch",
            )

            st.markdown("**Feature drift table**")
            disp = df_feats.copy()
            disp["psi"] = disp["psi"].apply(lambda p: "n/a" if pd.isna(p) else f"{p:.4f}")
            st.dataframe(disp, width="stretch", hide_index=True)
            st.caption("Structural features are shown but do not set the overall status (DR-09).")

    st.divider()

    # Score distribution overlay
    st.subheader("Score distribution: reference versus current window")
    rows = _score_bin_rows(drift_data.get("score_bins"), window_title)
    if rows:
        df_overlay = pd.DataFrame(rows)
        order = list(dict.fromkeys(df_overlay["Bin"]))
        overlay_chart = alt.Chart(df_overlay).mark_bar(opacity=0.85).encode(
            x=alt.X("Bin:N", sort=order, title="Score bin (reference deciles)"),
            y=alt.Y("Share (%):Q", title="Population share (%)",
                    scale=alt.Scale(domain=[0, max(df_overlay["Share (%)"].max() * 1.15, 15)])),
            color=alt.Color("Distribution:N", title="Population",
                            scale=alt.Scale(range=["#4E79A7", "#F28E2B"])),
            xOffset="Distribution:N",
            tooltip=[alt.Tooltip("Bin:N"), alt.Tooltip("Distribution:N"),
                     alt.Tooltip("Share (%):Q", format=".2f")],
        ).properties(title="Score distribution shift", height=320)
        st.altair_chart(overlay_chart, width="stretch")
        st.caption("Each reference decile holds 10% of the reference population by construction.")
        st.dataframe(df_overlay.pivot(index="Bin", columns="Distribution", values="Share (%)")
                     .reindex(order).round(2).reset_index(), width="stretch", hide_index=True)

    # Quarterly PSI trend
    st.divider()
    st.subheader("Quarterly score PSI (2018Q1 to 2021Q4)")
    q_data, _ = safe(get, "/drift", source="real", window="quarterly")
    if q_data and q_data.get("series"):
        df_q = pd.DataFrame(q_data["series"])
        if "score_psi" in df_q.columns and "quarter" in df_q.columns:
            keep = [c for c in ("quarter", "score_psi", "status", "n") if c in df_q.columns]
            df_q = df_q[keep].copy()          # drop nested columns such as 'features' and 'triggers'
            df_q["quarter"] = df_q["quarter"].astype(str)
            if "status" in df_q.columns:
                df_q["status"] = df_q["status"].astype(str)
            psi_max = df_q["score_psi"].dropna().max() if df_q["score_psi"].notna().any() else 0.0

            tooltip = [alt.Tooltip("quarter:N", title="Quarter"),
                       alt.Tooltip("score_psi:Q", title="Score PSI", format=".4f")]
            if "status" in df_q.columns:
                tooltip.append(alt.Tooltip("status:N", title="Overall status"))
            if "n" in df_q.columns:
                tooltip.append(alt.Tooltip("n:Q", title="Loans"))

            q_line = alt.Chart(df_q).mark_line(point=True, color="#4E79A7", strokeWidth=2).encode(
                x=alt.X("quarter:N", sort=None, title="Origination quarter"),
                y=alt.Y("score_psi:Q", title="Score PSI",
                        scale=alt.Scale(domain=[0, max(psi_max * 1.2, 0.25)])),
                tooltip=tooltip,
            )
            q_rule10 = alt.Chart(pd.DataFrame({"y": [0.10]})).mark_rule(
                color=AMBER, strokeDash=[4, 4]).encode(y="y:Q")
            q_rule20 = alt.Chart(pd.DataFrame({"y": [0.20]})).mark_rule(
                color=RED, strokeDash=[4, 4]).encode(y="y:Q")
            st.altair_chart((q_line + q_rule10 + q_rule20).properties(
                title="Quarterly score drift relative to the validation reference", height=300),
                width="stretch")
            st.caption("Quarters with fewer than 1,000 loans report no PSI (DR-06). "
                       "The overall status of a quarter can be 'retrain' because of feature PSI "
                       "even when its score PSI is low (DR-08).")