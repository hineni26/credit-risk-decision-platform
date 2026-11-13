import streamlit as st
import pandas as pd
import altair as alt
from dashboard.client import get, safe


def _val(g: dict, key: str):
    """Scalar value of a field, whether flat or {value, ci_lower, ci_upper}."""
    v = g.get(key)
    return v.get("value") if isinstance(v, dict) else v


def _first(g: dict, *keys):
    """First non-null scalar among several possible key names."""
    for k in keys:
        v = _val(g, k)
        if v is not None:
            return v
    return None


def _rate(g: dict, key: str):
    """Return (value, ci_lower, ci_upper) for a rate in either schema."""
    v = g.get(key)
    if isinstance(v, dict):
        return v.get("value"), v.get("ci_lower"), v.get("ci_upper")
    return v, g.get(f"{key}_ci_lower"), g.get(f"{key}_ci_upper")


def _pct(x, d=1):
    return "N/A" if x is None else f"{x * 100:.{d}f}%"


def _ci(lo, hi):
    if lo is None or hi is None:
        return "N/A"
    return f"[{lo * 100:.1f}%, {hi * 100:.1f}%]"


def _num(x, d=2):
    return "N/A" if x is None else f"{x:.{d}f}"


def _fmt_summary_value(v):
    if isinstance(v, dict):
        return "; ".join(f"{k}: {x:.2f}" if isinstance(x, (int, float)) else f"{k}: {x}" for k, x in v.items())
    if isinstance(v, (int, float)):
        return f"{v:.3f}"
    return str(v)


def _is_rated(g: dict) -> bool:
    if "rated" in g:
        return bool(g["rated"])
    return not g.get("too_small", False)


def render(params: dict, meta: dict):
    st.header("Fairness & Bias Audit")
    st.markdown(
        "Audit approval decisions and model error rates across loan-level proxy groups "
        "using Wilson 95% confidence intervals. A flag is a prompt to investigate, "
        "not proof of discrimination."
    )

    configured_attrs = meta.get("fairness", {}).get("attributes", [])
    attr_keys = [a["key"] if isinstance(a, dict) else str(a) for a in configured_attrs]
    attr_labels = {a["key"]: a.get("label", a["key"]) for a in configured_attrs if isinstance(a, dict)}

    sel_attr = st.selectbox(
        "Select proxy attribute",
        options=attr_keys,
        format_func=lambda k: attr_labels.get(k, k),
        key="fairness_attr_select",
    )

    fair_res, _ = safe(get, "/fairness", **params, attribute=sel_attr)
    if not fair_res or not fair_res.get("attributes"):
        return

    audit = fair_res["attributes"][sel_attr]
    thresh_used = fair_res["threshold_used"]
    split_name = str(fair_res.get("split", params.get("split", ""))).upper()
    st.caption(f"Evaluated on split **{split_name}** at operating threshold **{thresh_used * 100:.2f}%**")

    if audit.get("status") == "not_enough_rated_groups":
        st.warning(
            "Not enough rated groups. A group needs at least 500 loans and 30 defaults "
            "to be included in the gap metrics."
        )
        return

    # Flags panel
    flags = audit.get("flags", [])
    if flags:
        st.error(f"{len(flags)} fairness flag(s) triggered")
        for f in flags:
            label = str(f.get("type", "flag")).replace("_", " ").title()
            group = f" ({f['group']})" if f.get("group") else ""
            reason = f.get("reason") or f.get("message") or str(f)
            st.markdown(f"- **{label}**{group}: {reason}")
    else:
        st.success("No fairness flags triggered. All summary measures are within their tolerance levels.")

    # FA-02 caution note, always shown beside the flags
    notes = audit.get("notes")
    if isinstance(notes, (list, tuple)):
        notes = " ".join(str(n) for n in notes)
    st.info(
        "**Contextual note (FA-02):** "
        + (notes or "Some disparity is expected because occupancy and loan purpose are genuine risk drivers. "
                    "A flag is a prompt to investigate, not proof of bias.")
    )

    # Summary measures
    summary = audit.get("summary")
    if isinstance(summary, dict) and summary:
        st.subheader("Summary measures")
        st.dataframe(
            pd.DataFrame([{"Measure": str(k).replace("_", " ").title(), "Value": _fmt_summary_value(v)}
                          for k, v in summary.items()]),
            width="stretch", hide_index=True,
        )
    cal_by_group = summary.get("calibration_ratio", {}) if isinstance(summary, dict) else {}
    if not isinstance(cal_by_group, dict):
        cal_by_group = {}

    st.divider()

    # Group table
    st.subheader("Group metrics")
    groups_data = audit.get("groups", [])
    table_rows = []
    for g in groups_data:
        name = str(g.get("group"))
        dcr, dcr_lo, dcr_hi = _rate(g, "default_capture_rate")
        frr, frr_lo, frr_hi = _rate(g, "false_rejection_rate")
        obs, obs_lo, obs_hi = _rate(g, "observed_default_rate")
        if obs is None:
            obs = _first(g, "default_rate")
        table_rows.append({
            "Group": name,
            "Loans": g.get("loans", g.get("n_loans")),
            "Defaults": g.get("defaults", g.get("n_defaults")),
            "Observed Default Rate": _pct(obs, 2),
            "Mean Predicted Probability": _pct(_first(g, "mean_predicted_probability", "mean_predicted"), 2),
            "Calibration Ratio": _num(cal_by_group.get(name, _val(g, "calibration_ratio"))),
            "Rejection Rate": _pct(_val(g, "rejection_rate")),
            "Approval Rate": _pct(_val(g, "approval_rate")),
            "Default Capture Rate": _pct(dcr),
            "Default Capture 95% CI": _ci(dcr_lo, dcr_hi),
            "False Rejection Rate": _pct(frr),
            "False Rejection 95% CI": _ci(frr_lo, frr_hi),
            "Rated": "Yes" if _is_rated(g) else "No (too small or unknown)",
        })
    st.dataframe(pd.DataFrame(table_rows), width="stretch", hide_index=True)

    st.divider()

    # Charts, each paired with the table above (UI-05, UI-12)
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Rejection rate by group")
        rej_rows = []
        for g in groups_data:
            r = _val(g, "rejection_rate")
            if r is not None:
                rej_rows.append({"Group": str(g.get("group")),
                                 "Rejection Rate (%)": r * 100.0,
                                 "Status": "Rated" if _is_rated(g) else "Not rated"})
        if rej_rows:
            rej_chart = alt.Chart(pd.DataFrame(rej_rows)).mark_bar().encode(
                x=alt.X("Group:N", title="Group", sort=None),
                y=alt.Y("Rejection Rate (%):Q", title="Rejection rate (%)", scale=alt.Scale(domain=[0, 100])),
                color=alt.Color("Status:N",
                                scale=alt.Scale(domain=["Rated", "Not rated"], range=["#4E79A7", "#BAB0AC"])),
                tooltip=["Group", alt.Tooltip("Rejection Rate (%):Q", format=".1f"), "Status"],
            ).properties(height=320, title="Rejection rate by group")
            st.altair_chart(rej_chart, width="stretch")

    with col2:
        st.subheader("Error rates with Wilson 95% CIs")
        whisker = []
        for g in groups_data:
            if not _is_rated(g):
                continue
            for metric, key in (("Default capture rate", "default_capture_rate"),
                                ("False rejection rate", "false_rejection_rate")):
                v, lo, hi = _rate(g, key)
                if v is not None and lo is not None and hi is not None:
                    whisker.append({"Group": str(g.get("group")), "Metric": metric,
                                    "Rate": v * 100.0, "CI_Lower": lo * 100.0, "CI_Upper": hi * 100.0})
        if whisker:
            df_w = pd.DataFrame(whisker)
            points = alt.Chart(df_w).mark_point(filled=True, size=60).encode(
                x=alt.X("Group:N", title="Group"),
                y=alt.Y("Rate:Q", title="Rate (%)", scale=alt.Scale(domain=[0, 100])),
                color=alt.Color("Metric:N", title="Metric"),
                tooltip=["Group", "Metric", alt.Tooltip("Rate:Q", format=".1f"),
                         alt.Tooltip("CI_Lower:Q", format=".1f"), alt.Tooltip("CI_Upper:Q", format=".1f")],
            )
            bars = alt.Chart(df_w).mark_errorbar().encode(
                x=alt.X("Group:N"),
                y=alt.Y("CI_Lower:Q", title="Rate (%)"),
                y2="CI_Upper:Q",
                color=alt.Color("Metric:N"),
            )
            st.altair_chart((points + bars).properties(
                height=320, title="Default capture and false rejection rates"), width="stretch")

    # Baseline versus current
    st.divider()
    st.subheader("Baseline versus current threshold")
    st.caption("How rejection and capture rates moved from the default threshold to the current threshold.")

    base_data = (fair_res.get("baseline") or {}).get(sel_attr, {})
    if isinstance(base_data, dict) and base_data.get("groups"):
        base_groups = {str(bg.get("group")): bg for bg in base_data["groups"]}
        comp_rows = []
        for g in groups_data:
            name = str(g.get("group"))
            b_rej = _val(base_groups.get(name, {}), "rejection_rate")
            c_rej = _val(g, "rejection_rate")
            change = "N/A" if (b_rej is None or c_rej is None) else f"{(c_rej - b_rej) * 100:+.1f} pts"
            comp_rows.append({
                "Group": name,
                "Baseline Rejection Rate": _pct(b_rej),
                "Current Rejection Rate": _pct(c_rej),
                "Change": change,
                "Current Capture Rate": _pct(_rate(g, "default_capture_rate")[0]),
            })
        st.dataframe(pd.DataFrame(comp_rows), width="stretch", hide_index=True)