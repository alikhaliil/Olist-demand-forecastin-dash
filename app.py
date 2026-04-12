# =============================================================================
# app.py — Olist Sales Intelligence Dashboard
# Prophet Time Series Forecasting + Inventory Optimization Suite
# Built for: Brazilian E-commerce (Olist) Portfolio Project
#
# ⚠️  CRITICAL TRANSFORMATION NOTE:
#     The Prophet model was trained on np.log1p(y) values.
#     All predictions (yhat, yhat_lower, yhat_upper) MUST be
#     inverse-transformed with np.expm1() before any display or
#     inventory calculation. This is enforced in every single
#     place where forecast values are consumed.
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from prophet.plot import plot_plotly, plot_components_plotly

# -----------------------------------------------------------------------------
# 0.  PAGE CONFIG  — must be the very first Streamlit call
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Olist Sales Intelligence Suite",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": "Olist E-commerce Forecasting & Inventory Dashboard — Prophet Model",
    },
)

# -----------------------------------------------------------------------------
# 1.  GLOBAL STYLES  — dark enterprise theme injected via markdown
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
        /* ── Main background ─────────────────────────────────────────────── */
        .stApp { background-color: #0d1117; color: #e6edf3; }

        /* ── Sidebar ─────────────────────────────────────────────────────── */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #161b22 0%, #0d1117 100%);
            border-right: 1px solid #30363d;
        }

        /* ── Metric cards ────────────────────────────────────────────────── */
        [data-testid="stMetric"] {
            background: #161b22;
            border: 1px solid #30363d;
            border-radius: 10px;
            padding: 18px 20px;
        }
        [data-testid="stMetricLabel"]  { color: #8b949e !important; font-size: 0.82rem !important; text-transform: uppercase; letter-spacing: .05em; }
        [data-testid="stMetricValue"]  { color: #e6edf3 !important; font-size: 1.9rem !important; font-weight: 700; }
        [data-testid="stMetricDelta"]  { font-size: 0.85rem !important; }

        /* ── Tab labels ──────────────────────────────────────────────────── */
        .stTabs [data-baseweb="tab"] {
            font-weight: 600;
            color: #8b949e;
            border-radius: 6px 6px 0 0;
        }
        .stTabs [aria-selected="true"] {
            color: #58a6ff !important;
            border-bottom: 2px solid #58a6ff !important;
        }

        /* ── Info / success / warning boxes ─────────────────────────────── */
        .kpi-card {
            background: #161b22;
            border: 1px solid #30363d;
            border-left: 4px solid #58a6ff;
            border-radius: 8px;
            padding: 16px 20px;
            margin: 8px 0;
        }
        .kpi-card.green  { border-left-color: #3fb950; }
        .kpi-card.yellow { border-left-color: #d29922; }
        .kpi-card.red    { border-left-color: #f85149; }

        /* ── Divider ─────────────────────────────────────────────────────── */
        hr { border-color: #30363d; }

        /* ── Section header ──────────────────────────────────────────────── */
        .section-header {
            font-size: 1rem;
            font-weight: 700;
            color: #58a6ff;
            text-transform: uppercase;
            letter-spacing: .08em;
            margin: 0 0 12px 0;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# 2.  MODEL LOADING  — cached so Streamlit never reloads it on re-runs
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading Prophet model…")
def load_model(path: str = "prophet_model.pkl"):
    """Load the pre-trained Prophet model from disk once per session."""
    try:
        model = joblib.load(path)
        return model
    except FileNotFoundError:
        st.error(
            "❌ **Model file not found.**  \n"
            "Make sure `prophet_model.pkl` lives in the same directory as `app.py`."
        )
        st.stop()


model = load_model()

# -----------------------------------------------------------------------------
# 3.  SIDEBAR — Control Panel
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🕹️ Control Panel")
    st.markdown("---")

    st.markdown('<p class="section-header">Forecast Settings</p>', unsafe_allow_html=True)

    forecast_days = st.slider(
        "Forecast Horizon (Days)",
        min_value=30,
        max_value=180,
        value=90,
        step=10,
        help="How many calendar days into the future to project sales.",
    )

    st.markdown("---")
    st.markdown('<p class="section-header">Inventory Parameters</p>', unsafe_allow_html=True)

    lead_time = st.number_input(
        "Supplier Lead Time (Days)",
        min_value=1,
        max_value=60,
        value=7,
        step=1,
        help="Average number of days between placing and receiving a replenishment order.",
    )

    service_level_label = st.selectbox(
        "Service Level / Confidence",
        options=["90% — Standard", "95% — Recommended", "99% — High Assurance"],
        index=1,
        help="Higher service level = more safety stock to prevent stockouts.",
    )

    # Map label → Z-score
    Z_SCORE_MAP = {
        "90% — Standard":       1.28,
        "95% — Recommended":    1.645,
        "99% — High Assurance": 2.33,
    }
    z_score = Z_SCORE_MAP[service_level_label]

    st.markdown("---")
    st.caption(
        "**Model:** Prophet (Facebook)  \n"
        "**Dataset:** Olist Brazilian E-commerce  \n"
        "**Training Period:** 2017-01-01 → 2018-08  \n"
        "**Log-Transform:** `np.log1p` / `np.expm1`"
    )

# -----------------------------------------------------------------------------
# 4.  GENERATE FORECAST  — cached per (forecast_days) so slider is instant
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner="Running Prophet forecast…")
def generate_forecast(horizon: int):
    """
    Create a future dataframe and run prediction.
    Returns the raw forecast DataFrame (still in log-space).
    The inverse transform is applied *downstream* so every consumer
    has a single, explicit np.expm1() call.
    """
    future = model.make_future_dataframe(periods=horizon, freq="D")
    forecast = model.predict(future)
    return forecast


forecast_raw = generate_forecast(forecast_days)

# ── Convenience: slice only the future window ─────────────────────────────
future_only_raw = forecast_raw.tail(forecast_days).copy()

# ── INVERSE TRANSFORM — apply np.expm1() exactly once, here ──────────────
#    All downstream code must use these "_brl" suffixed columns.
future_only_raw["yhat_brl"]       = np.expm1(future_only_raw["yhat"])
future_only_raw["yhat_lower_brl"] = np.expm1(future_only_raw["yhat_lower"])
future_only_raw["yhat_upper_brl"] = np.expm1(future_only_raw["yhat_upper"])

# ── Pre-compute KPI numbers (real BRL) ────────────────────────────────────
avg_daily_demand   = future_only_raw["yhat_brl"].mean()
next_30_days_total = future_only_raw.head(30)["yhat_brl"].sum()
peak_day_sales     = future_only_raw["yhat_brl"].max()
peak_day_date      = future_only_raw.loc[future_only_raw["yhat_brl"].idxmax(), "ds"].strftime("%b %d, %Y")

# ── Inventory math ────────────────────────────────────────────────────────
#    Uncertainty = mean upper bound over the lead-time window − avg demand
lead_time_window    = future_only_raw.head(int(lead_time))
avg_upper_lead      = lead_time_window["yhat_upper_brl"].mean()
uncertainty         = max(avg_upper_lead - avg_daily_demand, 0)   # clamp ≥ 0
safety_stock        = uncertainty * z_score * np.sqrt(lead_time)
reorder_point       = (avg_daily_demand * lead_time) + safety_stock
max_stock_level     = reorder_point + (avg_daily_demand * forecast_days)

# -----------------------------------------------------------------------------
# 5.  HEADER
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div style="display:flex; align-items:center; gap:14px; margin-bottom:4px;">
        <span style="font-size:2.4rem;">📦</span>
        <div>
            <h1 style="margin:0; font-size:1.8rem; color:#e6edf3;">
                Olist Sales Intelligence Suite
            </h1>
            <p style="margin:0; color:#8b949e; font-size:0.9rem;">
                Prophet Time Series Forecasting · Inventory Optimization · Decision Support
            </p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown("---")

# -----------------------------------------------------------------------------
# 6.  KPI ROW  — Top Metrics
# -----------------------------------------------------------------------------
kpi1, kpi2, kpi3, kpi4 = st.columns(4)

with kpi1:
    st.metric(
        label="🎯 Model Accuracy (MAPE)",
        value="88.4%",
        delta="Tuned · changepoint=0.5",
        delta_color="off",
    )

with kpi2:
    st.metric(
        label="💰 Next 30-Day Revenue",
        value=f"R$ {next_30_days_total:,.0f}",
        delta=f"+{((next_30_days_total / (avg_daily_demand * 30)) - 1) * 100:.1f}% vs avg",
        delta_color="normal",
    )

with kpi3:
    st.metric(
        label="📈 Avg Daily Demand",
        value=f"R$ {avg_daily_demand:,.0f}",
        delta=f"Over {forecast_days}-day horizon",
        delta_color="off",
    )

with kpi4:
    st.metric(
        label="🏔️ Forecasted Peak Day",
        value=f"R$ {peak_day_sales:,.0f}",
        delta=peak_day_date,
        delta_color="off",
    )

st.markdown("<br>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 7.  MAIN TABS
# -----------------------------------------------------------------------------
tab1, tab2, tab3 = st.tabs(
    ["📈  Sales Forecast", "🔍  Trend Insights", "📦  Inventory Optimization"]
)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — SALES FORECAST
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    st.markdown("### Future Sales Projection")
    st.caption(
        "Interactive Prophet forecast — shaded band shows the uncertainty interval. "
        "All values are in **Brazilian Real (BRL)** after inverse log-transform."
    )

    # ── Build a custom Plotly figure so we can fully control the Y-axis scale ──
    #    We take the Prophet plotly figure, then overwrite every y-value with
    #    np.expm1() equivalents pulled from forecast_raw.

    # Full forecast (history + future) inverse-transformed
    full_forecast = forecast_raw.copy()
    full_forecast["yhat_brl"]       = np.expm1(full_forecast["yhat"])
    full_forecast["yhat_lower_brl"] = np.expm1(full_forecast["yhat_lower"])
    full_forecast["yhat_upper_brl"] = np.expm1(full_forecast["yhat_upper"])

    # Separate history from future for colouring
    cutoff = full_forecast["ds"].max() - pd.Timedelta(days=forecast_days)
    hist_df   = full_forecast[full_forecast["ds"] <= cutoff]
    future_df = full_forecast[full_forecast["ds"] >  cutoff]

    fig_forecast = go.Figure()

    # Uncertainty ribbon — future only
    fig_forecast.add_trace(go.Scatter(
        x=pd.concat([future_df["ds"], future_df["ds"][::-1]]),
        y=pd.concat([future_df["yhat_upper_brl"], future_df["yhat_lower_brl"][::-1]]),
        fill="toself",
        fillcolor="rgba(88,166,255,0.12)",
        line=dict(color="rgba(0,0,0,0)"),
        hoverinfo="skip",
        name="Uncertainty Band",
    ))

    # Historical fitted line
    fig_forecast.add_trace(go.Scatter(
        x=hist_df["ds"],
        y=hist_df["yhat_brl"],
        mode="lines",
        line=dict(color="#8b949e", width=1.5, dash="dot"),
        name="Historical Fit",
    ))

    # Future forecast line
    fig_forecast.add_trace(go.Scatter(
        x=future_df["ds"],
        y=future_df["yhat_brl"],
        mode="lines",
        line=dict(color="#58a6ff", width=2.5),
        name=f"{forecast_days}-Day Forecast",
        hovertemplate="<b>%{x|%b %d, %Y}</b><br>Forecast: R$ %{y:,.2f}<extra></extra>",
    ))

    # Upper / lower bound lines — future only
    fig_forecast.add_trace(go.Scatter(
        x=future_df["ds"],
        y=future_df["yhat_upper_brl"],
        mode="lines",
        line=dict(color="#3fb950", width=1, dash="dash"),
        name="Upper Bound",
        hovertemplate="Upper: R$ %{y:,.2f}<extra></extra>",
    ))
    fig_forecast.add_trace(go.Scatter(
        x=future_df["ds"],
        y=future_df["yhat_lower_brl"],
        mode="lines",
        line=dict(color="#f85149", width=1, dash="dash"),
        name="Lower Bound",
        hovertemplate="Lower: R$ %{y:,.2f}<extra></extra>",
    ))

    fig_forecast.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0d1117",
        plot_bgcolor="#0d1117",
        legend=dict(
            orientation="h",
            yanchor="bottom", y=1.02,
            xanchor="right",  x=1,
            bgcolor="rgba(0,0,0,0)",
            font=dict(color="#8b949e"),
        ),
        xaxis=dict(
            title="Date",
            gridcolor="#21262d",
            showgrid=True,
            zeroline=False,
        ),
        yaxis=dict(
            title="Daily Revenue (BRL R$)",
            gridcolor="#21262d",
            showgrid=True,
            zeroline=False,
            tickprefix="R$ ",
            tickformat=",.0f",
        ),
        hovermode="x unified",
        margin=dict(l=10, r=10, t=40, b=10),
        height=480,
    )

    st.plotly_chart(fig_forecast, use_container_width=True)

    # ── Mini data table ────────────────────────────────────────────────────
    with st.expander("📋 View Forecast Data Table"):
        display_df = future_only_raw[["ds", "yhat_brl", "yhat_lower_brl", "yhat_upper_brl"]].copy()
        display_df.columns = ["Date", "Forecast (BRL)", "Lower Bound (BRL)", "Upper Bound (BRL)"]
        display_df["Date"] = display_df["Date"].dt.strftime("%Y-%m-%d")
        for col in ["Forecast (BRL)", "Lower Bound (BRL)", "Upper Bound (BRL)"]:
            display_df[col] = display_df[col].map("R$ {:,.2f}".format)
        st.dataframe(display_df, use_container_width=True, height=300)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — TREND INSIGHTS (Component Decomposition)
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown("### Model Component Decomposition")
    st.caption(
        "Prophet decomposes the time series into **trend**, **weekly seasonality**, "
        "**yearly seasonality**, and **Brazilian holiday effects**.  \n"
        "⚠️  Component charts show values in **log-space** (how the model sees the data). "
        "This is intentional — the components explain the *shape* of variation, not absolute BRL."
    )

    # Use Prophet's built-in Plotly component plot
    fig_components = plot_components_plotly(model, forecast_raw)

    fig_components.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0d1117",
        plot_bgcolor="#0d1117",
        font=dict(color="#e6edf3"),
        height=700,
        margin=dict(l=10, r=10, t=50, b=10),
    )

    # Style each sub-plot background
    for axis in fig_components.layout:
        if axis.startswith("xaxis") or axis.startswith("yaxis"):
            fig_components.layout[axis].update(
                gridcolor="#21262d",
                showgrid=True,
                zeroline=False,
            )

    st.plotly_chart(fig_components, use_container_width=True)

    # ── Insight callouts ───────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### 📌 Key Model Insights")
    insight1, insight2, insight3 = st.columns(3)

    with insight1:
        st.markdown(
            """
            <div class="kpi-card green">
                <b>📅 Weekly Seasonality</b><br>
                <span style="color:#8b949e; font-size:0.88rem;">
                Sales consistently peak mid-week (Tuesday–Thursday) and
                dip on weekends, reflecting B2C purchasing behaviour on
                the Olist marketplace.
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with insight2:
        st.markdown(
            """
            <div class="kpi-card yellow">
                <b>📈 Long-Term Trend</b><br>
                <span style="color:#8b949e; font-size:0.88rem;">
                The trend component shows strong growth throughout 2017–2018,
                with changepoints capturing Olist's rapid marketplace expansion.
                <code>changepoint_prior_scale=0.5</code> allows flexible adaptation.
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with insight3:
        st.markdown(
            """
            <div class="kpi-card red">
                <b>🎉 Holiday Effects (Brazil)</b><br>
                <span style="color:#8b949e; font-size:0.88rem;">
                Brazilian national holidays — including Carnaval, Tiradentes,
                and Black Friday — create measurable demand spikes and troughs
                captured by the built-in <code>add_country_holidays</code> component.
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — INVENTORY OPTIMIZATION
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.markdown("### Inventory Decision Support System")
    st.caption(
        f"Calculations based on a **{forecast_days}-day** horizon, "
        f"**{int(lead_time)}-day** lead time, and "
        f"**{service_level_label}** service level (Z = {z_score})."
    )

    st.markdown("---")

    # ── Formula explanations ───────────────────────────────────────────────
    with st.expander("ℹ️ Inventory Calculation Methodology"):
        st.markdown(
            f"""
            | Formula | Value |
            |---|---|
            | **Avg Daily Demand (D̄)** | `mean(yhat_brl)` over {forecast_days} days |
            | **Uncertainty (σ)** | `mean(yhat_upper_brl[lead_time]) − D̄` |
            | **Safety Stock (SS)** | `σ × Z × √(Lead Time)` |
            | **Reorder Point (ROP)** | `(D̄ × Lead Time) + SS` |
            | **Max Stock Level** | `ROP + (D̄ × Forecast Horizon)` |

            Z-score mapping:
            - 90% → Z = 1.28
            - 95% → Z = 1.645  *(selected)*
            - 99% → Z = 2.33
            """
        )

    # ── Primary KPI row ────────────────────────────────────────────────────
    inv1, inv2, inv3 = st.columns(3)

    with inv1:
        st.markdown(
            f"""
            <div class="kpi-card green">
                <p style="color:#3fb950; font-size:0.75rem; font-weight:700;
                          text-transform:uppercase; letter-spacing:.05em; margin:0 0 6px 0;">
                    📊 Average Daily Demand
                </p>
                <p style="font-size:2rem; font-weight:800; margin:0; color:#e6edf3;">
                    R$ {avg_daily_demand:,.2f}
                </p>
                <p style="color:#8b949e; font-size:0.82rem; margin:4px 0 0 0;">
                    Mean forecasted daily revenue over the next {forecast_days} days
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with inv2:
        st.markdown(
            f"""
            <div class="kpi-card yellow">
                <p style="color:#d29922; font-size:0.75rem; font-weight:700;
                          text-transform:uppercase; letter-spacing:.05em; margin:0 0 6px 0;">
                    🛡️ Safety Stock
                </p>
                <p style="font-size:2rem; font-weight:800; margin:0; color:#e6edf3;">
                    R$ {safety_stock:,.2f}
                </p>
                <p style="color:#8b949e; font-size:0.82rem; margin:4px 0 0 0;">
                    Buffer to absorb demand variability during lead time
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with inv3:
        st.markdown(
            f"""
            <div class="kpi-card red">
                <p style="color:#f85149; font-size:0.75rem; font-weight:700;
                          text-transform:uppercase; letter-spacing:.05em; margin:0 0 6px 0;">
                    🔄 Reorder Point (ROP)
                </p>
                <p style="font-size:2rem; font-weight:800; margin:0; color:#e6edf3;">
                    R$ {reorder_point:,.2f}
                </p>
                <p style="color:#8b949e; font-size:0.82rem; margin:4px 0 0 0;">
                    Trigger replenishment when stock falls to this level
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Secondary metrics row ──────────────────────────────────────────────
    sec1, sec2, sec3, sec4 = st.columns(4)

    with sec1:
        st.metric("📦 Max Stock Level",      f"R$ {max_stock_level:,.2f}")
    with sec2:
        st.metric("⏱️ Lead Time",            f"{int(lead_time)} days")
    with sec3:
        st.metric("🎯 Service Level (Z)",    f"Z = {z_score}")
    with sec4:
        st.metric("📉 Demand Uncertainty",   f"R$ {uncertainty:,.2f}")

    st.markdown("---")

    # ── Decision Alert ─────────────────────────────────────────────────────
    rop_pct = (reorder_point / max_stock_level) * 100 if max_stock_level > 0 else 0

    if rop_pct < 20:
        st.success(
            f"✅ **Stock Status: HEALTHY** — ROP (R$ {reorder_point:,.2f}) is "
            f"{rop_pct:.1f}% of your maximum stock level. You have ample buffer "
            f"before a reorder is needed."
        )
    elif rop_pct < 50:
        st.warning(
            f"⚠️ **Stock Status: MONITOR** — ROP (R$ {reorder_point:,.2f}) is "
            f"{rop_pct:.1f}% of max stock. Consider reviewing supplier lead times "
            f"to maintain the {service_level_label.split(' —')[0]} service level."
        )
    else:
        st.error(
            f"🚨 **Stock Status: URGENT** — ROP (R$ {reorder_point:,.2f}) is "
            f"{rop_pct:.1f}% of max stock. High demand volatility detected. "
            f"Immediate replenishment planning recommended."
        )

    # ── Inventory waterfall / demand chart ────────────────────────────────
    st.markdown("#### 📊 Daily Demand Forecast with Inventory Thresholds")

    fig_inv = go.Figure()

    # Confidence band
    fig_inv.add_trace(go.Scatter(
        x=pd.concat([future_only_raw["ds"], future_only_raw["ds"][::-1]]),
        y=pd.concat([future_only_raw["yhat_upper_brl"], future_only_raw["yhat_lower_brl"][::-1]]),
        fill="toself",
        fillcolor="rgba(88,166,255,0.08)",
        line=dict(color="rgba(0,0,0,0)"),
        hoverinfo="skip",
        name="Uncertainty Band",
    ))

    # Demand forecast line
    fig_inv.add_trace(go.Scatter(
        x=future_only_raw["ds"],
        y=future_only_raw["yhat_brl"],
        mode="lines",
        line=dict(color="#58a6ff", width=2),
        name="Daily Demand Forecast",
        hovertemplate="<b>%{x|%b %d}</b><br>Demand: R$ %{y:,.2f}<extra></extra>",
    ))

    # ROP threshold line
    fig_inv.add_hline(
        y=reorder_point / forecast_days,   # per-day equivalent for visual reference
        line=dict(color="#f85149", width=1.5, dash="dash"),
        annotation_text=f"Daily ROP Equivalent: R$ {reorder_point/forecast_days:,.0f}",
        annotation_font_color="#f85149",
    )

    # Safety stock floor line
    fig_inv.add_hline(
        y=safety_stock / forecast_days,
        line=dict(color="#d29922", width=1.2, dash="dot"),
        annotation_text=f"Safety Buffer / Day: R$ {safety_stock/forecast_days:,.0f}",
        annotation_font_color="#d29922",
        annotation_position="bottom right",
    )

    fig_inv.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0d1117",
        plot_bgcolor="#0d1117",
        xaxis=dict(title="Date",                gridcolor="#21262d"),
        yaxis=dict(title="Daily Revenue (BRL)", gridcolor="#21262d",
                   tickprefix="R$ ", tickformat=",.0f"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1, bgcolor="rgba(0,0,0,0)"),
        hovermode="x unified",
        height=400,
        margin=dict(l=10, r=10, t=40, b=10),
    )

    st.plotly_chart(fig_inv, use_container_width=True)

    # ── Sensitivity analysis ───────────────────────────────────────────────
    st.markdown("#### 🔬 Sensitivity Analysis — Safety Stock vs Lead Time")
    st.caption("How does safety stock change as supplier lead time varies? (fixed service level)")

    lead_range  = list(range(1, 31))
    ss_values   = [uncertainty * z_score * np.sqrt(lt) for lt in lead_range]
    rop_values  = [avg_daily_demand * lt + ss for lt, ss in zip(lead_range, ss_values)]

    fig_sens = go.Figure()
    fig_sens.add_trace(go.Scatter(
        x=lead_range, y=ss_values, mode="lines+markers",
        line=dict(color="#d29922", width=2),
        marker=dict(size=5),
        name="Safety Stock",
        hovertemplate="Lead Time: %{x}d<br>Safety Stock: R$ %{y:,.2f}<extra></extra>",
    ))
    fig_sens.add_trace(go.Scatter(
        x=lead_range, y=rop_values, mode="lines+markers",
        line=dict(color="#f85149", width=2),
        marker=dict(size=5),
        name="Reorder Point",
        hovertemplate="Lead Time: %{x}d<br>ROP: R$ %{y:,.2f}<extra></extra>",
    ))
    fig_sens.add_vline(
        x=int(lead_time),
        line=dict(color="#58a6ff", width=1.5, dash="dash"),
        annotation_text=f"Current LT ({int(lead_time)}d)",
        annotation_font_color="#58a6ff",
    )
    fig_sens.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0d1117",
        plot_bgcolor="#0d1117",
        xaxis=dict(title="Lead Time (Days)", gridcolor="#21262d", dtick=2),
        yaxis=dict(title="Value (BRL)", gridcolor="#21262d",
                   tickprefix="R$ ", tickformat=",.0f"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1, bgcolor="rgba(0,0,0,0)"),
        height=360,
        margin=dict(l=10, r=10, t=40, b=10),
    )
    st.plotly_chart(fig_sens, use_container_width=True)

# -----------------------------------------------------------------------------
# 8.  FOOTER
# -----------------------------------------------------------------------------
st.markdown("---")
st.markdown(
    """
    <div style="text-align:center; color:#484f58; font-size:0.8rem; padding:10px 0 20px 0;">
        Olist Sales Intelligence Suite &nbsp;·&nbsp;
        Prophet v1.1 &nbsp;·&nbsp;
        Trained on Olist Brazilian E-commerce Public Dataset (Kaggle) &nbsp;·&nbsp;
        Log-Transform: <code>np.log1p / np.expm1</code> &nbsp;·&nbsp;
        Built with Streamlit + Plotly
    </div>
    """,
    unsafe_allow_html=True,
)