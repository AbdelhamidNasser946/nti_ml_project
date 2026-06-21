"""
Fraud Detection — Streamlit App
================================
Model  : RandomForestClassifier (100 trees, 12 features)
Run    : streamlit run fraud_detection_app.py

Install dependencies:
    pip install streamlit joblib scikit-learn pandas numpy matplotlib
"""

import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import streamlit as st

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fraud Detection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_PATH = r"D:\NTI\project\fraud_rf_model.joblib"

FEATURE_NAMES = [
    "transaction_amount", "hour_of_day", "is_weekend", "num_items",
    "customer_age", "prev_transactions", "distance_from_home",
    "device_type", "network_quality", "is_first_transaction",
    "store_type", "velocity_score",
]

FEATURE_IMPORTANCES = {
    "hour_of_day": 0.1721, "device_type": 0.1544, "store_type": 0.1370,
    "prev_transactions": 0.0840, "num_items": 0.0731,
    "transaction_amount": 0.0628, "distance_from_home": 0.0611,
    "customer_age": 0.0584, "network_quality": 0.0543,
    "velocity_score": 0.0526, "is_weekend": 0.0499,
    "is_first_transaction": 0.0404,
}

# ─────────────────────────────────────────────────────────────
# STYLING
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background-color: #0f1117; }

    [data-testid="stSidebar"] {
        background-color: #1a1d27;
        border-right: 1px solid #2d2f3e;
    }

    .metric-card {
        background: #1a1d27;
        border: 1px solid #2d2f3e;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-label {
        font-size: 13px;
        color: #8b8fa8;
        margin-bottom: 4px;
        text-transform: uppercase;
        letter-spacing: .05em;
    }
    .metric-value { font-size: 28px; font-weight: 700; }

    .result-fraud {
        background: linear-gradient(135deg, #3d1515, #5c1f1f);
        border: 1px solid #e53e3e;
        border-radius: 14px;
        padding: 1.5rem 2rem;
        text-align: center;
    }
    .result-legit {
        background: linear-gradient(135deg, #0f2d1f, #1a4d2e);
        border: 1px solid #38a169;
        border-radius: 14px;
        padding: 1.5rem 2rem;
        text-align: center;
    }
    .result-title { font-size: 26px; font-weight: 700; margin-bottom: 6px; }
    .result-sub   { font-size: 14px; color: #a0aec0; }

    .section-header {
        font-size: 13px;
        font-weight: 600;
        color: #8b8fa8;
        text-transform: uppercase;
        letter-spacing: .08em;
        margin: 1.2rem 0 .6rem 0;
        border-bottom: 1px solid #2d2f3e;
        padding-bottom: 6px;
    }

    .stTabs [data-baseweb="tab"] {
        background: #1a1d27;
        border-radius: 8px 8px 0 0;
        color: #8b8fa8;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background: #2d2f3e !important;
        color: #ffffff !important;
    }

    .stDataFrame { border-radius: 10px; }
    hr { border-color: #2d2f3e; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        model = joblib.load(MODEL_PATH)
        return model, None
    except FileNotFoundError:
        return None, f"Model file '{MODEL_PATH}' not found. Place it in the same folder."
    except Exception as e:
        return None, str(e)


# ─────────────────────────────────────────────────────────────
# PREDICTION
# ─────────────────────────────────────────────────────────────
def predict(model, features: dict):
    X = pd.DataFrame([features])[FEATURE_NAMES]
    proba = model.predict_proba(X)[0]
    pred  = model.predict(X)[0]
    return int(pred), float(proba[1])


# ─────────────────────────────────────────────────────────────
# GAUGE CHART  (matplotlib semicircle)
# ─────────────────────────────────────────────────────────────
def gauge_chart(fraud_prob: float) -> plt.Figure:
    pct   = fraud_prob * 100
    color = "#e53e3e" if pct >= 50 else "#f6ad55" if pct >= 30 else "#38a169"

    fig, ax = plt.subplots(figsize=(4, 2.4), subplot_kw={"aspect": "equal"})
    fig.patch.set_facecolor("#1a1d27")
    ax.set_facecolor("#1a1d27")

    # Background arc (grey track)
    theta_bg = np.linspace(np.pi, 0, 300)
    ax.plot(np.cos(theta_bg), np.sin(theta_bg), lw=18,
            color="#2d2f3e", solid_capstyle="butt")

    # Filled arc (value)
    fill_angle = np.pi * (1 - fraud_prob)
    theta_fill = np.linspace(np.pi, fill_angle, 300)
    ax.plot(np.cos(theta_fill), np.sin(theta_fill), lw=18,
            color=color, solid_capstyle="butt")

    # Zone markers
    for angle, label in [(np.pi, "0%"), (np.pi * 0.7, "30%"),
                          (np.pi * 0.4, "60%"), (0, "100%")]:
        ax.text(1.22 * np.cos(angle), 1.22 * np.sin(angle), label,
                ha="center", va="center", fontsize=7, color="#8b8fa8")

    # Center text
    ax.text(0, -0.05, f"{pct:.1f}%", ha="center", va="center",
            fontsize=22, fontweight="bold", color=color)
    ax.text(0, -0.32, "Fraud Probability", ha="center", va="center",
            fontsize=8, color="#8b8fa8")

    ax.set_xlim(-1.35, 1.35)
    ax.set_ylim(-0.5, 1.2)
    ax.axis("off")
    plt.tight_layout(pad=0.2)
    return fig


# ─────────────────────────────────────────────────────────────
# FEATURE IMPORTANCE CHART  (matplotlib horizontal bar)
# ─────────────────────────────────────────────────────────────
def importance_chart(model) -> plt.Figure:
    try:
        imps  = model.feature_importances_
        names = list(model.feature_names_in_)
    except Exception:
        imps  = list(FEATURE_IMPORTANCES.values())
        names = list(FEATURE_IMPORTANCES.keys())

    pairs = sorted(zip(names, imps), key=lambda x: x[1])
    feat_sorted, imp_sorted = zip(*pairs)

    colors = [
        "#e53e3e" if v > 0.15 else "#f6ad55" if v > 0.08 else "#4299e1"
        for v in imp_sorted
    ]

    fig, ax = plt.subplots(figsize=(7, 5))
    fig.patch.set_facecolor("#1a1d27")
    ax.set_facecolor("#1a1d27")

    bars = ax.barh(feat_sorted, imp_sorted, color=colors, height=0.6)

    for bar, val in zip(bars, imp_sorted):
        ax.text(val + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=9, color="#ffffff")

    ax.set_xlabel("Importance", color="#8b8fa8", fontsize=10)
    ax.tick_params(colors="#8b8fa8", labelsize=9)
    ax.spines[:].set_color("#2d2f3e")
    ax.xaxis.label.set_color("#8b8fa8")
    ax.set_facecolor("#1a1d27")
    ax.grid(axis="x", color="#2d2f3e", linestyle="--", linewidth=0.6)

    legend_items = [
        mpatches.Patch(color="#e53e3e", label="High (> 15%)"),
        mpatches.Patch(color="#f6ad55", label="Medium (8–15%)"),
        mpatches.Patch(color="#4299e1", label="Low (< 8%)"),
    ]
    ax.legend(handles=legend_items, loc="lower right",
              facecolor="#1a1d27", edgecolor="#2d2f3e",
              labelcolor="#8b8fa8", fontsize=8)

    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────
# BATCH DONUT CHART  (matplotlib pie)
# ─────────────────────────────────────────────────────────────
def batch_results_chart(results_df: pd.DataFrame) -> plt.Figure:
    counts = results_df["Prediction"].value_counts()
    labels = counts.index.tolist()
    values = counts.values.tolist()
    colors = ["#38a169" if l == "Legitimate" else "#e53e3e" for l in labels]

    fig, ax = plt.subplots(figsize=(4, 4))
    fig.patch.set_facecolor("#1a1d27")
    ax.set_facecolor("#1a1d27")

    wedges, texts, autotexts = ax.pie(
        values,
        labels=labels,
        colors=colors,
        autopct="%1.1f%%",
        startangle=90,
        wedgeprops={"width": 0.5, "edgecolor": "#1a1d27", "linewidth": 2},
        textprops={"color": "#ffffff", "fontsize": 11},
    )
    for at in autotexts:
        at.set_color("#ffffff")
        at.set_fontsize(10)

    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────
# SIDEBAR — INPUT FORM
# ─────────────────────────────────────────────────────────────
def render_sidebar() -> tuple:
    with st.sidebar:
        st.markdown("## 🔍 Transaction Details")
        st.markdown("Fill in the transaction fields below.")

        st.markdown('<div class="section-header">💳 Transaction</div>', unsafe_allow_html=True)
        transaction_amount = st.number_input(
            "Transaction Amount ($)", min_value=1.0, max_value=10000.0,
            value=85.0, step=0.5,
        )
        num_items = st.number_input(
            "Number of Items", min_value=1, max_value=50, value=3, step=1
        )
        store_type = st.selectbox(
            "Store Type",
            options=[(0, "Physical Store"), (1, "Online Store")],
            format_func=lambda x: x[1],
        )[0]

        st.markdown('<div class="section-header">👤 Customer</div>', unsafe_allow_html=True)
        customer_age = st.slider("Customer Age", 18, 85, 35)
        prev_transactions = st.number_input(
            "Previous Transactions", min_value=0, max_value=500, value=12, step=1
        )
        is_first_transaction = st.selectbox(
            "First Transaction?",
            options=[(0, "No — returning customer"), (1, "Yes — new customer")],
            format_func=lambda x: x[1],
        )[0]

        st.markdown('<div class="section-header">📍 Location & Device</div>', unsafe_allow_html=True)
        distance_from_home = st.number_input(
            "Distance from Home (km)", min_value=0.0, max_value=500.0,
            value=12.0, step=0.5,
        )
        device_type = st.selectbox(
            "Device Type",
            options=[(0, "Type 0 — Mobile"), (1, "Type 1 — Desktop"), (2, "Type 2 — Tablet")],
            format_func=lambda x: x[1],
        )[0]

        st.markdown('<div class="section-header">📡 Network & Timing</div>', unsafe_allow_html=True)
        network_quality = st.slider("Network Quality (0–100)", 0, 100, 75)
        hour_of_day = st.selectbox(
            "Time of Day",
            options=[(1, "Period 1 — Morning"), (2, "Period 2 — Afternoon"), (3, "Period 3 — Evening")],
            format_func=lambda x: x[1],
        )[0]
        is_weekend = st.selectbox(
            "Day Type",
            options=[(0, "Weekday"), (1, "Weekend")],
            format_func=lambda x: x[1],
        )[0]
        velocity_score = st.number_input(
            "Velocity Score", min_value=-5.0, max_value=20.0, value=5.0, step=0.1,
            help="Transaction velocity — higher = faster/more suspicious activity",
        )

        st.markdown("---")
        predict_btn = st.button("🔍 Predict", use_container_width=True, type="primary")

    return {
        "transaction_amount":  transaction_amount,
        "hour_of_day":         hour_of_day,
        "is_weekend":          is_weekend,
        "num_items":           num_items,
        "customer_age":        customer_age,
        "prev_transactions":   prev_transactions,
        "distance_from_home":  distance_from_home,
        "device_type":         device_type,
        "network_quality":     network_quality,
        "is_first_transaction": is_first_transaction,
        "store_type":          store_type,
        "velocity_score":      velocity_score,
    }, predict_btn


# ─────────────────────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────────────────────
def main():
    model, error = load_model()

    st.markdown("# 🔍 Fraud Detection System")
    st.markdown("Real-time transaction fraud prediction powered by Random Forest (100 trees).")
    st.markdown("---")

    if error:
        st.error(f"⚠️ {error}")
        st.info(f"Make sure `{MODEL_PATH}` is in the same directory as this script.")
        return

    features, predict_btn = render_sidebar()

    tab1, tab2, tab3 = st.tabs([
        "  🎯  Single Prediction  ",
        "  📂  Batch Prediction   ",
        "  📊  Model Insights     ",
    ])

    # ── TAB 1: Single Prediction ─────────────────────────────
    with tab1:
        label, fraud_prob = predict(model, features)
        legit_prob = 1 - fraud_prob

        # Result banner
        if label == 1:
            st.markdown("""
            <div class="result-fraud">
                <div class="result-title">🚨 Fraudulent Transaction</div>
                <div class="result-sub">This transaction has been flagged as potentially fraudulent</div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="result-legit">
                <div class="result-title">✅ Legitimate Transaction</div>
                <div class="result-sub">This transaction appears to be legitimate</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Gauge + metric cards
        col1, col2 = st.columns([1.3, 1])
        with col1:
            st.pyplot(gauge_chart(fraud_prob), use_container_width=True)
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Fraud Probability</div>
                <div class="metric-value" style="color:#e53e3e">{fraud_prob*100:.1f}%</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Legitimate Probability</div>
                <div class="metric-value" style="color:#38a169">{legit_prob*100:.1f}%</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Decision</div>
                <div class="metric-value" style="color:{'#e53e3e' if label==1 else '#38a169'}">
                    {'FRAUD' if label==1 else 'LEGIT'}
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Risk level
        st.markdown("#### Risk Level")
        if fraud_prob >= 0.70:
            st.error("🔴 **HIGH RISK** — Block this transaction immediately.")
        elif fraud_prob >= 0.40:
            st.warning("🟡 **MEDIUM RISK** — Flag for manual review.")
        else:
            st.success("🟢 **LOW RISK** — Transaction looks safe to approve.")

        with st.expander("📋 View input values"):
            df_in = pd.DataFrame([features]).T.rename(columns={0: "Value"})
            st.dataframe(df_in, use_container_width=True)

    # ── TAB 2: Batch Prediction ───────────────────────────────
    with tab2:
        st.markdown("### Upload a CSV file for batch prediction")
        st.markdown(f"The file must contain these columns: `{'`, `'.join(FEATURE_NAMES)}`")

        # Template download
        template_df = pd.DataFrame(columns=FEATURE_NAMES)
        template_df.loc[0] = [85.0,  2, 0, 3, 35, 12,   12.0, 1, 75, 0, 0, 5.0]
        template_df.loc[1] = [249.0, 1, 1, 8, 22,  0, 180.0,  0, 30, 1, 1, 1.2]
        st.download_button(
            "⬇️ Download CSV Template",
            data=template_df.to_csv(index=False),
            file_name="fraud_template.csv",
            mime="text/csv",
        )

        uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

        if uploaded_file:
            try:
                df_batch    = pd.read_csv(uploaded_file)
                missing_cols = set(FEATURE_NAMES) - set(df_batch.columns)
                if missing_cols:
                    st.error(f"Missing columns: {missing_cols}")
                else:
                    X_batch = df_batch[FEATURE_NAMES]
                    probs   = model.predict_proba(X_batch)[:, 1]
                    preds   = model.predict(X_batch)

                    df_batch["Fraud Probability (%)"] = (probs * 100).round(2)
                    df_batch["Prediction"] = ["Fraud" if p == 1 else "Legitimate" for p in preds]
                    df_batch["Risk Level"] = pd.cut(
                        probs,
                        bins=[0, 0.30, 0.60, 1.0],
                        labels=["🟢 Low", "🟡 Medium", "🔴 High"],
                    )

                    n_fraud    = int((preds == 1).sum())
                    n_legit    = int((preds == 0).sum())
                    fraud_rate = n_fraud / len(preds) * 100

                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Total Transactions", len(df_batch))
                    c2.metric("🔴 Flagged as Fraud", n_fraud)
                    c3.metric("✅ Legitimate",        n_legit)
                    c4.metric("Fraud Rate",           f"{fraud_rate:.1f}%")

                    col_a, col_b = st.columns([1, 1.8])
                    with col_a:
                        st.pyplot(batch_results_chart(df_batch), use_container_width=True)
                    with col_b:
                        st.markdown("#### Results")
                        st.dataframe(
                            df_batch[
                                ["Fraud Probability (%)", "Prediction", "Risk Level"]
                                + FEATURE_NAMES
                            ],
                            use_container_width=True,
                            height=280,
                        )

                    st.download_button(
                        "⬇️ Download Results CSV",
                        data=df_batch.to_csv(index=False),
                        file_name="fraud_predictions.csv",
                        mime="text/csv",
                    )

            except Exception as e:
                st.error(f"Error processing file: {e}")

    # ── TAB 3: Model Insights ─────────────────────────────────
    with tab3:
        st.markdown("### Feature Importances")
        st.markdown(
            "How much each feature contributed to the model's predictions "
            "across all 100 decision trees."
        )
        st.pyplot(importance_chart(model), use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Model Details")
            info = {
                "Algorithm":            "Random Forest",
                "Trees (n_estimators)": model.n_estimators,
                "Features":             model.n_features_in_,
                "Classes":              "0 = Legit, 1 = Fraud",
                "Max depth":            str(model.max_depth),
                "Max features":         str(model.max_features),
            }
            st.dataframe(
                pd.DataFrame(info.items(), columns=["Property", "Value"]),
                use_container_width=True, hide_index=True,
            )

        with col2:
            st.markdown("#### Top Risk Signals")
            top_feats = sorted(FEATURE_IMPORTANCES.items(), key=lambda x: -x[1])[:5]
            for feat, imp in top_feats:
                pct = int(imp * 100)
                st.markdown(f"**{feat}** — {pct}%")
                st.progress(pct)

        st.markdown("---")
        st.markdown("#### Interpretation Guide")
        cols = st.columns(3)
        cols[0].error("🔴 **High Risk (≥ 70%)**\nBlock transaction immediately.")
        cols[1].warning("🟡 **Medium Risk (40–70%)**\nFlag for manual review.")
        cols[2].success("🟢 **Low Risk (< 40%)**\nApprove transaction normally.")


if __name__ == "__main__":
    main()