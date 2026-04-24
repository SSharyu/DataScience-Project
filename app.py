import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ─────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🏦 Loan Eligibility Predictor",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .main { background-color: #f0f4f8; }

    /* Header */
    .header-container {
        background: linear-gradient(135deg, #1a237e 0%, #283593 50%, #3949ab 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(26,35,126,0.3);
    }
    .header-title {
        color: #ffffff;
        font-size: 2.2rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .header-subtitle {
        color: #c5cae9;
        font-size: 1rem;
        margin-top: 0.4rem;
    }

    /* Metric cards */
    .metric-card {
        background: white;
        border-radius: 14px;
        padding: 1.4rem 1.6rem;
        box-shadow: 0 2px 16px rgba(0,0,0,0.08);
        border-left: 5px solid #3949ab;
        margin-bottom: 1rem;
    }
    .metric-label {
        color: #607d8b;
        font-size: 0.82rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.8px;
    }
    .metric-value {
        color: #1a237e;
        font-size: 1.8rem;
        font-weight: 800;
        margin-top: 0.2rem;
    }

    /* Result boxes */
    .result-approved {
        background: linear-gradient(135deg, #e8f5e9, #c8e6c9);
        border: 2px solid #2e7d32;
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        box-shadow: 0 4px 20px rgba(46,125,50,0.15);
    }
    .result-rejected {
        background: linear-gradient(135deg, #ffebee, #ffcdd2);
        border: 2px solid #c62828;
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        box-shadow: 0 4px 20px rgba(198,40,40,0.15);
    }
    .result-title {
        font-size: 2rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
    }
    .result-subtitle {
        font-size: 1rem;
        font-weight: 500;
        opacity: 0.8;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a237e 0%, #283593 100%);
    }
    section[data-testid="stSidebar"] .stSelectbox label,
    section[data-testid="stSidebar"] .stSlider label,
    section[data-testid="stSidebar"] .stNumberInput label,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: #ffffff !important;
    }
    section[data-testid="stSidebar"] .stMarkdown { color: #c5cae9; }

    /* Section headers */
    .section-header {
        font-size: 1.1rem;
        font-weight: 700;
        color: #1a237e;
        border-bottom: 3px solid #3949ab;
        padding-bottom: 0.4rem;
        margin: 1.4rem 0 1rem 0;
    }

    /* Info pill */
    .info-pill {
        display: inline-block;
        background: #e8eaf6;
        color: #3949ab;
        border-radius: 20px;
        padding: 0.25rem 0.85rem;
        font-size: 0.82rem;
        font-weight: 600;
        margin: 0.15rem;
    }

    /* Probability bar */
    .prob-bar-container {
        background: #e0e0e0;
        border-radius: 12px;
        height: 22px;
        margin: 0.5rem 0;
        overflow: hidden;
    }
    .prob-bar-fill-green {
        background: linear-gradient(90deg, #2e7d32, #43a047);
        height: 100%;
        border-radius: 12px;
        transition: width 0.5s ease;
    }
    .prob-bar-fill-red {
        background: linear-gradient(90deg, #c62828, #e53935);
        height: 100%;
        border-radius: 12px;
    }

    /* Button */
    .stButton>button {
        background: linear-gradient(135deg, #1a237e, #3949ab);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 2rem;
        font-size: 1rem;
        font-weight: 700;
        width: 100%;
        cursor: pointer;
        transition: all 0.2s;
        box-shadow: 0 4px 15px rgba(26,35,126,0.3);
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(26,35,126,0.4);
    }

    /* Input card */
    .input-card {
        background: white;
        border-radius: 14px;
        padding: 1.5rem;
        box-shadow: 0 2px 12px rgba(0,0,0,0.07);
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# Load model artifacts
# ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    required = ['loan_model.pkl', 'scaler.pkl', 'feature_names.pkl',
                'le_education.pkl', 'le_self_employed.pkl', 'le_loan_status.pkl']
    missing = [f for f in required if not os.path.exists(f)]
    if missing:
        return None, None, None, None, None, None, missing

    model         = joblib.load('loan_model.pkl')
    scaler        = joblib.load('scaler.pkl')
    feature_names = joblib.load('feature_names.pkl')
    le_edu        = joblib.load('le_education.pkl')
    le_emp        = joblib.load('le_self_employed.pkl')
    le_target     = joblib.load('le_loan_status.pkl')
    return model, scaler, feature_names, le_edu, le_emp, le_target, []

model, scaler, feature_names, le_edu, le_emp, le_target, missing_files = load_artifacts()


# ─────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="header-container">
    <p class="header-title">🏦 Loan Eligibility Predictor</p>
    <p class="header-subtitle">AI-powered loan approval prediction using Machine Learning</p>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# Model not found warning
# ─────────────────────────────────────────────────────────────
if missing_files:
    st.warning(f"""
    ⚠️ **Model files not found:** `{'`, `'.join(missing_files)}`

    Please run the **Jupyter Notebook** first to train and save the model, then place the `.pkl` files in the same directory as `app.py`.
    """)
    st.stop()


# ─────────────────────────────────────────────────────────────
# Sidebar – Input Form
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📋 Applicant Details")
    st.markdown("---")

    st.markdown("### 👤 Personal Information")
    education     = st.selectbox("Education Level", ["Graduate", "Not Graduate"])
    self_employed = st.selectbox("Self Employed", ["No", "Yes"])
    no_of_dep     = st.slider("Number of Dependents", 0, 5, 1)

    st.markdown("---")
    st.markdown("### 💰 Financial Information")
    income_annum  = st.number_input("Annual Income (₹)", min_value=100_000,
                                     max_value=10_000_000, value=500_000,
                                     step=50_000, format="%d")
    loan_amount   = st.number_input("Loan Amount (₹)", min_value=100_000,
                                     max_value=40_000_000, value=2_000_000,
                                     step=100_000, format="%d")
    loan_term     = st.slider("Loan Term (months)", 2, 20, 12)
    cibil_score   = st.slider("CIBIL Score", 300, 900, 650)

    st.markdown("---")
    st.markdown("### 🏠 Asset Values (₹)")
    res_assets  = st.number_input("Residential Assets", 0, 30_000_000, 1_000_000,
                                   step=100_000, format="%d")
    comm_assets = st.number_input("Commercial Assets", 0, 20_000_000, 500_000,
                                   step=100_000, format="%d")
    lux_assets  = st.number_input("Luxury Assets", 0, 40_000_000, 500_000,
                                   step=100_000, format="%d")
    bank_assets = st.number_input("Bank Assets", 0, 20_000_000, 300_000,
                                   step=50_000, format="%d")

    st.markdown("---")
    predict_btn = st.button("🔍 Check Loan Eligibility", use_container_width=True)


# ─────────────────────────────────────────────────────────────
# Main content area
# ─────────────────────────────────────────────────────────────
col_left, col_right = st.columns([1, 1], gap="large")

# ── Left: Applicant Summary ──
with col_left:
    st.markdown('<p class="section-header">📊 Applicant Summary</p>', unsafe_allow_html=True)

    with st.container():
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"""
            <div class="metric-card">
                <p class="metric-label">Annual Income</p>
                <p class="metric-value">₹{income_annum:,.0f}</p>
            </div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class="metric-card">
                <p class="metric-label">Loan Amount</p>
                <p class="metric-value">₹{loan_amount:,.0f}</p>
            </div>""", unsafe_allow_html=True)

        c3, c4 = st.columns(2)
        with c3:
            total_assets = res_assets + comm_assets + lux_assets + bank_assets
            st.markdown(f"""
            <div class="metric-card">
                <p class="metric-label">Total Assets</p>
                <p class="metric-value">₹{total_assets:,.0f}</p>
            </div>""", unsafe_allow_html=True)
        with c4:
            lti = loan_amount / (income_annum + 1)
            st.markdown(f"""
            <div class="metric-card">
                <p class="metric-label">Loan-to-Income Ratio</p>
                <p class="metric-value">{lti:.2f}x</p>
            </div>""", unsafe_allow_html=True)

    # Tags
    st.markdown("**Profile Tags:**")
    tags_html = ""
    tags_html += f'<span class="info-pill">📚 {education}</span>'
    tags_html += f'<span class="info-pill">💼 {"Self-Employed" if self_employed=="Yes" else "Salaried"}</span>'
    tags_html += f'<span class="info-pill">👨‍👩‍👧 {no_of_dep} Dependents</span>'
    tags_html += f'<span class="info-pill">📅 {loan_term} Months</span>'

    cibil_color = "#2e7d32" if cibil_score >= 700 else "#f57c00" if cibil_score >= 600 else "#c62828"
    cibil_label = "Excellent" if cibil_score >= 750 else "Good" if cibil_score >= 700 else "Fair" if cibil_score >= 600 else "Poor"
    tags_html += f'<span class="info-pill" style="background:#e8f5e9;color:{cibil_color}">⭐ CIBIL {cibil_score} ({cibil_label})</span>'
    st.markdown(tags_html, unsafe_allow_html=True)

    # CIBIL gauge
    st.markdown('<p class="section-header" style="margin-top:1.5rem">📈 CIBIL Score Gauge</p>',
                unsafe_allow_html=True)
    cibil_pct = int(((cibil_score - 300) / 600) * 100)
    bar_color = "#2e7d32" if cibil_score >= 700 else "#f57c00" if cibil_score >= 600 else "#c62828"
    st.markdown(f"""
    <div style="display:flex;justify-content:space-between;font-size:0.75rem;color:#607d8b;margin-bottom:4px">
        <span>Poor (300)</span><span>Fair (600)</span><span>Good (700)</span><span>Excellent (900)</span>
    </div>
    <div class="prob-bar-container">
        <div style="background:linear-gradient(90deg,#c62828,#f57c00,{bar_color});
                    width:{cibil_pct}%;height:100%;border-radius:12px;"></div>
    </div>
    <p style="text-align:center;font-size:1rem;font-weight:700;color:{bar_color};margin-top:4px">
        {cibil_score} — {cibil_label}
    </p>
    """, unsafe_allow_html=True)


# ── Right: Prediction Result ──
with col_right:
    st.markdown('<p class="section-header">🎯 Prediction Result</p>', unsafe_allow_html=True)

    if not predict_btn:
        st.info("👈 Fill in the applicant details on the left sidebar and click **Check Loan Eligibility** to get the prediction.")
        st.markdown("""
        <div style="background:white;border-radius:14px;padding:1.5rem;box-shadow:0 2px 12px rgba(0,0,0,0.07)">
            <p style="font-weight:700;color:#1a237e;font-size:1rem">📌 How it works</p>
            <ol style="color:#455a64;font-size:0.9rem;line-height:1.9">
                <li>Enter applicant's personal & financial details</li>
                <li>Our ML model analyzes 16 features</li>
                <li>Instant prediction with probability scores</li>
                <li>Get actionable insights</li>
            </ol>
            <p style="font-weight:700;color:#1a237e;font-size:0.9rem;margin-top:1rem">🔑 Key Factors</p>
            <p style="color:#607d8b;font-size:0.85rem">CIBIL Score · Income · Assets · Loan-to-Income Ratio · Loan Term</p>
        </div>
        """, unsafe_allow_html=True)

    else:
        # Build feature vector — direct mapping avoids encoder label mismatch
        edu_map  = {"Graduate": 0, "Not Graduate": 1}
        emp_map  = {"No": 0, "Yes": 1}
        edu_enc  = edu_map[education]
        emp_enc  = emp_map[self_employed]
        total    = res_assets + comm_assets + lux_assets + bank_assets
        lti_r    = loan_amount / (income_annum + 1)
        atl_r    = total / (loan_amount + 1)
        ipd      = income_annum / (no_of_dep + 1)

        input_dict = {
            'no_of_dependents': no_of_dep,
            'education': edu_enc,
            'self_employed': emp_enc,
            'income_annum': income_annum,
            'loan_amount': loan_amount,
            'loan_term': loan_term,
            'cibil_score': cibil_score,
            'residential_assets_value': res_assets,
            'commercial_assets_value': comm_assets,
            'luxury_assets_value': lux_assets,
            'bank_asset_value': bank_assets,
            'total_assets': total,
            'loan_to_income_ratio': lti_r,
            'asset_to_loan_ratio': atl_r,
            'income_per_dependent': ipd
        }

        input_df = pd.DataFrame([input_dict])

        # Align to feature names
        for col in feature_names:
            if col not in input_df.columns:
                input_df[col] = 0
        input_df = input_df[feature_names]

        input_scaled = scaler.transform(input_df)
        pred_enc  = model.predict(input_scaled)[0]
        pred_prob = model.predict_proba(input_scaled)[0]

        # LabelEncoder encodes alphabetically: Approved=0, Rejected=1
        target_map = {0: 'Approved', 1: 'Rejected'}
        prediction = target_map[int(pred_enc)]
        prob_approved = pred_prob[0]
        prob_rejected = pred_prob[1]

        # Result banner
        if prediction == 'Approved':
            st.markdown(f"""
            <div class="result-approved">
                <p class="result-title" style="color:#2e7d32">✅ LOAN APPROVED</p>
                <p class="result-subtitle" style="color:#388e3c">
                    Congratulations! The application meets eligibility criteria.
                </p>
                <p style="font-size:2.2rem;font-weight:900;color:#1b5e20;margin-top:0.8rem">
                    {prob_approved*100:.1f}% Confidence
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-rejected">
                <p class="result-title" style="color:#c62828">❌ LOAN REJECTED</p>
                <p class="result-subtitle" style="color:#d32f2f">
                    This application does not meet the current eligibility criteria.
                </p>
                <p style="font-size:2.2rem;font-weight:900;color:#b71c1c;margin-top:0.8rem">
                    {prob_rejected*100:.1f}% Confidence
                </p>
            </div>
            """, unsafe_allow_html=True)

        # Probability breakdown
        st.markdown('<p class="section-header" style="margin-top:1.5rem">📊 Probability Breakdown</p>',
                    unsafe_allow_html=True)

        st.markdown(f"""
        <div style="background:white;border-radius:12px;padding:1.2rem;box-shadow:0 2px 10px rgba(0,0,0,0.07)">
            <p style="margin:0 0 4px 0;color:#2e7d32;font-weight:700;font-size:0.92rem">
                ✅ Approved — {prob_approved*100:.1f}%
            </p>
            <div class="prob-bar-container">
                <div class="prob-bar-fill-green" style="width:{prob_approved*100:.1f}%"></div>
            </div>
            <p style="margin:0.8rem 0 4px 0;color:#c62828;font-weight:700;font-size:0.92rem">
                ❌ Rejected — {prob_rejected*100:.1f}%
            </p>
            <div class="prob-bar-container">
                <div class="prob-bar-fill-red" style="width:{prob_rejected*100:.1f}%"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Key risk factors
        st.markdown('<p class="section-header" style="margin-top:1.5rem">💡 Key Insights</p>',
                    unsafe_allow_html=True)

        insights = []
        if cibil_score >= 700:
            insights.append(("✅", f"Strong CIBIL score ({cibil_score}) — positive signal", "#2e7d32"))
        elif cibil_score >= 600:
            insights.append(("⚠️", f"Average CIBIL score ({cibil_score}) — borderline", "#f57c00"))
        else:
            insights.append(("❌", f"Low CIBIL score ({cibil_score}) — high risk", "#c62828"))

        if lti_r < 3:
            insights.append(("✅", f"Healthy loan-to-income ratio ({lti_r:.1f}x)", "#2e7d32"))
        elif lti_r < 5:
            insights.append(("⚠️", f"Moderate loan-to-income ratio ({lti_r:.1f}x)", "#f57c00"))
        else:
            insights.append(("❌", f"High loan-to-income ratio ({lti_r:.1f}x) — risky", "#c62828"))

        if total >= loan_amount:
            insights.append(("✅", f"Total assets (₹{total:,.0f}) exceed loan amount", "#2e7d32"))
        else:
            insights.append(("⚠️", f"Total assets (₹{total:,.0f}) below loan amount", "#f57c00"))

        if education == "Graduate":
            insights.append(("✅", "Graduate — favorable profile", "#2e7d32"))
        if self_employed == "No":
            insights.append(("✅", "Salaried — stable income source", "#2e7d32"))

        insight_html = '<div style="background:white;border-radius:12px;padding:1rem;box-shadow:0 2px 10px rgba(0,0,0,0.07)">'
        for icon, text, color in insights:
            insight_html += f'<p style="margin:0.4rem 0;color:{color};font-size:0.88rem;font-weight:600">{icon} {text}</p>'
        insight_html += '</div>'
        st.markdown(insight_html, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center;color:#90a4ae;font-size:0.82rem;padding:0.5rem 0">
    🏦 Loan Eligibility Predictor &nbsp;|&nbsp; Powered by Machine Learning &nbsp;|&nbsp;
    <em>For demonstration purposes only. Not financial advice.</em>
</div>
""", unsafe_allow_html=True)