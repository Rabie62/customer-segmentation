import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Customer Persona Segmenter",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .persona-card {
        background-color: #F8FAFC;
        border-radius: 12px;
        padding: 22px;
        margin-top: 15px;
        border: 2px solid #E2E8F0;
    }
</style>
""", unsafe_allow_html=True)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "kmeans_model.joblib")

@st.cache_resource
def load_segmentation_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)

payload = load_segmentation_model()

# ---------------------------------------------------------
# Sidebar - Presets & Project Details
# ---------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/shopping-cart-loaded.png", width=70)
    st.title("Project Overview")
    st.markdown("""
    **Unsupervised Machine Learning**
    - **Task**: K-Means Customer Clustering
    - **Dataset**: Kaggle Mall Customer Segmentation
    - **Clusters**: k = 5 (Validated via Elbow & Silhouette)
    """)
    st.divider()

    st.subheader("⚡ Quick Customer Presets")
    preset = st.radio(
        "Auto-populate customer profile:",
        options=[
            "Custom Input",
            "🎯 VIP / High Spender ($88k, Score 82)",
            "⚡ Impulsive Trendsetter ($25k, Score 80)",
            "💼 High-Wealth Saver ($85k, Score 15)",
            "🏷️ Budget Shopper ($22k, Score 20)",
            "🛒 Mainstream Shopper ($55k, Score 50)"
        ],
        index=0
    )

# Assign default values based on preset
if "VIP" in preset:
    default_income = 88
    default_score = 82
    default_age = 32
    default_gender = "Female"
elif "Impulsive" in preset:
    default_income = 25
    default_score = 80
    default_age = 22
    default_gender = "Female"
elif "Saver" in preset:
    default_income = 85
    default_score = 15
    default_age = 42
    default_gender = "Male"
elif "Budget" in preset:
    default_income = 22
    default_score = 20
    default_age = 50
    default_gender = "Male"
elif "Mainstream" in preset:
    default_income = 55
    default_score = 50
    default_age = 40
    default_gender = "Female"
else:
    default_income = 70
    default_score = 75
    default_age = 28
    default_gender = "Female"

# ---------------------------------------------------------
# Main Page
# ---------------------------------------------------------
st.markdown('<div class="main-header">🛍️ Customer Persona Segmenter</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Behavioral market segmentation powered by K-Means unsupervised clustering.</div>', unsafe_allow_html=True)

# Default personas and strategies in case metadata was not packaged
DEFAULT_PERSONA_MAPPING = {
    0: 'Standard / Mainstream',
    1: 'VIP / High Spenders',
    2: 'Impulsive / Trendsetters',
    3: 'Careful / Savers',
    4: 'Budget / Sensible'
}

DEFAULT_PERSONA_DETAILS = {
    'VIP / High Spenders': {
        'tag': '🎯 Prime Target',
        'desc': 'High income and high spending appetite. Represents the most lucrative customer group.',
        'strategy': 'Promote premium collections, private sales events, and personalized loyalty tiers.'
    },
    'Careful / Savers': {
        'tag': '💼 High Wealth Savers',
        'desc': 'High income but conservative spending patterns. They prioritize quality and value.',
        'strategy': 'Focus on high-durability products, warranties, and educational value propositions.'
    },
    'Standard / Mainstream': {
        'tag': '🛒 Core Base',
        'desc': 'Moderate income with balanced spending behavior. Forms the volume backbone.',
        'strategy': 'Run seasonal discounts, cross-selling bundles, and membership retention rewards.'
    },
    'Impulsive / Trendsetters': {
        'tag': '⚡ Trend Seekers',
        'desc': 'Modest income with disproportionately high spending. Strong preference for trendy goods.',
        'strategy': 'Engage via social media influencers, flash sales, and flexible payment plans.'
    },
    'Budget / Sensible': {
        'tag': '🏷️ Price Sensitive',
        'desc': 'Lower income and low spending propensity. Highly price-conscious.',
        'strategy': 'Offer coupons, clearance promotions, and essential product bundles.'
    }
}

if payload is None:
    st.error("Model file `kmeans_model.joblib` not found. Please run `customer_segmentation.ipynb` first to train and export the model.")
    st.stop()

# Support both dict payload and raw KMeans estimator
if isinstance(payload, dict):
    kmeans = payload.get('model', payload)
    centroids = payload.get('centroids', getattr(kmeans, 'cluster_centers_', None))
    persona_mapping = payload.get('persona_mapping', DEFAULT_PERSONA_MAPPING)
    persona_details = payload.get('persona_details', DEFAULT_PERSONA_DETAILS)
else:
    kmeans = payload
    centroids = kmeans.cluster_centers_
    persona_mapping = DEFAULT_PERSONA_MAPPING
    persona_details = DEFAULT_PERSONA_DETAILS

# Robust background data loading for the scatter plot
DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "Mall_Customers.csv")
if isinstance(payload, dict) and 'data' in payload:
    df_mall = payload['data']
elif os.path.exists(DATA_PATH):
    df_mall = pd.read_csv(DATA_PATH)
    if 'Cluster' not in df_mall.columns:
        df_mall['Cluster'] = kmeans.predict(df_mall[['Annual Income (k$)', 'Spending Score (1-100)']])
else:
    df_mall = pd.DataFrame(columns=['Annual Income (k$)', 'Spending Score (1-100)', 'Cluster'])

col_input, col_output = st.columns([1, 1], gap="large")

with col_input:
    st.subheader("👤 Shopper Profile")

    with st.container():
        gender = st.selectbox("Gender", ["Female", "Male"], index=0 if default_gender == "Female" else 1)
        age = st.slider("Customer Age", min_value=18, max_value=75, value=default_age, step=1)
        income_k = st.slider("Annual Income ($k / year)", min_value=10, max_value=140, value=default_income, step=1, help="In thousands of USD ($50k = $50,000)")
        spending_score = st.slider("Spending Score (1 - 100)", min_value=1, max_value=100, value=default_score, step=1, help="Mall rating based on past purchasing frequency & basket size")

    annual_income_full = income_k * 1000

    # User input point
    shopper_df = pd.DataFrame([[income_k, spending_score]], columns=['Annual Income (k$)', 'Spending Score (1-100)'])
    assigned_cluster = kmeans.predict(shopper_df)[0]
    shopper_point = shopper_df.values
    assigned_persona = persona_mapping[assigned_cluster]
    cluster_info = persona_details[assigned_persona]

    # Calculate distance to centroid
    centroid_point = centroids[assigned_cluster]
    euclidean_dist = np.linalg.norm(shopper_point - centroid_point)

    st.write("")
    st.subheader("🏷️ Assigned Persona")

    palette_colors = {
        'Standard / Mainstream': '#3B82F6',
        'VIP / High Spenders': '#10B981',
        'Impulsive / Trendsetters': '#F59E0B',
        'Careful / Savers': '#EF4444',
        'Budget / Sensible': '#8B5CF6'
    }
    accent_color = palette_colors.get(assigned_persona, '#3B82F6')

    st.markdown(f"""
    <div style="background-color: #FFFFFF; border-left: 6px solid {accent_color}; border-radius: 8px; padding: 18px; box-shadow: 0 1px 3px rgba(0,0,0,0.08);">
        <span style="background-color: {accent_color}22; color: {accent_color}; font-size: 0.8rem; font-weight: 700; padding: 4px 10px; border-radius: 12px; text-transform: uppercase;">
            {cluster_info['tag']}
        </span>
        <h2 style="color: #0F172A; margin: 8px 0 4px 0; font-size: 1.6rem;">{assigned_persona}</h2>
        <p style="color: #475569; font-size: 0.95rem; margin: 0 0 10px 0;">{cluster_info['desc']}</p>
        <hr style="border: none; border-top: 1px solid #F1F5F9; margin: 10px 0;">
        <div style="color: #1E293B; font-weight: 600; font-size: 0.9rem; margin-bottom: 2px;">🎯 Recommended Marketing Strategy:</div>
        <div style="color: #334155; font-size: 0.9rem;">{cluster_info['strategy']}</div>
    </div>
    """, unsafe_allow_html=True)

with col_output:
    st.subheader("📊 Cluster Position & Customer Space")

    # Clean KPI Cards for the Shopper
    st.markdown(f"""
    <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin-bottom: 16px;">
        <div style="background-color: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 10px; padding: 16px 10px; text-align: center;">
            <div style="color: #64748B; font-size: 0.8rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.04em; margin-bottom: 6px;">
                Annual Income
            </div>
            <div style="color: #0F172A; font-size: 1.55rem; font-weight: 700; white-space: nowrap;">
                ${income_k}k
            </div>
        </div>
        <div style="background-color: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 10px; padding: 16px 10px; text-align: center;">
            <div style="color: #64748B; font-size: 0.8rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.04em; margin-bottom: 6px;">
                Spending Score
            </div>
            <div style="color: #0F172A; font-size: 1.55rem; font-weight: 700; white-space: nowrap;">
                {spending_score} / 100
            </div>
        </div>
        <div style="background-color: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 10px; padding: 16px 10px; text-align: center;">
            <div style="color: #64748B; font-size: 0.8rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.04em; margin-bottom: 6px;">
                Centroid Proximity
            </div>
            <div style="color: #0F172A; font-size: 1.55rem; font-weight: 700; white-space: nowrap;">
                {euclidean_dist:.1f}
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 2D Scatter Plot with Current Customer Placement
    fig, ax = plt.subplots(figsize=(8, 5.5))

    palette_list = ['#3B82F6', '#10B981', '#F59E0B', '#EF4444', '#8B5CF6']
    
    # Plot existing mall customer points
    for c_id in range(5):
        c_name = persona_mapping[c_id]
        c_subset = df_mall[df_mall['Cluster'] == c_id]
        ax.scatter(
            c_subset['Annual Income (k$)'],
            c_subset['Spending Score (1-100)'],
            color=palette_list[c_id],
            label=c_name,
            alpha=0.45,
            s=50,
            edgecolors='none'
        )

    # Plot Centroids
    ax.scatter(
        centroids[:, 0],
        centroids[:, 1],
        s=180,
        c='black',
        marker='X',
        linewidth=1.5,
        label='Cluster Centroids'
    )

    # Plot Current Shopper as a Golden Star
    ax.scatter(
        [income_k],
        [spending_score],
        s=380,
        c='#EAB308',
        marker='*',
        edgecolors='#000000',
        linewidth=1.8,
        label='Current Shopper (Your Input)',
        zorder=10
    )

    ax.set_title('Customer Segmentation Map (Annual Income vs. Spending)', fontsize=12, fontweight='bold', pad=12)
    ax.set_xlabel('Annual Income (k$)', fontsize=10)
    ax.set_ylabel('Spending Score (1-100)', fontsize=10)
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', frameon=True, fontsize=8.5)
    plt.tight_layout()

    st.pyplot(fig)
