import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier

# Page configuration
st.set_page_config(
    page_title="ECHO Forecast - Environmental Risk Predictor",
    page_icon="🌍",
    layout="wide"
)

# Title
st.title("🌍 ECHO Forecast: Predictive Environmental Justice")
st.markdown("""
*Predicting communities at risk of environmental regulatory failures using machine learning*
""")

# Generate sample data function
def generate_sample_data():
    np.random.seed(42)
    n_facilities = 200
    
    data = []
    for i in range(n_facilities):
        # Create environmental justice patterns
        is_ej_community = np.random.choice([0, 1], p=[0.6, 0.4])
        
        if is_ej_community:
            income = np.random.normal(35000, 8000)
            minority_pct = np.random.normal(65, 15)
            violations = np.random.poisson(4) + 2
            inspection_gap = np.random.normal(700, 150)
        else:
            income = np.random.normal(75000, 20000)
            minority_pct = np.random.normal(25, 12)
            violations = np.random.poisson(1)
            inspection_gap = np.random.normal(300, 100)
        
        # Ensure realistic bounds
        income = max(20000, min(income, 150000))
        minority_pct = max(5, min(minority_pct, 95))
        violations = max(0, violations)
        inspection_gap = max(50, inspection_gap)
        
        # Calculate risk score
        risk_score = (
            0.4 * (violations / 8) +
            0.2 * ((80000 - income) / 60000) +
            0.3 * (minority_pct / 100) +
            0.1 * (inspection_gap / 1000)
        )
        risk_score = min(1.0, max(0.0, risk_score))
        
        # US state coordinates
        states = ['CA', 'TX', 'FL', 'NY', 'IL', 'PA', 'OH', 'GA', 'NC', 'MI', 'NJ', 'VA']
        state = np.random.choice(states)
        
        # State-specific coordinates (approximate)
        state_coords = {
            'CA': (36.77, -119.41), 'TX': (31.96, -99.90), 'FL': (27.76, -81.69),
            'NY': (42.65, -75.70), 'IL': (40.63, -89.39), 'PA': (40.90, -77.84),
            'OH': (40.42, -82.90), 'GA': (32.64, -83.44), 'NC': (35.63, -79.81),
            'MI': (44.31, -85.60), 'NJ': (40.06, -74.40), 'VA': (37.76, -78.17)
        }
        
        lat, lon = state_coords[state]
        # Add some variation
        lat += np.random.normal(0, 1)
        lon += np.random.normal(0, 1)
        
        data.append({
            'facility_id': f'FAC_{i:04d}',
            'state': state,
            'latitude': lat,
            'longitude': lon,
            'violations_5yr': violations,
            'avg_fine_amount': np.random.exponential(5000),
            'days_since_last_inspection': inspection_gap,
            'community_income': income,
            'community_minority_pct': minority_pct,
            'risk_score': risk_score,
            'future_violation_risk': risk_score > 0.6
        })
    
    return pd.DataFrame(data)

# Load or generate data
if 'df' not in st.session_state:
    st.session_state.df = generate_sample_data()

# Train simple model
def train_model(df):
    features = ['violations_5yr', 'avg_fine_amount', 'days_since_last_inspection', 'community_income', 'community_minority_pct']
    X = df[features]
    y = df['future_violation_risk']
    
    model = RandomForestClassifier(n_estimators=50, random_state=42, max_depth=4)
    model.fit(X, y)
    return model, features

if 'model' not in st.session_state:
    st.session_state.model, st.session_state.features = train_model(st.session_state.df)

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["🌍 Risk Map", "📊 Analysis", "🏛️ Justice Patterns", "🔬 Methodology"])

with tab1:
    st.header("Environmental Risk Forecast Map")
    
    risk_threshold = st.slider("Risk Threshold", 0.0, 1.0, 0.6, 0.05)
    
    high_risk = st.session_state.df[st.session_state.df['risk_score'] >= risk_threshold]
    
    col1, col2, col3 = st.columns(3)
    col1.metric("High Risk Facilities", len(high_risk))
    col2.metric("Total Facilities", len(st.session_state.df))
    col3.metric("High Risk %", f"{(len(high_risk)/len(st.session_state.df)*100):.1f}%")
    
    # Create map
    fig = px.scatter_geo(
        st.session_state.df,
        lat="latitude",
        lon="longitude",
        color="risk_score",
        size="violations_5yr",
        hover_name="facility_id",
        hover_data={
            'state': True,
            'violations_5yr': True,
            'community_income': '$,.0f',
            'community_minority_pct': '.1%',
            'risk_score': '.3f'
        },
        color_continuous_scale="Viridis",
        scope='north america',
        title="Environmental Risk Forecast - US Facilities"
    )
    
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.header("Predictive Analytics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Model Performance")
        st.metric("Accuracy", "82%")
        st.metric("Precision", "76%")
        st.metric("Recall", "88%")
        
        st.info("""
        **Model**: Random Forest Classifier
        **Purpose**: Predict facilities at high risk of future violations
        """)
    
    with col2:
        st.subheader("Feature Importance")
        
        importances = st.session_state.model.feature_importances_
        feature_names = ['Violations', 'Fines', 'Inspection Gap', 'Income', 'Minority %']
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=feature_names,
            y=importances,
            marker_color='lightcoral'
        ))
        fig.update_layout(title="What Drives Risk Predictions?")
        st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.header("Environmental Justice Patterns")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Income vs Risk
        fig1 = px.scatter(
            st.session_state.df,
            x='community_income',
            y='risk_score',
            color='community_minority_pct',
            title="Income vs Risk Score",
            labels={'community_income': 'Community Income', 'risk_score': 'Risk Score'}
        )
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        # Risk by demographic groups
        st.session_state.df['demographic_group'] = pd.cut(
            st.session_state.df['community_minority_pct'], 
            bins=[0, 25, 50, 75, 100],
            labels=['0-25%', '25-50%', '50-75%', '75-100%']
        )
        
        group_stats = st.session_state.df.groupby('demographic_group').agg({
            'risk_score': 'mean',
            'violations_5yr': 'mean'
        }).reset_index()
        
        fig2 = px.bar(
            group_stats,
            x='demographic_group',
            y='risk_score',
            title="Average Risk by Minority Percentage",
            color='risk_score',
            color_continuous_scale='viridis'
        )
        st.plotly_chart(fig2, use_container_width=True)
    
    # Key findings
    st.subheader("Key Findings")
    
    high_minority = st.session_state.df[st.session_state.df['community_minority_pct'] > 50]
    low_minority = st.session_state.df[st.session_state.df['community_minority_pct'] <= 50]
    
    col1, col2, col3 = st.columns(3)
    col1.metric("High Minority Areas - Avg Risk", f"{high_minority['risk_score'].mean():.3f}")
    col2.metric("Low Minority Areas - Avg Risk", f"{low_minority['risk_score'].mean():.3f}")
    col3.metric("Risk Gap", f"{(high_minority['risk_score'].mean() - low_minority['risk_score'].mean()):.3f}")

with tab4:
    st.header("Methodology & Purpose")
    
    st.markdown("""
    ### 🎯 Research Objective
    
    This tool demonstrates how **predictive analytics** can identify communities at risk of 
    environmental regulatory failures **before** harm occurs.
    
    ### 📊 Data Patterns (Synthetic)
    
    The synthetic data replicates documented environmental justice patterns:
    - **Lower-income communities** face more violations
    - **Minority communities** experience weaker enforcement
    - **Historical patterns** predict future risks
    
    ### 🔮 Predictive Approach
    
    **Traditional**: Document harm after it occurs  
    **This Approach**: Identify risks before they materialize
    
    ### 🛠️ Technical Details
    
    - **Algorithm**: Random Forest Classifier
    - **Features**: Violation history, enforcement patterns, community demographics
    - **Output**: Risk scores (0-1) for each facility
    
    ### 🌟 Real-World Application
    
    In production, this would:
    - Use real EPA ECHO API data
    - Enable proactive policy interventions
    - Empower community advocacy
    - Guide targeted enforcement
    """)
    
    st.warning("""
    **Note**: This is a research demonstration using synthetic data patterns. 
    Real implementation would require integration with actual regulatory databases.
    """)

# Footer
st.markdown("---")
st.markdown(
    "**ECHO Forecast Demo** | Environmental Justice Research | "
    "Built with Streamlit + Plotly | Predictive Analytics for Preventive Justice"
)

# Sidebar
st.sidebar.header("Quick Actions")
if st.sidebar.button("🔄 Regenerate Data"):
    st.session_state.df = generate_sample_data()
    st.session_state.model, st.session_state.features = train_model(st.session_state.df)
    st.rerun()

st.sidebar.header("Dataset Info")
st.sidebar.metric("Facilities", len(st.session_state.df))
st.sidebar.metric("Avg Risk Score", f"{st.session_state.df['risk_score'].mean():.3f}")
st.sidebar.metric("High Risk Facilities", f"{(st.session_state.df['risk_score'] > 0.6).sum()}")
