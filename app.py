import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
from sklearn.ensemble import RandomForestClassifier
import plotly.express as px
import plotly.graph_objects as go

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

# Generate sample data with realistic US coordinates
def generate_sample_data():
    np.random.seed(42)
    n_facilities = 150
    
    # Real US city coordinates for more realistic distribution
    city_coordinates = {
        'Los Angeles, CA': (34.05, -118.24), 'Houston, TX': (29.76, -95.36), 
        'Chicago, IL': (41.87, -87.62), 'Phoenix, AZ': (33.44, -112.07),
        'Philadelphia, PA': (39.95, -75.16), 'San Antonio, TX': (29.42, -98.49),
        'San Diego, CA': (32.71, -117.16), 'Dallas, TX': (32.77, -96.79),
        'San Jose, CA': (37.33, -121.88), 'Detroit, MI': (42.33, -83.04),
        'Jacksonville, FL': (30.33, -81.65), 'Indianapolis, IN': (39.76, -86.15),
        'San Francisco, CA': (37.77, -122.41), 'Columbus, OH': (39.96, -82.99),
        'Charlotte, NC': (35.22, -80.84), 'Seattle, WA': (47.60, -122.33),
        'Denver, CO': (39.73, -104.99), 'Boston, MA': (42.36, -71.05),
        'Atlanta, GA': (33.74, -84.38), 'Miami, FL': (25.76, -80.19)
    }
    
    cities = list(city_coordinates.keys())
    
    data = []
    for i in range(n_facilities):
        city = np.random.choice(cities)
        base_lat, base_lon = city_coordinates[city]
        
        # Add some random variation around the city
        lat = base_lat + np.random.normal(0, 0.3)
        lon = base_lon + np.random.normal(0, 0.3)
        
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
        
        data.append({
            'facility_id': f'FAC_{i:04d}',
            'city': city,
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

# Create Folium map function
def create_folium_map(df, risk_threshold=0.6):
    # Create base map centered on US
    m = folium.Map(location=[39.8283, -98.5795], zoom_start=4)
    
    # Add markers for each facility
    for _, row in df.iterrows():
        risk = row['risk_score']
        
        # Determine color based on risk
        if risk >= risk_threshold:
            color = 'red'
            size = 12
        elif risk >= risk_threshold - 0.2:
            color = 'orange'
            size = 8
        else:
            color = 'green'
            size = 6
        
        # Create popup text
        popup_text = f"""
        <b>Facility:</b> {row['facility_id']}<br>
        <b>City:</b> {row['city']}<br>
        <b>Risk Score:</b> {risk:.3f}<br>
        <b>Violations (5yr):</b> {row['violations_5yr']}<br>
        <b>Community Income:</b> ${row['community_income']:,.0f}<br>
        <b>Minority %:</b> {row['community_minority_pct']:.1f}%
        """
        
        # Add marker to map
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=size,
            popup=folium.Popup(popup_text, max_width=300),
            color=color,
            fillColor=color,
            fillOpacity=0.7,
            weight=1
        ).add_to(m)
    
    # Add legend
    legend_html = '''
    <div style="position: fixed; 
                top: 10px; left: 50px; width: 200px; height: 120px; 
                background-color: white; border:2px solid grey; z-index:9999; 
                font-size:14px; padding: 10px">
    <p><b>Risk Legend</b></p>
    <p><span style="color:red;">●</span> High Risk (≥{threshold})</p>
    <p><span style="color:orange;">●</span> Medium Risk</p>
    <p><span style="color:green;">●</span> Low Risk</p>
    <p><i>Size = Violation History</i></p>
    </div>
    '''.format(threshold=risk_threshold)
    
    m.get_root().html.add_child(folium.Element(legend_html))
    
    return m

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

# Main app layout
col1, col2 = st.columns([2, 1])

with col1:
    st.header("Environmental Risk Forecast Map")
    
    risk_threshold = st.slider(
        "Risk Threshold", 
        0.0, 1.0, 0.6, 0.05,
        help="Adjust to highlight facilities above a certain risk level"
    )
    
    # Create and display Folium map
    folium_map = create_folium_map(st.session_state.df, risk_threshold)
    st_folium(folium_map, width=700, height=500)

with col2:
    st.header("Risk Analysis")
    
    high_risk = st.session_state.df[st.session_state.df['risk_score'] >= risk_threshold]
    
    st.metric("High Risk Facilities", len(high_risk))
    st.metric("Total Facilities", len(st.session_state.df))
    st.metric("High Risk %", f"{(len(high_risk)/len(st.session_state.df)*100):.1f}%")
    
    # Risk distribution
    st.subheader("Risk Distribution")
    risk_bins = pd.cut(st.session_state.df['risk_score'], bins=[0, 0.3, 0.6, 1.0])
    risk_counts = risk_bins.value_counts().sort_index()
    
    for bin_range, count in risk_counts.items():
        st.write(f"{bin_range}: {count} facilities")
    
    # Environmental justice insights
    st.subheader("Justice Insights")
    high_minority = st.session_state.df[st.session_state.df['community_minority_pct'] > 50]
    low_minority = st.session_state.df[st.session_state.df['community_minority_pct'] <= 50]
    
    st.write(f"High minority areas: {high_minority['risk_score'].mean():.3f} avg risk")
    st.write(f"Low minority areas: {low_minority['risk_score'].mean():.3f} avg risk")
    
    risk_gap = high_minority['risk_score'].mean() - low_minority['risk_score'].mean()
    st.write(f"Risk gap: {risk_gap:.3f}")

# Additional analysis tabs
tab1, tab2 = st.tabs(["📊 Detailed Analysis", "🔍 Facility Details"])

with tab1:
    st.header("Predictive Model Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Feature Importance")
        importances = st.session_state.model.feature_importances_
        features = ['Violations', 'Fines', 'Inspection Gap', 'Income', 'Minority %']
        
        importance_df = pd.DataFrame({
            'Feature': features,
            'Importance': importances
        }).sort_values('Importance', ascending=True)
        
        st.bar_chart(importance_df.set_index('Feature'))
    
    with col2:
        st.subheader("Risk Patterns")
        
        # Create a scatter plot using Plotly (fixed version)
        chart_data = st.session_state.df[['community_income', 'risk_score', 'community_minority_pct']].copy()
        chart_data['minority_high'] = chart_data['community_minority_pct'] > 50
        
        scatter_fig = px.scatter(
            chart_data,
            x='community_income',
            y='risk_score',
            color='minority_high',
            title='Income vs Risk Score',
            labels={
                'community_income': 'Community Income ($)',
                'risk_score': 'Risk Score',
                'minority_high': 'High Minority %'
            }
        )
        st.plotly_chart(scatter_fig, use_container_width=True)

with tab2:
    st.header("Facility Details")
    
    # Sort by risk score
    display_df = st.session_state.df.sort_values('risk_score', ascending=False)
    
    # Format for display
    display_df_detailed = display_df[[
        'facility_id', 'city', 'risk_score', 'violations_5yr', 
        'community_income', 'community_minority_pct'
    ]].copy()
    
    display_df_detailed['community_income'] = display_df_detailed['community_income'].apply(lambda x: f"${x:,.0f}")
    display_df_detailed['community_minority_pct'] = display_df_detailed['community_minority_pct'].apply(lambda x: f"{x:.1f}%")
    display_df_detailed['risk_score'] = display_df_detailed['risk_score'].apply(lambda x: f"{x:.3f}")
    
    st.dataframe(display_df_detailed, use_container_width=True)

# Methodology section
with st.expander("🔬 Methodology & Research Context"):
    st.markdown("""
    ### Research Objective
    This tool demonstrates how **predictive analytics** can identify communities at risk of 
    environmental regulatory failures **before** harm occurs, enabling preventive environmental justice.
    
    ### Data Patterns (Synthetic)
    The synthetic data replicates documented environmental justice patterns:
    - **Lower-income communities** face more violations
    - **Minority communities** experience weaker enforcement  
    - **Historical patterns** predict future risks
    
    ### Technical Approach
    - **Algorithm**: Random Forest Classifier
    - **Features**: Violation history, enforcement patterns, community demographics
    - **Output**: Risk scores (0-1) for proactive intervention
    
    ### Real-World Application
    In production, this would enable:
    - Proactive policy interventions
    - Targeted enforcement resources
    - Community-led advocacy
    - Preventive environmental justice
    """)

# Footer
st.markdown("---")
st.markdown(
    "**ECHO Forecast Demo** | Environmental Justice Research | "
    "Folium + Streamlit | Predictive Analytics for Preventive Justice"
)

# Sidebar
st.sidebar.header("Controls")
if st.sidebar.button("🔄 Generate New Data"):
    st.session_state.df = generate_sample_data()
    st.session_state.model, st.session_state.features = train_model(st.session_state.df)
    st.rerun()

st.sidebar.header("Dataset Info")
st.sidebar.metric("Facilities", len(st.session_state.df))
st.sidebar.metric("Avg Risk Score", f"{st.session_state.df['risk_score'].mean():.3f}")
st.sidebar.metric("High Risk Facilities", f"{(st.session_state.df['risk_score'] > 0.6).sum()}")

st.sidebar.header("About")
st.sidebar.info(
    "This demo shows how machine learning can predict environmental "
    "regulatory failures before they occur, enabling preventive justice."
)
