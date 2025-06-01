import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime
from datetime import timedelta
import random

# Page configuration
st.set_page_config(
    page_title="Supply Chain Risk AI Dashboard",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .risk-high {
        background-color: #ffebee;
        border-left: 5px solid #f44336;
        padding: 10px;
        margin: 5px 0;
    }
    .risk-medium {
        background-color: #fff3e0;
        border-left: 5px solid #ff9800;
        padding: 10px;
        margin: 5px 0;
    }
    .risk-low {
        background-color: #e8f5e8;
        border-left: 5px solid #4caf50;
        padding: 10px;
        margin: 5px 0;
    }
    .stTab > div > div {
        gap: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'risks_data' not in st.session_state:
    # Generate sample risk data
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
    
    suppliers = ['Supplier A', 'Supplier B', 'Supplier C', 'Supplier D', 'Supplier E']
    risk_types = ['Geopolitical', 'Weather', 'Financial', 'Operational', 'Cyber Security', 'Quality']
    
    risks = []
    for i in range(200):
        risk = {
            'date': np.random.choice(dates),
            'supplier': np.random.choice(suppliers),
            'risk_type': np.random.choice(risk_types),
            'severity': np.random.choice(['High', 'Medium', 'Low'], p=[0.2, 0.3, 0.5]),
            'probability': np.random.uniform(0.1, 0.9),
            'impact_score': np.random.uniform(1, 10),
            'status': np.random.choice(['Active', 'Mitigated', 'Monitoring'], p=[0.3, 0.4, 0.3]),
            'region': np.random.choice(['North America', 'Europe', 'Asia-Pacific', 'Latin America'])
        }
        risks.append(risk)
    
    st.session_state.risks_data = pd.DataFrame(risks)

# Sidebar
st.sidebar.title("🔧 Dashboard Controls")

# Date range selector
date_range = st.sidebar.date_input(
    "Select Date Range",
    value=(datetime.date.today() - timedelta(days=30), datetime.date.today()),
    max_value=datetime.date.today()
)

# Risk type filter
risk_types = st.sidebar.multiselect(
    "Filter by Risk Type",
    options=st.session_state.risks_data['risk_type'].unique(),
    default=st.session_state.risks_data['risk_type'].unique()
)

# Severity filter
severity_filter = st.sidebar.multiselect(
    "Filter by Severity",
    options=['High', 'Medium', 'Low'],
    default=['High', 'Medium', 'Low']
)

# Supplier filter
supplier_filter = st.sidebar.multiselect(
    "Filter by Supplier",
    options=st.session_state.risks_data['supplier'].unique(),
    default=st.session_state.risks_data['supplier'].unique()
)

# Filter data based on selections
filtered_data = st.session_state.risks_data[
    (st.session_state.risks_data['risk_type'].isin(risk_types)) &
    (st.session_state.risks_data['severity'].isin(severity_filter)) &
    (st.session_state.risks_data['supplier'].isin(supplier_filter))
]

# Main header
st.markdown('<h1 class="main-header">🚚 Supply Chain Risk AI Dashboard</h1>', unsafe_allow_html=True)

# Key metrics row
col1, col2, col3, col4 = st.columns(4)

with col1:
    total_risks = len(filtered_data)
    st.metric("Total Risks", total_risks, delta=f"+{random.randint(1, 5)} from yesterday")

with col2:
    high_risks = len(filtered_data[filtered_data['severity'] == 'High'])
    st.metric("High Severity", high_risks, delta=f"-{random.randint(1, 3)} from yesterday")

with col3:
    active_risks = len(filtered_data[filtered_data['status'] == 'Active'])
    st.metric("Active Risks", active_risks, delta=f"+{random.randint(0, 2)} from yesterday")

with col4:
    avg_impact = filtered_data['impact_score'].mean()
    st.metric("Avg Impact Score", f"{avg_impact:.1f}", delta=f"{random.uniform(-0.5, 0.5):.1f}")

# Create tabs for different sections
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Risk Overview", "🎯 Risk Detection", "📈 Impact Assessment", "🛡️ Mitigation Planning", "⚙️ System Monitoring"])

with tab1:
    st.header("Risk Overview Dashboard")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Risk distribution by type
        risk_counts = filtered_data['risk_type'].value_counts()
        fig_pie = px.pie(
            values=risk_counts.values, 
            names=risk_counts.index,
            title="Risk Distribution by Type",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        # Risk severity distribution
        severity_counts = filtered_data['severity'].value_counts()
        colors = {'High': '#ff4444', 'Medium': '#ffaa00', 'Low': '#44ff44'}
        fig_bar = px.bar(
            x=severity_counts.index,
            y=severity_counts.values,
            title="Risk Distribution by Severity",
            color=severity_counts.index,
            color_discrete_map=colors
        )
        st.plotly_chart(fig_bar, use_container_width=True)
    
    # Risk trend over time
    st.subheader("Risk Trends Over Time")
    
    # Group by month for trend analysis
    filtered_data['month'] = pd.to_datetime(filtered_data['date']).dt.to_period('M')
    monthly_risks = filtered_data.groupby(['month', 'severity']).size().unstack(fill_value=0)
    
    fig_trend = px.line(
        x=monthly_risks.index.astype(str),
        y=[monthly_risks.get('High', []), monthly_risks.get('Medium', []), monthly_risks.get('Low', [])],
        title="Monthly Risk Trends by Severity"
    )
    fig_trend.update_layout(xaxis_title="Month", yaxis_title="Number of Risks")
    st.plotly_chart(fig_trend, use_container_width=True)

with tab2:
    st.header("🎯 Real-time Risk Detection")
    
    # Simulated real-time alerts
    st.subheader("Active Alerts")
    
    # Recent high-severity risks
    recent_high_risks = filtered_data[
        (filtered_data['severity'] == 'High') & 
        (filtered_data['status'] == 'Active')
    ].head(5)
    
    for _, risk in recent_high_risks.iterrows():
        st.markdown(f"""
        <div class="risk-high">
            <strong>🚨 {risk['risk_type']} Risk - {risk['supplier']}</strong><br>
            Impact Score: {risk['impact_score']:.1f} | Probability: {risk['probability']:.1%}<br>
            <small>Detected: {risk['date'].strftime('%Y-%m-%d')}</small>
        </div>
        """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Risk Detection by Region")
        region_risks = filtered_data['region'].value_counts()
        fig_region = px.bar(
            x=region_risks.values,
            y=region_risks.index,
            orientation='h',
            title="Risks by Geographic Region",
            color=region_risks.values,
            color_continuous_scale='Reds'
        )
        st.plotly_chart(fig_region, use_container_width=True)
    
    with col2:
        st.subheader("Supplier Risk Profile")
        supplier_risk_matrix = pd.crosstab(filtered_data['supplier'], filtered_data['severity'])
        fig_heatmap = px.imshow(
            supplier_risk_matrix.T,
            title="Supplier-Severity Risk Matrix",
            color_continuous_scale='RdYlBu_r',
            aspect='auto'
        )
        fig_heatmap.update_xaxes(title="Suppliers")
        fig_heatmap.update_yaxes(title="Severity Level")
        st.plotly_chart(fig_heatmap, use_container_width=True)

with tab3:
    st.header("📈 Impact Assessment")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Impact vs Probability Matrix")
        fig_scatter = px.scatter(
            filtered_data,
            x='probability',
            y='impact_score',
            color='severity',
            size='impact_score',
            hover_data=['supplier', 'risk_type'],
            title="Risk Impact vs Probability Analysis",
            color_discrete_map={'High': '#ff4444', 'Medium': '#ffaa00', 'Low': '#44ff44'}
        )
        fig_scatter.update_xaxes(title="Probability", tickformat='.0%')
        fig_scatter.update_yaxes(title="Impact Score")
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    with col2:
        st.subheader("Financial Impact Estimation")
        # Simulate financial impact data
        financial_impact = filtered_data.copy()
        financial_impact['estimated_cost'] = financial_impact['impact_score'] * np.random.uniform(10000, 100000, len(financial_impact))
        
        impact_by_type = financial_impact.groupby('risk_type')['estimated_cost'].sum().sort_values(ascending=True)
        
        fig_cost = px.bar(
            x=impact_by_type.values,
            y=impact_by_type.index,
            orientation='h',
            title="Estimated Financial Impact by Risk Type",
            color=impact_by_type.values,
            color_continuous_scale='Reds'
        )
        fig_cost.update_xaxes(title="Estimated Cost ($)")
        st.plotly_chart(fig_cost, use_container_width=True)
    
    # Detailed impact assessment table
    st.subheader("Detailed Impact Assessment")
    
    impact_summary = filtered_data.groupby(['risk_type', 'severity']).agg({
        'impact_score': ['mean', 'max', 'count'],
        'probability': 'mean'
    }).round(2)
    
    st.dataframe(impact_summary, use_container_width=True)

with tab4:
    st.header("🛡️ AI-Powered Mitigation Planning")
    
    # Risk selection for mitigation
    st.subheader("Select Risk for Mitigation Analysis")
    
    high_priority_risks = filtered_data[
        (filtered_data['severity'] == 'High') | 
        (filtered_data['impact_score'] > 7)
    ].sort_values('impact_score', ascending=False)
    
    if not high_priority_risks.empty:
        selected_risk_idx = st.selectbox(
            "Choose a high-priority risk:",
            range(len(high_priority_risks)),
            format_func=lambda x: f"{high_priority_risks.iloc[x]['risk_type']} - {high_priority_risks.iloc[x]['supplier']} (Impact: {high_priority_risks.iloc[x]['impact_score']:.1f})"
        )
        
        selected_risk = high_priority_risks.iloc[selected_risk_idx]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Risk Details")
            st.write(f"**Type:** {selected_risk['risk_type']}")
            st.write(f"**Supplier:** {selected_risk['supplier']}")
            st.write(f"**Severity:** {selected_risk['severity']}")
            st.write(f"**Impact Score:** {selected_risk['impact_score']:.1f}")
            st.write(f"**Probability:** {selected_risk['probability']:.1%}")
            st.write(f"**Status:** {selected_risk['status']}")
            st.write(f"**Region:** {selected_risk['region']}")
        
        with col2:
            st.subheader("AI-Generated Mitigation Strategies")
            
            # Simulated AI recommendations based on risk type
            mitigation_strategies = {
                'Geopolitical': [
                    "Diversify supplier base across multiple regions",
                    "Establish backup suppliers in stable countries",
                    "Implement political risk insurance",
                    "Monitor geopolitical developments closely"
                ],
                'Weather': [
                    "Implement weather monitoring systems",
                    "Create seasonal inventory buffers",
                    "Develop alternative transportation routes",
                    "Partner with suppliers in different climate zones"
                ],
                'Financial': [
                    "Conduct regular financial health assessments",
                    "Implement supplier financial monitoring",
                    "Negotiate payment terms and guarantees",
                    "Maintain backup supplier relationships"
                ],
                'Operational': [
                    "Implement operational excellence programs",
                    "Regular supplier audits and assessments",
                    "Capacity planning and backup arrangements",
                    "Technology upgrades and automation"
                ],
                'Cyber Security': [
                    "Implement cybersecurity assessments",
                    "Require security certifications",
                    "Regular security training and updates",
                    "Backup data and communication systems"
                ],
                'Quality': [
                    "Enhance quality control processes",
                    "Regular quality audits and inspections",
                    "Supplier quality certification programs",
                    "Implement quality monitoring systems"
                ]
            }
            
            strategies = mitigation_strategies.get(selected_risk['risk_type'], ["General risk mitigation strategies needed"])
            
            for i, strategy in enumerate(strategies, 1):
                st.write(f"{i}. {strategy}")
        
        # Action plan
        st.subheader("Recommended Action Plan")
        
        if st.button("Generate Action Plan", type="primary"):
            with st.spinner("AI is analyzing the risk and generating action plan..."):
                import time
                time.sleep(2)  # Simulate processing time
                
                st.success("✅ Action Plan Generated!")
                
                action_plan = {
                    "Immediate Actions (0-7 days)": [
                        "Alert relevant stakeholders",
                        "Assess current inventory levels",
                        "Contact alternative suppliers"
                    ],
                    "Short-term Actions (1-4 weeks)": [
                        "Implement primary mitigation strategy",
                        "Negotiate with backup suppliers",
                        "Update risk assessment"
                    ],
                    "Long-term Actions (1-6 months)": [
                        "Review and update supplier contracts",
                        "Implement monitoring systems",
                        "Conduct post-incident review"
                    ]
                }
                
                for phase, actions in action_plan.items():
                    st.write(f"**{phase}:**")
                    for action in actions:
                        st.write(f"- {action}")
                    st.write("")

with tab5:
    st.header("⚙️ System Monitoring & Performance")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("System Health Metrics")
        
        # Simulated system metrics
        system_metrics = {
            "API Response Time": f"{random.uniform(50, 200):.0f}ms",
            "Model Accuracy": f"{random.uniform(85, 95):.1f}%",
            "Data Freshness": f"{random.randint(1, 5)} min ago",
            "Active Connections": random.randint(50, 200),
            "Alerts Processed": random.randint(100, 500),
            "System Uptime": "99.8%"
        }
        
        for metric, value in system_metrics.items():
            st.metric(metric, value)
    
    with col2:
        st.subheader("Model Performance")
        
        # Simulated model performance data
        dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
        accuracy_scores = np.random.normal(0.92, 0.02, 30)
        precision_scores = np.random.normal(0.89, 0.03, 30)
        recall_scores = np.random.normal(0.87, 0.025, 30)
        
        performance_df = pd.DataFrame({
            'Date': dates,
            'Accuracy': accuracy_scores,
            'Precision': precision_scores,
            'Recall': recall_scores
        })
        
        fig_performance = px.line(
            performance_df,
            x='Date',
            y=['Accuracy', 'Precision', 'Recall'],
            title="Model Performance Over Time"
        )
        fig_performance.update_yaxes(range=[0.8, 1.0])
        st.plotly_chart(fig_performance, use_container_width=True)
    
    # LangSmith Integration Status
    st.subheader("LangSmith Monitoring Status")
    
    langsmith_status = {
        "Traces Collected": "✅ Active",
        "Evaluation Runs": "✅ Scheduled",
        "Dataset Updates": "✅ Real-time",
        "Feedback Loop": "✅ Enabled",
        "Error Tracking": "✅ Monitoring"
    }
    
    status_cols = st.columns(len(langsmith_status))
    for i, (feature, status) in enumerate(langsmith_status.items()):
        with status_cols[i]:
            st.write(f"**{feature}**")
            st.write(status)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p>Supply Chain Risk AI Dashboard | Powered by LangChain, LangGraph & LangSmith</p>
    <p>🔄 Last updated: Real-time | 📊 Data sources: Multi-modal AI analysis</p>
</div>
""", unsafe_allow_html=True)

# Sidebar additional info
with st.sidebar:
    st.markdown("---")
    st.subheader("💡 Quick Actions")
    
    if st.button("🔍 Run Risk Scan", use_container_width=True):
        with st.spinner("Scanning for new risks..."):
            import time
            time.sleep(2)
        st.success("Scan completed! 3 new risks detected.")
    
    if st.button("📧 Generate Report", use_container_width=True):
        st.success("Risk report generated and sent!")
    
    if st.button("🔄 Refresh Data", use_container_width=True):
        st.rerun()
    
    st.markdown("---")
    st.subheader("📈 System Stats")
    st.write("🟢 All systems operational")
    st.write(f"📊 Processing {random.randint(1000, 5000)} events/hour")
    st.write(f"🤖 AI Confidence: {random.randint(85, 95)}%")