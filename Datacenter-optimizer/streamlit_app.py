import streamlit as st
import pandas as pd
import numpy as np
from app.models import DataCenterOptimizer
import plotly.graph_objects as go
import plotly.express as px

# Page configuration
st.set_page_config(
    page_title="Data Center Optimizer",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'optimizer' not in st.session_state:
    st.session_state.optimizer = DataCenterOptimizer()
    st.session_state.optimizer.train_models()
    st.session_state.history = []

def create_efficiency_gauge(efficiency):
    """Create energy efficiency gauge chart"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=efficiency,
        title={'text': "Energy Efficiency"},
        gauge={
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 50], 'color': "red"},
                {'range': [50, 75], 'color': "yellow"},
                {'range': [75, 100], 'color': "green"}
            ]
        }
    ))
    return fig

def create_resource_usage_chart(metrics):
    """Create resource usage comparison chart"""
    metrics_df = pd.DataFrame({
        'Resource': ['CPU', 'Memory', 'Network', 'Disk I/O', 'Power'],
        'Usage': [
            metrics['cpu_usage'],
            metrics['memory_usage'],
            metrics['network_traffic']/10,  # Normalized for visualization
            metrics['disk_io']/5,          # Normalized for visualization
            metrics['power_consumption']/5  # Normalized for visualization
        ]
    })
    
    fig = px.bar(metrics_df, x='Resource', y='Usage',
                 title="Resource Usage Overview",
                 color='Usage',
                 color_continuous_scale='Viridis')
    return fig

def main():
    # Title and Introduction
    st.title("Data Center Cost Optimizer")
    st.write("""
    This tool uses machine learning to analyze VM metrics and provide cost optimization
    recommendations. It combines energy efficiency prediction, workload classification,
    and anomaly detection.
    """)
    
    # Sidebar inputs
    st.sidebar.header("VM Metrics Input")
    with st.sidebar.form("vm_metrics_form"):
        cpu_usage = st.slider("CPU Usage (%)", 0, 100, 50)
        memory_usage = st.slider("Memory Usage (%)", 0, 100, 50)
        network_traffic = st.slider("Network Traffic (Mbps)", 0, 1000, 200)
        disk_io = st.slider("Disk I/O (IOPS)", 0, 500, 100)
        power_consumption = st.slider("Power Consumption (Watts)", 50, 500, 200)
        
        submitted = st.form_submit_button("Analyze VM")

    if submitted:
        metrics = {
            'cpu_usage': cpu_usage,
            'memory_usage': memory_usage,
            'network_traffic': network_traffic,
            'disk_io': disk_io,
            'power_consumption': power_consumption
        }

        # Get analysis results
        with st.spinner('Analyzing VM metrics...'):
            results = st.session_state.optimizer.analyze_vm(metrics)
            st.session_state.history.append({**metrics, **results})

        # Display results in columns
        col1, col2 = st.columns(2)

        with col1:
            st.header("Analysis Results")
            
            # Metrics Summary
            st.subheader("Metrics Summary")
            metrics_df = pd.DataFrame([metrics])
            st.dataframe(metrics_df)
            
            # Key Findings
            st.subheader("Key Findings")
            st.metric(
                label="Energy Efficiency",
                value=f"{results['energy_efficiency']:.1f}%",
                delta=f"{results['energy_efficiency']-70:.1f}% from baseline"
            )
            st.metric(
                label="Workload Type",
                value=results['workload_type']
            )
            st.metric(
                label="Anomaly Status",
                value="Normal" if not results['is_anomaly'] else "Anomaly Detected",
                delta="Alert!" if results['is_anomaly'] else "Good",
                delta_color="inverse"
            )

        with col2:
            st.header("Visualizations")
            # Energy Efficiency Gauge
            st.plotly_chart(create_efficiency_gauge(results['energy_efficiency']))
            # Resource Usage Chart
            st.plotly_chart(create_resource_usage_chart(metrics))

        # Recommendations
        st.header("Recommendations")
        col3, col4 = st.columns(2)

        with col3:
            st.subheader("Placement Recommendation")
            st.write(f"**Primary:** {results['placement_recommendation']['primary']}")
            st.write(f"**Reasoning:** {results['placement_recommendation']['reasoning']}")
            st.write(f"**Potential Cost Savings:** {results['placement_recommendation']['potential_cost_savings']}")

        with col4:
            if results['additional_notes']:
                st.subheader("Additional Notes")
                for note in results['additional_notes']:
                    st.info(note)

        # Optimization Suggestions
        if results['optimization_suggestions']:
            st.subheader("Optimization Suggestions")
            for suggestion in results['optimization_suggestions']:
                with st.expander(f"{suggestion['type'].title()} - {suggestion['priority']} Priority"):
                    st.write(f"**Action:** {suggestion['action']}")
                    st.write(f"**Potential Savings:** {suggestion['potential_savings']}")

        # History
        if st.session_state.history:
            st.header("Analysis History")
            history_df = pd.DataFrame(st.session_state.history)
            st.dataframe(history_df)

if __name__ == "__main__":
    main()