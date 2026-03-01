import streamlit as st
import plotly.graph_objects as go
import numpy as np

def render():
    st.header("📉 The Degradation Problem")
    st.markdown("""
        Before ResNet, adding more layers to a Convolutional Neural Network actually **increased** training error. 
        This wasn't caused by overfitting—adding more layers led to higher *training* error, indicating that deeper networks are harder to optimize.
    """)
    
    layers = st.slider("Number of Layers (Simulated network depth)", min_value=20, max_value=56, value=20, step=2)
    
    # Generate mock training curve based on ResNet paper's observations
    # A 20-layer network converges better/faster than a 56-layer plain network.
    epochs = np.arange(1, 101)
    
    # Base error curve (20 layers)
    np.random.seed(42) # For deterministic noise
    error_20 = 1.0 * np.exp(-epochs / 20) + 0.1
    error_20 += np.random.normal(0, 0.005, size=len(epochs))
    
    # Error for current layers (simulated)
    # The more layers (up to 56), the worse it gets in a MULTIPLICATIVE exponential way
    # At 56 layers, error should be significantly higher
    base_factor = (layers - 20) / 36.0 
    
    # Model degradation where deeper networks don't learn as well
    current_error = 1.0 * np.exp(-epochs / (20 + base_factor * 10)) + 0.1 + (base_factor * 0.15)
    current_error += np.random.normal(0, 0.008, size=len(epochs))
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=epochs, y=error_20, mode='lines', name='20-layer (Plain)', line=dict(color='deepskyblue', dash='dash')))
    
    if layers != 20:
        fig.add_trace(go.Scatter(x=epochs, y=current_error, mode='lines', name=f'{layers}-layer (Plain)', line=dict(color='crimson')))
    
    fig.update_layout(
        title=f"Simulated Training Error on CIFAR-10",
        xaxis_title="Epochs",
        yaxis_title="Training Error",
        template="plotly_dark",
        hovermode="x unified",
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99)
    )
    
    st.plotly_chart(fig, width="stretch")
    
    st.info("💡 **Insight:** Notice how increasing the depth (layers) from 20 to 56 causes the training error to go *up*. This is counterintuitive! A deeper network should hypothetically be able to perfectly mimic the shallower one (by learning identity mappings for extra layers). The fact that it performs worse shows optimization algorithms struggle with deep plain networks.")
