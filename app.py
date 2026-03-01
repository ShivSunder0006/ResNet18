import streamlit as st

st.set_page_config(
    page_title="ResNet-18 Interactive Explainer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Sidebar Navigation
with st.sidebar:
    st.title("🧠 ResNet-18 Explainer")
    st.markdown("Explore the breakthroughs of **Deep Residual Learning for Image Recognition**.")
    page = st.radio(
        "Navigation",
        [
            "🏠 Introduction",
            "📉 The Degradation Problem",
            "🔗 The Residual Block",
            "🔍 Architecture Explorer",
            "💬 Chat with the Paper",
            "👀 Live Feature Maps"
        ]
    )
    
    st.markdown("---")
    st.markdown("Built with ❤️ using Streamlit & PyTorch")

# Render the selected page
if page == "🏠 Introduction":
    st.header("Welcome to the ResNet-18 Explainer! 🎉")
    st.markdown("""
        **Deep Residual Learning for Image Recognition** (ResNet) is one of the most influential papers in modern deep learning, allowing networks to become substantially deeper without sacrificing trainability.
        
        Use the sidebar to navigate through the interactive components:
        
        * **📉 The Degradation Problem:** See why simply stacking layers fails.
        * **🔗 The Residual Block:** Understand the core formula: $H(x) = F(x) + x$.
        * **🔍 Architecture Explorer:** Dive into the structure of ResNet-18.
        * **💬 Chat with the Paper:** Ask an AI questions about the original text.
        * **👀 Live Feature Maps:** See what a real ResNet-18 sees inside its layers.
        
        *Ready to begin? Click on the next section in the sidebar!*
    """)
elif page == "📉 The Degradation Problem":
    from components.degradation_problem import render
    render()
elif page == "🔗 The Residual Block":
    from components.residual_block import render
    render()
elif page == "🔍 Architecture Explorer":
    from components.architecture import render
    render()
elif page == "💬 Chat with the Paper":
    from components.rag_chatbot import render
    render()
elif page == "👀 Live Feature Maps":
    from components.feature_maps import render
    render()
