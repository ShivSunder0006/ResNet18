import streamlit as st

def render():
    st.header("🔗 The Residual Block")
    st.markdown("""
        To solve the degradation problem, ResNet introduced the **Residual Block**. 
        Instead of hoping the stacked layers directly fit the desired underlying mapping $H(x)$, 
        ResNet explicitly lets these layers fit a **residual mapping**, $F(x) := H(x) - x$. 
        The original mapping is then recast as:
        
        $$H(x) = F(x) + x$$
        
        This means if the optimal mapping is an identity mapping, it's easier to push the weights to zero ($F(x) = 0$) 
        than to fit an identity mapping by a stack of non-linear layers.
    """)
    
    st.subheader("Interactive Flow")
    st.markdown("Let's simulate a simplified scalar input $x$ flowing through a basic Residual Block.")
    
    x_val = st.number_input("Input value $x$:", value=1.0, step=0.1)
    
    col1, col2, col3 = st.columns([2, 1, 2])
    
    with col1:
        st.markdown(f"**Main Path $F(x)$**")
        st.info(f"Input: {x_val:.2f}")
        w1 = st.slider("Weight Layer 1 ($W_1$)", min_value=-2.0, max_value=2.0, value=0.5, step=0.1)
        out1 = max(0, x_val * w1) # ReLU
        st.warning(f"After Layer 1 & ReLU: max(0, {x_val:.2f} * {w1:.2f}) = {out1:.2f}")
        
        w2 = st.slider("Weight Layer 2 ($W_2$)", min_value=-2.0, max_value=2.0, value=-0.2, step=0.1)
        f_x = out1 * w2
        st.success(f"Output $F(x)$: {out1:.2f} * {w2:.2f} = {f_x:.2f}")
        
    with col2:
        st.markdown("<h1 style='text-align: center; margin-top: 150px;'>+</h1>", unsafe_allow_html=True)
        
    with col3:
        st.markdown(f"**Shortcut (Identity) Path**")
        st.info(f"Input: {x_val:.2f}")
        st.markdown("<div style='height: 180px; border-left: 2px dashed gray; margin-left: 50%;'></div>", unsafe_allow_html=True)
        st.success(f"Shortcut Output $x$: {x_val:.2f}")
        
    st.divider()
    
    h_x_pre_relu = f_x + x_val
    h_x_final = max(0, h_x_pre_relu)
    
    st.markdown(f"### Final Output $H(x)$")
    st.markdown(f"**$H(x) = ReLU(F(x) + x)$**")
    st.markdown(f"**$H(x) = ReLU({f_x:.2f} + {x_val:.2f})$ = {h_x_final:.2f}**")
    
    st.info("💡 **Key Takeaway:** Notice that even if the weights $W_1$ and $W_2$ become zero (meaning $F(x) = 0$), the signal **$x$ STILL flows through the network perfectly via the shortcut**. This solves the degradation problem by allowing gradients to flow freely backward through the identity connections during training!")
