import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
from utils.model_utils import load_resnet_and_hooks, preprocess_image

def render():
    st.header("👀 Live Feature Maps")
    st.markdown("""
        Let's see what ResNet-18 actually "sees". Upload an image and we'll visualize the activations (feature maps) 
        from the very first convolutional layer and the final residual block.
    """)
    
    model, activations = load_resnet_and_hooks()
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file).convert('RGB')
            # Display original image
            st.image(image, caption='Uploaded Image', width="stretch")
            
            with st.spinner("Processing image through ResNet-18..."):
                input_batch = preprocess_image(image)
                # Forward pass - the hooks will populate the activations dict
                _ = model(input_batch)
                
            st.subheader("1️⃣ Early Layer (Conv1) Features")
            st.markdown("*Early layers act as edge, color, and basic texture detectors.*")
            fig_conv1 = plot_feature_maps(activations['conv1'], num_maps=16)
            st.pyplot(fig_conv1)
            
            st.subheader("4️⃣ Deep Layer (Layer 4) Features")
            st.markdown("*Deeper layers detect highly abstract semantic concepts and object parts.*")
            fig_layer4 = plot_feature_maps(activations['layer4'], num_maps=16)
            st.pyplot(fig_layer4)
            
        except Exception as e:
            st.error(f"Error processing image: {e}")

def plot_feature_maps(activation_tensor, num_maps=16):
    # activation_tensor shape is [1, Channels, Height, Width]
    maps = activation_tensor[0].cpu().numpy()
    
    # We'll plot a grid
    cols = 4
    rows = num_maps // cols
    fig, axes = plt.subplots(rows, cols, figsize=(10, 10))
    for i, ax in enumerate(axes.flat):
        if i < maps.shape[0]:
            m = maps[i]
            # Min-Max normalize the map for visualization
            m_min, m_max = m.min(), m.max()
            if m_max - m_min > 0:
                m = (m - m_min) / (m_max - m_min)
            ax.imshow(m, cmap='viridis')
        ax.axis('off')
        
    plt.tight_layout()
    return fig
