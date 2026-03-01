import streamlit as st

def render():
    st.header("🔍 Architecture Explorer: ResNet-18")
    st.markdown("""
        **ResNet-18** is built primarily from the residual blocks we just learned about. 
        It has 18 parameterized layers in total (17 convolutional layers and 1 fully-connected layer at the end).
        
        Click on the blocks below to explore the architecture layer by layer!
    """)
    
    st.markdown("### Structure Walkthrough")
    
    st.info("**Input Image** (3 channels, 224x224)")
    
    with st.expander("1️⃣ Conv1 & MaxPool (The entry point)"):
        st.markdown("""
        **Conv1:** 
        - Filters: 64, Kernel: $7\\times7$, Stride: 2
        - Output size: $112\\times112\\times64$
        
        **MaxPool:**
        - Window: $3\\times3$, Stride: 2
        - Output size: $56\\times56\\times64$
        
        *This aggressively downsamples the spatial resolution while extracting early features like edges and textures.*
        """)
        st.markdown("<div style='background-color: #2e3b4e; padding: 10px; border-radius: 5px; text-align: center;'>🟩 <b>Input 224x224</b> ➔ 🟦 <b>Conv 7x7, 64</b> ➔ 🟨 <b>MaxPool 3x3</b> ➔ 🟩 <b>Output 56x56</b></div>", unsafe_allow_html=True)
        
    with st.expander("2️⃣ Conv2_x (The first residual blocks)"):
        st.markdown("""
        Contains **2 Residual Blocks**.
        Each block is:
        - Conv $3\\times3$, 64 filters
        - Conv $3\\times3$, 64 filters
        
        Since spatial size is already $56\\times56$, the stride here is 1.
        
        **Output size:** $56\\times56\\times64$
        """)
        st.markdown("<div style='background-color: #3b2e4e; padding: 10px; border-radius: 5px; text-align: center;'>🟩 <b>Input 56x56</b> ➔ 🟪 [<b>Conv 3x3, 64</b> ➔ <b>Conv 3x3, 64</b>] x 2 ➔ 🟩 <b>Output 56x56</b></div>", unsafe_allow_html=True)
        
    with st.expander("3️⃣ Conv3_x (Downsampling begins)"):
        st.markdown("""
        Contains **2 Residual Blocks**.
        Each block is:
        - Conv $3\\times3$, 128 filters
        - Conv $3\\times3$, 128 filters
        
        The first block here uses a stride of 2 to halve the spatial resolution while doubling the number of filters.
        
        **Output size:** $28\\times28\\times128$
        """)
        st.markdown("<div style='background-color: #4e2e3b; padding: 10px; border-radius: 5px; text-align: center;'>🟩 <b>Input 56x56</b> ➔ 🟥 [<b>Conv 3x3, 128</b> ➔ <b>Conv 3x3, 128</b>] x 2 ➔ 🟩 <b>Output 28x28</b></div>", unsafe_allow_html=True)
        
    with st.expander("4️⃣ Conv4_x"):
        st.markdown("""
        Contains **2 Residual Blocks**.
        Each block is:
        - Conv $3\\times3$, 256 filters
        - Conv $3\\times3$, 256 filters
        
        Again, the first block uses a stride of 2.
        
        **Output size:** $14\\times14\\times256$
        """)
        st.markdown("<div style='background-color: #4e442e; padding: 10px; border-radius: 5px; text-align: center;'>🟩 <b>Input 28x28</b> ➔ 🟧 [<b>Conv 3x3, 256</b> ➔ <b>Conv 3x3, 256</b>] x 2 ➔ 🟩 <b>Output 14x14</b></div>", unsafe_allow_html=True)
        
    with st.expander("5️⃣ Conv5_x (Deepest features)"):
        st.markdown("""
        Contains **2 Residual Blocks**.
        Each block is:
        - Conv $3\\times3$, 512 filters
        - Conv $3\\times3$, 512 filters
        
        The first block uses a stride of 2.
        
        **Output size:** $7\\times7\\times512$
        """)
        st.markdown("<div style='background-color: #2e4e37; padding: 10px; border-radius: 5px; text-align: center;'>🟩 <b>Input 14x14</b> ➔ 🟩 [<b>Conv 3x3, 512</b> ➔ <b>Conv 3x3, 512</b>] x 2 ➔ 🟩 <b>Output 7x7</b></div>", unsafe_allow_html=True)
        
    with st.expander("🏁 Output Layers (Classification)"):
        st.markdown("""
        **Average Pooling:**
        - Pools the $7\\times7$ spatial dimensions down to $1\\times1$.
        - Flattens the tensor to a 512-dimensional vector.
        
        **Fully Connected (FC) Layer:**
        - Maps the 512-dimensional vector to the number of classes (e.g., 1000 for ImageNet).
        - Followed by a Softmax to produce probabilities.
        """)
        st.markdown("<div style='background-color: #2e4e4d; padding: 10px; border-radius: 5px; text-align: center;'>🟩 <b>Input 7x7x512</b> ➔ 🧊 <b>AvgPool</b> ➔ 🧱 <b>FC 1000</b> ➔ 🎯 <b>Softmax</b></div>", unsafe_allow_html=True)
        
    st.success("🎉 By using residual connections throughout `Conv2_x` to `Conv5_x`, ResNet-18 allows gradients to bypass these individual layers if they aren't needed, completely solving the degradation problem and allowing clean training of these 18 layers!")
