import streamlit as st
from utils.rag_setup import setup_rag_pipeline, generate_answer

def render():
    st.header("💬 Chat with the ResNet Paper")
    st.markdown("""
        Have a question about the paper? 
        This is an interactive RAG (Retrieval-Augmented Generation) pipeline. 
        It searches the text of *Deep Residual Learning for Image Recognition* (He et al., 2015) 
        and uses a local LLM to synthesize the answer based on retrieved paragraphs.
    """)
    
    # Load model and DB
    retriever, llm_pipeline, error = setup_rag_pipeline()
    
    if error:
        st.error(error)
        return
        
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # React to user input
    if prompt := st.chat_input("Ask a question (e.g., 'Why didn't they use dropout?')"):
        # Display user message in chat message container
        st.chat_message("user").markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer, docs = generate_answer(prompt, retriever, llm_pipeline)
                
                # We can also show the source material as an expander
                st.markdown(answer)
                
                with st.expander("📚 View source excerpts from the paper"):
                    for i, doc in enumerate(docs):
                        st.markdown(f"**Excerpt {i+1} (Page {doc.metadata.get('page', 'Unknown')}):**")
                        st.caption(doc.page_content)
                
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": answer})
