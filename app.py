import streamlit as st
import plotly.express as px
import os
from rag_backend import RAGBackend  # Assumes RAGBackend is in rag_backend.py
from dotenv import load_dotenv

load_dotenv(dotenv_path=".env")

# Initialize RAGBackend in session state
if 'rag_backend' not in st.session_state:
    # Replace with your Groq API key or use os.getenv("GROQ_API_KEY") for security
    st.session_state.rag_backend = RAGBackend(groq_api_key=os.getenv("GROQ_API_KEY"))

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Choose a page", ["Chat", "Upload", "Statistics"])

# Upload Page
if page == "Upload":
    st.header("Upload Documents or URLs")
    
    # File upload section
    uploaded_files = st.file_uploader(
        "Upload .txt or .pdf files",
        accept_multiple_files=True,
        type=['txt', 'pdf']
    )
    if uploaded_files and st.button("Process Uploaded Files"):
        for uploaded_file in uploaded_files:
            # Save file temporarily
            temp_path = uploaded_file.name
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            # Process file with RAGBackend
            success = st.session_state.rag_backend.process_file(temp_path, temp_path)
            file_size = f"{uploaded_file.size / 1024:.1f}KB"
            if success:
                st.success(f"Successfully processed {uploaded_file.name} ({file_size})")
            else:
                st.error(f"Failed to process {uploaded_file.name} ({file_size})")
            # Clean up temporary file
            os.remove(temp_path)
    
    # URL input section
    url = st.text_input("Enter a web URL")
    if url and st.button("Process URL"):
        success = st.session_state.rag_backend.process_url(url)
        if success:
            st.success(f"Successfully processed URL: {url}")
        else:
            st.error(f"Failed to process URL: {url}")
    
    # Display document stats
    st.subheader("Uploaded Documents Stats")
    doc_count = st.session_state.rag_backend.get_document_count()
    st.write(f"Total documents: {doc_count}")
    if doc_count > 0:
        sources = list(set([doc.metadata['source'] for doc in st.session_state.rag_backend.documents]))
        st.write("Document sources:")
        for source in sources:
            st.write(f"- {source}")
    
    # Option to clear knowledge base
    if st.button("Clear Knowledge Base"):
        st.session_state.rag_backend.clear_knowledge_base()
        st.session_state.chat_history = []
        if 'retrieval_counts' in st.session_state:
            del st.session_state.retrieval_counts
        st.success("Knowledge base cleared.")

# Chat Page
elif page == "Chat":
    doc_count = st.session_state.rag_backend.get_document_count()
    if doc_count == 0:
        st.write("No documents uploaded. Please upload documents to start chatting.")
        st.write("Select 'Upload' from the sidebar to add documents.")
    else:
        st.header("Chat with Your Documents")
        
        # Initialize chat history
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        
        # Create containers for layout
        history_container = st.container()
        input_container = st.container()
        
        # Process input in input container (bottom of UI)
        with input_container:
            user_input = st.chat_input("Ask a question about your documents")
            if user_input:
                # Add user message to history
                st.session_state.chat_history.append({"role": "user", "content": user_input})
                
                # Query RAGBackend
                try:
                    answer, docs = st.session_state.rag_backend.query(user_input, return_sources=True)
                    sources = list(set([doc.metadata['source'] for doc in docs]))
                    # Add assistant response to history
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    })
                    
                    # Update retrieval counts
                    if 'retrieval_counts' not in st.session_state:
                        st.session_state.retrieval_counts = {}
                    for doc in docs:
                        source = doc.metadata['source']
                        st.session_state.retrieval_counts[source] = (
                            st.session_state.retrieval_counts.get(source, 0) + 1
                        )
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        
        # Display chat history in history container (top of UI)
        with history_container:
            for message in st.session_state.chat_history:
                with st.chat_message(message["role"]):
                    st.write(message["content"])
                    if "sources" in message:
                        with st.expander("Sources"):
                            for source in message["sources"]:
                                st.write(f"- {source}")

# Statistics Page
elif page == "Statistics":
    st.header("Statistics")
    
    # Uploaded documents stats
    st.subheader("Uploaded Documents")
    doc_count = st.session_state.rag_backend.get_document_count()
    st.write(f"Total documents: {doc_count}")
    if doc_count > 0:
        sources = list(set([doc.metadata['source'] for doc in st.session_state.rag_backend.documents]))
        st.write("Document sources:")
        for source in sources:
            st.write(f"- {source}")
    
    # Retrieval statistics
    st.subheader("Retrieval Statistics")
    if 'retrieval_counts' in st.session_state and st.session_state.retrieval_counts:
        sources = list(st.session_state.retrieval_counts.keys())
        counts = list(st.session_state.retrieval_counts.values())
        
        # Format source labels to be more readable
        formatted_sources = []
        for source in sources:
            if source.startswith("URL_"):
                # Extract domain from URL
                domain = source.split("//")[1].split("/")[0] if "//" in source else source
                formatted_sources.append(f"URL: {domain}")
            elif source.endswith(".pdf") or source.endswith(".txt"):
                # Just use the filename
                formatted_sources.append(os.path.basename(source))
            else:
                # Use as is but truncate if too long
                formatted_sources.append(source[:20] + "..." if len(source) > 20 else source)
        
        fig = px.bar(
            x=formatted_sources,
            y=counts,
            labels={'x': 'Document Source', 'y': 'Retrieval Count'},
            template='plotly_white'
        )
        
        # Improve layout for better readability
        fig.update_layout(
            xaxis=dict(
                tickangle=0,  # Horizontal text
                tickmode='array',
                tickvals=list(range(len(formatted_sources))),
                ticktext=formatted_sources
            ),
            margin=dict(b=100),  # Add bottom margin for labels
            autosize=True,
            height=500  # Increase height to accommodate labels
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.write("No retrieval statistics available yet. Ask questions in the Chat page to generate data.")