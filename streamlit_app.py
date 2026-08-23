import os
import sys
import streamlit as st

# Set Streamlit Page Configuration (Zero Emojis)
st.set_page_config(
    page_title="Atlas — PDF Retrieval & Context Engine",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Ensure backend modules are discoverable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "backend")))

from backend.src.atlas_rag.pipeline import AtlasRAGPipeline

@st.cache_resource
def load_rag_pipeline():
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(data_dir, exist_ok=True)
    return AtlasRAGPipeline(data_dir=data_dir)

pipeline = load_rag_pipeline()

# Title Header
st.title("Atlas — PDF Retrieval & Context Engine")
st.caption("Domain-specific PDF contextualization and grounded answer generation engine.")

st.markdown("---")

# Sidebar: PDF Library & Ingestion
st.sidebar.header("PDF Context Library")

uploaded_file = st.sidebar.file_uploader("Upload New PDF Document", type=["pdf"])

if uploaded_file is not None:
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    save_path = os.path.join(data_dir, uploaded_file.name)
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    with st.sidebar.spinner(f"Ingesting {uploaded_file.name}..."):
        pipeline.ingest_pdf(save_path, uploaded_file.name)
    st.sidebar.success(f"Successfully ingested `{uploaded_file.name}`!")

# Display PDF Library with Checkboxes
all_docs = pipeline.get_all_documents()
if all_docs:
    st.sidebar.subheader("Active Context Selection")
    for doc in all_docs:
        col1, col2 = st.sidebar.columns([0.75, 0.25])
        is_checked = col1.checkbox(
            doc["filename"],
            value=doc.get("is_active", True),
            key=f"cb_{doc['filename']}"
        )
        if is_checked != doc.get("is_active", True):
            pipeline.toggle_document_active(doc["filename"], is_checked)
            st.rerun()

        if col2.button("Delete", key=f"del_{doc['filename']}", help="Delete from disk"):
            pipeline.delete_document_context(doc["filename"])
            st.rerun()
else:
    st.sidebar.info("No PDFs in library. Upload a PDF to get started.")

# Main Area: Question & Answer
col_left, col_right = st.columns([1, 1])

with col_left:
    st.subheader("Ask a Question")
    page_scope = st.text_input("PDF Page Scope (Optional)", placeholder="e.g. 3, or 1-5 (leave blank for all)")
    question = st.text_area("Your Question", placeholder="Type your question about the PDF context...", height=120)
    ask_btn = st.button("Ask Question", type="primary", use_container_width=True)

with col_right:
    st.subheader("Answer & Retrieved Context")
    
    if ask_btn:
        if not question.strip():
            st.warning("Please type a question before submitting.")
        else:
            with st.spinner("Searching active PDF context..."):
                res = pipeline.query(question, page_selection=page_scope or None)
                
                st.markdown(res.get("output_text", "No response generated."))
                
                retrieved_chunks = res.get("retrieved_chunks", [])
                if retrieved_chunks:
                    st.markdown("### Retrieved Context Passages")
                    for chunk in retrieved_chunks:
                        with st.expander(f"Page {chunk.get('page_number', 1)} | Similarity Match: {chunk.get('similarity_score', 0.9)*100:.1f}%"):
                            st.write(chunk.get("content", ""))
