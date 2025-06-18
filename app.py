import streamlit as st
import subprocess
import os

def convert_to_pdf(input_path):
    """Convert DOCX, XLSX, PPTX, etc. to PDF using unoconv."""
    output_path = os.path.splitext(input_path)[0] + ".pdf"
    
    try:
        subprocess.run(["unoconv", "-f", "pdf", input_path], check=True)
        print(f"✅ Converted: {input_path} → {output_path}")
        return output_path
    except subprocess.CalledProcessError as e:
        print(f"❌ Error converting {input_path}: {e}")
        return None


# Configure the page
st.set_page_config(page_title="Multi-RAG Interface", layout="wide")

st.title("Multi-RAG System Interface")
st.markdown(
    """
    This app lets you switch between two Retrieval-Augmented Generation (RAG) systems:
    
    - **Google Gemini RAG**  
    - **Ollama RAG**
    
    Use the sidebar to navigate between the systems.  
    Any common state (e.g. an uploaded document) can be stored in `st.session_state` to persist between pages.
    """
)

# (Optional) Initialize session state variables if you wish to share state across pages.
if "uploaded_file" not in st.session_state:
    st.session_state.uploaded_file = None
if "selected_dataset" not in st.session_state:
    st.session_state.selected_dataset = None

st.info("Upload your document and select a dataset in each RAG page as needed.")
