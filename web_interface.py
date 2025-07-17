import streamlit as st
import os
import tempfile
import json
from typing import List, Dict

# Import our RAG system
from rag_system_improved import ImprovedResearchRAGSystem as ResearchRAGSystem
from config import Config

# Page configuration
st.set_page_config(
    page_title="ResearchGPT",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

def initialize_rag_system():
    """Initialize the RAG system"""
    try:
        return ResearchRAGSystem()
    except Exception as e:
        st.error(f"Failed to initialize RAG system: {str(e)}")
        return None

def main():
    st.title("📚 ResearchGPT")
    st.markdown("A comprehensive system for processing research papers and answering questions based on their content.")
    
    # Initialize RAG system
    rag_system = initialize_rag_system()
    if not rag_system:
        st.error("Please check your configuration and try again.")
        return
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["🏠 Home", "📄 Process PDFs", "💾 Local Storage", "❓ Ask Questions", "📊 System Stats", "📝 Research Summary"]
    )
    
    if page == "🏠 Home":
        show_home_page(rag_system)
    elif page == "📄 Process PDFs":
        show_process_pdfs_page(rag_system)
    elif page == "💾 Local Storage":
        show_local_storage_page(rag_system)
    elif page == "❓ Ask Questions":
        show_ask_questions_page(rag_system)
    elif page == "📊 System Stats":
        show_system_stats_page(rag_system)
    elif page == "📝 Research Summary":
        show_research_summary_page(rag_system)

def show_home_page(rag_system):
    """Display the home page"""
    st.header("Welcome to ResearchGPT")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 What is this system?")
        st.markdown("""
        This Research RAG (Retrieval-Augmented Generation) system allows you to:
        
        - **Process research papers** from Azure Blob Storage or local files
        - **Store and organize PDFs** in local storage with automatic backup
        - **Extract and index** text content from PDF documents
        - **Ask questions** about the research papers
        - **Generate summaries** of research findings
        - **Search and retrieve** relevant information from your paper collection
        """)
    
    with col2:
        st.subheader("🚀 Quick Start")
        st.markdown("""
        1. **Add PDFs**: Upload papers to local storage or connect to Azure
        2. **Process PDFs**: Extract and index the content
        3. **Ask Questions**: Query the system about your research
        4. **View Stats**: Monitor your system's performance
        5. **Get Summaries**: Generate research summaries
        """)
    
    # System status
    st.subheader("📊 System Status")
    try:
        stats = rag_system.get_system_stats()
        if 'error' not in stats:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Documents", stats['vector_store'].get('total_documents', 0))
            
            with col2:
                st.metric("Unique Files", stats['vector_store'].get('unique_files', 0))
            
            with col3:
                azure_pdfs = stats['azure_storage'].get('total_pdfs', 0) if stats['azure_storage'].get('enabled') is not False else 0
                st.metric("Azure PDFs", azure_pdfs)
            
            with col4:
                local_files = stats['local_storage'].get('total_files', 0) if stats['local_storage'].get('enabled') is not False else 0
                st.metric("Local PDFs", local_files)
        else:
            st.warning("Unable to retrieve system statistics")
    except Exception as e:
        st.error(f"Error getting system stats: {str(e)}")

def show_process_pdfs_page(rag_system):
    """Display the PDF processing page"""
    st.header("📄 Process Research Papers")
    
    tab1, tab2, tab3 = st.tabs(["Azure Blob Storage", "Local Files", "Local Storage"])
    
    with tab1:
        st.subheader("Process PDFs from Azure Blob Storage")
        st.markdown("Connect to your Azure Blob Storage container to process research papers.")
        
        if st.button("🔄 Process Azure PDFs", type="primary"):
            with st.spinner("Processing PDFs from Azure..."):
                result = rag_system.process_azure_pdfs()
                
                if result['success']:
                    st.success(f"✅ Successfully processed {result['pdfs_processed']} PDFs!")
                    st.info(f"Added {result['chunks_added']} text chunks to vector store")
                else:
                    st.error(f"❌ Processing failed: {result.get('error', 'Unknown error')}")
    
    with tab2:
        st.subheader("Upload Local PDF Files")
        st.markdown("Upload PDF files from your local machine for processing.")
        
        uploaded_files = st.file_uploader(
            "Choose PDF files",
            type=['pdf'],
            accept_multiple_files=True,
            key="local_upload"
        )
        
        if uploaded_files and st.button("🔄 Process Uploaded PDFs", type="primary"):
            with st.spinner("Processing uploaded PDFs..."):
                # Save uploaded files to temporary directory
                temp_dir = tempfile.mkdtemp()
                saved_files = []
                
                for uploaded_file in uploaded_files:
                    file_path = os.path.join(temp_dir, uploaded_file.name)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    saved_files.append(file_path)
                
                # Process the files
                result = rag_system.process_local_pdfs(temp_dir)
                
                # Clean up
                for file_path in saved_files:
                    if os.path.exists(file_path):
                        os.remove(file_path)
                os.rmdir(temp_dir)
                
                if result['success']:
                    st.success(f"✅ Successfully processed {result['pdfs_processed']} PDFs!")
                    st.info(f"Added {result['chunks_added']} text chunks to vector store")
                else:
                    st.error(f"❌ Processing failed: {result.get('error', 'Unknown error')}")
    
    with tab3:
        st.subheader("Process PDFs from Local Storage")
        st.markdown("Process PDFs that are already stored in your local storage.")
        
        if st.button("🔄 Process Local Storage PDFs", type="primary"):
            with st.spinner("Processing PDFs from local storage..."):
                result = rag_system.process_local_storage_pdfs()
                
                if result['success']:
                    st.success(f"✅ Successfully processed {result['pdfs_processed']} PDFs!")
                    st.info(f"Added {result['chunks_added']} text chunks to vector store")
                else:
                    st.error(f"❌ Processing failed: {result.get('error', 'Unknown error')}")

def show_local_storage_page(rag_system):
    """Display the local storage management page"""
    st.header("💾 Local Storage Management")
    
    tab1, tab2, tab3 = st.tabs(["📤 Add PDFs", "📋 View PDFs", "🗑️ Manage PDFs"])
    
    with tab1:
        st.subheader("Add PDFs to Local Storage")
        st.markdown("Upload PDF files to your local storage for organization and processing.")
        
        uploaded_files = st.file_uploader(
            "Choose PDF files to add to local storage",
            type=['pdf'],
            accept_multiple_files=True,
            key="local_storage_upload"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            organize_files = st.checkbox("Organize files by date", value=True)
        with col2:
            if uploaded_files and st.button("📤 Add to Local Storage", type="primary"):
                with st.spinner("Adding PDFs to local storage..."):
                    results = []
                    for uploaded_file in uploaded_files:
                        # Save to temporary file first
                        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
                            tmp_file.write(uploaded_file.getbuffer())
                            tmp_path = tmp_file.name
                        
                        # Add to local storage
                        result = rag_system.add_pdf_to_local_storage(tmp_path, organize_files)
                        results.append(result)
                        
                        # Clean up temp file
                        os.unlink(tmp_path)
                    
                    # Show results
                    successful = [r for r in results if r['success']]
                    failed = [r for r in results if not r['success']]
                    
                    if successful:
                        st.success(f"✅ Successfully added {len(successful)} PDFs to local storage")
                        for result in successful:
                            st.info(f"📄 {result['file_name']} → {result['local_path']}")
                    
                    if failed:
                        st.error(f"❌ Failed to add {len(failed)} PDFs")
                        for result in failed:
                            st.error(f"📄 {result.get('original_path', 'Unknown')}: {result.get('error', 'Unknown error')}")
    
    with tab2:
        st.subheader("View PDFs in Local Storage")
        st.markdown("Browse and view PDFs stored in your local storage.")
        
        if st.button("🔄 Refresh PDF List", type="primary"):
            with st.spinner("Loading PDF list..."):
                pdfs = rag_system.get_local_storage_pdfs()
                
                if not pdfs:
                    st.info("📭 No PDFs found in local storage")
                else:
                    st.success(f"📚 Found {len(pdfs)} PDFs in local storage")
                    
                    # Create a dataframe for better display
                    import pandas as pd
                    pdf_data = []
                    for pdf in pdfs:
                        if 'error' not in pdf:
                            pdf_data.append({
                                'Name': pdf['name'],
                                'Size (MB)': f"{pdf['size'] / (1024*1024):.2f}",
                                'Modified': pdf['modified'].strftime('%Y-%m-%d %H:%M'),
                                'Path': pdf['relative_path']
                            })
                    
                    if pdf_data:
                        df = pd.DataFrame(pdf_data)
                        st.dataframe(df, use_container_width=True)
                    else:
                        st.warning("No valid PDFs found")
    
    with tab3:
        st.subheader("Manage PDFs in Local Storage")
        st.markdown("Delete PDFs and manage local storage.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🗑️ Delete PDF")
            pdfs = rag_system.get_local_storage_pdfs()
            if pdfs:
                pdf_names = [pdf['name'] for pdf in pdfs if 'error' not in pdf]
                selected_pdf = st.selectbox("Select PDF to delete:", pdf_names)
                
                if st.button("🗑️ Delete Selected PDF", type="primary"):
                    with st.spinner("Deleting PDF..."):
                        result = rag_system.delete_local_pdf(selected_pdf)
                        
                        if result['success']:
                            st.success(f"✅ Successfully deleted {selected_pdf}")
                            if result.get('backup_created'):
                                st.info("📦 Backup created before deletion")
                        else:
                            st.error(f"❌ Failed to delete: {result.get('error', 'Unknown error')}")
            else:
                st.info("No PDFs available for deletion")
        
        with col2:
            st.subheader("🧹 Cleanup Backups")
            days_to_keep = st.slider("Keep backups for (days):", 1, 90, 30)
            
            if st.button("🧹 Cleanup Old Backups", type="primary"):
                with st.spinner("Cleaning up old backups..."):
                    result = rag_system.cleanup_local_backups(days_to_keep)
                    
                    if result['success']:
                        st.success(f"✅ Cleaned up {result['files_deleted']} old backup files")
                    else:
                        st.error(f"❌ Cleanup failed: {result.get('error', 'Unknown error')}")

def show_ask_questions_page(rag_system):
    """Display the question asking page"""
    st.header("❓ Ask Questions About Research")
    
    # Question input
    question = st.text_area(
        "Enter your question about the research papers:",
        placeholder="e.g., What are the main findings about machine learning in healthcare?",
        height=100
    )
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        include_sources = st.checkbox("Include source information", value=True)
    
    with col2:
        if st.button("🔍 Ask Question", type="primary"):
            if question.strip():
                with st.spinner("Searching and generating answer..."):
                    result = rag_system.ask_question(question, include_sources)
                    
                    if result['success']:
                        st.success("✅ Answer generated!")
                        
                        # Display answer
                        st.subheader("Answer")
                        st.write(result['answer'])
                        
                        # Display sources if requested
                        if include_sources and result['sources']:
                            st.subheader("📚 Sources")
                            for i, source in enumerate(result['sources']):
                                with st.expander(f"Source {i+1}: {source['file_name']}"):
                                    st.write(f"**File:** {source['file_name']}")
                                    st.write(f"**Chunk:** {source['chunk_index']}")
                                    st.write(f"**Similarity Score:** {source['similarity_score']:.3f}")
                        
                        # Display metadata
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Sources Used", result['num_sources'])
                        with col2:
                            st.metric("Context Length", f"{result['context_length']:,} chars")
                        with col3:
                            avg_score = sum(result['similarity_scores']) / len(result['similarity_scores']) if result['similarity_scores'] else 0
                            st.metric("Avg Similarity", f"{avg_score:.3f}")
                    else:
                        st.error(f"❌ Failed to generate answer: {result.get('answer', 'Unknown error')}")
            else:
                st.warning("Please enter a question.")

def show_system_stats_page(rag_system):
    """Display the system statistics page"""
    st.header("📊 System Statistics")
    
    if st.button("🔄 Refresh Statistics", type="primary"):
        with st.spinner("Loading system statistics..."):
            stats = rag_system.get_system_stats()
            
            if 'error' not in stats:
                # Vector Store Stats
                st.subheader("🗄️ Vector Store")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Total Documents", stats['vector_store'].get('total_documents', 0))
                
                with col2:
                    st.metric("Unique Files", stats['vector_store'].get('unique_files', 0))
                
                with col3:
                    st.metric("Estimated Chunks", stats['vector_store'].get('estimated_total_chunks', 0))
                
                # Azure Storage Stats
                st.subheader("☁️ Azure Blob Storage")
                azure_stats = stats['azure_storage']
                if azure_stats.get('enabled') is False:
                    st.info("Azure storage is disabled")
                elif 'error' in azure_stats:
                    st.error(f"Azure storage error: {azure_stats['error']}")
                else:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("Total PDFs", azure_stats.get('total_pdfs', 0))
                    
                    with col2:
                        pdf_names = azure_stats.get('pdf_names', [])
                        if pdf_names:
                            st.write("**PDF Files:**")
                            for name in pdf_names[:10]:  # Show first 10
                                st.write(f"- {name}")
                            if len(pdf_names) > 10:
                                st.write(f"... and {len(pdf_names) - 10} more")
                
                # Local Storage Stats
                st.subheader("💾 Local Storage")
                local_stats = stats['local_storage']
                if local_stats.get('enabled') is False:
                    st.info("Local storage is disabled")
                elif 'error' in local_stats:
                    st.error(f"Local storage error: {local_stats['error']}")
                else:
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Total Files", local_stats.get('total_files', 0))
                    
                    with col2:
                        st.metric("Main Directory", local_stats.get('main_directory_files', 0))
                    
                    with col3:
                        st.metric("Organized Files", local_stats.get('organized_files', 0))
                    
                    with col4:
                        st.metric("Backup Files", local_stats.get('backup_files', 0))
                    
                    st.info(f"**Storage Path:** {local_stats.get('storage_path', 'N/A')}")
                    st.info(f"**Total Size:** {local_stats.get('total_size_mb', 0):.2f} MB")
                
                # Configuration
                st.subheader("⚙️ Configuration")
                config = stats['configuration']
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Chunk Size", config.get('chunk_size', 0))
                
                with col2:
                    st.metric("Chunk Overlap", config.get('chunk_overlap', 0))
                
                with col3:
                    st.metric("Top K Results", config.get('top_k_results', 0))
                
                with col4:
                    st.metric("Similarity Threshold", config.get('similarity_threshold', 0))
            else:
                st.error(f"Failed to load statistics: {stats['error']}")

def show_research_summary_page(rag_system):
    """Display the research summary page"""
    st.header("📝 Research Summary")
    
    # Topic input
    topic = st.text_input(
        "Enter a specific topic (optional):",
        placeholder="e.g., machine learning, healthcare, climate change"
    )
    
    if st.button("📋 Generate Summary", type="primary"):
        with st.spinner("Generating research summary..."):
            result = rag_system.summarize_research_findings(topic)
            
            if result['success']:
                st.success("✅ Summary generated!")
                
                st.subheader("📝 Research Summary")
                st.write(result['summary'])
                
                if result['sources_used']:
                    st.subheader("📚 Sources Used")
                    for i, source in enumerate(result['sources_used']):
                        with st.expander(f"Source {i+1}: {source['file_name']}"):
                            st.write(f"**File:** {source['file_name']}")
                            st.write(f"**Similarity Score:** {source['similarity_score']:.3f}")
                
                st.info(f"Summary generated for topic: {result['topic']}")
            else:
                st.error(f"❌ Failed to generate summary: {result.get('error', 'Unknown error')}")

if __name__ == "__main__":
    main() 