# ResearchGPT

A comprehensive Retrieval-Augmented Generation (RAG) system designed specifically for research papers. This system can process PDF research papers from Azure Blob Storage or local files, extract and index their content, and provide intelligent question-answering capabilities based on the research content.

## 🚀 Features

### Core Functionality
- **PDF Processing**: Extract text from research papers using multiple PDF parsing methods
- **Azure Integration**: Seamlessly connect to Azure Blob Storage for paper storage
- **Local Storage**: Store and organize PDFs locally with automatic backup and organization
- **Vector Indexing**: Create semantic embeddings and store in FAISS
- **Intelligent Q&A**: Ask questions about research papers and get contextual answers
- **Research Summarization**: Generate comprehensive summaries of research findings
- **Web Interface**: User-friendly Streamlit web application (now uses the improved system)
- **CLI Interface**: Command-line tools for automation

### Advanced Features
- **Multi-method PDF parsing**: Uses both PyPDF2 and pdfplumber for better text extraction
- **Smart text chunking**: Intelligent document splitting with overlap
- **Similarity search**: Find relevant research content using semantic similarity
- **Source attribution**: Track which papers and sections were used for answers
- **Batch processing**: Handle large collections of research papers efficiently
- **Local file organization**: Automatic organization by date and backup management
- **Hybrid storage**: Use both Azure and local storage simultaneously

## 📋 Prerequisites

- Python 3.8 or higher
- Azure Blob Storage account (optional, for cloud storage)
- OpenAI API key (for LLM capabilities)
- Sufficient disk space for vector database and PDF storage

## 🛠️ Installation

### Standard Installation (Linux/macOS)

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd research_gpt
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**:
   ```bash
   cp env_example.txt .env
   # Edit .env with your actual credentials
   ```

### Windows Installation (WSL)

#### Prerequisites
1. **Install WSL 2**:
   ```powershell
   # Open PowerShell as Administrator and run:
   wsl --install
   ```
   This will install Ubuntu by default. Restart your computer when prompted.

2. **Update WSL**:
   ```bash
   # In WSL terminal:
   sudo apt update && sudo apt upgrade -y
   ```

3. **Install Python and pip**:
   ```bash
   sudo apt install python3 python3-pip python3-venv -y
   ```

#### Installation Steps

1. **Clone the repository in WSL**:
   ```bash
   # Navigate to your preferred directory
   cd ~
   git clone <repository-url>
   cd research_gpt
   ```

2. **Create a virtual environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**:
   ```bash
   cp env_example.txt .env
   # Edit .env with your actual credentials
   nano .env  # or use your preferred editor
   ```

#### Running the System on Windows

1. **Start the web interface**:
   ```bash
   # In WSL with activated virtual environment:
   streamlit run web_interface.py
   ```

2. **Access from Windows**:
   - The Streamlit app will be available at `http://localhost:8501`
   - You can access it from your Windows browser
   - WSL automatically forwards the port to Windows

3. **File Management**:
   - Place your PDF files in the `pdfs/` directory within the WSL project
   - Access WSL files from Windows Explorer: `\\wsl$\Ubuntu\home\yourusername\research_gpt\`
   - Or access Windows files from WSL: `/mnt/c/Users/yourusername/`

#### Troubleshooting WSL

**If you encounter permission issues**:
```bash
# Fix ownership of the project directory
   sudo chown -R $USER:$USER ~/research_gpt
```

**If you need to install additional packages**:
```bash
sudo apt install build-essential python3-dev -y
```

**If Streamlit doesn't start**:
```bash
# Check if the port is available
netstat -tulpn | grep 8501
# Kill any existing process on port 8501
sudo kill -9 $(lsof -t -i:8501)
```

**For better performance**:
- Store your project in the WSL filesystem (not Windows filesystem)
- Use WSL 2 for better performance
- Allocate more memory to WSL if needed

#### Windows Integration Tips

1. **VS Code Integration**:
   - Install the "Remote - WSL" extension in VS Code
   - Open the project folder in WSL for seamless development

2. **File Sharing**:
   - Use `/mnt/c/` to access Windows files from WSL
   - Use `\\wsl$\` to access WSL files from Windows

3. **Port Forwarding**:
   - WSL automatically forwards ports to Windows
   - Access web interfaces at `localhost:port` from Windows browser

4. **Environment Variables**:
   - Set up your `.env` file in WSL
   - Use WSL-compatible paths for file references

## ⚙️ Configuration

Create a `.env` file with the following variables:

```env
# Azure Blob Storage Configuration (Optional)
AZURE_STORAGE_CONNECTION_STRING=your_azure_storage_connection_string_here
AZURE_CONTAINER_NAME=research-papers

# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4
OPENAI_EMBEDDING_MODEL=text-embedding-ada-002

# Vector Database Configuration
VECTOR_DB_PATH=./vector_db
CHUNK_SIZE=1000
CHUNK_OVERLAP=200

# RAG Configuration
TOP_K_RESULTS=5
SIMILARITY_THRESHOLD=0.7

# Local Storage Configuration
LOCAL_PDF_DIR=./pdfs
PROCESSED_DATA_DIR=./processed_data
ENABLE_LOCAL_STORAGE=true
LOCAL_STORAGE_BACKUP=true
```

### Required Services

1. **OpenAI API**: Get an API key from [OpenAI Platform](https://platform.openai.com/)
2. **Azure Blob Storage**: Optional - Create a storage account and container for your research papers

## 🚀 Usage

### Web Interface

Start the Streamlit web application (now uses the improved system):

```bash
streamlit run web_interface.py
```

Navigate to `http://localhost:8501` to access the web interface.

### Command Line Interface

#### Local Storage Management
```bash
# Add PDF to local storage
python cli.py add-pdf /path/to/paper.pdf

# List PDFs in local storage
python cli.py list-local-pdfs

# Process PDFs from local storage
python cli.py process-local-storage

# Delete PDF from local storage
python cli.py delete-local-pdf paper.pdf

# Clean up old backups
python cli.py cleanup-backups --days 30
```

#### Azure Storage Operations
```bash
# Process PDFs from Azure
python cli.py process-azure

# Process local PDF files
python cli.py process-local /path/to/pdfs
```

#### Question Answering
```bash
# Ask questions about research
python cli.py ask "What are the main findings about machine learning in healthcare?"

# Generate research summary
python cli.py summary "artificial intelligence"

# View system statistics
python cli.py stats
```

### Programmatic Usage

```python
from rag_system_improved import ImprovedResearchRAGSystem

# Initialize the system
rag = ImprovedResearchRAGSystem()

# Add PDF to local storage
result = rag.add_pdf_to_local_storage('/path/to/paper.pdf', organize=True)

# Process PDFs from local storage
result = rag.process_local_storage_pdfs()

# Process PDFs from Azure
result = rag.process_azure_pdfs()

# Ask a question
answer = rag.ask_question("What are the key findings about climate change?")

# Generate summary
summary = rag.summarize_research_findings("machine learning")

# Get local storage statistics
pdfs = rag.get_local_storage_pdfs()
```

## 📁 Project Structure

```
research_gpt/
├── config.py              # Configuration management
├── pdf_processor.py       # PDF text extraction
├── azure_storage.py       # Azure Blob Storage operations
├── local_storage.py       # Local storage management
├── vector_store_basic_semantic.py # Vector database and embeddings (used by improved system)
├── rag_system_improved.py # Improved RAG system orchestration
├── web_interface.py       # Streamlit web application (uses improved system)
├── cli.py                 # Command-line interface
├── requirements.txt       # Python dependencies
├── env_example.txt        # Environment variables template
└── README.md              # This file
```

## 🔧 System Components

### 1. PDF Processor (`pdf_processor.py`)
- Extracts text from PDF files using multiple methods
- Handles different PDF formats and structures
- Cleans and preprocesses extracted text
- Validates PDF files before processing

### 2. Azure Storage Handler (`azure_storage.py`)
- Manages PDF files in Azure Blob Storage
- Upload, download, and list PDF files
- Handle container creation and management
- Stream processing for large files

### 3. Local Storage Handler (`local_storage.py`)
- Manages PDF files in local directory structure
- Automatic file organization by date
- Backup management and cleanup
- File validation and size limits

### 4. Vector Store (`vector_store_basic_semantic.py`)
- Creates semantic embeddings using OpenAI or sentence transformers
- Stores vectors in FAISS for similarity search
- Implements intelligent text chunking
- Provides similarity search functionality

### 5. Improved RAG System (`rag_system_improved.py`)
- Orchestrates all components
- Implements question-answering using LangChain
- Manages the complete RAG pipeline
- Provides research summarization capabilities

## 💾 Local Storage Features

### File Organization
- **Automatic organization**: Files are organized by year/month based on modification date
- **Backup system**: Automatic backup creation before any file operations
- **File validation**: Checks file size, format, and integrity
- **Duplicate handling**: Prevents file conflicts with timestamp suffixes

### Directory Structure
```
pdfs/
├── backup/           # Automatic backups
├── organized/        # Organized by date
│   ├── 2024/
│   │   ├── 01/      # January 2024
│   │   └── 02/      # February 2024
│   └── 2023/
│       └── 12/      # December 2023
└── *.pdf            # Direct PDF files
```

### Management Commands
- **Add PDFs**: Upload and organize files automatically
- **List PDFs**: View all stored files with metadata
- **Delete PDFs**: Remove files with backup creation
- **Cleanup**: Remove old backup files
- **Process**: Extract and index content from stored PDFs

## 🎯 Use Cases

### Academic Research
- Process large collections of research papers
- Find relevant papers for specific topics
- Generate literature reviews
- Extract key findings from multiple studies

### Industry Research
- Analyze competitor research and publications
- Track technology trends
- Generate market research summaries
- Answer technical questions from documentation

### Personal Research
- Organize personal research paper collections
- Create knowledge bases from PDFs
- Generate summaries of research areas
- Build personal research assistants

## 🔍 Example Queries

Here are some example questions you can ask the system:

- "What are the main findings about machine learning in healthcare?"
- "Summarize the research on climate change mitigation strategies"
- "What methods are used for natural language processing?"
- "What are the latest developments in quantum computing?"
- "Compare different approaches to computer vision"

## 📊 Performance Considerations

### Processing Speed
- PDF processing: ~1-5 seconds per page depending on complexity
- Vector indexing: ~100-500 documents per minute
- Question answering: ~2-10 seconds depending on query complexity

### Storage Requirements
- Vector database: ~1-5 MB per research paper
- PDF storage: Varies by paper size (typically 1-10 MB per paper)
- Local storage: Plan for 2-10x the original PDF size (including backups)
- Total storage: Plan for 2-10x the original PDF size

### Memory Usage
- Embedding model: ~500MB RAM
- Vector database: ~100-500MB RAM depending on collection size
- PDF processing: ~50-200MB RAM per concurrent process

## 🛡️ Security Considerations

- Store API keys securely in environment variables
- Use Azure managed identities when possible
- Implement proper access controls for Azure storage
- Consider data encryption for sensitive research papers
- Monitor API usage and costs
- Local backups provide additional data protection

## 🔧 Troubleshooting

### Common Issues

1. **PDF Processing Errors**
   - Ensure PDFs are not password-protected
   - Check file size limits (default: 50MB)
   - Verify PDF format compatibility

2. **Azure Connection Issues**
   - Verify connection string format
   - Check Azure storage account permissions
   - Ensure container exists or can be created

3. **OpenAI API Errors**
   - Verify API key is valid
   - Check API usage limits and billing
   - Ensure model names are correct

4. **Local Storage Issues**
   - Check disk space availability
   - Verify directory permissions
   - Ensure backup directory is writable

5. **Memory Issues**
   - Reduce batch sizes for large collections
   - Process PDFs in smaller batches
   - Monitor system memory usage

### Debug Mode

Enable detailed logging by setting the log level:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [LangChain](https://langchain.com/) for the RAG framework
- [ChromaDB](https://www.trychroma.com/) for vector storage
- [Streamlit](https://streamlit.io/) for the web interface
- [Azure Storage](https://azure.microsoft.com/services/storage/) for cloud storage
- [OpenAI](https://openai.com/) for language models