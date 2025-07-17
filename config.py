import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Configuration class for the Research RAG system"""
    
    # Azure Blob Storage Configuration
    AZURE_STORAGE_CONNECTION_STRING = os.getenv('AZURE_STORAGE_CONNECTION_STRING')
    AZURE_CONTAINER_NAME = os.getenv('AZURE_CONTAINER_NAME', 'research-papers')
    
    # Azure Cognitive Search Configuration
    AZURE_SEARCH_ENDPOINT = os.getenv('AZURE_SEARCH_ENDPOINT')
    AZURE_SEARCH_KEY = os.getenv('AZURE_SEARCH_KEY')
    AZURE_SEARCH_INDEX_NAME = os.getenv('AZURE_SEARCH_INDEX_NAME', 'research-papers-index')
    
    # OpenAI Configuration
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    OPENAI_MODEL = os.getenv('OPENAI_MODEL', 'gpt-4')
    OPENAI_EMBEDDING_MODEL = os.getenv('OPENAI_EMBEDDING_MODEL', 'text-embedding-ada-002')
    
    # Vector Database Configuration
    VECTOR_DB_PATH = os.getenv('VECTOR_DB_PATH', './vector_db')
    CHUNK_SIZE = int(os.getenv('CHUNK_SIZE', '1000'))
    CHUNK_OVERLAP = int(os.getenv('CHUNK_OVERLAP', '200'))
    
    # Processing Configuration
    MAX_FILE_SIZE_MB = int(os.getenv('MAX_FILE_SIZE_MB', '50'))
    SUPPORTED_LANGUAGES = ['en']  # Add more languages as needed
    
    # RAG Configuration
    TOP_K_RESULTS = int(os.getenv('TOP_K_RESULTS', '5'))
    SIMILARITY_THRESHOLD = float(os.getenv('SIMILARITY_THRESHOLD', '0.7'))
    
    # Local Storage Configuration
    LOCAL_PDF_DIR = os.getenv('LOCAL_PDF_DIR', './pdfs')
    PROCESSED_DATA_DIR = os.getenv('PROCESSED_DATA_DIR', './processed_data')
    ENABLE_LOCAL_STORAGE = os.getenv('ENABLE_LOCAL_STORAGE', 'true').lower() == 'true'
    LOCAL_STORAGE_BACKUP = os.getenv('LOCAL_STORAGE_BACKUP', 'true').lower() == 'true'
    
    @classmethod
    def validate_config(cls):
        """Validate that all required configuration is present"""
        required_vars = [
            'OPENAI_API_KEY'
        ]
        
        # Azure storage is optional if local storage is enabled
        if not cls.ENABLE_LOCAL_STORAGE and not cls.AZURE_STORAGE_CONNECTION_STRING:
            required_vars.append('AZURE_STORAGE_CONNECTION_STRING')
        
        missing_vars = []
        for var in required_vars:
            if not getattr(cls, var):
                missing_vars.append(var)
        
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
        
        return True 