import os
import logging
from typing import List, Dict, Optional, Tuple
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.retrievers import ContextualCompressionRetriever, EnsembleRetriever, MultiQueryRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from pydantic import SecretStr
import json
import tempfile
import numpy as np
from datetime import datetime

from config import Config
from pdf_processor import PDFProcessor
from local_storage import LocalStorage

# Conditional Azure import
AZURE_AVAILABLE = False
AzureBlobStorage = None
try:
    from azure_storage import AzureBlobStorage
    AZURE_AVAILABLE = True
except ImportError as e:
    print(f"Azure storage not available: {e}")
except Exception as e:
    print(f"Azure storage initialization failed: {e}")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImprovedResearchRAGSystem:
    """Enhanced RAG system with better semantic understanding and retrieval"""
    
    def __init__(self):
        self.config = Config()
        self.config.validate_config()
        
        # Initialize components
        self.pdf_processor = PDFProcessor()
        
        # Initialize storage components
        self.azure_storage = None
        self.local_storage = None
        
        if self.config.ENABLE_LOCAL_STORAGE:
            self.local_storage = LocalStorage()
            logger.info("Local storage enabled")
        
        if self.config.AZURE_STORAGE_CONNECTION_STRING and AZURE_AVAILABLE and AzureBlobStorage:
            try:
                self.azure_storage = AzureBlobStorage()
                logger.info("Azure storage enabled")
            except Exception as e:
                logger.warning(f"Azure storage initialization failed: {str(e)}")
        
        # Initialize LLM components with better models
        self.embeddings = OpenAIEmbeddings(
            api_key=SecretStr(self.config.OPENAI_API_KEY) if self.config.OPENAI_API_KEY else None,
            model=self.config.OPENAI_EMBEDDING_MODEL
        )
        
        self.llm = ChatOpenAI(
            api_key=SecretStr(self.config.OPENAI_API_KEY) if self.config.OPENAI_API_KEY else None,
            model=self.config.OPENAI_MODEL,
            temperature=0.1
        )
        
        # Create text splitter with better parameters
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.CHUNK_SIZE,
            chunk_overlap=self.config.CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        # Initialize vector store
        self.vector_store = None
        self.vector_store_path = os.path.join(self.config.VECTOR_DB_PATH, "faiss_index")
        self._initialize_vector_store()
        
        # Initialize retrievers
        self._initialize_retrievers()
        
        # Enhanced prompt templates
        self._initialize_prompts()
    
    def _initialize_vector_store(self):
        """Initialize FAISS vector store"""
        try:
            if os.path.exists(self.vector_store_path):
                # Allow dangerous deserialization for trusted local files
                self.vector_store = FAISS.load_local(
                    self.vector_store_path, 
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                logger.info("Loaded existing FAISS vector store")
            else:
                # Create empty vector store
                self.vector_store = FAISS.from_texts(["Initial document"], self.embeddings)
                # Remove the initial document
                self.vector_store.delete([self.vector_store.index_to_docstore_id[0]])
                logger.info("Created new FAISS vector store")
        except Exception as e:
            logger.error(f"Failed to initialize vector store: {e}")
            # Create a simple in-memory store as fallback
            self.vector_store = FAISS.from_texts(["Initial document"], self.embeddings)
            self.vector_store.delete([self.vector_store.index_to_docstore_id[0]])
    
    def _initialize_retrievers(self):
        """Initialize multiple retrievers for ensemble approach"""
        try:
            # Create multiple retrievers for ensemble
            self.retrievers = {}
            
            # 1. Semantic retriever (FAISS)
            if self.vector_store:
                self.retrievers['semantic'] = self.vector_store.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": self.config.TOP_K_RESULTS}
                )
            
            # 2. Multi-query retriever for better query understanding
            if self.vector_store:
                self.retrievers['multi_query'] = MultiQueryRetriever.from_llm(
                    retriever=self.vector_store.as_retriever(
                        search_type="similarity",
                        search_kwargs={"k": self.config.TOP_K_RESULTS}
                    ),
                    llm=self.llm
                )
            
            # 3. Contextual compression retriever
            if self.vector_store:
                compressor_prompt = """Given the following question and context, extract only the relevant information that helps answer the question.

Question: {question}

Context: {context}

Relevant information:"""
                
                compressor_prompt_template = PromptTemplate(
                    template=compressor_prompt,
                    input_variables=["question", "context"]
                )
                
                compressor = LLMChainExtractor.from_llm(llm=self.llm, prompt=compressor_prompt_template)
                
                self.retrievers['contextual'] = ContextualCompressionRetriever(
                    base_compressor=compressor,
                    base_retriever=self.vector_store.as_retriever(
                        search_type="similarity",
                        search_kwargs={"k": self.config.TOP_K_RESULTS * 2}
                    )
                )
            
            # 4. Ensemble retriever (combines multiple retrievers)
            if len(self.retrievers) > 1:
                self.ensemble_retriever = EnsembleRetriever(
                    retrievers=list(self.retrievers.values()),
                    weights=[0.4, 0.3, 0.3]  # Weight the retrievers
                )
            else:
                self.ensemble_retriever = list(self.retrievers.values())[0] if self.retrievers else None
            
            logger.info(f"Initialized {len(self.retrievers)} retrievers")
            
        except Exception as e:
            logger.error(f"Failed to initialize retrievers: {e}")
            self.retrievers = {}
            self.ensemble_retriever = None
    
    def _initialize_prompts(self):
        """Initialize enhanced prompt templates"""
        self.qa_prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""You are a research assistant with expertise in academic papers. Answer the following question based on the provided research context.

IMPORTANT INSTRUCTIONS:
1. Only use information from the provided context to answer the question
2. If the context doesn't contain enough information to answer the question, say so clearly
3. Provide specific citations and references when possible
4. Be precise and accurate in your response
5. If the question is ambiguous, ask for clarification or provide multiple interpretations

Context:
{context}

Question: {question}

Answer:"""
        )
        
        self.query_enhancement_prompt = PromptTemplate(
            input_variables=["question"],
            template="""Given the following question about research papers, generate 3 different ways to ask the same question to improve search results. Focus on different aspects, synonyms, and related concepts.

Original question: {question}

Generate 3 variations:
1. """
        )
    
    def _enhance_query(self, question: str) -> List[str]:
        """Enhance the query with multiple variations for better retrieval"""
        try:
            # Generate query variations using LLM
            chain = LLMChain(llm=self.llm, prompt=self.query_enhancement_prompt)
            response = chain.run(question=question)
            
            # Parse the response to extract variations
            lines = response.strip().split('\n')
            variations = [question]  # Include original question
            
            for line in lines:
                if line.strip() and any(line.strip().startswith(str(i)) for i in range(1, 4)):
                    # Extract the variation after the number
                    variation = line.split('.', 1)[1].strip() if '.' in line else line.strip()
                    if variation and variation != question:
                        variations.append(variation)
            
            return variations[:4]  # Return up to 4 variations including original
            
        except Exception as e:
            logger.error(f"Query enhancement failed: {e}")
            return [question]  # Return original question if enhancement fails
    
    def _get_relevant_documents(self, question: str) -> List[Document]:
        """Get relevant documents using ensemble retrieval"""
        try:
            if not self.ensemble_retriever:
                return []
            
            # Get documents from ensemble retriever
            docs = self.ensemble_retriever.get_relevant_documents(question)
            
            # Remove duplicates while preserving order
            seen = set()
            unique_docs = []
            for doc in docs:
                doc_hash = hash(doc.page_content)
                if doc_hash not in seen:
                    seen.add(doc_hash)
                    unique_docs.append(doc)
            
            return unique_docs[:self.config.TOP_K_RESULTS * 2]  # Return more docs for better context
            
        except Exception as e:
            logger.error(f"Document retrieval failed: {e}")
            return []
    
    def add_pdf_to_local_storage(self, file_path: str, organize: bool = True) -> Dict:
        """Add a PDF file to local storage"""
        if not self.local_storage:
            return {
                'success': False,
                'error': 'Local storage is not enabled'
            }
        
        return self.local_storage.add_pdf(file_path, organize)
    
    def process_local_storage_pdfs(self) -> Dict:
        """Process all PDFs from local storage with improved vector storage"""
        if not self.local_storage:
            return {
                'success': False,
                'error': 'Local storage is not enabled'
            }
        
        try:
            logger.info("Starting enhanced local storage PDF processing pipeline")
            
            # Get all PDFs from local storage
            pdfs_in_local = self.local_storage.list_pdfs()
            if not pdfs_in_local:
                return {
                    'success': False,
                    'error': 'No PDF files found in local storage'
                }
            
            logger.info(f"Found {len(pdfs_in_local)} PDF files in local storage")
            
            # Process each PDF
            processed_documents = []
            for pdf_info in pdfs_in_local:
                if 'error' not in pdf_info:
                    doc_result = self.pdf_processor.extract_text_from_pdf(pdf_info['path'])
                    if doc_result['processing_info'].get('success'):
                        processed_documents.append(doc_result)
                        logger.info(f"Successfully processed: {pdf_info['name']}")
                    else:
                        logger.error(f"Failed to process: {pdf_info['name']}")
            
            # Add documents to vector store
            if processed_documents:
                vector_result = self._add_documents_to_vector_store(processed_documents)
                
                return {
                    'success': True,
                    'pdfs_found': len(pdfs_in_local),
                    'pdfs_processed': len(processed_documents),
                    'chunks_added': vector_result.get('total_chunks_added', 0),
                    'vector_store_result': vector_result
                }
            else:
                return {
                    'success': False,
                    'error': 'No documents were successfully processed'
                }
                
        except Exception as e:
            logger.error(f"Local storage PDF processing failed: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _add_documents_to_vector_store(self, documents: List[Dict]) -> Dict:
        """Add documents to the FAISS vector store"""
        try:
            total_added = 0
            
            for doc in documents:
                if not doc.get('text'):
                    continue
                
                # Chunk the document
                chunks = self.text_splitter.split_text(doc['text'])
                
                # Create LangChain Document objects
                langchain_docs = []
                for i, chunk in enumerate(chunks):
                    metadata = {
                        'file_name': doc['file_name'],
                        'file_path': doc.get('file_path', ''),
                        'chunk_index': i,
                        'total_chunks': len(chunks),
                        'chunk_size': len(chunk),
                        'processing_method': doc.get('processing_info', {}).get('method', 'unknown')
                    }
                    
                    # Add original document metadata
                    if 'metadata' in doc:
                        metadata.update(doc['metadata'])
                    
                    langchain_docs.append(Document(
                        page_content=chunk,
                        metadata=metadata
                    ))
                
                # Add to vector store
                if langchain_docs and self.vector_store:
                    self.vector_store.add_documents(langchain_docs)
                    total_added += len(langchain_docs)
            
            # Save vector store
            if self.vector_store:
                self.vector_store.save_local(self.vector_store_path)
            
            logger.info(f"Successfully added {total_added} chunks to FAISS vector store")
            
            return {
                'success': True,
                'total_chunks_added': total_added,
                'total_documents_processed': len(documents)
            }
            
        except Exception as e:
            logger.error(f"Failed to add documents to vector store: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def ask_question(self, question: str, include_sources: bool = True) -> Dict:
        """Ask a question with enhanced semantic understanding"""
        try:
            logger.info(f"Processing enhanced question: {question}")
            
            # Get relevant documents using ensemble retrieval
            relevant_docs = self._get_relevant_documents(question)
            
            if not relevant_docs:
                return {
                    'success': False,
                    'answer': 'No relevant documents found to answer your question.',
                    'sources': [],
                    'similarity_scores': []
                }
            
            # Prepare context from relevant documents
            context_parts = []
            sources = []
            
            for i, doc in enumerate(relevant_docs):
                context_parts.append(doc.page_content)
                sources.append({
                    'file_name': doc.metadata.get('file_name', 'Unknown'),
                    'chunk_index': doc.metadata.get('chunk_index', 0),
                    'rank': i + 1
                })
            
            context = "\n\n---\n\n".join(context_parts)
            
            # Generate answer using enhanced prompt
            chain = LLMChain(llm=self.llm, prompt=self.qa_prompt_template)
            response = chain.run(context=context, question=question)
            
            return {
                'success': True,
                'answer': response,
                'sources': sources if include_sources else [],
                'context_length': len(context),
                'num_sources': len(sources),
                'retrieval_method': 'ensemble'
            }
            
        except Exception as e:
            logger.error(f"Enhanced question answering failed: {str(e)}")
            return {
                'success': False,
                'answer': f'Error processing your question: {str(e)}',
                'sources': [],
                'similarity_scores': []
            }
    
    def get_system_stats(self) -> Dict:
        """Get comprehensive system statistics"""
        try:
            stats = {
                'vector_store': {
                    'type': 'FAISS',
                    'documents_count': self.vector_store.index.ntotal if self.vector_store else 0,
                    'index_size': self.vector_store.index.ntotal if self.vector_store else 0
                },
                'configuration': {
                    'chunk_size': self.config.CHUNK_SIZE,
                    'chunk_overlap': self.config.CHUNK_OVERLAP,
                    'top_k_results': self.config.TOP_K_RESULTS,
                    'similarity_threshold': self.config.SIMILARITY_THRESHOLD
                },
                'retrievers': {
                    'count': len(self.retrievers),
                    'types': list(self.retrievers.keys())
                }
            }
            
            # Add storage stats
            if self.azure_storage:
                try:
                    azure_pdfs = self.azure_storage.list_pdfs()
                    stats['azure_storage'] = {
                        'total_pdfs': len(azure_pdfs),
                        'pdf_names': [pdf['name'] for pdf in azure_pdfs]
                    }
                except Exception as e:
                    stats['azure_storage'] = {'error': str(e)}
            else:
                stats['azure_storage'] = {'enabled': False}
            
            if self.local_storage:
                try:
                    local_stats = self.local_storage.get_storage_stats()
                    stats['local_storage'] = local_stats
                except Exception as e:
                    stats['local_storage'] = {'error': str(e)}
            else:
                stats['local_storage'] = {'enabled': False}
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get system stats: {str(e)}")
            return {'error': str(e)}
    
    def summarize_research_findings(self, topic: Optional[str] = None) -> Dict:
        """Generate a comprehensive summary of research findings"""
        try:
            if topic:
                question = f"Provide a comprehensive summary of the key research findings related to {topic} from the available papers. Include main conclusions, methodologies, and important insights."
            else:
                question = "Provide a comprehensive summary of the key research findings from all available papers. Organize by themes and highlight the most important contributions."
            
            result = self.ask_question(question)
            
            if result['success']:
                return {
                    'success': True,
                    'summary': result['answer'],
                    'sources_used': result['sources'],
                    'topic': topic or 'general'
                }
            else:
                return result
                
        except Exception as e:
            logger.error(f"Research summarization failed: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_local_storage_pdfs(self) -> List[Dict]:
        """Get list of PDFs in local storage"""
        if not self.local_storage:
            return []
        
        return self.local_storage.list_pdfs()
    
    def delete_local_pdf(self, file_name: str) -> Dict:
        """Delete a PDF from local storage"""
        if not self.local_storage:
            return {
                'success': False,
                'error': 'Local storage is not enabled'
            }
        
        return self.local_storage.delete_pdf(file_name)
    
    def cleanup_local_backups(self, days_to_keep: int = 30) -> Dict:
        """Clean up old backup files in local storage"""
        if not self.local_storage:
            return {
                'success': False,
                'error': 'Local storage is not enabled'
            }
        
        return self.local_storage.cleanup_old_backups(days_to_keep) 