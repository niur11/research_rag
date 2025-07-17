import os
import logging
from typing import List, Dict, Optional, Tuple
import json
import re
from datetime import datetime
from collections import Counter

from config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BasicSemanticVectorStore:
    """Basic semantic vector store using TF-IDF and keyword expansion"""
    
    def __init__(self):
        self.config = Config()
        self.vector_db_path = self.config.VECTOR_DB_PATH
        self.chunk_size = self.config.CHUNK_SIZE
        self.chunk_overlap = self.config.CHUNK_OVERLAP
        
        # Create directory if it doesn't exist
        os.makedirs(self.vector_db_path, exist_ok=True)
        
        # File to store documents
        self.documents_file = os.path.join(self.vector_db_path, "documents.json")
        self.documents = self._load_documents()
        
        # Semantic keyword mappings
        self.semantic_mappings = {
            'attention': ['attention', 'self-attention', 'attention mechanism', 'attentional'],
            'transformer': ['transformer', 'transformer model', 'attention is all you need'],
            'neural': ['neural', 'neural network', 'deep learning', 'machine learning'],
            'encoder': ['encoder', 'decoder', 'encoder-decoder', 'sequence'],
            'reference': ['reference', 'citation', 'cite', 'paper', 'publication'],
            'list': ['list', 'enumeration', 'items', 'references', 'bibliography'],
            'model': ['model', 'architecture', 'network', 'system'],
            'learning': ['learning', 'training', 'machine learning', 'deep learning'],
            'network': ['network', 'neural network', 'architecture', 'model']
        }
    
    def _load_documents(self) -> List[Dict]:
        """Load documents from file"""
        if os.path.exists(self.documents_file):
            try:
                with open(self.documents_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load documents: {e}")
                return []
        return []
    
    def _save_documents(self):
        """Save documents to file"""
        try:
            with open(self.documents_file, 'w', encoding='utf-8') as f:
                json.dump(self.documents, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save documents: {e}")
    
    def _expand_query_semantically(self, query: str) -> List[str]:
        """Expand query with semantic keywords"""
        query_lower = query.lower()
        expanded_terms = [query_lower]
        
        # Add semantic expansions
        for base_term, expansions in self.semantic_mappings.items():
            if base_term in query_lower:
                expanded_terms.extend(expansions)
        
        return expanded_terms
    
    def _calculate_semantic_similarity(self, query_terms: List[str], text: str) -> float:
        """Calculate semantic similarity using expanded terms"""
        text_lower = text.lower()
        
        # Count matches for each expanded term
        total_matches = 0
        total_terms = len(query_terms)
        
        for term in query_terms:
            # Count occurrences of the term
            matches = text_lower.count(term)
            if matches > 0:
                total_matches += matches
        
        # Calculate similarity score
        if total_terms > 0:
            # Normalize by text length and term count
            text_words = len(text.split())
            if text_words > 0:
                similarity = (total_matches / text_words) * (total_matches / total_terms)
                return min(similarity * 10, 1.0)  # Scale up and cap at 1.0
        
        return 0.0
    
    def chunk_text(self, text: str, chunk_size: Optional[int] = None, 
                   chunk_overlap: Optional[int] = None) -> List[str]:
        """Split text into overlapping chunks"""
        if chunk_size is None:
            chunk_size = self.chunk_size
        if chunk_overlap is None:
            chunk_overlap = self.chunk_overlap
        
        if not text or len(text) <= chunk_size:
            return [text] if text else []
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            # If this isn't the last chunk, try to break at a sentence boundary
            if end < len(text):
                # Look for sentence endings within the last 100 characters
                search_start = max(start + chunk_size - 100, start)
                search_end = min(end + 100, len(text))
                
                # Find the last sentence ending
                last_period = text.rfind('.', search_start, search_end)
                last_exclamation = text.rfind('!', search_start, search_end)
                last_question = text.rfind('?', search_start, search_end)
                
                # Use the latest sentence ending
                sentence_end = max(last_period, last_exclamation, last_question)
                
                if sentence_end > start + chunk_size // 2:  # Only if it's not too early
                    end = sentence_end + 1
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            # Move start position, accounting for overlap
            start = end - chunk_overlap
            if start >= len(text):
                break
        
        return chunks
    
    def add_documents(self, documents: List[Dict]) -> Dict:
        """Add documents to the vector store"""
        try:
            total_added = 0
            
            for doc in documents:
                if not doc.get('text'):
                    continue
                
                # Chunk the document
                chunks = self.chunk_text(doc['text'])
                
                # Create metadata for each chunk
                for i, chunk in enumerate(chunks):
                    chunk_id = f"{doc['file_name']}_{i}"
                    
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
                    
                    # Store document
                    document_entry = {
                        'id': chunk_id,
                        'text': chunk,
                        'metadata': metadata,
                        'added_at': datetime.now().isoformat()
                    }
                    
                    self.documents.append(document_entry)
                    total_added += 1
            
            # Save to file
            self._save_documents()
            
            logger.info(f"Successfully added {total_added} chunks to vector store")
            
            return {
                'success': True,
                'total_chunks_added': total_added,
                'total_documents_processed': len(documents)
            }
            
        except Exception as e:
            logger.error(f"Failed to add documents: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def search_similar(self, query: str, top_k: Optional[int] = None, 
                      threshold: Optional[float] = None) -> List[Dict]:
        """Search for similar documents using semantic keyword expansion"""
        try:
            if top_k is None:
                top_k = self.config.TOP_K_RESULTS
            if threshold is None:
                threshold = self.config.SIMILARITY_THRESHOLD
            
            # Expand query semantically
            expanded_terms = self._expand_query_semantically(query)
            
            # Calculate similarities for all documents
            similarities = []
            for i, doc in enumerate(self.documents):
                similarity_score = self._calculate_semantic_similarity(expanded_terms, doc['text'])
                
                if similarity_score >= threshold:
                    similarities.append((i, similarity_score))
            
            # Sort by similarity and take top_k
            similarities.sort(key=lambda x: x[1], reverse=True)
            results = []
            
            for i, similarity_score in similarities[:top_k]:
                doc = self.documents[i]
                results.append({
                    'document': doc['text'],
                    'metadata': doc['metadata'],
                    'similarity_score': similarity_score,
                    'distance': 1 - similarity_score,
                    'rank': len(results) + 1
                })
            
            logger.info(f"Found {len(results)} similar documents for query")
            return results
            
        except Exception as e:
            logger.error(f"Search failed: {str(e)}")
            return []
    
    def get_collection_stats(self) -> Dict:
        """Get statistics about the vector store collection"""
        try:
            total_documents = len(self.documents)
            
            # Analyze metadata
            file_names = set()
            total_chunks = 0
            
            for doc in self.documents:
                if doc.get('metadata'):
                    file_names.add(doc['metadata'].get('file_name', 'unknown'))
                    total_chunks += doc['metadata'].get('total_chunks', 0)
            
            return {
                'total_documents': total_documents,
                'unique_files': len(file_names),
                'file_names': list(file_names),
                'estimated_total_chunks': total_chunks
            }
            
        except Exception as e:
            logger.error(f"Failed to get collection stats: {str(e)}")
            return {}
    
    def delete_documents(self, file_names: List[str]) -> Dict:
        """Delete documents by file name"""
        try:
            original_count = len(self.documents)
            
            # Filter out documents with matching file names
            self.documents = [
                doc for doc in self.documents 
                if doc.get('metadata', {}).get('file_name') not in file_names
            ]
            
            deleted_count = original_count - len(self.documents)
            
            if deleted_count > 0:
                self._save_documents()
                logger.info(f"Deleted {deleted_count} chunks for files: {file_names}")
                
                return {
                    'success': True,
                    'deleted_chunks': deleted_count,
                    'deleted_files': file_names
                }
            else:
                return {
                    'success': False,
                    'error': 'No matching documents found'
                }
                
        except Exception as e:
            logger.error(f"Failed to delete documents: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def clear_collection(self) -> Dict:
        """Clear all documents from the collection"""
        try:
            self.documents = []
            self._save_documents()
            logger.info("Cleared all documents from collection")
            
            return {
                'success': True,
                'message': 'Collection cleared'
            }
            
        except Exception as e:
            logger.error(f"Failed to clear collection: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            } 