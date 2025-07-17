import os
import logging
from typing import List, Dict, Optional, Any
from pathlib import Path
import PyPDF2  # type: ignore
import pdfplumber  # type: ignore
from tqdm import tqdm
import re

from config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PDFProcessor:
    """Handles PDF processing and text extraction"""
    
    def __init__(self):
        self.config = Config()
        self.supported_extensions = ['.pdf']

    def extract_text_from_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """
        Extract text from a PDF file using multiple methods for better coverage.

        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Dictionary containing extracted text, metadata, and processing info
        """
        try:
            result = {
                'file_path': pdf_path,
                'file_name': os.path.basename(pdf_path),
                'text': '',
                'pages': [],
                'metadata': {},
                'processing_info': {}
            }
            
            # Try PyPDF2 first
            text_pypdf2 = self._extract_with_pypdf2(pdf_path)
            
            # Try pdfplumber as backup
            text_pdfplumber = self._extract_with_pdfplumber(pdf_path)
            
            # Use the method that extracted more text
            if len(text_pdfplumber) > len(text_pypdf2):
                result['text'] = text_pdfplumber
                result['processing_info']['method'] = 'pdfplumber'
            else:
                result['text'] = text_pypdf2
                result['processing_info']['method'] = 'pypdf2'
            
            # Extract metadata
            result['metadata'] = self._extract_metadata(pdf_path)
            
            # Clean and preprocess text
            result['text'] = self._clean_text(result['text'])
            
            # Split into pages
            result['pages'] = self._split_into_pages(result['text'])
            
            result['processing_info']['success'] = True
            result['processing_info']['text_length'] = len(result['text'])
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing PDF {pdf_path}: {str(e)}")
            return {
                'file_path': pdf_path,
                'file_name': os.path.basename(pdf_path),
                'text': '',
                'pages': [],
                'metadata': {},
                'processing_info': {
                    'success': False,
                    'error': str(e)
                }
            }
    
    def _extract_with_pypdf2(self, pdf_path: str) -> str:
        """Extract text using PyPDF2"""
        text = ""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
        except Exception as e:
            logger.warning(f"PyPDF2 extraction failed for {pdf_path}: {str(e)}")
        return text
    
    def _extract_with_pdfplumber(self, pdf_path: str) -> str:
        """Extract text using pdfplumber"""
        text = ""
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
        except Exception as e:
            logger.warning(f"pdfplumber extraction failed for {pdf_path}: {str(e)}")
        return text
    
    def _extract_metadata(self, pdf_path: str) -> Dict:
        """Extract metadata from PDF"""
        metadata = {}
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                if pdf_reader.metadata:
                    metadata = dict(pdf_reader.metadata)
        except Exception as e:
            logger.warning(f"Failed to extract metadata from {pdf_path}: {str(e)}")
        return metadata
    
    def _clean_text(self, text: str) -> str:
        """Clean and preprocess extracted text"""
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters that might interfere with processing
        text = re.sub(r'[^\w\s\.\,\;\:\!\?\-\(\)\[\]\{\}]', ' ', text)
        
        # Remove page numbers and headers/footers
        text = re.sub(r'\b\d+\s*$', '', text, flags=re.MULTILINE)
        
        # Remove common PDF artifacts
        text = re.sub(r'[^\x00-\x7F]+', '', text)  # Remove non-ASCII characters
        
        return text.strip()
    
    def _split_into_pages(self, text: str) -> List[str]:
        """Split text into pages (approximate)"""
        if not text:
            return []
        
        # Simple splitting by double newlines (common page breaks)
        pages = re.split(r'\n\s*\n', text)
        return [page.strip() for page in pages if page.strip()]
    
    def process_directory(self, directory_path: str) -> List[Dict]:
        """
        Process all PDF files in a directory
        
        Args:
            directory_path: Path to directory containing PDF files
            
        Returns:
            List of processed PDF results
        """
        results = []
        pdf_files = []
        
        # Find all PDF files
        for root, dirs, files in os.walk(directory_path):
            for file in files:
                if file.lower().endswith('.pdf'):
                    pdf_files.append(os.path.join(root, file))
        
        logger.info(f"Found {len(pdf_files)} PDF files to process")
        
        # Process each PDF file
        for pdf_file in tqdm(pdf_files, desc="Processing PDFs"):
            result = self.extract_text_from_pdf(pdf_file)
            results.append(result)
            
            if result['processing_info'].get('success'):
                logger.info(f"Successfully processed: {result['file_name']}")
            else:
                logger.error(f"Failed to process: {result['file_name']}")
        
        return results
    
    def validate_pdf(self, pdf_path: str) -> bool:
        """
        Validate if a PDF file can be processed
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            True if PDF is valid and can be processed
        """
        try:
            # Check file exists and is readable
            if not os.path.exists(pdf_path):
                return False
            
            # Check file size
            file_size = os.path.getsize(pdf_path) / (1024 * 1024)  # MB
            if file_size > self.config.MAX_FILE_SIZE_MB:
                logger.warning(f"File {pdf_path} is too large: {file_size:.2f}MB")
                return False
            
            # Try to open with PyPDF2
            with open(pdf_path, 'rb') as file:
                PyPDF2.PdfReader(file)
            
            return True
            
        except Exception as e:
            logger.error(f"PDF validation failed for {pdf_path}: {str(e)}")
            return False 