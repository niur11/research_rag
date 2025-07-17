import os
import shutil
import logging
from typing import List, Dict, Optional
from pathlib import Path
import tempfile
from datetime import datetime
import hashlib

from config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LocalStorage:
    """Handles local storage operations for PDF files"""
    
    def __init__(self):
        self.config = Config()
        self.local_pdf_dir = self.config.LOCAL_PDF_DIR
        self.processed_data_dir = self.config.PROCESSED_DATA_DIR
        self.max_file_size = self.config.MAX_FILE_SIZE_MB * 1024 * 1024  # Convert to bytes
        
        # Create directories if they don't exist
        self._ensure_directories()
    
    def _ensure_directories(self):
        """Ensure all required directories exist"""
        directories = [
            self.local_pdf_dir,
            self.processed_data_dir,
            os.path.join(self.local_pdf_dir, 'backup'),
            os.path.join(self.local_pdf_dir, 'organized')
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            logger.info(f"Ensured directory exists: {directory}")
    
    def add_pdf(self, file_path: str, organize: bool = True) -> Dict:
        """
        Add a PDF file to local storage
        
        Args:
            file_path: Path to the PDF file to add
            organize: Whether to organize files by date/type
            
        Returns:
            Dictionary with operation result
        """
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
            
            # Validate file
            if not self._validate_pdf(file_path):
                return {
                    'success': False,
                    'error': 'Invalid PDF file or file too large'
                }
            
            # Generate destination path
            file_name = os.path.basename(file_path)
            if organize:
                dest_path = self._get_organized_path(file_path)
            else:
                dest_path = os.path.join(self.local_pdf_dir, file_name)
            
            # Copy file to local storage
            shutil.copy2(file_path, dest_path)
            
            # Create backup if enabled
            if self.config.LOCAL_STORAGE_BACKUP:
                self._create_backup(dest_path)
            
            logger.info(f"Successfully added {file_name} to local storage")
            
            return {
                'success': True,
                'original_path': file_path,
                'local_path': dest_path,
                'file_name': file_name,
                'file_size': os.path.getsize(dest_path)
            }
            
        except Exception as e:
            logger.error(f"Failed to add PDF {file_path}: {str(e)}")
            return {
                'success': False,
                'original_path': file_path,
                'error': str(e)
            }
    
    def _validate_pdf(self, file_path: str) -> bool:
        """Validate PDF file"""
        try:
            # Check file size
            file_size = os.path.getsize(file_path)
            if file_size > self.max_file_size:
                logger.warning(f"File {file_path} is too large: {file_size / (1024*1024):.2f}MB")
                return False
            
            # Check file extension
            if not file_path.lower().endswith('.pdf'):
                logger.warning(f"File {file_path} is not a PDF")
                return False
            
            # Try to open with PyPDF2 to validate
            try:
                import PyPDF2
                with open(file_path, 'rb') as file:
                    PyPDF2.PdfReader(file)
                return True
            except ImportError:
                # If PyPDF2 is not available, just check if file is readable
                with open(file_path, 'rb') as file:
                    header = file.read(4)
                    return header == b'%PDF'
            
        except Exception as e:
            logger.error(f"PDF validation failed for {file_path}: {str(e)}")
            return False
    
    def _get_organized_path(self, file_path: str) -> str:
        """Get organized path based on file metadata"""
        try:
            # Get file modification time
            mtime = os.path.getmtime(file_path)
            date = datetime.fromtimestamp(mtime)
            
            # Create year/month directory structure
            year_month = date.strftime('%Y/%m')
            organized_dir = os.path.join(self.local_pdf_dir, 'organized', year_month)
            os.makedirs(organized_dir, exist_ok=True)
            
            # Generate unique filename if needed
            file_name = os.path.basename(file_path)
            dest_path = os.path.join(organized_dir, file_name)
            
            # If file exists, add timestamp
            counter = 1
            while os.path.exists(dest_path):
                name, ext = os.path.splitext(file_name)
                dest_path = os.path.join(organized_dir, f"{name}_{counter}{ext}")
                counter += 1
            
            return dest_path
            
        except Exception as e:
            logger.warning(f"Failed to organize path for {file_path}: {str(e)}")
            # Fallback to simple copy
            return os.path.join(self.local_pdf_dir, os.path.basename(file_path))
    
    def _create_backup(self, file_path: str):
        """Create backup of the file"""
        try:
            backup_dir = os.path.join(self.local_pdf_dir, 'backup')
            backup_path = os.path.join(backup_dir, os.path.basename(file_path))
            
            # Add timestamp to backup filename
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            name, ext = os.path.splitext(os.path.basename(file_path))
            backup_path = os.path.join(backup_dir, f"{name}_{timestamp}{ext}")
            
            shutil.copy2(file_path, backup_path)
            logger.info(f"Created backup: {backup_path}")
            
        except Exception as e:
            logger.error(f"Failed to create backup for {file_path}: {str(e)}")
    
    def list_pdfs(self, include_organized: bool = True) -> List[Dict]:
        """
        List all PDF files in local storage
        
        Args:
            include_organized: Whether to include organized subdirectories
            
        Returns:
            List of PDF file information
        """
        try:
            pdfs = []
            
            # List files in main directory
            for file in os.listdir(self.local_pdf_dir):
                file_path = os.path.join(self.local_pdf_dir, file)
                if os.path.isfile(file_path) and file.lower().endswith('.pdf'):
                    pdfs.append(self._get_file_info(file_path))
            
            # List files in organized directories
            if include_organized:
                organized_dir = os.path.join(self.local_pdf_dir, 'organized')
                if os.path.exists(organized_dir):
                    for root, dirs, files in os.walk(organized_dir):
                        for file in files:
                            if file.lower().endswith('.pdf'):
                                file_path = os.path.join(root, file)
                                pdfs.append(self._get_file_info(file_path))
            
            logger.info(f"Found {len(pdfs)} PDF files in local storage")
            return pdfs
            
        except Exception as e:
            logger.error(f"Failed to list PDFs: {str(e)}")
            return []
    
    def _get_file_info(self, file_path: str) -> Dict:
        """Get file information"""
        try:
            stat = os.stat(file_path)
            return {
                'name': os.path.basename(file_path),
                'path': file_path,
                'size': stat.st_size,
                'modified': datetime.fromtimestamp(stat.st_mtime),
                'created': datetime.fromtimestamp(stat.st_ctime),
                'relative_path': os.path.relpath(file_path, self.local_pdf_dir)
            }
        except Exception as e:
            logger.error(f"Failed to get file info for {file_path}: {str(e)}")
            return {
                'name': os.path.basename(file_path),
                'path': file_path,
                'error': str(e)
            }
    
    def get_pdf(self, file_name: str) -> Optional[str]:
        """
        Get the full path of a PDF file by name
        
        Args:
            file_name: Name of the PDF file
            
        Returns:
            Full path to the file or None if not found
        """
        try:
            # Check main directory
            main_path = os.path.join(self.local_pdf_dir, file_name)
            if os.path.exists(main_path):
                return main_path
            
            # Check organized directories
            organized_dir = os.path.join(self.local_pdf_dir, 'organized')
            if os.path.exists(organized_dir):
                for root, dirs, files in os.walk(organized_dir):
                    if file_name in files:
                        return os.path.join(root, file_name)
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get PDF {file_name}: {str(e)}")
            return None
    
    def delete_pdf(self, file_name: str) -> Dict:
        """
        Delete a PDF file from local storage
        
        Args:
            file_name: Name of the PDF file to delete
            
        Returns:
            Dictionary with deletion result
        """
        try:
            file_path = self.get_pdf(file_name)
            if not file_path:
                return {
                    'success': False,
                    'error': f'File not found: {file_name}'
                }
            
            # Move to backup before deletion if backup is enabled
            if self.config.LOCAL_STORAGE_BACKUP:
                backup_dir = os.path.join(self.local_pdf_dir, 'backup')
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                name, ext = os.path.splitext(file_name)
                backup_path = os.path.join(backup_dir, f"{name}_deleted_{timestamp}{ext}")
                shutil.move(file_path, backup_path)
                logger.info(f"Moved {file_name} to backup before deletion")
            else:
                os.remove(file_path)
                logger.info(f"Deleted {file_name}")
            
            return {
                'success': True,
                'file_name': file_name,
                'backup_created': self.config.LOCAL_STORAGE_BACKUP
            }
            
        except Exception as e:
            logger.error(f"Failed to delete {file_name}: {str(e)}")
            return {
                'success': False,
                'file_name': file_name,
                'error': str(e)
            }
    
    def get_storage_stats(self) -> Dict:
        """Get local storage statistics"""
        try:
            total_files = 0
            total_size = 0
            organized_files = 0
            backup_files = 0
            
            # Count files in main directory
            for file in os.listdir(self.local_pdf_dir):
                file_path = os.path.join(self.local_pdf_dir, file)
                if os.path.isfile(file_path) and file.lower().endswith('.pdf'):
                    total_files += 1
                    total_size += os.path.getsize(file_path)
            
            # Count organized files
            organized_dir = os.path.join(self.local_pdf_dir, 'organized')
            if os.path.exists(organized_dir):
                for root, dirs, files in os.walk(organized_dir):
                    for file in files:
                        if file.lower().endswith('.pdf'):
                            organized_files += 1
                            total_size += os.path.getsize(os.path.join(root, file))
            
            # Count backup files
            backup_dir = os.path.join(self.local_pdf_dir, 'backup')
            if os.path.exists(backup_dir):
                for file in os.listdir(backup_dir):
                    if file.lower().endswith('.pdf'):
                        backup_files += 1
            
            return {
                'total_files': total_files + organized_files,
                'main_directory_files': total_files,
                'organized_files': organized_files,
                'backup_files': backup_files,
                'total_size_mb': total_size / (1024 * 1024),
                'storage_path': self.local_pdf_dir
            }
            
        except Exception as e:
            logger.error(f"Failed to get storage stats: {str(e)}")
            return {'error': str(e)}
    
    def cleanup_old_backups(self, days_to_keep: int = 30) -> Dict:
        """
        Clean up old backup files
        
        Args:
            days_to_keep: Number of days to keep backups
            
        Returns:
            Dictionary with cleanup results
        """
        try:
            backup_dir = os.path.join(self.local_pdf_dir, 'backup')
            if not os.path.exists(backup_dir):
                return {
                    'success': True,
                    'files_deleted': 0,
                    'message': 'No backup directory found'
                }
            
            cutoff_date = datetime.now().timestamp() - (days_to_keep * 24 * 3600)
            deleted_count = 0
            
            for file in os.listdir(backup_dir):
                file_path = os.path.join(backup_dir, file)
                if os.path.isfile(file_path) and file.lower().endswith('.pdf'):
                    if os.path.getmtime(file_path) < cutoff_date:
                        os.remove(file_path)
                        deleted_count += 1
            
            logger.info(f"Cleaned up {deleted_count} old backup files")
            
            return {
                'success': True,
                'files_deleted': deleted_count,
                'days_kept': days_to_keep
            }
            
        except Exception as e:
            logger.error(f"Failed to cleanup backups: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            } 