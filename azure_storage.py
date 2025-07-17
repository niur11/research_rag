import os
import logging
from typing import List, Dict, Optional, BinaryIO
from azure.storage.blob import BlobServiceClient, ContainerClient
from azure.core.exceptions import ResourceNotFoundError, ClientAuthenticationError
import tempfile
from pathlib import Path

from config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AzureBlobStorage:
    """Handles Azure Blob Storage operations for PDF files"""
    
    def __init__(self):
        self.config = Config()
        self.connection_string = self.config.AZURE_STORAGE_CONNECTION_STRING
        self.container_name = self.config.AZURE_CONTAINER_NAME
        
        if not self.connection_string:
            raise ValueError("Azure Storage connection string is required")
        
        self.blob_service_client = BlobServiceClient.from_connection_string(
            self.connection_string
        )
        self.container_client = self.blob_service_client.get_container_client(
            self.container_name
        )
    
    def create_container_if_not_exists(self) -> bool:
        """Create the container if it doesn't exist"""
        try:
            self.container_client.get_container_properties()
            logger.info(f"Container '{self.container_name}' already exists")
            return True
        except ResourceNotFoundError:
            try:
                self.container_client.create_container()
                logger.info(f"Created container '{self.container_name}'")
                return True
            except Exception as e:
                logger.error(f"Failed to create container: {str(e)}")
                return False
    
    def upload_pdf(self, local_file_path: str, blob_name: Optional[str] = None) -> Dict:
        """
        Upload a PDF file to Azure Blob Storage
        
        Args:
            local_file_path: Path to the local PDF file
            blob_name: Optional custom name for the blob
            
        Returns:
            Dictionary with upload result information
        """
        try:
            if not os.path.exists(local_file_path):
                raise FileNotFoundError(f"File not found: {local_file_path}")
            
            if not blob_name:
                blob_name = os.path.basename(local_file_path)
            
            # Get blob client
            blob_client = self.container_client.get_blob_client(blob_name)
            
            # Upload the file
            with open(local_file_path, "rb") as data:
                blob_client.upload_blob(data, overwrite=True)
            
            logger.info(f"Successfully uploaded {local_file_path} as {blob_name}")
            
            return {
                'success': True,
                'local_path': local_file_path,
                'blob_name': blob_name,
                'blob_url': blob_client.url
            }
            
        except Exception as e:
            logger.error(f"Failed to upload {local_file_path}: {str(e)}")
            return {
                'success': False,
                'local_path': local_file_path,
                'error': str(e)
            }
    
    def download_pdf(self, blob_name: str, local_file_path: Optional[str] = None) -> Dict:
        """
        Download a PDF file from Azure Blob Storage
        
        Args:
            blob_name: Name of the blob to download
            local_file_path: Optional local path to save the file
            
        Returns:
            Dictionary with download result information
        """
        try:
            if not local_file_path:
                # Create a temporary file
                temp_dir = tempfile.gettempdir()
                local_file_path = os.path.join(temp_dir, blob_name)
            
            # Get blob client
            blob_client = self.container_client.get_blob_client(blob_name)
            
            # Download the blob
            with open(local_file_path, "wb") as download_file:
                download_stream = blob_client.download_blob()
                download_file.write(download_stream.readall())
            
            logger.info(f"Successfully downloaded {blob_name} to {local_file_path}")
            
            return {
                'success': True,
                'blob_name': blob_name,
                'local_path': local_file_path,
                'file_size': os.path.getsize(local_file_path)
            }
            
        except ResourceNotFoundError:
            logger.error(f"Blob not found: {blob_name}")
            return {
                'success': False,
                'blob_name': blob_name,
                'error': 'Blob not found'
            }
        except Exception as e:
            logger.error(f"Failed to download {blob_name}: {str(e)}")
            return {
                'success': False,
                'blob_name': blob_name,
                'error': str(e)
            }
    
    def list_pdfs(self) -> List[Dict]:
        """
        List all PDF files in the container
        
        Returns:
            List of dictionaries containing blob information
        """
        try:
            pdfs = []
            blobs = self.container_client.list_blobs()
            
            for blob in blobs:
                if blob.name.lower().endswith('.pdf'):
                    pdfs.append({
                        'name': blob.name,
                        'size': blob.size,
                        'last_modified': blob.last_modified,
                        'url': f"{self.container_client.url}/{blob.name}"
                    })
            
            logger.info(f"Found {len(pdfs)} PDF files in container")
            return pdfs
            
        except Exception as e:
            logger.error(f"Failed to list PDFs: {str(e)}")
            return []
    
    def delete_pdf(self, blob_name: str) -> Dict:
        """
        Delete a PDF file from Azure Blob Storage
        
        Args:
            blob_name: Name of the blob to delete
            
        Returns:
            Dictionary with delete result information
        """
        try:
            blob_client = self.container_client.get_blob_client(blob_name)
            blob_client.delete_blob()
            
            logger.info(f"Successfully deleted {blob_name}")
            
            return {
                'success': True,
                'blob_name': blob_name
            }
            
        except ResourceNotFoundError:
            logger.error(f"Blob not found for deletion: {blob_name}")
            return {
                'success': False,
                'blob_name': blob_name,
                'error': 'Blob not found'
            }
        except Exception as e:
            logger.error(f"Failed to delete {blob_name}: {str(e)}")
            return {
                'success': False,
                'blob_name': blob_name,
                'error': str(e)
            }
    
    def get_pdf_metadata(self, blob_name: str) -> Dict:
        """
        Get metadata for a PDF blob
        
        Args:
            blob_name: Name of the blob
            
        Returns:
            Dictionary with blob metadata
        """
        try:
            blob_client = self.container_client.get_blob_client(blob_name)
            properties = blob_client.get_blob_properties()
            
            return {
                'success': True,
                'blob_name': blob_name,
                'size': properties.size,
                'last_modified': properties.last_modified,
                'content_type': properties.content_settings.content_type,
                'metadata': properties.metadata
            }
            
        except ResourceNotFoundError:
            logger.error(f"Blob not found: {blob_name}")
            return {
                'success': False,
                'blob_name': blob_name,
                'error': 'Blob not found'
            }
        except Exception as e:
            logger.error(f"Failed to get metadata for {blob_name}: {str(e)}")
            return {
                'success': False,
                'blob_name': blob_name,
                'error': str(e)
            }
    
    def download_all_pdfs(self, local_directory: str) -> List[Dict]:
        """
        Download all PDF files from the container to a local directory
        
        Args:
            local_directory: Local directory to save PDF files
            
        Returns:
            List of download results
        """
        try:
            # Create local directory if it doesn't exist
            os.makedirs(local_directory, exist_ok=True)
            
            # Get list of PDFs
            pdfs = self.list_pdfs()
            results = []
            
            for pdf in pdfs:
                local_path = os.path.join(local_directory, pdf['name'])
                result = self.download_pdf(pdf['name'], local_path)
                results.append(result)
            
            logger.info(f"Downloaded {len([r for r in results if r['success']])} PDF files")
            return results
            
        except Exception as e:
            logger.error(f"Failed to download all PDFs: {str(e)}")
            return [] 