"""
Cloud Storage service for experiment results and artifacts.

Enhanced version with robust error handling, metadata tracking, and versioning.
"""

from google.cloud import storage
from google.cloud.exceptions import GoogleCloudError
from src.utils.settings import settings
import json
import datetime
import hashlib
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path

logger = logging.getLogger(__name__)


class CloudStorageBackend:
    """
    Cloud Storage backend for storing experiment results,  raw instrument files,
    and analysis artifacts.
    
    Features:
    - Automatic timestamping and metadata
    - SHA-256 integrity hashing
    - Versioning support
    - Structured folder organization (experiments/, raw/, analyses/)
    """
    
    def __init__(self, bucket_name: str = None):
        """
        Initialize Cloud Storage backend.
        
        Args:
            bucket_name: GCS bucket name (defaults to settings.GCS_BUCKET)
        """
        self.bucket_name = bucket_name or settings.GCS_BUCKET
        try:
            self.client = storage.Client(project=settings.PROJECT_ID)
            self.bucket = self.client.bucket(self.bucket_name)
            logger.info(f"Initialized CloudStorageBackend for bucket: {self.bucket_name}")
        except GoogleCloudError as e:
            logger.error(f"Failed to initialize Cloud Storage: {e}")
            raise
    
    def store_experiment_result(
        self, 
        experiment_id: str, 
        result: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Store an experiment result as a JSON file with metadata.
        
        Args:
            experiment_id: Unique experiment identifier
            result: Result data (must be JSON-serializable)
            metadata: Optional additional metadata (tags, user, instrument, etc.)
        
        Returns:
            Public URL or gs:// URI of the stored blob
        """
        timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
        file_name = f"experiments/{experiment_id}/{timestamp}.json"
        
        # Calculate content hash for integrity
        result_json = json.dumps(result, sort_keys=True)
        content_hash = hashlib.sha256(result_json.encode()).hexdigest()
        
        # Build full payload with metadata
        payload = {
            "experiment_id": experiment_id,
            "timestamp": timestamp,
            "content_hash": content_hash,
            "metadata": metadata or {},
            "data": result
        }
        
        try:
            blob = self.bucket.blob(file_name)
            
            # Set custom metadata
            blob.metadata = {
                "experiment_id": experiment_id,
                "content_hash": content_hash,
                "uploaded_at": timestamp
            }
            
            # Upload with content type
            blob.upload_from_string(
                json.dumps(payload, indent=2),
                content_type="application/json"
            )
            
            uri = f"gs://{self.bucket_name}/{file_name}"
            logger.info(f"Stored experiment result: {uri}")
            return uri
            
        except GoogleCloudError as e:
            logger.error(f"Failed to store experiment result: {e}")
            raise
    
    def store_raw_file(
        self,
        experiment_id: str,
        file_path: str,
        file_name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Store a raw instrument file (e.g., XRD spectrum, NMR data).
        
        Args:
            experiment_id: Associated experiment ID
            file_path: Local path to the file to upload
            file_name: Optional custom name (defaults to original filename)
            metadata: Optional metadata
        
        Returns:
            gs:// URI of the stored blob
        """
        if not file_name:
            file_name = Path(file_path).name
        
        timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
        blob_path = f"raw/{experiment_id}/{timestamp}_{file_name}"
        
        try:
            blob = self.bucket.blob(blob_path)
            
            # Set metadata
            blob.metadata = {
                "experiment_id": experiment_id,
                "original_filename": file_name,
                "uploaded_at": timestamp,
                **(metadata or {})
            }
            
            # Upload file
            blob.upload_from_filename(file_path)
            
            uri = f"gs://{self.bucket_name}/{blob_path}"
            logger.info(f"Stored raw file: {uri}")
            return uri
            
        except (GoogleCloudError, IOError) as e:
            logger.error(f"Failed to store raw file: {e}")
            raise
    
    def retrieve_result(self, experiment_id: str, timestamp: Optional[str] = None) -> Dict[str, Any]:
        """
        Retrieve an experiment result.
        
        Args:
            experiment_id: Experiment ID
            timestamp: Optional specific timestamp (ISO format). If None, gets latest.
        
        Returns:
            Parsed JSON result
        """
        try:
            if timestamp:
                blob_path = f"experiments/{experiment_id}/{timestamp}.json"
                blob = self.bucket.blob(blob_path)
                if not blob.exists():
                    raise FileNotFoundError(f"Result not found: {blob_path}")
            else:
                # Get latest result for this experiment
                prefix = f"experiments/{experiment_id}/"
                blobs = list(self.client.list_blobs(self.bucket_name, prefix=prefix))
                
                if not blobs:
                    raise FileNotFoundError(f"No results found for experiment: {experiment_id}")
                
                # Sort by name (timestamp) descending
                blob = sorted(blobs, key=lambda b: b.name, reverse=True)[0]
            
            content = blob.download_as_text()
            result = json.loads(content)
            
            logger.info(f"Retrieved result for experiment: {experiment_id}")
            return result
            
        except (GoogleCloudError, json.JSONDecodeError) as e:
            logger.error(f"Failed to retrieve result: {e}")
            raise
    
    def list_experiments(self, prefix: str = "experiments/", max_results: int = 100) -> List[Dict[str, Any]]:
        """
        List all experiments in storage.
        
        Args:
            prefix: Blob prefix to filter (default: "experiments/")
            max_results: Maximum number of results
        
        Returns:
            List of dicts with experiment metadata
        """
        try:
            blobs = self.client.list_blobs(
                self.bucket_name, 
                prefix=prefix, 
                max_results=max_results
            )
            
            experiments = []
            for blob in blobs:
                experiments.append({
                    "name": blob.name,
                    "size_bytes": blob.size,
                    "created": blob.time_created.isoformat() if blob.time_created else None,
                    "updated": blob.updated.isoformat() if blob.updated else None,
                    "metadata": blob.metadata or {},
                    "uri": f"gs://{self.bucket_name}/{blob.name}"
                })
            
            logger.info(f"Listed {len(experiments)} experiment files")
            return experiments
            
        except GoogleCloudError as e:
            logger.error(f"Failed to list experiments: {e}")
            raise
    
    def delete_result(self, experiment_id: str, timestamp: str) -> bool:
        """
        Delete a specific experiment result.
        
        Args:
            experiment_id: Experiment ID
            timestamp: ISO timestamp of the result to delete
        
        Returns:
            True if deleted successfully
        """
        blob_path = f"experiments/{experiment_id}/{timestamp}.json"
        
        try:
            blob = self.bucket.blob(blob_path)
            if blob.exists():
                blob.delete()
                logger.info(f"Deleted result: {blob_path}")
                return True
            else:
                logger.warning(f"Result not found for deletion: {blob_path}")
                return False
                
        except GoogleCloudError as e:
            logger.error(f"Failed to delete result: {e}")
            raise


# Global instance (can be imported elsewhere)
storage_backend = None

def get_storage() -> CloudStorageBackend:
    """
    Get or create global storage backend instance.
    
    Lazy initialization to avoid errors if GCS is not configured.
    """
    global storage_backend
    if storage_backend is None:
        try:
            storage_backend = CloudStorageBackend()
        except Exception as e:
            logger.warning(f"Cloud Storage not available: {e}")
            return None
    return storage_backend
