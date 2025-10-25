"""
Cloud Storage service for experiment results and artifacts.

Enhanced version with robust error handling, metadata tracking, and versioning.
"""

import datetime
import hashlib
import json
import logging
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

from google.cloud import storage
from google.cloud.exceptions import GoogleCloudError

from src.utils.settings import settings

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


class LocalStorageBackend:
    """Filesystem backed storage for offline simulation runs."""

    def __init__(self, root: Optional[Path] = None) -> None:
        self.root = Path(root or settings.LOCAL_STORAGE_PATH)
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "experiments").mkdir(exist_ok=True)
        (self.root / "raw").mkdir(exist_ok=True)

    def store_experiment_result(
        self,
        experiment_id: str,
        result: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
        exp_dir = self.root / "experiments" / experiment_id
        exp_dir.mkdir(parents=True, exist_ok=True)
        file_path = exp_dir / f"{timestamp}.json"

        payload = {
            "experiment_id": experiment_id,
            "timestamp": timestamp,
            "metadata": metadata or {},
            "data": result,
        }

        file_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return str(file_path)

    def store_raw_file(
        self,
        experiment_id: str,
        file_path: str,
        file_name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        destination_name = file_name or Path(file_path).name
        dest_dir = self.root / "raw" / experiment_id
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_path = dest_dir / destination_name
        shutil.copy2(file_path, dest_path)
        return str(dest_path)

    def retrieve_result(self, experiment_id: str, timestamp: Optional[str] = None) -> Dict[str, Any]:
        exp_dir = self.root / "experiments" / experiment_id
        if not exp_dir.exists():
            raise FileNotFoundError(f"No results for experiment {experiment_id}")

        if timestamp:
            file_path = exp_dir / f"{timestamp}.json"
        else:
            files = sorted(exp_dir.glob("*.json"), reverse=True)
            if not files:
                raise FileNotFoundError(f"No result files for experiment {experiment_id}")
            file_path = files[0]

        return json.loads(file_path.read_text(encoding="utf-8"))

    def list_experiments(self, prefix: str = "experiments/", max_results: int = 100) -> List[Dict[str, Any]]:
        if not prefix.startswith("experiments"):
            return []
        exp_dir = self.root / "experiments"
        experiments: List[Dict[str, Any]] = []
        for experiment_folder in sorted(exp_dir.iterdir()):
            if not experiment_folder.is_dir():
                continue
            for file_path in sorted(experiment_folder.glob("*.json"), reverse=True)[:max_results]:
                stat = file_path.stat()
                experiments.append(
                    {
                        "name": f"{experiment_folder.name}/{file_path.name}",
                        "size_bytes": stat.st_size,
                        "created": datetime.datetime.fromtimestamp(stat.st_ctime, tz=datetime.timezone.utc).isoformat(),
                        "updated": datetime.datetime.fromtimestamp(stat.st_mtime, tz=datetime.timezone.utc).isoformat(),
                        "metadata": {},
                        "uri": str(file_path),
                    }
                )
        return experiments[:max_results]

    def delete_result(self, experiment_id: str, timestamp: str) -> bool:
        file_path = self.root / "experiments" / experiment_id / f"{timestamp}.json"
        if file_path.exists():
            file_path.unlink()
            return True
        return False
    
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
        if settings.GCS_BUCKET:
            try:
                storage_backend = CloudStorageBackend()
            except Exception as exc:
                logger.warning(f"Falling back to local storage: {exc}")
                storage_backend = LocalStorageBackend()
        else:
            logger.info("GCS bucket not configured - using local filesystem storage")
            storage_backend = LocalStorageBackend()
    return storage_backend
