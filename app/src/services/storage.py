"""
Cloud Storage service for experiment results.
"""

from google.cloud import storage
from datetime import datetime
import json
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class CloudStorageBackend:
    """
    Store experiment results in Cloud Storage.
    
    Moat: DATA - Provenance tracking with metadata.
    """
    
    def __init__(self, bucket_name: str):
        self.bucket_name = bucket_name
        self.client = storage.Client()
        self.bucket = self.client.bucket(bucket_name)
        
        logger.info(f"CloudStorageBackend initialized: bucket={bucket_name}")
    
    def store_result(
        self, 
        experiment_id: str, 
        result: Dict[str, Any],
        metadata: Optional[Dict[str, str]] = None
    ) -> str:
        """
        Store experiment result with provenance.
        
        Args:
            experiment_id: Unique experiment identifier
            result: Result data (will be JSON serialized)
            metadata: Optional metadata tags
        
        Returns:
            GCS URI of stored object
        """
        try:
            # Construct blob path
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            blob_name = f"experiments/{experiment_id}/{timestamp}_result.json"
            
            blob = self.bucket.blob(blob_name)
            
            # Upload content
            blob.upload_from_string(
                json.dumps(result, indent=2),
                content_type="application/json"
            )
            
            # Add metadata
            if metadata is None:
                metadata = {}
            
            metadata.update({
                "experiment_id": experiment_id,
                "timestamp": datetime.utcnow().isoformat(),
                "content_type": "experiment_result"
            })
            
            blob.metadata = metadata
            blob.patch()
            
            uri = f"gs://{self.bucket_name}/{blob_name}"
            logger.info(f"Result stored: {uri}")
            
            return uri
        
        except Exception as e:
            logger.error(f"Failed to store result: {e}")
            raise
    
    def retrieve_result(self, experiment_id: str, timestamp: str = None) -> Dict[str, Any]:
        """
        Retrieve experiment result.
        
        Args:
            experiment_id: Experiment identifier
            timestamp: Optional specific timestamp (defaults to latest)
        
        Returns:
            Result data
        """
        try:
            if timestamp:
                blob_name = f"experiments/{experiment_id}/{timestamp}_result.json"
            else:
                # List all results for this experiment and get latest
                prefix = f"experiments/{experiment_id}/"
                blobs = list(self.bucket.list_blobs(prefix=prefix))
                
                if not blobs:
                    raise FileNotFoundError(f"No results found for {experiment_id}")
                
                # Sort by name (timestamp is in name)
                blobs.sort(key=lambda b: b.name, reverse=True)
                blob_name = blobs[0].name
            
            blob = self.bucket.blob(blob_name)
            content = blob.download_as_text()
            
            return json.loads(content)
        
        except Exception as e:
            logger.error(f"Failed to retrieve result: {e}")
            raise
    
    def list_experiments(self, limit: int = 100) -> list:
        """
        List recent experiments.
        
        Args:
            limit: Maximum number of experiments to return
        
        Returns:
            List of experiment IDs
        """
        try:
            prefix = "experiments/"
            blobs = self.bucket.list_blobs(prefix=prefix, delimiter="/")
            
            # Extract unique experiment IDs from prefixes
            experiment_ids = set()
            for prefix in blobs.prefixes:
                exp_id = prefix.replace("experiments/", "").rstrip("/")
                experiment_ids.add(exp_id)
            
            return sorted(list(experiment_ids))[:limit]
        
        except Exception as e:
            logger.error(f"Failed to list experiments: {e}")
            raise

