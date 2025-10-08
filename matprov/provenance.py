"""
Provenance Tracker: Merkle tree-based experiment lineage tracking

Implements append-only ledger with cryptographic verification using Merkle trees.
"""

import json
import hashlib
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any


class ProvenanceTracker:
    """
    Tracks experiment provenance using an append-only Merkle tree ledger.
    
    Each entry in the ledger contains:
    - timestamp: When the entry was created
    - entry_type: "experiment", "link", "signature", etc.
    - content_hash: SHA-256 hash of the content
    - prev_root: Previous Merkle root
    - merkle_root: New Merkle root after adding this entry
    - metadata: Additional metadata
    """
    
    def __init__(self, matprov_dir: Path):
        """Initialize provenance tracker."""
        self.matprov_dir = matprov_dir
        self.ledger_path = matprov_dir / "ledger.jsonl"
        self.root_path = matprov_dir / "merkle_root.txt"
        
        # Ensure directories exist
        matprov_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize ledger if doesn't exist
        if not self.ledger_path.exists():
            self.ledger_path.write_text("")
        
        # Initialize root if doesn't exist
        if not self.root_path.exists():
            self.root_path.write_text("0" * 64)  # Genesis root
    
    def _get_current_root(self) -> str:
        """Get the current Merkle root."""
        return self.root_path.read_text().strip()
    
    def _update_root(self, new_root: str):
        """Update the Merkle root file."""
        self.root_path.write_text(new_root)
    
    def _compute_entry_hash(self, entry: Dict[str, Any]) -> str:
        """Compute SHA-256 hash of an entry (excluding merkle_root)."""
        # Create deterministic representation
        entry_copy = {k: v for k, v in entry.items() if k != 'merkle_root'}
        entry_str = json.dumps(entry_copy, sort_keys=True)
        return hashlib.sha256(entry_str.encode()).hexdigest()
    
    def _compute_merkle_root(self, prev_root: str, entry_hash: str) -> str:
        """Compute new Merkle root from previous root and entry hash."""
        combined = f"{prev_root}{entry_hash}"
        return hashlib.sha256(combined.encode()).hexdigest()
    
    def _append_entry(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        """Append entry to ledger and update Merkle root."""
        # Get current root
        prev_root = self._get_current_root()
        entry['prev_root'] = prev_root
        
        # Compute entry hash
        entry_hash = self._compute_entry_hash(entry)
        
        # Compute new Merkle root
        new_root = self._compute_merkle_root(prev_root, entry_hash)
        entry['merkle_root'] = new_root
        
        # Append to ledger
        with open(self.ledger_path, 'a') as f:
            f.write(json.dumps(entry) + '\n')
        
        # Update root file
        self._update_root(new_root)
        
        return entry
    
    def add_experiment(
        self,
        experiment_id: str,
        content_hash: str,
        file_path: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Add an experiment to the provenance ledger."""
        entry = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'entry_type': 'experiment',
            'experiment_id': experiment_id,
            'content_hash': content_hash,
            'file_path': file_path,
            'metadata': metadata or {}
        }
        return self._append_entry(entry)
    
    def add_link(
        self,
        link_type: str,
        source_id: str,
        target_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Add a link between entities (e.g., prediction → experiment)."""
        entry = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'entry_type': 'link',
            'link_type': link_type,
            'source_id': source_id,
            'target_id': target_id,
            'metadata': metadata or {}
        }
        return self._append_entry(entry)
    
    def add_signature(
        self,
        entity_id: str,
        signature_type: str,
        signature_data: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Add a cryptographic signature."""
        entry = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'entry_type': 'signature',
            'entity_id': entity_id,
            'signature_type': signature_type,
            'signature_data': signature_data,
            'metadata': metadata or {}
        }
        return self._append_entry(entry)
    
    def read_ledger(self) -> List[Dict[str, Any]]:
        """Read all entries from the ledger."""
        entries = []
        if self.ledger_path.exists():
            with open(self.ledger_path, 'r') as f:
                for line in f:
                    if line.strip():
                        entries.append(json.loads(line))
        return entries
    
    def find_experiment(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """Find an experiment entry in the ledger."""
        entries = self.read_ledger()
        for entry in entries:
            if entry.get('entry_type') == 'experiment' and entry.get('experiment_id') == experiment_id:
                return entry
        return None
    
    def verify_ledger(self) -> bool:
        """Verify the integrity of the entire Merkle chain."""
        entries = self.read_ledger()
        
        if not entries:
            return True  # Empty ledger is valid
        
        prev_root = "0" * 64  # Genesis root
        
        for entry in entries:
            # Verify that prev_root matches
            if entry['prev_root'] != prev_root:
                return False
            
            # Recompute entry hash
            entry_hash = self._compute_entry_hash(entry)
            
            # Recompute Merkle root
            computed_root = self._compute_merkle_root(prev_root, entry_hash)
            
            # Verify it matches
            if computed_root != entry['merkle_root']:
                return False
            
            # Move to next entry
            prev_root = entry['merkle_root']
        
        # Verify final root matches file
        current_root = self._get_current_root()
        if prev_root != current_root:
            return False
        
        return True
    
    def get_lineage(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the full lineage tree for an experiment.
        
        Returns a dict with:
        - experiment: The experiment entry
        - predictions: List of linked predictions
        - related_experiments: List of related experiments
        - signatures: List of signatures
        """
        entries = self.read_ledger()
        
        # Find the experiment
        experiment_entry = None
        for entry in entries:
            if entry.get('entry_type') == 'experiment' and entry.get('experiment_id') == experiment_id:
                experiment_entry = entry
                break
        
        if not experiment_entry:
            return None
        
        # Find related entries
        predictions = []
        related_experiments = []
        signatures = []
        
        for entry in entries:
            if entry.get('entry_type') == 'link':
                if entry.get('target_id') == experiment_id:
                    if 'prediction' in entry.get('link_type', ''):
                        predictions.append(entry)
                    else:
                        related_experiments.append(entry)
            elif entry.get('entry_type') == 'signature' and entry.get('entity_id') == experiment_id:
                signatures.append(entry)
        
        return {
            'experiment': experiment_entry,
            'predictions': predictions,
            'related_experiments': related_experiments,
            'signatures': signatures
        }
    
    def lineage_to_tree(self, lineage: Dict[str, Any]) -> str:
        """Convert lineage to ASCII tree format."""
        lines = []
        exp = lineage['experiment']
        
        lines.append(f"🧪 Experiment: {exp['experiment_id']}")
        lines.append(f"   ├─ Hash: {exp['content_hash'][:16]}...")
        lines.append(f"   ├─ Timestamp: {exp['timestamp']}")
        lines.append(f"   └─ Merkle Root: {exp['merkle_root'][:16]}...")
        
        if lineage['predictions']:
            lines.append(f"\n🔮 Predictions ({len(lineage['predictions'])})")
            for i, pred in enumerate(lineage['predictions']):
                prefix = "└─" if i == len(lineage['predictions']) - 1 else "├─"
                lines.append(f"   {prefix} {pred['source_id']}")
        
        if lineage['signatures']:
            lines.append(f"\n🔐 Signatures ({len(lineage['signatures'])})")
            for i, sig in enumerate(lineage['signatures']):
                prefix = "└─" if i == len(lineage['signatures']) - 1 else "├─"
                lines.append(f"   {prefix} {sig['signature_type']}: {sig['signature_data'][:16]}...")
        
        if lineage['related_experiments']:
            lines.append(f"\n🔗 Related Experiments ({len(lineage['related_experiments'])})")
            for i, rel in enumerate(lineage['related_experiments']):
                prefix = "└─" if i == len(lineage['related_experiments']) - 1 else "├─"
                lines.append(f"   {prefix} {rel['target_id']}")
        
        return '\n'.join(lines)
    
    def lineage_to_dot(self, lineage: Dict[str, Any]) -> str:
        """Convert lineage to Graphviz DOT format."""
        lines = []
        lines.append('digraph lineage {')
        lines.append('  rankdir=LR;')
        lines.append('  node [shape=box];')
        
        exp_id = lineage['experiment']['experiment_id']
        lines.append(f'  "{exp_id}" [style=filled, fillcolor=lightblue];')
        
        for pred in lineage['predictions']:
            pred_id = pred['source_id']
            lines.append(f'  "{pred_id}" [shape=ellipse, fillcolor=lightgreen];')
            lines.append(f'  "{pred_id}" -> "{exp_id}" [label="{pred.get("link_type", "link")}"];')
        
        for rel in lineage['related_experiments']:
            rel_id = rel['target_id']
            lines.append(f'  "{exp_id}" -> "{rel_id}" [style=dashed];')
        
        lines.append('}')
        return '\n'.join(lines)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the provenance ledger."""
        entries = self.read_ledger()
        
        experiments = [e for e in entries if e.get('entry_type') == 'experiment']
        links = [e for e in entries if e.get('entry_type') == 'link']
        signatures = [e for e in entries if e.get('entry_type') == 'signature']
        
        return {
            'total_entries': len(entries),
            'total_experiments': len(experiments),
            'total_links': len(links),
            'total_signatures': len(signatures),
            'current_merkle_root': self._get_current_root(),
            'ledger_valid': self.verify_ledger()
        }


# Example usage
if __name__ == "__main__":
    import tempfile
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tracker = ProvenanceTracker(Path(tmpdir))
        
        # Add experiments
        exp1 = tracker.add_experiment(
            experiment_id="EXP-001",
            content_hash="abc123",
            file_path="exp1.json",
            metadata={"target": "YBCO"}
        )
        print(f"✅ Added EXP-001, root: {exp1['merkle_root'][:16]}...")
        
        exp2 = tracker.add_experiment(
            experiment_id="EXP-002",
            content_hash="def456",
            file_path="exp2.json",
            metadata={"target": "LSCO"}
        )
        print(f"✅ Added EXP-002, root: {exp2['merkle_root'][:16]}...")
        
        # Add link
        link = tracker.add_link(
            link_type="prediction→experiment",
            source_id="PRED-001",
            target_id="EXP-001",
            metadata={"model": "v1.0"}
        )
        print(f"✅ Linked PRED-001 → EXP-001, root: {link['merkle_root'][:16]}...")
        
        # Verify ledger
        is_valid = tracker.verify_ledger()
        print(f"\n🔍 Ledger verification: {'✅ VALID' if is_valid else '❌ INVALID'}")
        
        # Get lineage
        lineage = tracker.get_lineage("EXP-001")
        print(f"\n🌳 Lineage for EXP-001:")
        print(tracker.lineage_to_tree(lineage))
        
        # Statistics
        stats = tracker.get_statistics()
        print(f"\n📊 Statistics:")
        print(f"   Experiments: {stats['total_experiments']}")
        print(f"   Links: {stats['total_links']}")
        print(f"   Ledger valid: {'✅' if stats['ledger_valid'] else '❌'}")

