"""
UCI Superconductivity Dataset Loader

Loads the UCI Superconductivity dataset (21,263 compounds, 81 features).
This is the validated baseline dataset used throughout Phase 10.

Dataset: https://archive.ics.uci.edu/dataset/464/superconductivty+data
License: CC BY 4.0

Author: GOATnote Autonomous Research Lab Initiative
Contact: b@thegoatnote.com
License: MIT
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from sklearn.model_selection import train_test_split
from typing import Tuple

logger = logging.getLogger(__name__)


def load_uci_superconductor(
    cache_dir: Path = Path('data/raw/uci'),
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load UCI Superconductivity dataset.
    
    Args:
        cache_dir: Directory to cache downloaded data
        random_state: Random seed for reproducibility
    
    Returns:
        Tuple of (train_df, val_df, test_df)
        Each DataFrame has 81 features + 'Tc' target column
    """
    logger.info("📂 Loading UCI Superconductivity dataset...")
    
    try:
        from ucimlrepo import fetch_ucirepo
        
        # Fetch dataset
        cache_dir.mkdir(parents=True, exist_ok=True)
        superconductivity = fetch_ucirepo(id=464)
        
        # Extract features and targets
        X = superconductivity.data.features
        y = superconductivity.data.targets
        
        # Combine into DataFrame
        df = X.copy()
        df['Tc'] = y.values.squeeze() if len(y.shape) > 1 else y.values
        
        logger.info(f"✅ Loaded {len(df)} compounds with {len(X.columns)} features")
        
        # Split: 70% train, 15% val, 15% test (stratified by Tc quartiles)
        df['Tc_bin'] = pd.qcut(df['Tc'], q=4, labels=False, duplicates='drop')
        
        train_df, temp_df = train_test_split(
            df, test_size=0.3, random_state=random_state, stratify=df['Tc_bin']
        )
        val_df, test_df = train_test_split(
            temp_df, test_size=0.5, random_state=random_state, stratify=temp_df['Tc_bin']
        )
        
        # Drop stratification column
        train_df = train_df.drop(columns=['Tc_bin'])
        val_df = val_df.drop(columns=['Tc_bin'])
        test_df = test_df.drop(columns=['Tc_bin'])
        
        logger.info(f"✅ Split: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
        
        return train_df, val_df, test_df
        
    except ImportError:
        logger.error("❌ Missing ucimlrepo package. Install: pip install ucimlrepo")
        raise
    
    except Exception as e:
        logger.error(f"❌ Failed to load UCI dataset: {e}")
        raise


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test loader
    train_df, val_df, test_df = load_uci_superconductor()
    
    print(f"\n✅ UCI Loader Test:")
    print(f"   Train: {train_df.shape}")
    print(f"   Val: {val_df.shape}")
    print(f"   Test: {test_df.shape}")
    print(f"   Tc range: [{train_df['Tc'].min():.1f}, {train_df['Tc'].max():.1f}] K")
    print(f"✅ Test passed!")

