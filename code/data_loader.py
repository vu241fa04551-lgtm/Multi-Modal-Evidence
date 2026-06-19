"""
Data loader for claims processing.
Handles loading CSV files, images, and user history.
"""

import os
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from PIL import Image
import io


class DataLoader:
    """Loads and manages claim data, images, and user history."""
    
    def __init__(self, base_path: str):
        """
        Initialize data loader.
        
        Args:
            base_path: Base directory containing dataset/
        """
        self.base_path = Path(base_path)
        self.dataset_path = self.base_path / "dataset"
        
        # Load all CSV files
        self.sample_claims_df = None
        self.claims_df = None
        self.user_history_df = None
        self.evidence_requirements_df = None
        
    def load_all_data(self) -> bool:
        """
        Load all required data files.
        
        Returns:
            True if all files loaded successfully
        """
        try:
            # Load sample claims if exists
            sample_path = self.dataset_path / "sample_claims.csv"
            if sample_path.exists():
                self.sample_claims_df = pd.read_csv(sample_path)
                print(f"✓ Loaded sample_claims: {len(self.sample_claims_df)} rows")
            
            # Load test claims
            claims_path = self.dataset_path / "claims.csv"
            if claims_path.exists():
                self.claims_df = pd.read_csv(claims_path)
                print(f"✓ Loaded claims: {len(self.claims_df)} rows")
            
            # Load user history
            history_path = self.dataset_path / "user_history.csv"
            if history_path.exists():
                self.user_history_df = pd.read_csv(history_path)
                self.user_history_df.set_index('user_id', inplace=True)
                print(f"✓ Loaded user_history: {len(self.user_history_df)} users")
            
            # Load evidence requirements
            evidence_path = self.dataset_path / "evidence_requirements.csv"
            if evidence_path.exists():
                self.evidence_requirements_df = pd.read_csv(evidence_path)
                print(f"✓ Loaded evidence_requirements: {len(self.evidence_requirements_df)} rows")
            
            return True
        except Exception as e:
            print(f"✗ Error loading data: {e}")
            return False
    
    def load_image(self, image_path: str) -> Optional[Image.Image]:
        """
        Load a single image.
        
        Args:
            image_path: Relative path to image from dataset/
            
        Returns:
            PIL Image object or None if failed
        """
        try:
            full_path = self.dataset_path / image_path
            if full_path.exists():
                return Image.open(full_path)
            else:
                print(f"✗ Image not found: {image_path}")
                return None
        except Exception as e:
            print(f"✗ Error loading image {image_path}: {e}")
            return None
    
    def load_images_for_claim(self, image_paths_str: str) -> Dict[str, Optional[Image.Image]]:
        """
        Load all images for a claim.
        
        Args:
            image_paths_str: Semicolon-separated image paths
            
        Returns:
            Dictionary mapping image_id -> Image object
        """
        images = {}
        if pd.isna(image_paths_str):
            return images
            
        paths = str(image_paths_str).split(';')
        for path in paths:
            path = path.strip()
            if path:
                # Extract image ID (filename without extension)
                image_id = Path(path).stem
                img = self.load_image(path)
                if img:
                    images[image_id] = img
        
        return images
    
    def get_user_history(self, user_id: str) -> Optional[Dict]:
        """
        Get user history for risk assessment.
        
        Args:
            user_id: User ID to look up
            
        Returns:
            Dictionary of user history or None
        """
        if self.user_history_df is not None:
            try:
                return self.user_history_df.loc[user_id].to_dict()
            except KeyError:
                return None
        return None
    
    def get_evidence_requirements(self, claim_object: str) -> List[Dict]:
        """
        Get evidence requirements for claim object type.
        
        Args:
            claim_object: 'car', 'laptop', or 'package'
            
        Returns:
            List of requirement dictionaries
        """
        if self.evidence_requirements_df is None:
            return []
        
        # Get requirements for this object type or 'all'
        reqs = self.evidence_requirements_df[
            (self.evidence_requirements_df['claim_object'] == claim_object) |
            (self.evidence_requirements_df['claim_object'] == 'all')
        ]
        
        return reqs.to_dict('records')
    
    def validate_paths(self) -> bool:
        """Validate that required directories exist."""
        required = [self.dataset_path]
        for path in required:
            if not path.exists():
                print(f"✗ Missing directory: {path}")
                return False
        return True
