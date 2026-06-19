"""
Output generator that formats analysis results into required CSV format.
"""

import pandas as pd
from typing import Dict, List
from pathlib import Path


class OutputGenerator:
    """Generates output CSV in required format."""
    
    # Allowed values from problem statement
    ALLOWED_ISSUE_TYPES = {
        'dent', 'scratch', 'crack', 'glass_shatter', 'broken_part',
        'missing_part', 'torn_packaging', 'crushed_packaging',
        'water_damage', 'stain', 'none', 'unknown'
    }
    
    ALLOWED_OBJECT_PARTS = {
        'car': {'front_bumper', 'rear_bumper', 'door', 'hood', 'windshield',
                'side_mirror', 'headlight', 'taillight', 'fender', 'quarter_panel',
                'body', 'unknown'},
        'laptop': {'screen', 'keyboard', 'trackpad', 'hinge', 'lid', 'corner',
                  'port', 'base', 'body', 'unknown'},
        'package': {'box', 'package_corner', 'package_side', 'seal', 'label',
                   'contents', 'item', 'unknown'},
    }
    
    ALLOWED_CLAIM_STATUS = {'supported', 'contradicted', 'not_enough_information'}
    
    ALLOWED_RISK_FLAGS = {
        'none', 'blurry_image', 'cropped_or_obstructed', 'low_light_or_glare',
        'wrong_angle', 'wrong_object', 'wrong_object_part', 'damage_not_visible',
        'claim_mismatch', 'possible_manipulation', 'non_original_image',
        'text_instruction_present', 'user_history_risk', 'manual_review_required'
    }
    
    ALLOWED_SEVERITY = {'none', 'low', 'medium', 'high', 'unknown'}
    
    def __init__(self):
        """Initialize output generator."""
        self.results = []
    
    def add_result(self, row_data: Dict, analysis_result: Dict) -> Dict:
        """
        Format a single analysis result for output.
        
        Args:
            row_data: Original row from claims.csv
            analysis_result: Analysis result from ClaimAnalyzer
            
        Returns:
            Formatted output row
        """
        
        decision = analysis_result.get("decision", {})
        claim_obj = analysis_result.get("claim_extraction", {})
        img_analysis = analysis_result.get("image_analysis", {})
        evidence = analysis_result.get("evidence_assessment", {})
        
        # Extract values
        user_id = row_data.get("user_id", "")
        image_paths = row_data.get("image_paths", "")
        user_claim = row_data.get("user_claim", "")
        claim_object = row_data.get("claim_object", "")
        
        # Evidence standard
        evidence_standard_met = evidence.get("evidence_sufficient", False)
        evidence_reason = evidence.get("evidence_reason", "Unknown")
        
        # Risk flags
        risk_flags = decision.get("risk_flags", ["none"])
        risk_flags_str = ";".join([self._validate_risk_flag(f) for f in risk_flags])
        
        # Issue type - map from claimed issue
        issue_type = self._infer_issue_type(
            claim_obj.get("main_issue", "unknown"),
            img_analysis.get("visible_issues", [])
        )
        
        # Object part
        object_part = self._infer_object_part(
            claim_object,
            claim_obj.get("affected_part", "unknown")
        )
        
        # Claim status and justification
        claim_status = self._validate_claim_status(decision.get("claim_status", "not_enough_information"))
        claim_justification = decision.get("claim_status_justification", "No justification available")
        
        # Supporting image IDs
        supporting_ids = decision.get("supporting_image_ids", [])
        supporting_ids_str = ";".join(supporting_ids) if supporting_ids else "none"
        
        # Valid image
        valid_image = True
        quality_issues = evidence.get("quality_issues", [])
        if any(q in quality_issues for q in ["wrong_object", "wrong_object_part"]):
            valid_image = False
        
        # Severity
        severity = self._validate_severity(decision.get("severity", "unknown"))
        
        output_row = {
            "user_id": user_id,
            "image_paths": image_paths,
            "user_claim": user_claim,
            "claim_object": claim_object,
            "evidence_standard_met": "true" if evidence_standard_met else "false",
            "evidence_standard_met_reason": evidence_reason,
            "risk_flags": risk_flags_str,
            "issue_type": issue_type,
            "object_part": object_part,
            "claim_status": claim_status,
            "claim_status_justification": claim_justification,
            "supporting_image_ids": supporting_ids_str,
            "valid_image": "true" if valid_image else "false",
            "severity": severity
        }
        
        self.results.append(output_row)
        return output_row
    
    def _validate_risk_flag(self, flag: str) -> str:
        """Validate and normalize risk flag."""
        flag = flag.strip().lower()
        if flag in self.ALLOWED_RISK_FLAGS:
            return flag
        # Try to map common variants
        mapping = {
            "blurry": "blurry_image",
            "cropped": "cropped_or_obstructed",
            "obstructed": "cropped_or_obstructed",
            "low_light": "low_light_or_glare",
            "glare": "low_light_or_glare",
            "angle": "wrong_angle",
            "wrong_obj": "wrong_object",
            "manipulation": "possible_manipulation",
        }
        for key, val in mapping.items():
            if key in flag:
                return val
        return "manual_review_required"
    
    def _infer_issue_type(self, claimed_issue: str, visible_issues: List[str]) -> str:
        """Infer issue type from claim and visible damage."""
        claimed = claimed_issue.lower()
        
        # Direct matches
        if claimed in self.ALLOWED_ISSUE_TYPES:
            return claimed
        
        # Try to find in visible issues
        for issue in visible_issues:
            issue_lower = issue.lower()
            for allowed in self.ALLOWED_ISSUE_TYPES:
                if allowed in issue_lower or issue_lower in allowed:
                    return allowed
        
        # Fuzzy matching
        mapping = {
            "dent": ["dent", "bent", "indentation"],
            "scratch": ["scratch", "scrape", "scuff"],
            "crack": ["crack", "fractured", "broken"],
            "glass_shatter": ["glass", "shatter", "broken glass"],
            "broken_part": ["broken", "snapped", "split"],
            "missing_part": ["missing", "absent"],
            "torn_packaging": ["torn", "ripped"],
            "crushed_packaging": ["crushed", "smashed"],
            "water_damage": ["water", "wet", "moisture"],
            "stain": ["stain", "mark", "discoloration"]
        }
        
        for issue_type, keywords in mapping.items():
            for keyword in keywords:
                if keyword in claimed:
                    return issue_type
        
        return "unknown"
    
    def _infer_object_part(self, claim_object: str, affected_part: str) -> str:
        """Infer object part from claim object and affected part."""
        part_lower = affected_part.lower()
        
        if claim_object not in self.ALLOWED_OBJECT_PARTS:
            return "unknown"
        
        allowed_parts = self.ALLOWED_OBJECT_PARTS[claim_object]
        
        # Direct match
        if part_lower in allowed_parts:
            return part_lower
        
        # Fuzzy match
        for allowed in allowed_parts:
            if allowed in part_lower or part_lower in allowed:
                return allowed
        
        # Try substring matching
        for allowed in allowed_parts:
            if allowed in part_lower:
                return allowed
        
        return "unknown"
    
    def _validate_claim_status(self, status: str) -> str:
        """Validate claim status."""
        status = status.lower().strip()
        if status in self.ALLOWED_CLAIM_STATUS:
            return status
        return "not_enough_information"
    
    def _validate_severity(self, severity: str) -> str:
        """Validate severity value."""
        severity = severity.lower().strip()
        if severity in self.ALLOWED_SEVERITY:
            return severity
        return "unknown"
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert results to DataFrame."""
        return pd.DataFrame(self.results)
    
    def save_csv(self, output_path: str):
        """
        Save results to CSV file.
        
        Args:
            output_path: Path to save output.csv
        """
        df = self.to_dataframe()
        
        # Ensure column order
        column_order = [
            "user_id", "image_paths", "user_claim", "claim_object",
            "evidence_standard_met", "evidence_standard_met_reason",
            "risk_flags", "issue_type", "object_part",
            "claim_status", "claim_status_justification",
            "supporting_image_ids", "valid_image", "severity"
        ]
        
        df = df[column_order]
        df.to_csv(output_path, index=False)
        print(f"✓ Saved output to {output_path}")
