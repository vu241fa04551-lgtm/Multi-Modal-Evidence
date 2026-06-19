"""
Claim analyzer using Google Gemini Vision API.
Analyzes images and extracted claims to determine support level.
"""

import json
import time
from typing import Dict, List, Optional, Tuple
from PIL import Image
import google.generativeai as genai
from datetime import datetime


class ClaimAnalyzer:
    """Analyzes claims using vision and language models."""
    
    def __init__(self, api_key: str):
        """
        Initialize the claim analyzer with Gemini API.
        
        Args:
            api_key: Google GenAI API key
        """
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash')
        self.vision_model = genai.GenerativeModel('gemini-2.0-flash')
        self.call_count = 0
        self.token_usage = {"input": 0, "output": 0}
        
    def analyze_claim(
        self,
        user_claim: str,
        images: Dict[str, Image.Image],
        claim_object: str,
        user_history: Optional[Dict] = None,
        evidence_requirements: Optional[List[Dict]] = None
    ) -> Dict:
        """
        Analyze a single claim with images.
        
        Args:
            user_claim: Conversation/claim text from user
            images: Dict mapping image_id -> Image object
            claim_object: 'car', 'laptop', or 'package'
            user_history: User history dict for risk context
            evidence_requirements: List of evidence requirement dicts
            
        Returns:
            Analysis result dictionary
        """
        
        # Step 1: Extract claim details
        claim_extraction = self._extract_claim_details(user_claim, claim_object)
        
        # Step 2: Analyze images
        image_analysis = self._analyze_images(images, claim_object)
        
        # Step 3: Assess evidence
        evidence_assessment = self._assess_evidence(
            claim_extraction,
            image_analysis,
            claim_object,
            evidence_requirements
        )
        
        # Step 4: Make decision
        decision = self._make_decision(
            claim_extraction,
            image_analysis,
            evidence_assessment,
            user_history
        )
        
        return {
            "claim_extraction": claim_extraction,
            "image_analysis": image_analysis,
            "evidence_assessment": evidence_assessment,
            "decision": decision
        }
    
    def _extract_claim_details(self, user_claim: str, claim_object: str) -> Dict:
        """Extract structured claim details from conversation."""
        
        prompt = f"""Analyze this user claim about a {claim_object} damage:

"{user_claim}"

Extract and return a JSON with:
- main_issue: Primary damage type mentioned (e.g., "dent", "crack", "screen damage")
- affected_part: Which part is affected (be specific)
- severity_claimed: How severe does the user claim it is? (none/low/medium/high/unknown)
- is_damage_claim: Boolean - is this actually claiming damage exists?
- additional_context: Any other relevant details

Return only valid JSON, no other text."""

        try:
            response = self.vision_model.generate_content(prompt)
            self.call_count += 1
            if response.usage_metadata:
                self.token_usage["input"] += response.usage_metadata.prompt_token_count
                self.token_usage["output"] += response.usage_metadata.candidates_token_count
            
            result = json.loads(response.text)
            return result
        except Exception as e:
            print(f"✗ Error extracting claim: {e}")
            return {
                "main_issue": "unknown",
                "affected_part": "unknown",
                "severity_claimed": "unknown",
                "is_damage_claim": False,
                "additional_context": f"Error: {str(e)}"
            }
    
    def _analyze_images(self, images: Dict[str, Image.Image], claim_object: str) -> Dict:
        """Analyze images using vision model."""
        
        if not images:
            return {
                "image_count": 0,
                "image_analyses": {},
                "overall_quality": "no_images",
                "visible_issues": []
            }
        
        analyses = {}
        visible_issues = []
        quality_flags = []
        
        for image_id, img in images.items():
            analysis = self._analyze_single_image(img, claim_object, image_id)
            analyses[image_id] = analysis
            
            # Collect quality flags
            if analysis.get("quality_issues"):
                quality_flags.extend(analysis["quality_issues"])
            
            # Collect visible issues
            if analysis.get("visible_damage"):
                visible_issues.extend(analysis.get("visible_damage", []))
        
        return {
            "image_count": len(images),
            "image_analyses": analyses,
            "quality_issues": list(set(quality_flags)),
            "visible_issues": visible_issues
        }
    
    def _analyze_single_image(self, img: Image.Image, claim_object: str, image_id: str) -> Dict:
        """Analyze a single image."""
        
        prompt = f"""Analyze this image of a {claim_object}.

For the {claim_object} shown:
1. What visible damage or issues can you see? List specific damage types and locations.
2. What is the condition of the object? Is it in good condition, or is there damage?
3. What parts can you clearly see?
4. Are there any quality issues with the image itself? (blurry, cropped, low light, glare, wrong angle, etc.)
5. Does the image clearly show a {claim_object}? (Or is it potentially a different object?)
6. Rate the visibility of any damage on a scale: clear, partially_visible, not_visible
7. Any signs of manipulation or non-original image?

Return JSON with:
- visible_damage: list of damage types observed with location
- condition: "intact", "damaged", "unknown"
- visible_parts: list of clearly visible parts
- quality_issues: list of image quality problems (empty if none)
- object_match: true/false - is this clearly the right object type?
- damage_visibility: "clear", "partial", "not_visible"
- authenticity_concerns: "none", "suspected_manipulation", "non_original"
- severity_observed: "none", "low", "medium", "high", "unknown"

Return only valid JSON."""

        try:
            response = self.vision_model.generate_content([prompt, img])
            self.call_count += 1
            if response.usage_metadata:
                self.token_usage["input"] += response.usage_metadata.prompt_token_count
                self.token_usage["output"] += response.usage_metadata.candidates_token_count
            
            result = json.loads(response.text)
            result["image_id"] = image_id
            return result
        except Exception as e:
            print(f"✗ Error analyzing image {image_id}: {e}")
            return {
                "image_id": image_id,
                "visible_damage": [],
                "condition": "unknown",
                "visible_parts": [],
                "quality_issues": ["analysis_error"],
                "object_match": False,
                "damage_visibility": "unknown",
                "authenticity_concerns": "none",
                "severity_observed": "unknown",
                "error": str(e)
            }
    
    def _assess_evidence(
        self,
        claim_extraction: Dict,
        image_analysis: Dict,
        claim_object: str,
        evidence_requirements: Optional[List[Dict]] = None
    ) -> Dict:
        """Assess whether image evidence meets requirements."""
        
        image_count = image_analysis.get("image_count", 0)
        quality_issues = image_analysis.get("quality_issues", [])
        
        # Determine if evidence is sufficient
        has_critical_quality_issues = any(
            issue in quality_issues
            for issue in ["blurry_image", "cropped_or_obstructed", "wrong_object"]
        )
        
        sufficient = (
            image_count > 0 and 
            not has_critical_quality_issues and
            image_analysis.get("damage_visibility") != "not_visible"
        )
        
        # Build reason
        if image_count == 0:
            reason = "No images provided"
        elif has_critical_quality_issues:
            reason = f"Critical quality issues: {', '.join(quality_issues)}"
        elif image_analysis.get("damage_visibility") == "not_visible":
            reason = "No visible damage in provided images"
        else:
            reason = "Images sufficient for evaluation"
        
        return {
            "evidence_sufficient": sufficient,
            "evidence_reason": reason,
            "image_count": image_count,
            "quality_issues": quality_issues
        }
    
    def _make_decision(
        self,
        claim_extraction: Dict,
        image_analysis: Dict,
        evidence_assessment: Dict,
        user_history: Optional[Dict] = None
    ) -> Dict:
        """Make final claim decision."""
        
        claim_status = "not_enough_information"
        justification = ""
        supporting_images = []
        risk_flags = []
        
        # Check if evidence is sufficient
        if not evidence_assessment.get("evidence_sufficient"):
            claim_status = "not_enough_information"
            justification = evidence_assessment.get("evidence_reason", "Insufficient evidence")
        else:
            # Compare claim with visible damage
            visible_damage = image_analysis.get("visible_issues", [])
            claimed_issue = claim_extraction.get("main_issue", "")
            damage_visibility = image_analysis.get("damage_visibility", "unknown")
            
            # Get images with best visibility
            for img_id, img_analysis in image_analysis.get("image_analyses", {}).items():
                if img_analysis.get("damage_visibility") in ["clear", "partial"]:
                    supporting_images.append(img_id)
            
            if damage_visibility == "clear" and visible_damage:
                claim_status = "supported"
                justification = f"Visible damage matches claim: {', '.join(visible_damage)}"
            elif damage_visibility == "partial" and visible_damage:
                claim_status = "supported"
                justification = f"Damage partially visible: {', '.join(visible_damage)}"
            elif damage_visibility == "not_visible":
                claim_status = "contradicted"
                justification = "No visible damage despite claim of damage"
            else:
                claim_status = "not_enough_information"
                justification = "Unable to determine from available images"
        
        # Assess risk flags
        quality_issues = evidence_assessment.get("quality_issues", [])
        for issue in quality_issues:
            if issue in ["blurry_image", "cropped_or_obstructed", "low_light_or_glare"]:
                risk_flags.append(issue)
        
        # Check user history
        if user_history:
            history_flags = user_history.get("history_flags", "")
            if history_flags and history_flags != "none":
                risk_flags.append("user_history_risk")
        
        if image_analysis.get("image_analyses"):
            for img_analysis in image_analysis["image_analyses"].values():
                if img_analysis.get("authenticity_concerns") != "none":
                    risk_flags.append("possible_manipulation")
                    break
        
        return {
            "claim_status": claim_status,
            "claim_status_justification": justification,
            "supporting_image_ids": supporting_images,
            "risk_flags": risk_flags if risk_flags else ["none"],
            "severity": image_analysis.get("image_analyses", {})
                        .get(list(image_analysis.get("image_analyses", {}).keys())[0], {})
                        .get("severity_observed", "unknown") if image_analysis.get("image_analyses") else "unknown"
        }
    
    def get_stats(self) -> Dict:
        """Get API usage statistics."""
        return {
            "total_calls": self.call_count,
            "input_tokens": self.token_usage["input"],
            "output_tokens": self.token_usage["output"],
            "total_tokens": self.token_usage["input"] + self.token_usage["output"]
        }
