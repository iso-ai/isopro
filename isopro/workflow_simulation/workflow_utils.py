"""
Workflow Utilities Module

Provides utility functions for UI element detection and motion tracking.
"""

import cv2
import numpy as np
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from ultralytics import YOLO
import logging

logger = logging.getLogger(__name__)

@dataclass
class UIElement:
    """Represents a detected UI element."""
    type: str
    bbox: List[float]  # [x1, y1, x2, y2]
    confidence: float
    id: str = ''
    is_interactive: bool = True

class UIElementDetector:
    """Detects UI elements in frames using YOLO."""
    
    def __init__(self, model_path: str = 'yolov8x.pt'):
        """Initialize the UI element detector."""
        self.model = YOLO(model_path)
        
        # Common UI element classes
        self.ui_classes = [
            'button',
            'text',
            'checkbox',
            'dropdown',
            'input',
            'icon',
            'menu',
            'window',
            'link'
        ]
        
        logger.info("Initialized UI element detector")
        
    def detect_elements(self, frame: np.ndarray) -> List[UIElement]:
        """Detect UI elements in a frame."""
        results = self.model(frame)
        detected_elements = []
        
        for result in results:
            boxes = result.boxes
            for i, box in enumerate(boxes):
                if box.conf[0].item() > 0.5:  # Confidence threshold
                    element = UIElement(
                        type=self.ui_classes[int(box.cls[0].item())],
                        bbox=box.xyxy[0].tolist(),
                        confidence=box.conf[0].item(),
                        id=f"{self.ui_classes[int(box.cls[0].item())]}-{i}"
                    )
                    detected_elements.append(element)
        
        return detected_elements

class MotionDetector:
    """Detects motion between frames."""
    
    def __init__(self, min_area: int = 500):
        """Initialize motion detector."""
        self.min_area = min_area
        self.prev_frame = None
        
    def detect_motion(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Detect motion in frame."""
        # Convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        
        # Initialize previous frame if needed
        if self.prev_frame is None:
            self.prev_frame = gray
            return []
        
        # Calculate frame difference
        frame_diff = cv2.absdiff(self.prev_frame, gray)
        thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(
            thresh.copy(),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Process motion areas
        motion_areas = []
        for contour in contours:
            if cv2.contourArea(contour) > self.min_area:
                x, y, w, h = cv2.boundingRect(contour)
                motion_areas.append({
                    'bbox': [x, y, x + w, y + h],
                    'center': (x + w//2, y + h//2)
                })
        
        self.prev_frame = gray
        return motion_areas

def analyze_interaction(
    ui_elements: List[UIElement],
    motion_areas: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Analyze interaction between motion and UI elements."""
    if not motion_areas or not ui_elements:
        return None
    
    for motion in motion_areas:
        motion_bbox = motion['bbox']
        
        for element in ui_elements:
            if check_overlap(motion_bbox, element.bbox):
                # Determine interaction type
                action_type = classify_interaction(motion_bbox)
                
                return {
                    'action_type': action_type,
                    'element': element,
                    'motion_area': motion
                }
    
    return None

def check_overlap(bbox1: List[float], bbox2: List[float]) -> bool:
    """Check if two bounding boxes overlap."""
    x1_min, y1_min, x1_max, y1_max = bbox1
    x2_min, y2_min, x2_max, y2_max = bbox2
    
    return not (
        x1_max < x2_min or
        x1_min > x2_max or
        y1_max < y2_min or
        y1_min > y2_max
    )

def classify_interaction(motion_bbox: List[float]) -> str:
    """Classify the type of interaction based on motion pattern."""
    width = motion_bbox[2] - motion_bbox[0]
    height = motion_bbox[3] - motion_bbox[1]
    
    if width < 10 and height < 10:
        return 'click'
    elif width > 50 or height > 50:
        return 'drag'
    else:
        return 'double_click'

def estimate_cursor_position(motion_areas: List[Dict[str, Any]]) -> Tuple[float, float]:
    """Estimate cursor position from motion areas."""
    if not motion_areas:
        return (0.0, 0.0)
    
    # Use center of the most recent motion area
    return motion_areas[-1]['center']

# Example usage
def process_frame(frame: np.ndarray) -> Dict[str, Any]:
    """Process a single frame for UI interactions."""
    # Initialize detectors
    ui_detector = UIElementDetector()
    motion_detector = MotionDetector()
    
    # Detect UI elements
    ui_elements = ui_detector.detect_elements(frame)
    
    # Detect motion
    motion_areas = motion_detector.detect_motion(frame)
    
    # Analyze interactions
    interaction = analyze_interaction(ui_elements, motion_areas)
    
    # Estimate cursor position
    cursor_pos = estimate_cursor_position(motion_areas)
    
    return {
        'ui_elements': ui_elements,
        'motion_areas': motion_areas,
        'interaction': interaction,
        'cursor_position': cursor_pos
    }