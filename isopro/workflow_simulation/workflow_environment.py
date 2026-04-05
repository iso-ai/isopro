"""
WorkflowEnvironment Module

Provides a gymnasium-compatible environment for learning and replicating UI workflows
from video demonstrations. Handles video analysis, UI element detection, and state management.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cv2
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import logging
from datetime import datetime
import os

logger = logging.getLogger(__name__)

@dataclass
class UIElement:
    """Represents a detected UI element with its properties."""
    id: str
    type: str 
    bbox: List[float]
    confidence: float
    state: str = 'default'
    enabled: bool = True
    visible: bool = True

@dataclass
class WorkflowState:
    """Represents the complete state of a workflow."""
    ui_elements: List[UIElement]
    cursor_position: Tuple[float, float]
    timestamp: float
    last_action: Optional[str] = None
    last_element_interacted: Optional[UIElement] = None
    sequence_position: int = 0

class UIElementDetector:
    """Detects UI elements in video frames using computer vision heuristics.

    Strategy (no ML model required):
      1. Convert to HSV and detect saturated colored regions — typical for
         buttons, icons, and interactive widgets.
      2. Find rectangular contours in the edge map that match button/panel
         aspect ratios.
      3. Detect text-like regions using morphological operations on the
         grayscale frame.

    Falls back to YOLO if a model_path to a trained YOLO weights file
    is provided and ultralytics is installed.

    Args:
        model_path: Optional path to a YOLO .pt weights file.
            If None or if ultralytics is not installed, CV heuristics are used.
    """

    # HSV saturation threshold — pixels above this are considered "colored"
    # (buttons, icons) rather than background.
    _SAT_THRESHOLD: int = 60
    # Minimum contour area to consider as a UI element (pixels²).
    _MIN_CONTOUR_AREA: int = 400
    # Aspect ratio bounds for button-like rectangles [min, max].
    _BUTTON_ASPECT_RANGE: tuple = (0.2, 8.0)

    def __init__(self, model_path: Optional[str] = None) -> None:
        self.model_path = model_path
        self._yolo_model = self._load_yolo(model_path)

    def detect_elements(self, frame: np.ndarray) -> List[UIElement]:
        """Detect UI elements in a single BGR frame.

        Args:
            frame: BGR numpy array from cv2.VideoCapture.read().

        Returns:
            List of UIElement instances with bounding boxes and types.
        """
        if self._yolo_model is not None:
            return self._detect_yolo(frame)
        return self._detect_cv(frame)

    # ------------------------------------------------------------------
    # CV heuristic detection
    # ------------------------------------------------------------------

    def _detect_cv(self, frame: np.ndarray) -> List[UIElement]:
        """Detect UI elements using color segmentation and contour analysis.

        Args:
            frame: BGR numpy array.

        Returns:
            List of UIElement instances.
        """
        elements: List[UIElement] = []
        height, width = frame.shape[:2]

        # --- Pass 1: colored rectangle detection (buttons, panels) ---
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # Isolate high-saturation pixels — UI widgets are typically more
        # saturated than neutral backgrounds.
        sat_mask = cv2.inRange(hsv, (0, self._SAT_THRESHOLD, 50), (180, 255, 255))
        sat_mask = cv2.morphologyEx(sat_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        colored_contours, _ = cv2.findContours(
            sat_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        for i, contour in enumerate(colored_contours):
            elem = self._contour_to_element(
                contour, i, width, height, default_type="button"
            )
            if elem is not None:
                elements.append(elem)

        # --- Pass 2: edge-based rectangular region detection (panels, inputs) ---
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, threshold1=50, threshold2=150)
        edges = cv2.dilate(edges, np.ones((2, 2), np.uint8), iterations=1)
        edge_contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        offset = len(elements)
        for i, contour in enumerate(edge_contours):
            elem = self._contour_to_element(
                contour, offset + i, width, height, default_type="panel"
            )
            if elem is not None and not _overlaps_any(elem, elements):
                elements.append(elem)

        # --- Pass 3: text region detection (labels, inputs) ---
        text_elements = self._detect_text_regions(gray, width, height, len(elements))
        elements.extend(text_elements)

        return elements

    def _contour_to_element(
        self,
        contour,
        idx: int,
        frame_width: int,
        frame_height: int,
        default_type: str,
    ) -> Optional[UIElement]:
        """Convert an OpenCV contour to a UIElement if it passes filters.

        Args:
            contour: OpenCV contour array.
            idx: Index used to generate the element ID.
            frame_width: Frame width in pixels.
            frame_height: Frame height in pixels.
            default_type: Element type label if no better guess can be made.

        Returns:
            UIElement or None if the contour fails area/aspect ratio filters.
        """
        area = cv2.contourArea(contour)
        if area < self._MIN_CONTOUR_AREA:
            return None

        x, y, w, h = cv2.boundingRect(contour)
        aspect = w / max(h, 1)
        if not (self._BUTTON_ASPECT_RANGE[0] <= aspect <= self._BUTTON_ASPECT_RANGE[1]):
            return None

        # Skip elements that cover most of the frame — likely background.
        if w * h > 0.6 * frame_width * frame_height:
            return None

        # Normalize bbox to [0, 1] range.
        norm_bbox = [
            x / frame_width,
            y / frame_height,
            (x + w) / frame_width,
            (y + h) / frame_height,
        ]

        # Refine type guess based on aspect ratio.
        if 2.0 < aspect <= 8.0:
            elem_type = "text_input"
        elif aspect <= 2.0 and h < 50:
            elem_type = "button"
        else:
            elem_type = default_type

        # Confidence heuristic: larger, more rectangular contours score higher.
        rect_area = w * h
        fill_ratio = area / max(rect_area, 1)
        confidence = min(fill_ratio * 0.8 + min(area / 5000.0, 0.2), 0.95)

        return UIElement(
            id=f"elem_{idx}",
            type=elem_type,
            bbox=norm_bbox,
            confidence=round(confidence, 3),
        )

    def _detect_text_regions(
        self,
        gray: np.ndarray,
        frame_width: int,
        frame_height: int,
        id_offset: int,
    ) -> List[UIElement]:
        """Find text-like regions using morphological operations.

        Dilates characters horizontally to merge them into word/line blocks,
        then finds contours of those blocks.

        Args:
            gray: Grayscale frame.
            frame_width: Frame width in pixels.
            frame_height: Frame height in pixels.
            id_offset: ID offset so IDs don't clash with other elements.

        Returns:
            List of UIElement instances typed as "text".
        """
        # Threshold to isolate dark text on light background.
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        # Horizontal dilation merges characters into words.
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 3))
        dilated = cv2.dilate(thresh, kernel, iterations=1)
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        elements: List[UIElement] = []
        for i, contour in enumerate(contours):
            elem = self._contour_to_element(
                contour, id_offset + i, frame_width, frame_height, default_type="text"
            )
            if elem is not None:
                elem.type = "text"
                elements.append(elem)
        return elements

    # ------------------------------------------------------------------
    # YOLO detection (optional)
    # ------------------------------------------------------------------

    @staticmethod
    def _load_yolo(model_path: Optional[str]):
        """Load YOLO model if path is given and ultralytics is installed.

        Args:
            model_path: Path to YOLO weights file (.pt).

        Returns:
            YOLO model or None.
        """
        if model_path is None:
            return None
        try:
            from ultralytics import YOLO
            return YOLO(model_path)
        except ImportError:
            logging.getLogger(__name__).warning(
                "ultralytics not installed; falling back to CV heuristics."
            )
            return None

    def _detect_yolo(self, frame: np.ndarray) -> List[UIElement]:
        """Run YOLO detection on a frame.

        Args:
            frame: BGR numpy array.

        Returns:
            List of UIElement instances from YOLO predictions.
        """
        height, width = frame.shape[:2]
        results = self._yolo_model(frame, verbose=False)
        elements: List[UIElement] = []
        for i, box in enumerate(results[0].boxes):
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            cls_name = self._yolo_model.names.get(cls_id, "unknown")
            elements.append(UIElement(
                id=f"yolo_{i}",
                type=cls_name,
                bbox=[x1 / width, y1 / height, x2 / width, y2 / height],
                confidence=round(conf, 3),
            ))
        return elements


# ---------------------------------------------------------------------------
# Module-level helpers for UIElementDetector
# ---------------------------------------------------------------------------


def _overlaps_any(candidate: UIElement, existing: List[UIElement], threshold: float = 0.5) -> bool:
    """Check if a candidate element significantly overlaps any existing element.

    Args:
        candidate: The element to test.
        existing: List of already-accepted elements.
        threshold: IoU threshold above which the candidate is considered a duplicate.

    Returns:
        True if the candidate overlaps an existing element above the threshold.
    """
    cx0, cy0, cx1, cy1 = candidate.bbox
    for elem in existing:
        ex0, ey0, ex1, ey1 = elem.bbox
        ix0, iy0 = max(cx0, ex0), max(cy0, ey0)
        ix1, iy1 = min(cx1, ex1), min(cy1, ey1)
        if ix1 <= ix0 or iy1 <= iy0:
            continue
        intersection = (ix1 - ix0) * (iy1 - iy0)
        union = (
            (cx1 - cx0) * (cy1 - cy0)
            + (ex1 - ex0) * (ey1 - ey0)
            - intersection
        )
        if union > 0 and intersection / union > threshold:
            return True
    return False

class MotionDetector:
    """Detects cursor motion between frames."""
    
    def __init__(self, min_area: int = 500):
        self.min_area = min_area
        self.prev_frame = None
        
    def detect_motion(self, frame: np.ndarray) -> Optional[Tuple[float, float]]:
        """Detect motion and return cursor position if found."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        
        if self.prev_frame is None:
            self.prev_frame = gray
            return None
            
        # Calculate frame difference
        frame_diff = cv2.absdiff(self.prev_frame, gray)
        thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)
        
        # Find motion areas
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, 
                                     cv2.CHAIN_APPROX_SIMPLE)
        
        # Get largest motion area
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest_contour) > self.min_area:
                M = cv2.moments(largest_contour)
                if M["m00"] > 0:
                    cx = M["m10"] / M["m00"]
                    cy = M["m01"] / M["m00"]
                    self.prev_frame = gray
                    return (cx, cy)
        
        self.prev_frame = gray
        return None

class WorkflowEnvironment(gym.Env):
    """Environment for learning and replicating UI workflows."""
    
    def __init__(
        self,
        video_path: str,
        output_dir: str = "output",
        anthropic_api_key: Optional[str] = None,
        model_path: Optional[str] = None,
        viz_enabled: bool = False
    ):
        super().__init__()
        
        # Initialize paths - remove quotes if present and resolve path
        self.video_path = Path(video_path.strip('"')).resolve()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if video file exists
        if not self.video_path.exists():
            raise ValueError(
                f"Video file not found at: {self.video_path}\n"
                f"Current working directory: {os.getcwd()}"
            )
        
        # Initialize video capture
        self.cap = cv2.VideoCapture(str(self.video_path))
        if not self.cap.isOpened():
            raise ValueError(f"Failed to open video file: {self.video_path}")
        
        # Initialize components
        self.ui_detector = UIElementDetector(model_path)
        self.motion_detector = MotionDetector()
        self.anthropic_api_key = anthropic_api_key
        self.viz_enabled = viz_enabled
        
        # Setup spaces
        self._setup_spaces()
        
        # Initialize state
        self.current_frame = None
        self.current_step = 0
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        logger.info(f"Initialized WorkflowEnvironment with video: {self.video_path}")

    def _detect_ui_elements(self) -> List[Dict]:
        """Detect UI elements in current frame."""
        if self.current_frame is None:
            return []

        elements = self.ui_detector.detect_elements(self.current_frame)
        return [
            {
                'type': elem.type,
                'bbox': elem.bbox,
                'state': elem.state
            }
            for elem in elements
        ]

    def _create_initial_state(self) -> 'WorkflowState':
        """Create initial workflow state."""
        ui_elements = self.ui_detector.detect_elements(self.current_frame)
        height, width = self.current_frame.shape[:2]
    
        return WorkflowState(
            ui_elements=ui_elements,
            cursor_position=(width/2, height/2),
            timestamp=0.0,
            sequence_position=0
        )
        
    def _setup_spaces(self):
        """Setup action and observation spaces."""
        self.action_space = spaces.Dict({
            'action_type': spaces.Discrete(4),  # click, double_click, drag, type
            'target_element': spaces.Box(
                low=0,
                high=1,
                shape=(4,),
                dtype=np.float32
            ),
            'parameters': spaces.Dict({
                'text_input': spaces.Text(max_length=100),
                'drag_end': spaces.Box(
                    low=0,
                    high=1,
                    shape=(2,),
                    dtype=np.float32
                )
            })
        })
        
        self.observation_space = spaces.Dict({
            'ui_elements': spaces.Sequence(
                spaces.Dict({
                    'type': spaces.Text(max_length=20),
                    'bbox': spaces.Box(low=0, high=1, shape=(4,)),
                    'state': spaces.Text(max_length=20)
                })
            ),
            'cursor_pos': spaces.Box(low=0, high=1, shape=(2,)),
            'last_action': spaces.Text(max_length=50),
            'progress': spaces.Box(low=0, high=1, shape=(1,))
        })
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None
    ) -> Tuple[Dict, Dict]:
        """Reset environment to initial state."""
        super().reset(seed=seed)
        
        # Reset video capture
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        # Get first frame
        success, self.current_frame = self.cap.read()
        if not success:
            raise RuntimeError("Failed to read first frame from video")
        
        # Initialize state
        self.current_state = self._create_initial_state()
        
        return self._get_observation(), {}
    
    def step(self, action: Dict) -> Tuple[Dict, float, bool, bool, Dict]:
        """Execute action and return next state."""
        if self.current_frame is None:
            raise RuntimeError("Environment needs to be reset")
        
        # Read next frame
        success, self.current_frame = self.cap.read()
        if not success:
            return self._get_observation(), 0.0, True, False, {}
        
        # Process action and update state
        reward = self._process_action(action)
        self.current_state = self._update_state(action)
        
        # Check if episode is done
        frame_position = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
        done = frame_position >= self.total_frames
        
        return self._get_observation(), reward, done, False, self._get_info()
    
    def render(self):
        """Render current environment state."""
        if not self.viz_enabled or self.current_frame is None:
            return
        
        frame = self.current_frame.copy()
        
        # Draw UI elements
        for element in self.current_state.ui_elements:
            x1, y1, x2, y2 = map(int, element.bbox)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, element.type, (x1, y1-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Draw cursor
        cx, cy = map(int, self.current_state.cursor_position)
        cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)
        
        cv2.imshow('WorkflowEnvironment', frame)
        cv2.waitKey(1)
    
    def close(self):
        """Clean up resources."""
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()
    
    def _update_state(self, action: Dict) -> WorkflowState:
        """Update workflow state based on action and new frame."""
        # Detect UI elements in new frame
        ui_elements = self.ui_detector.detect_elements(self.current_frame)
        
        # Detect cursor motion
        cursor_pos = self.motion_detector.detect_motion(self.current_frame)
        if cursor_pos is None:
            cursor_pos = self.current_state.cursor_position
        
        # Find interacted element
        target_element = None
        if 'target_element' in action:
            target_bbox = action['target_element']
            for element in ui_elements:
                if self._check_overlap(target_bbox, element.bbox):
                    target_element = element
                    break
        
        return WorkflowState(
            ui_elements=ui_elements,
            cursor_position=cursor_pos,
            timestamp=self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0,
            last_action=self._get_action_type(action),
            last_element_interacted=target_element,
            sequence_position=int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
        )
    
    def _get_observation(self) -> Dict:
        """Get current observation."""
        if self.current_state is None:
            return self._get_empty_observation()
        
        return {
            'ui_elements': [
                {
                    'type': elem.type,
                    'bbox': elem.bbox,
                    'state': elem.state
                }
                for elem in self.current_state.ui_elements
            ],
            'cursor_pos': self.current_state.cursor_position,
            'last_action': self.current_state.last_action or '',
            'progress': [self.current_state.sequence_position / self.total_frames]
        }
    
    def _get_empty_observation(self) -> Dict:
        """Return empty observation with correct structure."""
        return {
            'ui_elements': [],
            'cursor_pos': (0.0, 0.0),
            'last_action': '',
            'progress': [0.0]
        }
    
    def _get_info(self) -> Dict:
        """Get additional information about current state."""
        return {
            'frame_position': self.current_state.sequence_position,
            'timestamp': self.current_state.timestamp,
            'total_frames': self.total_frames
        }
    
    def _process_action(self, action: Dict) -> float:
        """Process action and calculate reward."""
        # Simple reward implementation - can be enhanced based on needs
        if self.current_state.last_element_interacted:
            return 1.0
        return 0.0
    
    @staticmethod
    def _check_overlap(bbox1: List[float], bbox2: List[float]) -> bool:
        """Check if two bounding boxes overlap."""
        x1_min, y1_min, x1_max, y1_max = bbox1
        x2_min, y2_min, x2_max, y2_max = bbox2
        
        return not (x1_max < x2_min or x1_min > x2_max or
                   y1_max < y2_min or y1_min > y2_max)
    
    @staticmethod
    def _get_action_type(action: Dict) -> str:
        """Convert action type from index to string."""
        action_types = ['click', 'double_click', 'drag', 'type']
        action_idx = action.get('action_type', 0)
        return action_types[action_idx]