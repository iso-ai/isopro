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
    """Handles detection of UI elements in video frames."""
    
    def __init__(self, model_path: Optional[str] = None):
        """Initialize the detector with optional custom model."""
        self.model_path = model_path
        # Placeholder for actual model initialization
        
    def detect_elements(self, frame: np.ndarray) -> List[UIElement]:
        """Detect UI elements in a single frame."""
        # Placeholder implementation - replace with actual detection logic
        height, width = frame.shape[:2]
        
        # Example element for testing
        element = UIElement(
            id="test_element",
            type="button",
            bbox=[0, 0, width/4, height/4],
            confidence=0.95
        )
        
        return [element]

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