"""
Workflow Validator Module -
Validates workflow reasoning using IsoZero, a LLM-based reasoning system.
This module is responsible for comparing the executed workflow steps against the provided demonstrations
and calculating a validation score based on the matching action, target accuracy, timing accuracy, and reasoning score.
"""

import logging
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from datetime import datetime
import json
import numpy as np
from tqdm import tqdm
from isozero import ClaudeAgent, QuestionAnswerer

logger = logging.getLogger(__name__)

@dataclass
class ValidationConfig:
    """Configuration for workflow validation."""
    sequence_matching: bool = True  # Enable sequence matching validation
    state_validation: bool = True   # Enable state validation
    save_visualizations: bool = True  # Save validation visualizations
    action_weight: float = 0.4      # Weight for action matching in scoring
    target_weight: float = 0.3      # Weight for target accuracy in scoring
    timing_weight: float = 0.1      # Weight for timing accuracy in scoring
    reasoning_weight: float = 0.2   # Weight for reasoning score in scoring
    timing_window: float = 0.5      # Timing window for accuracy (in seconds)
    min_target_iou: float = 0.5     # Minimum IoU for target matching
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ValidationConfig':
        """Create config from dictionary."""
        return cls(
            sequence_matching=config_dict.get('sequence_matching', True),
            state_validation=config_dict.get('state_validation', True),
            save_visualizations=config_dict.get('save_visualizations', True),
            action_weight=config_dict.get('action_weight', 0.4),
            target_weight=config_dict.get('target_weight', 0.3),
            timing_weight=config_dict.get('timing_weight', 0.1),
            reasoning_weight=config_dict.get('reasoning_weight', 0.2),
            timing_window=config_dict.get('timing_window', 0.5),
            min_target_iou=config_dict.get('min_target_iou', 0.5)
        )

@dataclass
class ValidationResult:
    """Stores validation results for a workflow step."""
    step_id: int
    action_match: bool
    target_accuracy: float
    timing_accuracy: float
    reasoning_score: float
    overall_score: float
    messages: List[str]

class WorkflowValidator:
    """Validates executed workflows against demonstrations."""
    
    def __init__(
        self,
        config: ValidationConfig,
        output_dir: str = "output",
        log_dir: str = "logs",
        anthropic_api_key: Optional[str] = None
    ):
        self.config = config
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(output_dir)
        self.log_dir = Path(log_dir)
        
        # Create directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self._setup_logging()
        
        # Initialize IsoZero for reasoning validation if key provided
        if anthropic_api_key:
            self.claude_agent = ClaudeAgent(api_key=anthropic_api_key)
            self.qa_system = QuestionAnswerer(self.claude_agent)
            logger.info("Initialized IsoZero reasoning system")
            
    def _setup_logging(self):
        """Configure logging to file."""
        log_file = self.log_dir / f"validation_{self.timestamp}.log"
        handler = logging.FileHandler(log_file)
        handler.setFormatter(
            logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        )
        logger.addHandler(handler)
        
    def validate_workflow(
        self,
        demo_sequence: List[Dict[str, Any]],
        exec_sequence: List[Dict[str, Any]]
    ) -> List[ValidationResult]:
        """Validate executed workflow against demonstration."""
        if not self.config.sequence_matching:
            logger.info("Sequence matching validation disabled")
            return []
            
        results = []
        logger.info("Starting workflow validation")
        
        for step_id, (demo, exec) in enumerate(tqdm(zip(demo_sequence, exec_sequence))):
            result = self._validate_step(step_id, demo, exec)
            results.append(result)
            
            logger.info(f"Step {step_id} validation score: {result.overall_score:.2f}")
            
        self._save_results(results)
        return results
        
    def _validate_step(
        self,
        step_id: int,
        demo_step: Dict[str, Any],
        exec_step: Dict[str, Any]
    ) -> ValidationResult:
        """Validate a single workflow step."""
        messages = []
        
        # Check action matching
        action_match = demo_step['action_type'] == exec_step['action_type']
        if not action_match:
            messages.append(f"Action mismatch: expected {demo_step['action_type']}, "
                          f"got {exec_step['action_type']}")
            
        # Check target accuracy
        target_accuracy = self._calculate_target_accuracy(
            demo_step['target_element'],
            exec_step['target_element']
        )
        
        if target_accuracy < self.config.min_target_iou:
            messages.append(f"Low target accuracy: {target_accuracy:.2f}")
        
        # Check timing
        timing_accuracy = self._calculate_timing_accuracy(
            demo_step.get('timestamp', 0),
            exec_step.get('timestamp', 0)
        )
        
        # Get reasoning score if available
        reasoning_score = 0.0
        if hasattr(self, 'qa_system'):
            reasoning_score = self._validate_reasoning(demo_step, exec_step)
            
        # Calculate overall score
        overall_score = self._calculate_score(
            action_match,
            target_accuracy,
            timing_accuracy,
            reasoning_score
        )
        
        return ValidationResult(
            step_id=step_id,
            action_match=action_match,
            target_accuracy=target_accuracy,
            timing_accuracy=timing_accuracy,
            reasoning_score=reasoning_score,
            overall_score=overall_score,
            messages=messages
        )
        
    def _calculate_target_accuracy(
        self,
        demo_target: Dict[str, Any],
        exec_target: Dict[str, Any]
    ) -> float:
        """Calculate accuracy of target element matching using IoU."""
        if not demo_target or not exec_target:
            return 0.0
            
        demo_bbox = demo_target.get('bbox', [0, 0, 0, 0])
        exec_bbox = exec_target.get('bbox', [0, 0, 0, 0])
        
        # Calculate intersection over union
        x1 = max(demo_bbox[0], exec_bbox[0])
        y1 = max(demo_bbox[1], exec_bbox[1])
        x2 = min(demo_bbox[2], exec_bbox[2])
        y2 = min(demo_bbox[3], exec_bbox[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        
        area1 = (demo_bbox[2] - demo_bbox[0]) * (demo_bbox[3] - demo_bbox[1])
        area2 = (exec_bbox[2] - exec_bbox[0]) * (exec_bbox[3] - exec_bbox[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
        
    def _calculate_timing_accuracy(
        self,
        demo_time: float,
        exec_time: float
    ) -> float:
        """Calculate timing accuracy between demonstration and execution."""
        diff = abs(demo_time - exec_time)
        return max(0, 1 - (diff / self.config.timing_window))
        
    def _validate_reasoning(
        self,
        demo_step: Dict[str, Any],
        exec_step: Dict[str, Any]
    ) -> float:
        """Validate reasoning about the action using IsoZero."""
        context = f"""
        Demonstration: {demo_step['action_type']} on {demo_step['target_element']['type']}
        Execution: {exec_step['action_type']} on {exec_step['target_element']['type']}
        """
        
        question = "On a scale of 0 to 1, how well does the executed action match the demonstrated action's intent?"
        
        response = self.qa_system.answer_questions([(question, context)])
        try:
            return float(response[question]['solution'])
        except (ValueError, KeyError):
            return 0.0
            
    def _calculate_score(
        self,
        action_match: bool,
        target_accuracy: float,
        timing_accuracy: float,
        reasoning_score: float
    ) -> float:
        """Calculate overall validation score."""
        return (
            self.config.action_weight * float(action_match) +
            self.config.target_weight * target_accuracy +
            self.config.timing_weight * timing_accuracy +
            self.config.reasoning_weight * reasoning_score
        )
        
    def _save_results(self, results: List[ValidationResult]):
        """Save validation results and generate report."""
        if not self.config.save_visualizations:
            return
            
        # Save detailed results
        output_path = self.output_dir / f"validation_{self.timestamp}.json"
        with open(output_path, 'w') as f:
            json.dump(
                [
                    {
                        'step_id': r.step_id,
                        'action_match': r.action_match,
                        'target_accuracy': r.target_accuracy,
                        'timing_accuracy': r.timing_accuracy,
                        'reasoning_score': r.reasoning_score,
                        'overall_score': r.overall_score,
                        'messages': r.messages
                    }
                    for r in results
                ],
                f,
                indent=2
            )
            
        # Generate summary report
        report_path = self.output_dir / f"report_{self.timestamp}.txt"
        with open(report_path, 'w') as f:
            f.write("Workflow Validation Report\n")
            f.write("=========================\n\n")
            
            avg_score = np.mean([r.overall_score for r in results])
            f.write(f"Overall Score: {avg_score:.2f}\n\n")
            
            for result in results:
                f.write(f"Step {result.step_id}:\n")
                f.write(f"  Score: {result.overall_score:.2f}\n")
                if result.messages:
                    f.write("  Messages:\n")
                    for msg in result.messages:
                        f.write(f"    - {msg}\n")
                f.write("\n")
                
        logger.info(f"Results saved to {output_path}")
        logger.info(f"Report generated at {report_path}")