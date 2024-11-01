"""
Workflow Visualizer Module

Provides visualization capabilities for workflow execution and validation results.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class VisualizationConfig:
    """Configuration for visualization settings."""
    show_ui_elements: bool = True
    show_cursor: bool = True
    show_actions: bool = True
    save_frames: bool = True
    save_plots: bool = True
    real_time_display: bool = False

class WorkflowVisualizer:
    """Visualizes workflow execution and validation results."""
    
    def __init__(self, output_dir: str = "output", config: VisualizationConfig = None):
        """Initialize workflow visualizer."""
        self.config = config or VisualizationConfig()
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(output_dir) / "visualizations" / self.timestamp
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        if self.config.real_time_display:
            cv2.namedWindow("Workflow Visualization", cv2.WINDOW_NORMAL)
            
        logger.info("Initialized WorkflowVisualizer")
        
    def visualize_step(
        self,
        frame: np.ndarray,
        ui_elements: List[Dict[str, Any]],
        cursor_pos: tuple,
        action: str,
        step_num: int
    ):
        """Visualize a single workflow step."""
        viz_frame = frame.copy()
        
        # Draw UI elements
        if self.config.show_ui_elements:
            for element in ui_elements:
                self._draw_ui_element(viz_frame, element)
                
        # Draw cursor
        if self.config.show_cursor:
            self._draw_cursor(viz_frame, cursor_pos)
            
        # Draw action label
        if self.config.show_actions:
            self._draw_action_label(viz_frame, action)
            
        # Show real-time display if enabled
        if self.config.real_time_display:
            cv2.imshow("Workflow Visualization", viz_frame)
            cv2.waitKey(1)
            
        # Save frame if enabled
        if self.config.save_frames:
            frame_path = self.output_dir / f"step_{step_num:04d}.jpg"
            cv2.imwrite(str(frame_path), viz_frame)
            
    def visualize_validation_results(self, validation_results: List[Dict[str, Any]]):
        """Create visualization of validation results."""
        if not self.config.save_plots:
            return
            
        plt.style.use('seaborn')
        
        # Create subplots for different metrics
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot scores over time
        self._plot_scores(validation_results, axes[0, 0])
        
        # Plot accuracy heatmap
        self._plot_accuracy_heatmap(validation_results, axes[0, 1])
        
        # Plot success metrics
        self._plot_success_metrics(validation_results, axes[1, 0])
        
        # Plot timing analysis
        self._plot_timing_analysis(validation_results, axes[1, 1])
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "validation_results.png")
        plt.close()
        
    def _draw_ui_element(self, frame: np.ndarray, element: Dict[str, Any]):
        """Draw a UI element on the frame."""
        x1, y1, x2, y2 = [int(coord) for coord in element['bbox']]
        
        # Draw box
        color = (0, 255, 0) if element.get('enabled', True) else (0, 0, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Draw label
        label = f"{element['type']} ({element.get('confidence', 1.0):.2f})"
        cv2.putText(
            frame,
            label,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2
        )
        
    def _draw_cursor(self, frame: np.ndarray, cursor_pos: tuple):
        """Draw cursor position on frame."""
        x, y = [int(coord) for coord in cursor_pos]
        cv2.circle(frame, (x, y), 5, (255, 0, 0), -1)
        
    def _draw_action_label(self, frame: np.ndarray, action: str):
        """Draw action label on frame."""
        cv2.putText(
            frame,
            f"Action: {action}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2
        )
        
    def _plot_scores(self, results: List[Dict[str, Any]], ax: plt.Axes):
        """Plot validation scores over time."""
        scores = [r['overall_score'] for r in results]
        ax.plot(scores, marker='o')
        ax.set_title('Validation Scores Over Time')
        ax.set_xlabel('Step')
        ax.set_ylabel('Score')
        
    def _plot_accuracy_heatmap(self, results: List[Dict[str, Any]], ax: plt.Axes):
        """Plot accuracy heatmap."""
        data = np.array([
            [r['action_match'], r['target_accuracy'], r['timing_accuracy']]
            for r in results
        ])
        
        sns.heatmap(
            data.T,
            ax=ax,
            yticklabels=['Action', 'Target', 'Timing'],
            cmap='YlOrRd'
        )
        ax.set_title('Accuracy Heatmap')
        
    def _plot_success_metrics(self, results: List[Dict[str, Any]], ax: plt.Axes):
        """Plot success metrics."""
        metrics = {
            'Action Match': np.mean([r['action_match'] for r in results]),
            'Target Accuracy': np.mean([r['target_accuracy'] for r in results]),
            'Timing Accuracy': np.mean([r['timing_accuracy'] for r in results])
        }
        
        ax.bar(metrics.keys(), metrics.values())
        ax.set_title('Average Success Metrics')
        ax.set_ylim(0, 1)
        
    def _plot_timing_analysis(self, results: List[Dict[str, Any]], ax: plt.Axes):
        """Plot timing analysis."""
        timing_accuracies = [r['timing_accuracy'] for r in results]
        sns.histplot(timing_accuracies, ax=ax, bins=20)
        ax.set_title('Timing Accuracy Distribution')
        ax.set_xlabel('Accuracy')
        
    def close(self):
        """Clean up visualization resources."""
        if self.config.real_time_display:
            cv2.destroyAllWindows()