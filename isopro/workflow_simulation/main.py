"""
Main Module for Workflow Automation

A streamlined entry point for the workflow automation system that captures, learns,
and replicates UI workflows from video demonstrations.
"""

import argparse
import logging
from pathlib import Path
import json
from datetime import datetime
from typing import Dict

from .workflow_simulator import WorkflowSimulator
from .workflow_agent import AgentConfig
from .workflow_visualizer import VisualizationConfig
from .workflow_validator import ValidationConfig
from .agent_config import AgentConfig

class WorkflowAutomation:
    """Main class for handling workflow automation setup and execution."""
    
    def __init__(self, args: argparse.Namespace):
        self.video_path = args.video
        self.config_path = args.config
        self.output_dir = Path(args.output)
        self.log_dir = Path(args.logs)
        
        # Create necessary directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self._setup_logging()
        self.logger = logging.getLogger(__name__)
        
        # Load configuration
        self.config = self._load_config()
        
    def _setup_logging(self):
        """Configure logging with file and console output."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = self.log_dir / f"workflow_{timestamp}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_path),
                logging.StreamHandler()
            ]
        )
        
    def _load_config(self) -> Dict:
        """Load and validate configuration file."""
        try:
            with open(self.config_path) as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Failed to load config file: {e}")
            raise
            
    def _create_configs(self):
        """Create configuration objects from loaded config."""
        # Agent configuration
        self.agent_config = AgentConfig(
            learning_rate=self.config.get('learning_rate', 3e-4),
            pretrain_epochs=self.config.get('pretrain_epochs', 10),
            use_demonstration=True,
            use_reasoning=True,
            reward_threshold=self.config.get('reward_threshold', 0.8)  # Added this line
        )
        
        # Visualization configuration
        self.viz_config = VisualizationConfig(
            show_ui_elements=True,
            show_cursor=True,
            show_actions=True,
            save_frames=True,
            real_time_display=self.config.get('show_visualization', True)
        )
        
        # Validation configuration
        self.validation_config = ValidationConfig.from_dict(
            self.config.get('validation', {})
        )
        
    def run(self):
        """Execute the workflow automation process."""
        self.logger.info("Starting workflow automation")
        
        try:
            # Create configurations
            self._create_configs()
            
            # Initialize simulator
            simulator = WorkflowSimulator(
                video_path=self.video_path,
                anthropic_api_key=self.config.get('anthropic_api_key'),
                agent_config=self.agent_config,
                viz_config=self.viz_config,
                validation_config=self.validation_config,
                output_dir=str(self.output_dir)
            )
            
            # Train
            self.logger.info("Starting training")
            simulator.train_agents()
            
            # Evaluate
            self.logger.info("Starting evaluation")
            results = simulator.evaluate_agents()
            
            # Save results
            self._save_results(results)
            
            self.logger.info("Workflow automation completed successfully")
            return results
            
        except Exception as e:
            self.logger.error(f"Workflow automation failed: {e}", exc_info=True)
            raise
            
    def _save_results(self, results: Dict):
        """Save evaluation results to file."""
        results_path = self.output_dir / "results.json"
        try:
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2)
            self.logger.info(f"Results saved to {results_path}")
        except Exception as e:
            self.logger.error(f"Failed to save results: {e}")
            raise

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Workflow Automation System",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--video",
        required=True,
        help="Path to workflow video"
    )
    
    parser.add_argument(
        "--config",
        default="config.json",
        help="Path to config file"
    )
    
    parser.add_argument(
        "--output",
        default="output",
        help="Output directory for results and artifacts"
    )
    
    parser.add_argument(
        "--logs",
        default="logs",
        help="Directory for log files"
    )
    
    return parser.parse_args()

def main():
    """Main entry point for the workflow automation system."""
    args = parse_arguments()
    automation = WorkflowAutomation(args)
    automation.run()

if __name__ == "__main__":
    main()