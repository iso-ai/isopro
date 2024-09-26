# isopro/orchestration_simulation/evaluator.py

import logging
from typing import Dict, Any
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

class Evaluator:
    def __init__(self):
        self.metrics = ['execution_time', 'memory_usage', 'coherence']

    def evaluate(self, results: Dict[str, Dict[str, Any]]) -> str:
        """
        Evaluate the results from different execution modes and determine the best mode.
        
        :param results: A dictionary with execution modes as keys and their results as values
        :return: The name of the best execution mode
        """
        evaluations = {}
        for mode, result in results.items():
            evaluations[mode] = self._evaluate_mode(result)

        normalized_scores = self._normalize_scores(evaluations)
        total_scores = {mode: sum(scores.values()) for mode, scores in normalized_scores.items()}

        best_mode = max(total_scores, key=total_scores.get)
        
        self._log_evaluation_results(evaluations, best_mode)
        
        return best_mode

    def _evaluate_mode(self, result: Dict[str, Any]) -> Dict[str, float]:
        """
        Evaluate a single execution mode based on the metrics.
        
        :param result: The result dictionary for a single execution mode
        :return: A dictionary of evaluation scores for each metric
        """
        evaluation = {
            'execution_time': result.get('execution_time', 0),
            'memory_usage': result.get('memory_usage', 0),
            'coherence': self._calculate_coherence(result.get('output', ''))
        }
        return evaluation

    def _normalize_scores(self, evaluations: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
        """
        Normalize the evaluation scores across all modes.
        
        :param evaluations: A dictionary of evaluation scores for each mode
        :return: A dictionary of normalized scores
        """
        normalized = {}
        for metric in self.metrics:
            min_val = min(eval[metric] for eval in evaluations.values())
            max_val = max(eval[metric] for eval in evaluations.values())
            for mode in evaluations:
                if mode not in normalized:
                    normalized[mode] = {}
                if max_val - min_val == 0:
                    normalized[mode][metric] = 1  # Avoid division by zero
                else:
                    if metric in ['execution_time', 'memory_usage']:
                        normalized[mode][metric] = 1 - (evaluations[mode][metric] - min_val) / (max_val - min_val)
                    else:
                        normalized[mode][metric] = (evaluations[mode][metric] - min_val) / (max_val - min_val)
        return normalized

    def _calculate_coherence(self, text: str) -> float:
        """
        Calculate the coherence of the output text.
        
        :param text: The output text to evaluate
        :return: A coherence score
        """
        sentences = text.split('.')
        if len(sentences) < 2:
            return 0.0
        
        vectorizer = TfidfVectorizer().fit_transform(sentences)
        similarity_matrix = cosine_similarity(vectorizer)
        
        return np.mean(similarity_matrix)

    def _log_evaluation_results(self, evaluations: Dict[str, Dict[str, float]], best_mode: str):
        """
        Log the evaluation results and the best mode.
        
        :param evaluations: A dictionary of evaluation scores for each mode
        :param best_mode: The name of the best execution mode
        """
        logger.info("Evaluation Results:")
        for mode, scores in evaluations.items():
            logger.info(f"{mode} mode:")
            for metric, score in scores.items():
                logger.info(f"  {metric}: {score:.2f}")
        
        logger.info(f"\nBest mode: {best_mode}")
        logger.info("Reasoning:")
        logger.info(f"  Execution Time: {evaluations[best_mode]['execution_time']:.2f} seconds")
        logger.info(f"  Memory Usage: {evaluations[best_mode]['memory_usage']:.2f} MB")
        logger.info(f"  Coherence: {evaluations[best_mode]['coherence']:.2f}")