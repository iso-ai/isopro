"""
Analyze Adversarial Simulation

This module provides functions for analyzing the results of adversarial simulations.
"""

from typing import List, Dict, Any
from .llm_metrics import evaluate_llm_metrics
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import logging

logger = logging.getLogger(__name__)

def calculate_text_similarity(text1: str, text2: str) -> float:
    """
    Calculate the cosine similarity between two texts using sentence embeddings.

    Args:
        text1 (str): The first text.
        text2 (str): The second text.

    Returns:
        float: The cosine similarity between the two texts.
    """
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode([text1, text2])
    similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    return similarity

def analyze_adversarial_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze the results of an adversarial simulation.

    Args:
        results (List[Dict[str, Any]]): The list of simulation results to analyze.

    Returns:
        Dict[str, Any]: A dictionary containing various analysis metrics.
    """
    original_inputs = [r["original_input"] for r in results]
    perturbed_inputs = [r["perturbed_input"] for r in results]
    original_outputs = [r["original_output"] for r in results]
    perturbed_outputs = [r["perturbed_output"] for r in results]

    # Calculate input perturbation metrics
    input_similarities = [calculate_text_similarity(orig, pert) for orig, pert in zip(original_inputs, perturbed_inputs)]
    avg_input_similarity = np.mean(input_similarities)

    # Calculate output perturbation metrics
    output_similarities = [calculate_text_similarity(orig, pert) for orig, pert in zip(original_outputs, perturbed_outputs)]
    avg_output_similarity = np.mean(output_similarities)

    # Calculate LLM metrics for original and perturbed outputs
    original_metrics = evaluate_llm_metrics(original_inputs, original_outputs)
    perturbed_metrics = evaluate_llm_metrics(original_inputs, perturbed_outputs)

    # Calculate relative changes in LLM metrics
    metric_changes = {
        f"{metric}_change": (perturbed_metrics[metric] - original_metrics[metric]) / original_metrics[metric]
        for metric in original_metrics.keys()
    }

    analysis_results = {
        "avg_input_similarity": avg_input_similarity,
        "avg_output_similarity": avg_output_similarity,
        "original_metrics": original_metrics,
        "perturbed_metrics": perturbed_metrics,
        "metric_changes": metric_changes
    }

    logger.info("Completed analysis of adversarial simulation results")
    return analysis_results

def summarize_adversarial_impact(analysis_results: Dict[str, Any]) -> str:
    """
    Generate a summary of the impact of adversarial attacks based on the analysis results.

    Args:
        analysis_results (Dict[str, Any]): The results of the adversarial analysis.

    Returns:
        str: A summary of the adversarial impact.
    """
    summary = []
    summary.append(f"Input Perturbation: The average similarity between original and perturbed inputs is {analysis_results['avg_input_similarity']:.2f}")
    summary.append(f"Output Perturbation: The average similarity between original and perturbed outputs is {analysis_results['avg_output_similarity']:.2f}")

    for metric, change in analysis_results['metric_changes'].items():
        impact = "increased" if change > 0 else "decreased"
        summary.append(f"{metric.capitalize()}: {impact} by {abs(change)*100:.2f}%")

    most_affected_metric = max(analysis_results['metric_changes'], key=lambda k: abs(analysis_results['metric_changes'][k]))
    summary.append(f"The most affected metric was {most_affected_metric}, with a change of {abs(analysis_results['metric_changes'][most_affected_metric])*100:.2f}%")

    logger.info("Generated summary of adversarial impact")
    return "\n".join(summary)