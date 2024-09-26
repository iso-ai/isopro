# evaluation.py

import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging

logger = logging.getLogger(__name__)

def measure_coherence(texts):
    try:
        # Ensure texts is a list of strings
        if not all(isinstance(text, str) for text in texts):
            raise ValueError("All texts must be strings")

        # Tokenize the texts into sentences
        sentences = [sent for text in texts for sent in nltk.sent_tokenize(text)]

        # Create TF-IDF vectors
        vectorizer = TfidfVectorizer().fit_transform(sentences)

        # Calculate cosine similarity
        similarity_matrix = cosine_similarity(vectorizer)

        # Calculate average similarity
        coherence = similarity_matrix.mean()

        return coherence
    except Exception as e:
        logger.error(f"Error measuring coherence: {e}")
        return 0.0

def evaluate_results(evaluations):
    try:
        # Normalize scores
        normalized_scores = {}
        for metric in ['execution_time', 'memory_usage', 'coherence']:
            min_val = min(eval[metric] for eval in evaluations.values())
            max_val = max(eval[metric] for eval in evaluations.values())
            for mode in evaluations:
                if mode not in normalized_scores:
                    normalized_scores[mode] = {}
                if max_val - min_val == 0:
                    normalized_scores[mode][metric] = 1  # Avoid division by zero
                else:
                    if metric in ['execution_time', 'memory_usage']:
                        normalized_scores[mode][metric] = 1 - (evaluations[mode][metric] - min_val) / (max_val - min_val)
                    else:
                        normalized_scores[mode][metric] = (evaluations[mode][metric] - min_val) / (max_val - min_val)

        # Calculate total scores
        total_scores = {mode: sum(scores.values()) for mode, scores in normalized_scores.items()}

        best_mode = max(total_scores, key=total_scores.get)
        
        logger.info(f"\nBest mode for this task: {best_mode}")
        logger.info("Reasoning:")
        logger.info(f"  Execution Time: {evaluations[best_mode]['execution_time']:.2f} seconds")
        logger.info(f"  Memory Usage: {evaluations[best_mode]['memory_usage']:.2f} MB")
        logger.info(f"  Coherence: {evaluations[best_mode]['coherence']:.2f}")
        
        return best_mode
    except Exception as e:
        logger.error(f"Error in evaluate_results: {e}")
        return "Error in evaluation"