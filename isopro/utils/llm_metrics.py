"""
LLM Metrics Utilities

This module provides functions for calculating traditional LLM metrics
such as BLEU, ROUGE, Perplexity, Coherence, and others, with the ability
to choose custom Hugging Face transformers for certain metrics.
"""

import logging
import numpy as np
import torch
from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

# Default models
DEFAULT_PERPLEXITY_MODEL = "gpt2"
DEFAULT_COHERENCE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def load_model(model_name, model_class):
    """
    Load a model from Hugging Face's model hub.

    Args:
        model_name (str): The name of the model to load.
        model_class: The class of the model (e.g., AutoModelForCausalLM, AutoModel).

    Returns:
        tuple: The loaded model and tokenizer.
    """
    try:
        model = model_class.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        return model, tokenizer
    except Exception as e:
        logger.error(f"Error loading model {model_name}: {e}")
        raise

def calculate_bleu(reference, candidate):
    """
    Calculate the BLEU score for a given reference and candidate.

    Args:
        reference (str): The reference sentence.
        candidate (str): The candidate sentence to evaluate.

    Returns:
        float: The BLEU score.
    """
    try:
        from nltk.translate.bleu_score import SmoothingFunction
        smoothing = SmoothingFunction().method1
        score = sentence_bleu([reference.split()], candidate.split(), smoothing_function=smoothing)
        return max(0.0, min(1.0, score))  # Clamp between 0 and 1
    except Exception as e:
        logger.warning(f"Error calculating BLEU score: {e}")
        return 0.0

def calculate_rouge(reference, candidate):
    """
    Calculate the ROUGE score for a given reference and candidate.

    Args:
        reference (str): The reference sentence.
        candidate (str): The candidate sentence to evaluate.

    Returns:
        dict: A dictionary containing ROUGE-1, ROUGE-2, and ROUGE-L scores.
    """
    try:
        if not reference.strip() or not candidate.strip():
            return {'rouge-1': 0.0, 'rouge-2': 0.0, 'rouge-l': 0.0}
        
        rouge = Rouge()
        scores = rouge.get_scores(candidate, reference)
        return {
            'rouge-1': max(0.0, min(1.0, scores[0]['rouge-1']['f'])),
            'rouge-2': max(0.0, min(1.0, scores[0]['rouge-2']['f'])),
            'rouge-l': max(0.0, min(1.0, scores[0]['rouge-l']['f']))
        }
    except Exception as e:
        logger.warning(f"Error calculating ROUGE scores: {e}")
        return {'rouge-1': 0.0, 'rouge-2': 0.0, 'rouge-l': 0.0}

def calculate_perplexity(text, model_name=None):
    """
    Calculate the perplexity of a given text using a specified or default language model.

    Args:
        text (str): The text to evaluate.
        model_name (str, optional): The name of the Hugging Face model to use for perplexity calculation.

    Returns:
        float: The perplexity score.
    """
    try:
        if not text.strip():
            return float('inf')
        
        model_name = model_name or DEFAULT_PERPLEXITY_MODEL
        model, tokenizer = load_model(model_name, AutoModelForCausalLM)
        
        inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs.input_ids)
        
        perplexity = np.exp(outputs.loss.item())
        # Clamp to reasonable range
        return max(1.0, min(10000.0, perplexity))
    except Exception as e:
        logger.warning(f"Error calculating perplexity: {e}")
        return 100.0  # Default moderate perplexity value

def calculate_coherence(text, model_name=None):
    """
    Calculate the coherence of a given text using sentence embeddings.

    Args:
        text (str): The text to evaluate.
        model_name (str, optional): The name of the Sentence Transformer model to use for coherence calculation.

    Returns:
        float: The coherence score.
    """
    try:
        if not text.strip():
            return 0.0
        
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        if len(sentences) < 2:
            return 1.0
        
        model_name = model_name or DEFAULT_COHERENCE_MODEL
        model = SentenceTransformer(model_name)
        embeddings = model.encode(sentences)
        
        coherence_scores = []
        for i in range(len(embeddings) - 1):
            similarity = cosine_similarity([embeddings[i]], [embeddings[i+1]])[0][0]
            coherence_scores.append(max(0.0, min(1.0, similarity)))
        
        if not coherence_scores:
            return 1.0
        
        return np.mean(coherence_scores)
    except Exception as e:
        logger.warning(f"Error calculating coherence: {e}")
        return 0.5  # Default moderate coherence value

def calculate_f1_precision_recall(true_labels, predicted_labels):
    """
    Calculate F1 score, precision, and recall.

    Args:
        true_labels (list): The true labels.
        predicted_labels (list): The predicted labels.

    Returns:
        dict: A dictionary containing F1 score, precision, and recall.
    """
    return {
        'f1_score': f1_score(true_labels, predicted_labels, average='weighted'),
        'precision': precision_score(true_labels, predicted_labels, average='weighted'),
        'recall': recall_score(true_labels, predicted_labels, average='weighted')
    }

def evaluate_llm_metrics(reference_texts, generated_texts, true_labels=None, predicted_labels=None, perplexity_model=None, coherence_model=None):
    """
    Evaluate various LLM metrics for given reference and generated texts.

    Args:
        reference_texts (list): A list of reference texts.
        generated_texts (list): A list of generated texts to evaluate.
        true_labels (list, optional): True labels for classification metrics.
        predicted_labels (list, optional): Predicted labels for classification metrics.
        perplexity_model (str, optional): The name of the model to use for perplexity calculation.
        coherence_model (str, optional): The name of the model to use for coherence calculation.

    Returns:
        dict: A dictionary containing various LLM metrics.
    """
    metrics = {}
    
    # BLEU
    metrics['bleu'] = np.mean([calculate_bleu(ref, gen) for ref, gen in zip(reference_texts, generated_texts)])
    
    # ROUGE
    rouge_scores = [calculate_rouge(ref, gen) for ref, gen in zip(reference_texts, generated_texts)]
    metrics['rouge-1'] = np.mean([score['rouge-1'] for score in rouge_scores])
    metrics['rouge-2'] = np.mean([score['rouge-2'] for score in rouge_scores])
    metrics['rouge-l'] = np.mean([score['rouge-l'] for score in rouge_scores])
    
    # Perplexity
    metrics['perplexity'] = np.mean([calculate_perplexity(text, perplexity_model) for text in generated_texts])
    
    # Coherence
    metrics['coherence'] = np.mean([calculate_coherence(text, coherence_model) for text in generated_texts])
    
    # F1, Precision, Recall (if labels are provided)
    if true_labels and predicted_labels:
        classification_metrics = calculate_f1_precision_recall(true_labels, predicted_labels)
        metrics.update(classification_metrics)
    
    logger.info("Completed LLM metrics evaluation")
    return metrics