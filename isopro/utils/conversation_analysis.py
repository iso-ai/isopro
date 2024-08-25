"""
Conversation Analysis Utilities

This module provides functions for analyzing conversation simulations,
including sentiment analysis, response time analysis, and context adaptation analysis.
"""

import logging
from textblob import TextBlob
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

def analyze_sentiment(text):
    """
    Perform sentiment analysis on the given text.

    Args:
        text (str): The text to analyze.

    Returns:
        float: The sentiment polarity (-1 to 1, where -1 is very negative and 1 is very positive).
    """
    blob = TextBlob(text)
    return blob.sentiment.polarity

def analyze_response_time(conversation_history):
    """
    Analyze the response time in a conversation.

    Args:
        conversation_history (list): A list of dictionaries containing the conversation history.

    Returns:
        dict: A dictionary containing average response times for the AI and the user.
    """
    ai_response_times = []
    user_response_times = []
    
    for i in range(1, len(conversation_history)):
        time_diff = conversation_history[i]['timestamp'] - conversation_history[i-1]['timestamp']
        if conversation_history[i]['role'] == 'assistant':
            ai_response_times.append(time_diff)
        else:
            user_response_times.append(time_diff)
    
    return {
        'avg_ai_response_time': np.mean(ai_response_times),
        'avg_user_response_time': np.mean(user_response_times)
    }

def analyze_context_adaptation(conversation_history):
    """
    Analyze how well the AI adapts to the conversational context.

    Args:
        conversation_history (list): A list of dictionaries containing the conversation history.

    Returns:
        float: A score representing the AI's context adaptation (0 to 1, where 1 is perfect adaptation).
    """
    ai_responses = [msg['content'] for msg in conversation_history if msg['role'] == 'assistant']
    user_messages = [msg['content'] for msg in conversation_history if msg['role'] == 'user']
    
    if len(ai_responses) < 2 or len(user_messages) < 2:
        return 0.0
    
    vectorizer = TfidfVectorizer()
    user_vectors = vectorizer.fit_transform(user_messages)
    ai_vectors = vectorizer.transform(ai_responses)
    
    context_scores = []
    for i in range(1, len(ai_responses)):
        user_context = user_vectors[i-1:i+1]
        ai_response = ai_vectors[i]
        similarity = cosine_similarity(user_context, ai_response)
        context_scores.append(np.mean(similarity))
    
    return np.mean(context_scores)

def analyze_conversation(conversation_history):
    """
    Perform a comprehensive analysis of the conversation.

    Args:
        conversation_history (list): A list of dictionaries containing the conversation history.

    Returns:
        dict: A dictionary containing various analysis results.
    """
    sentiment_scores = [analyze_sentiment(msg['content']) for msg in conversation_history]
    response_times = analyze_response_time(conversation_history)
    context_adaptation = analyze_context_adaptation(conversation_history)
    
    analysis_results = {
        'overall_sentiment': np.mean(sentiment_scores),
        'sentiment_trend': np.polyfit(range(len(sentiment_scores)), sentiment_scores, 1)[0],
        'avg_ai_response_time': response_times['avg_ai_response_time'],
        'avg_user_response_time': response_times['avg_user_response_time'],
        'context_adaptation_score': context_adaptation
    }
    
    logger.info("Completed conversation analysis")
    return analysis_results