"""
Attack Utilities

This module provides utility functions for creating and managing adversarial attacks.
"""

import torch
from typing import Tuple, Callable
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from isoadverse.attacks.text_fgsm import text_fgsm_attack
from isoadverse.attacks.text_pgd import text_pgd_attack
from isoadverse.attacks.textbugger import textbugger_attack
from isoadverse.attacks.deepwordbug import deepwordbug_attack
import logging

logger = logging.getLogger(__name__)

def get_model_and_tokenizer(model_name: str = 'bert-base-uncased') -> Tuple[torch.nn.Module, torch.nn.Module]:
    """
    Load a pre-trained model and tokenizer.

    Args:
        model_name (str): The name of the model to load.

    Returns:
        Tuple[torch.nn.Module, torch.nn.Module]: The loaded model and tokenizer.
    """
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    logger.info(f"Loaded model {model_name} on {device}")
    return model, tokenizer

def create_attack(attack_type: str, model: torch.nn.Module, tokenizer: torch.nn.Module) -> Callable:
    """
    Create an attack function based on the specified attack type.

    Args:
        attack_type (str): The type of attack to create.
        model (torch.nn.Module): The model to use for the attack.
        tokenizer (torch.nn.Module): The tokenizer to use for the attack.

    Returns:
        Callable: The attack function.
    """
    if attack_type == "fgsm":
        return lambda x: text_fgsm_attack(model, tokenizer, x, torch.tensor([1]), epsilon=0.3)
    elif attack_type == "pgd":
        return lambda x: text_pgd_attack(model, tokenizer, x, torch.tensor([1]), epsilon=0.3, alpha=0.1, num_steps=10)
    elif attack_type == "textbugger":
        return lambda x: textbugger_attack(x, num_bugs=5)
    elif attack_type == "deepwordbug":
        return lambda x: deepwordbug_attack(x, num_bugs=5)
    else:
        raise ValueError(f"Unknown attack type: {attack_type}")

def get_available_attacks() -> list:
    """
    Get a list of available attack types.

    Returns:
        list: A list of available attack types.
    """
    return ["fgsm", "pgd", "textbugger", "deepwordbug"]