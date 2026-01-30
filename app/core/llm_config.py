"""
LLM Engine Configuration - DSPy Integration
Simplified function-based configuration without unnecessary wrapper classes.
"""

import dspy

from app.core.config import settings


def initialize_llm_engine() -> dspy.LM:
    """
    Initialize and configure DSPy Language Model.
    
    Returns:
        Configured DSPy LM instance
        
    Raises:
        ValueError: If configuration fails
    """
    lm = dspy.OllamaLocal(
        model=settings.ollama_model,
        base_url=settings.ollama_base_url,
        max_tokens=2000,
        temperature=0.7,
        timeout_s=300,  # 5 minutes timeout
    )
    
    # Configure DSPy globally
    dspy.configure(lm=lm)
    
    return lm