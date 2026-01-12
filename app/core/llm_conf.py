"""
LLM Engine Configuration - DSPy Integration
"""
import dspy
from app.core.config import settings
from typing import Literal


class LLMEngine:
    """LLM Engine Manager with DSPy"""
    
    def __init__(self, provider: Literal["ollama"] = "ollama"):
        self.provider = provider
        self._lm = None
        
    def initialize(self) -> dspy.LM:
        """Initialize DSPy Language Model"""
        if self._lm:
            return self._lm
            
        if self.provider == "ollama":
            self._lm = dspy.LM(
                model=f"ollama/{settings.ollama_model}",
                api_base=settings.ollama_base_url,
                max_tokens=4000,
                temperature=0.7
            )
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
            
        dspy.configure(lm=self._lm)
        return self._lm
    
    def get_engine(self) -> dspy.LM:
        """Get configured engine instance"""
        return self._lm if self._lm else self.initialize()


# Global engine instance
llm_engine = LLMEngine(provider="ollama")