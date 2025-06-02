"""
LLM Provider Module

This module provides a flexible interface for working with different LLM providers
including DeepSeek and OpenAI.
"""
import os
import logging
from enum import Enum
from typing import Optional, Dict, Any, Union

# Import from langchain_openai instead of the deprecated langchain_community.chat_models
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

class LLMProvider(Enum):
    """Supported LLM providers"""
    DEEPSEEK = "deepseek"
    OPENAI = "openai"
    
class LLMFactory:
    """Factory class for creating LLM instances based on provider"""
    
    @staticmethod
    def create_llm(
        provider: Union[str, LLMProvider] = "deepseek", 
        model_name: Optional[str] = None,
        temperature: float = 0, 
        json_mode: bool = True, 
        max_tokens: int = 2000,
        request_timeout: float = 60.0
    ) -> Optional[ChatOpenAI]:
        """
        Create an LLM instance based on the specified provider
        
        Args:
            provider: LLM provider ("deepseek" or "openai")
            model_name: Model name to use (provider-specific)
                - For DeepSeek: deepseek-chat, deepseek-coder, etc.
                - For OpenAI: gpt-4.1-mini, gpt-4, gpt-3.5-turbo, etc.
            temperature: Sampling temperature (0-1)
            json_mode: Whether to force JSON output format
            max_tokens: Maximum tokens to generate 
            request_timeout: Request timeout in seconds
            
        Returns:
            Configured LLM instance or None if configuration failed
        """
        # Convert string provider to enum if needed
        if isinstance(provider, str):
            try:
                provider = LLMProvider(provider.lower())
            except ValueError:
                logging.error(f"Unsupported LLM provider: {provider}")
                return None
        
        # Configure model kwargs including response format if json_mode is True
        model_kwargs = {}
        if json_mode:
            model_kwargs["response_format"] = {"type": "json_object"}
        
        # Create LLM based on provider
        if provider == LLMProvider.DEEPSEEK:
            # Default model for DeepSeek
            if model_name is None:
                model_name = "deepseek-chat"
                
            # Get DeepSeek API key
            api_key = os.getenv("DEEPSEEK_API_KEY")
            if not api_key:
                logging.error("DEEPSEEK_API_KEY not found in environment variables")
                return None
                
            logging.info(f"Initializing DeepSeek API with model={model_name}, temp={temperature}")
            
            return ChatOpenAI(
                model=model_name,
                openai_api_key=api_key,
                openai_api_base="https://api.deepseek.com/v1",
                temperature=temperature,
                max_tokens=max_tokens,
                model_kwargs=model_kwargs,
                request_timeout=request_timeout
            )
            
        elif provider == LLMProvider.OPENAI:
            # Default model for OpenAI
            if model_name is None:
                model_name = "gpt-3.5-turbo"
                
            # Get OpenAI API key
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                logging.error("OPENAI_API_KEY not found in environment variables")
                return None
                
            logging.info(f"Initializing OpenAI API with model={model_name}, temp={temperature}")
            
            return ChatOpenAI(
                model=model_name,
                openai_api_key=api_key,
                temperature=temperature,
                max_tokens=max_tokens,
                model_kwargs=model_kwargs,
                request_timeout=request_timeout
            )
        
        return None
    
    @staticmethod
    def get_provider_from_env() -> LLMProvider:
        """
        Determine which provider to use based on environment variables.
        Prefers DeepSeek if both API keys are available.
        
        Returns:
            LLMProvider enum value
        """
        if os.getenv("DEEPSEEK_API_KEY"):
            return LLMProvider.DEEPSEEK
        elif os.getenv("OPENAI_API_KEY"):
            return LLMProvider.OPENAI
        else:
            # Default to OpenAI, but it will fail without API key
            return LLMProvider.OPENAI
            
def create_llm(
    provider: Optional[Union[str, LLMProvider]] = None,
    model_name: Optional[str] = None,
    temperature: float = 0, 
    json_mode: bool = True, 
    max_tokens: int = 2000,
    request_timeout: float = 60.0
) -> Optional[ChatOpenAI]:
    """
    Convenience function to create an LLM instance
    
    If provider is None, automatically selects based on available API keys
    """
    if provider is None:
        provider = LLMFactory.get_provider_from_env()
        
    return LLMFactory.create_llm(
        provider=provider,
        model_name=model_name,
        temperature=temperature,
        json_mode=json_mode,
        max_tokens=max_tokens,
        request_timeout=request_timeout
    )
