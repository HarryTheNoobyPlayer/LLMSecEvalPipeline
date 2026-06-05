"""Code generation stage, turn prompts into Python code via an LLM."""

from llmseceval.code_generator.base import BaseCodeGenerator, GenerationResult
from llmseceval.code_generator.extractor import extract_code, strip_think
from llmseceval.code_generator.ollama_generator import OllamaGenerator
from llmseceval.code_generator.runner import CodeGeneratorRunner

__all__ = [
    "BaseCodeGenerator",
    "GenerationResult",
    "OllamaGenerator",
    "CodeGeneratorRunner",
    "extract_code",
    "strip_think",
]
