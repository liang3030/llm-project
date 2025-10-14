"""Custom exceptions for the application."""


class LLMProjectException(Exception):
    """Base exception for the LLM project."""
    pass


class ConfigurationError(LLMProjectException):
    """Raised when there's a configuration error."""
    pass


class ModelError(LLMProjectException):
    """Raised when there's an error with the LLM model."""
    pass


class APIError(LLMProjectException):
    """Raised when there's an API-related error."""
    pass