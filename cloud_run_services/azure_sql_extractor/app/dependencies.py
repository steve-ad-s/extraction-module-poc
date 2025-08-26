"""FastAPI dependencies for dependency injection.

Following FastAPI best practices for dependency management.
"""

from functools import lru_cache

from fastapi import Depends

from .config import Settings, get_settings
from .services import DLTExtractionService


@lru_cache()
def get_dlt_service(settings: Settings = Depends(get_settings)) -> DLTExtractionService:
    """Get DLT extraction service instance.
    
    Using dependency injection pattern for clean architecture.
    
    Args:
        settings: Application settings
        
    Returns:
        DLT extraction service instance
    """
    return DLTExtractionService(settings)
