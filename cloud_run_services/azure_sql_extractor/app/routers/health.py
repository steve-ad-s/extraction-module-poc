"""FastAPI router for health check endpoints.

Following FastAPI best practices for health monitoring.
"""

from fastapi import APIRouter, Depends, status

from ..config import get_settings, Settings
from ..models import HealthResponse

# Create router with tags
router = APIRouter(
    tags=["health"]
)


@router.get(
    "/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="Health Check",
    description="Check the health status of the FastAPI DLT extraction service",
    response_description="Service health information"
)
async def health_check(
    settings: Settings = Depends(get_settings)
) -> HealthResponse:
    """Health check endpoint.
    
    Returns the current health status of the service along with configuration 
    information. This endpoint is useful for container orchestration health checks.
    
    Args:
        settings: Application settings (injected)
        
    Returns:
        Health status and service information
    """
    return HealthResponse(
        status="healthy",
        service="fastapi-azure-sql-dlt-parquet", 
        destination="filesystem",
        file_format=settings.loader_file_format,
        version=settings.api_version
    )


@router.get(
    "/",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="Root Health Check",
    description="Root endpoint that returns health status",
    response_description="Service health information"
)
async def root_health_check(
    settings: Settings = Depends(get_settings)
) -> HealthResponse:
    """Root endpoint that also serves as health check.
    
    Args:
        settings: Application settings (injected)
        
    Returns:
        Health status and service information
    """
    return await health_check(settings)
