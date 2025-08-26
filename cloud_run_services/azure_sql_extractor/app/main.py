"""FastAPI application factory and configuration.

Following FastAPI best practices for application structure.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import get_settings
from .routers import extraction, health


def create_app() -> FastAPI:
    """Create and configure FastAPI application.
    
    Following factory pattern for clean application setup.
    
    Returns:
        Configured FastAPI application
    """
    # Get settings
    settings = get_settings()
    
    # Create FastAPI app with metadata
    app = FastAPI(
        title=settings.api_title,
        description=settings.api_description,
        version=settings.api_version,
        docs_url="/docs" if settings.environment != "production" else None,
        redoc_url="/redoc" if settings.environment != "production" else None,
        openapi_url="/openapi.json" if settings.environment != "production" else None,
    )
    
    # Add CORS middleware for Cloud Run
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Include routers
    app.include_router(health.router)
    app.include_router(extraction.router)
    
    return app


# Create app instance
app = create_app()
