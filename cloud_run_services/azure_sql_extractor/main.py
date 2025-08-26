"""Main entry point for FastAPI DLT extractor.

This is the main entry point that Cloud Run will use.
"""

import uvicorn

from app.main import app
from app.config import get_settings


if __name__ == "__main__":
    settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=settings.port,
        reload=settings.environment == "development"
    )
