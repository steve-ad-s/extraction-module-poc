"""Configuration management for FastAPI DLT extractor.

Following FastAPI best practices for settings management.
"""

import os
from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings using Pydantic BaseSettings."""
    
    # Azure SQL Configuration
    azure_sql_connection_string: str = Field(
        ..., 
        env="AZURE_SQL_CONNECTION_STRING",
        description="Azure SQL database connection string"
    )
    
    # GCS Configuration
    gcs_bucket_url: str = Field(
        "gs://default-bucket/azure-sql-data/",
        env="GCS_BUCKET_URL", 
        description="Google Cloud Storage bucket URL"
    )
    
    # Database Configuration
    sql_schema: str = Field(
        "dbo",
        env="SQL_SCHEMA",
        description="Default SQL schema name"
    )
    
    # File Configuration
    loader_file_format: str = Field(
        "parquet",
        env="LOADER_FILE_FORMAT",
        description="Default file format for DLT loader"
    )
    
    # API Configuration
    api_title: str = Field(
        "Azure SQL DLT Extractor API",
        description="FastAPI application title"
    )
    
    api_description: str = Field(
        "Functional FastAPI service for extracting Azure SQL data to Parquet files using DLT",
        description="FastAPI application description"
    )
    
    api_version: str = Field(
        "1.0.0",
        description="API version"
    )
    
    # Cloud Run Configuration
    port: int = Field(
        8080,
        env="PORT",
        description="Port for the FastAPI application"
    )
    
    # Environment Configuration
    environment: str = Field(
        "production",
        env="ENVIRONMENT",
        description="Current environment (development, staging, production)"
    )
    
    class Config:
        """Pydantic configuration."""
        env_file = ".env"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance.
    
    Using lru_cache for dependency injection performance.
    FastAPI best practice for settings management.
    """
    return Settings()


# Validation on import
_settings = get_settings()
if not _settings.azure_sql_connection_string:
    raise ValueError("AZURE_SQL_CONNECTION_STRING environment variable is required")
