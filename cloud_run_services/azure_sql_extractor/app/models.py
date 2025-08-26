"""Pydantic models for FastAPI DLT extractor.

Following FastAPI best practices for request/response validation.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, ConfigDict


class LoadType(str, Enum):
    """Enum for DLT load types."""
    INCREMENTAL = "incremental"
    FULL = "full"


class FileFormat(str, Enum):
    """Enum for supported file formats."""
    PARQUET = "parquet"
    JSONL = "jsonl"
    CSV = "csv"


class ExtractionStatus(str, Enum):
    """Enum for extraction statuses."""
    SUCCESS = "success"
    FAILED = "failed"
    IN_PROGRESS = "in_progress"


# Request Models
class SingleTableRequest(BaseModel):
    """Request model for single table extraction."""
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "table": "customers",
                    "load_type": "incremental",
                    "schema": "dbo"
                }
            ]
        }
    )
    
    table: str = Field(..., description="Name of the table to extract")
    load_type: LoadType = Field(LoadType.INCREMENTAL, description="Type of load operation")
    schema: str = Field("dbo", description="Database schema name")


class MultiTableRequest(BaseModel):
    """Request model for multiple table extraction."""
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "tables": ["customers", "orders", "products"],
                    "load_type": "incremental",
                    "schema": "dbo"
                }
            ]
        }
    )
    
    tables: list[str] = Field(..., description="List of table names to extract", min_length=1)
    load_type: LoadType = Field(LoadType.INCREMENTAL, description="Type of load operation")
    schema: str = Field("dbo", description="Database schema name")


class AllTablesRequest(BaseModel):
    """Request model for all tables extraction."""
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "load_type": "incremental",
                    "schema": "dbo"
                }
            ]
        }
    )
    
    load_type: LoadType = Field(LoadType.INCREMENTAL, description="Type of load operation")
    schema: str = Field("dbo", description="Database schema name")


# Response Models
class ExtractionResult(BaseModel):
    """Base model for extraction results."""
    status: ExtractionStatus = Field(..., description="Status of the extraction")
    message: Optional[str] = Field(None, description="Additional information")


class SuccessfulExtractionResult(ExtractionResult):
    """Model for successful extraction results."""
    status: ExtractionStatus = Field(ExtractionStatus.SUCCESS, description="Success status")
    load_info: str = Field(..., description="DLT load information")
    bucket_url: str = Field(..., description="GCS bucket URL where files were saved")
    file_format: FileFormat = Field(..., description="Format of the extracted files")
    pipeline_name: str = Field(..., description="DLT pipeline name used")


class FailedExtractionResult(ExtractionResult):
    """Model for failed extraction results."""
    status: ExtractionStatus = Field(ExtractionStatus.FAILED, description="Failed status")
    error: str = Field(..., description="Error message describing the failure")


class SingleTableResponse(BaseModel):
    """Response model for single table extraction."""
    table: str = Field(..., description="Name of the extracted table")
    result: ExtractionResult = Field(..., description="Extraction result")


class MultiTableResponse(BaseModel):
    """Response model for multiple table extraction."""
    tables: list[str] = Field(..., description="List of table names that were extracted")
    result: ExtractionResult = Field(..., description="Extraction result")


class AllTablesResponse(BaseModel):
    """Response model for all tables extraction."""
    schema: str = Field(..., description="Database schema that was extracted")
    result: ExtractionResult = Field(..., description="Extraction result")


class HealthResponse(BaseModel):
    """Response model for health check."""
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "status": "healthy",
                    "service": "fastapi-azure-sql-dlt-parquet",
                    "destination": "filesystem",
                    "file_format": "parquet",
                    "version": "1.0.0"
                }
            ]
        }
    )
    
    status: str = Field(..., description="Health status")
    service: str = Field(..., description="Service name")
    destination: str = Field(..., description="DLT destination type")
    file_format: FileFormat = Field(..., description="Default file format")
    version: str = Field("1.0.0", description="API version")


class ErrorResponse(BaseModel):
    """Standard error response model."""
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Additional error details")
