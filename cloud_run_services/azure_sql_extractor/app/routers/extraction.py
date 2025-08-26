"""FastAPI router for DLT extraction endpoints.

Following FastAPI best practices for modular routing.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import JSONResponse

from ..dependencies import get_dlt_service
from ..models import (
    AllTablesRequest,
    AllTablesResponse,
    ErrorResponse,
    MultiTableRequest,
    MultiTableResponse,
    SingleTableRequest,
    SingleTableResponse,
)
from ..services import DLTExtractionService

# Create router with prefix and tags
router = APIRouter(
    prefix="/extract",
    tags=["extraction"],
    responses={
        500: {"model": ErrorResponse, "description": "Internal server error"},
        400: {"model": ErrorResponse, "description": "Bad request"},
    }
)


@router.post(
    "/table",
    response_model=SingleTableResponse,
    status_code=status.HTTP_200_OK,
    summary="Extract Single Table",
    description="Extract a single table from Azure SQL Database to Parquet files in GCS",
    response_description="Single table extraction result"
)
async def extract_single_table(
    request: SingleTableRequest,
    service: DLTExtractionService = Depends(get_dlt_service)
) -> SingleTableResponse:
    """Extract a single table to Parquet files.
    
    This endpoint extracts data from a single Azure SQL table and saves it as 
    Parquet files in Google Cloud Storage using DLT (Data Load Tool).
    
    Args:
        request: Single table extraction parameters
        service: DLT extraction service (injected)
        
    Returns:
        Extraction result with status and file information
        
    Raises:
        HTTPException: If extraction fails
    """
    try:
        return await service.extract_single_table(request)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to extract table '{request.table}': {str(e)}"
        ) from e


@router.post(
    "/tables",
    response_model=MultiTableResponse,
    status_code=status.HTTP_200_OK,
    summary="Extract Multiple Tables",
    description="Extract multiple tables from Azure SQL Database to Parquet files in GCS",
    response_description="Multiple table extraction result"
)
async def extract_multiple_tables(
    request: MultiTableRequest,
    service: DLTExtractionService = Depends(get_dlt_service)
) -> MultiTableResponse:
    """Extract multiple tables to Parquet files.
    
    This endpoint extracts data from multiple Azure SQL tables and saves them as 
    Parquet files in Google Cloud Storage using DLT (Data Load Tool).
    
    Args:
        request: Multiple table extraction parameters
        service: DLT extraction service (injected)
        
    Returns:
        Extraction result with status and file information
        
    Raises:
        HTTPException: If extraction fails
    """
    try:
        return await service.extract_multiple_tables(request)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to extract tables {request.tables}: {str(e)}"
        ) from e


@router.post(
    "/all",
    response_model=AllTablesResponse,
    status_code=status.HTTP_200_OK,
    summary="Extract All Tables",
    description="Extract all tables from Azure SQL Database schema to Parquet files in GCS",
    response_description="All tables extraction result"
)
async def extract_all_tables(
    request: AllTablesRequest,
    service: DLTExtractionService = Depends(get_dlt_service)
) -> AllTablesResponse:
    """Extract all tables from a schema to Parquet files.
    
    This endpoint uses DLT's auto-discovery feature to extract all tables from 
    the specified Azure SQL schema and saves them as Parquet files in Google 
    Cloud Storage.
    
    Args:
        request: All tables extraction parameters
        service: DLT extraction service (injected)
        
    Returns:
        Extraction result with status and file information
        
    Raises:
        HTTPException: If extraction fails
    """
    try:
        return await service.extract_all_tables(request)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to extract all tables from schema '{request.schema}': {str(e)}"
        ) from e
