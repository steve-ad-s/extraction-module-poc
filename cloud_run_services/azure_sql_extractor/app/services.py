"""Business logic services for DLT extraction.

Following separation of concerns principle - pure business logic.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path for dlt_core import
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from dlt_core import (
    run_all_tables_extraction,
    run_multi_table_extraction, 
    run_single_table_extraction,
)

from .config import Settings
from .models import (
    AllTablesRequest,
    AllTablesResponse,
    ExtractionResult,
    FailedExtractionResult,
    FileFormat,
    LoadType,
    MultiTableRequest,
    MultiTableResponse,
    SingleTableRequest,
    SingleTableResponse,
    SuccessfulExtractionResult,
)


class DLTExtractionService:
    """Service class for DLT extraction operations.
    
    Encapsulates business logic and provides clean interface.
    """
    
    def __init__(self, settings: Settings) -> None:
        """Initialize the service with settings."""
        self.settings = settings
    
    async def extract_single_table(self, request: SingleTableRequest) -> SingleTableResponse:
        """Extract a single table using DLT functional core.
        
        Args:
            request: Single table extraction request
            
        Returns:
            Single table extraction response
        """
        # Use the functional DLT core
        result = run_single_table_extraction(
            connection_string=self.settings.azure_sql_connection_string,
            table_name=request.table,
            bucket_url=self.settings.gcs_bucket_url,
            schema=request.schema,
            load_type=request.load_type.value,
            loader_file_format=self.settings.loader_file_format,
        )
        
        # Convert result to appropriate response model
        extraction_result = self._convert_result(result)
        
        return SingleTableResponse(
            table=request.table,
            result=extraction_result
        )
    
    async def extract_multiple_tables(self, request: MultiTableRequest) -> MultiTableResponse:
        """Extract multiple tables using DLT functional core.
        
        Args:
            request: Multiple table extraction request
            
        Returns:
            Multiple table extraction response
        """
        # Use the functional DLT core
        result = run_multi_table_extraction(
            connection_string=self.settings.azure_sql_connection_string,
            tables=request.tables,
            bucket_url=self.settings.gcs_bucket_url,
            schema=request.schema,
            load_type=request.load_type.value,
            loader_file_format=self.settings.loader_file_format,
        )
        
        # Convert result to appropriate response model
        extraction_result = self._convert_result(result)
        
        return MultiTableResponse(
            tables=request.tables,
            result=extraction_result
        )
    
    async def extract_all_tables(self, request: AllTablesRequest) -> AllTablesResponse:
        """Extract all tables using DLT functional core.
        
        Args:
            request: All tables extraction request
            
        Returns:
            All tables extraction response
        """
        # Use the functional DLT core
        result = run_all_tables_extraction(
            connection_string=self.settings.azure_sql_connection_string,
            bucket_url=self.settings.gcs_bucket_url,
            schema=request.schema,
            load_type=request.load_type.value,
            loader_file_format=self.settings.loader_file_format,
        )
        
        # Convert result to appropriate response model
        extraction_result = self._convert_result(result)
        
        return AllTablesResponse(
            schema=request.schema,
            result=extraction_result
        )
    
    def _convert_result(self, result: dict[str, str]) -> ExtractionResult:
        """Convert DLT core result to Pydantic model.
        
        Args:
            result: Result dictionary from DLT core
            
        Returns:
            Appropriate ExtractionResult model
        """
        if result["status"] == "success":
            return SuccessfulExtractionResult(
                load_info=result["load_info"],
                bucket_url=result.get("bucket_url", self.settings.gcs_bucket_url),
                file_format=FileFormat(result.get("file_format", self.settings.loader_file_format)),
                pipeline_name=result.get("pipeline_name", "unknown"),
            )
        else:
            return FailedExtractionResult(
                error=result.get("error", "Unknown error occurred")
            )
