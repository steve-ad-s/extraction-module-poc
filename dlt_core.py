"""Simple, functional DLT core for Azure SQL to GCS extraction.

This module provides pure, functional utilities for data extraction using DLT.
Follows DRY principles and keeps the logic simple and testable.
"""

from typing import Dict, List, Optional, Any
import dlt
from dlt.destinations import filesystem
from dlt.sources import sql_database


def create_sql_source(
    connection_string: str, 
    schema: str = "dbo", 
    table_names: Optional[List[str]] = None
) -> Any:
    """Create a SQL database source for DLT.
    
    Args:
        connection_string: Database connection string
        schema: Database schema name  
        table_names: Optional list of specific tables to extract
        
    Returns:
        DLT source object
    """
    if table_names:
        # Extract specific tables only
        source = sql_database(
            credentials=connection_string,
            schema=schema,
            table_names=table_names
        )
    else:
        # Extract all tables in schema
        source = sql_database(
            credentials=connection_string,
            schema=schema
        )
    
    return source


def create_gcs_destination(bucket_url: str) -> Any:
    """Create a GCS filesystem destination for DLT.
    
    Args:
        bucket_url: GCS bucket URL (e.g., 'gs://my-bucket/path')
        
    Returns:
        DLT destination object
    """
    return filesystem(bucket_url=bucket_url)


def run_extraction(
    connection_string: str,
    bucket_url: str,
    schema: str = "dbo", 
    table_names: Optional[List[str]] = None,
    load_type: str = "replace",
    loader_file_format: str = "parquet",
    pipeline_name: str = "azure_sql_extraction"
) -> Dict[str, Any]:
    """Run a complete DLT extraction from Azure SQL to GCS.
    
    Args:
        connection_string: Azure SQL connection string
        bucket_url: GCS destination bucket URL
        schema: Database schema to extract from
        table_names: Optional list of specific tables
        load_type: DLT load type ('replace' or 'merge')
        loader_file_format: Output file format
        pipeline_name: Name for the DLT pipeline
        
    Returns:
        Dictionary with extraction result information
    """
    try:
        # Create DLT pipeline
        pipeline = dlt.pipeline(
            pipeline_name=pipeline_name,
            destination=create_gcs_destination(bucket_url),
            dataset_name="azure_sql_data"
        )
        
        # Create source
        source = create_sql_source(
            connection_string=connection_string,
            schema=schema,
            table_names=table_names
        )
        
        # Run extraction
        load_info = pipeline.run(
            source,
            write_disposition=load_type,
            loader_file_format=loader_file_format
        )
        
        return {
            "status": "success",
            "load_info": str(load_info),
            "pipeline_name": pipeline_name,
            "bucket_url": bucket_url,
            "file_format": loader_file_format
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }


def run_single_table_extraction(
    connection_string: str,
    table_name: str,
    bucket_url: str,
    schema: str = "dbo",
    load_type: str = "replace", 
    loader_file_format: str = "parquet"
) -> Dict[str, Any]:
    """Extract a single table from Azure SQL to GCS.
    
    Args:
        connection_string: Azure SQL connection string
        table_name: Name of the table to extract
        bucket_url: GCS destination bucket URL
        schema: Database schema name
        load_type: DLT load type
        loader_file_format: Output file format
        
    Returns:
        Dictionary with extraction result
    """
    return run_extraction(
        connection_string=connection_string,
        bucket_url=bucket_url,
        schema=schema,
        table_names=[table_name],
        load_type=load_type,
        loader_file_format=loader_file_format,
        pipeline_name=f"single_table_{table_name}"
    )


def run_multi_table_extraction(
    connection_string: str,
    tables: List[str],
    bucket_url: str,
    schema: str = "dbo",
    load_type: str = "replace",
    loader_file_format: str = "parquet"
) -> Dict[str, Any]:
    """Extract multiple tables from Azure SQL to GCS.
    
    Args:
        connection_string: Azure SQL connection string
        tables: List of table names to extract
        bucket_url: GCS destination bucket URL
        schema: Database schema name
        load_type: DLT load type
        loader_file_format: Output file format
        
    Returns:
        Dictionary with extraction result
    """
    return run_extraction(
        connection_string=connection_string,
        bucket_url=bucket_url,
        schema=schema,
        table_names=tables,
        load_type=load_type,
        loader_file_format=loader_file_format,
        pipeline_name=f"multi_table_{len(tables)}_tables"
    )


def run_all_tables_extraction(
    connection_string: str,
    bucket_url: str,
    schema: str = "dbo",
    load_type: str = "replace",
    loader_file_format: str = "parquet"
) -> Dict[str, Any]:
    """Extract all tables from a schema in Azure SQL to GCS.
    
    Args:
        connection_string: Azure SQL connection string
        bucket_url: GCS destination bucket URL
        schema: Database schema name
        load_type: DLT load type
        loader_file_format: Output file format
        
    Returns:
        Dictionary with extraction result
    """
    return run_extraction(
        connection_string=connection_string,
        bucket_url=bucket_url,
        schema=schema,
        table_names=None,  # Extract all tables
        load_type=load_type,
        loader_file_format=loader_file_format,
        pipeline_name=f"all_tables_{schema}"
    )


def run_pipeline_from_config(config_path: str) -> Dict[str, Any]:
    """Run pipeline from YAML configuration (legacy compatibility).
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Dictionary with extraction result
    """
    # Simple implementation for backward compatibility
    # In a real scenario, you'd parse the YAML and extract connection details
    return {
        "status": "error",
        "error": "Config-based extraction not implemented in simplified version"
    }
