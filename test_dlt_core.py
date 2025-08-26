"""Simple unit tests for dlt_core functions.

Tests the core extraction functions in a minimal, functional way.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from dlt_core import (
    create_sql_source,
    create_gcs_destination, 
    run_extraction,
    run_single_table_extraction,
    run_multi_table_extraction,
    run_all_tables_extraction
)


def test_create_sql_source_with_table_names():
    """Test creating SQL source with specific tables."""
    with patch('dlt_core.sql_database') as mock_sql_db:
        mock_source = Mock()
        mock_sql_db.return_value = mock_source
        
        result = create_sql_source(
            connection_string="test_conn",
            schema="test_schema", 
            table_names=["table1", "table2"]
        )
        
        assert result == mock_source
        mock_sql_db.assert_called_once_with(
            credentials="test_conn",
            schema="test_schema",
            table_names=["table1", "table2"]
        )


def test_create_sql_source_all_tables():
    """Test creating SQL source for all tables."""
    with patch('dlt_core.sql_database') as mock_sql_db:
        mock_source = Mock()
        mock_sql_db.return_value = mock_source
        
        result = create_sql_source(
            connection_string="test_conn",
            schema="test_schema"
        )
        
        assert result == mock_source
        mock_sql_db.assert_called_once_with(
            credentials="test_conn",
            schema="test_schema"
        )


def test_create_gcs_destination():
    """Test creating GCS destination."""
    with patch('dlt_core.filesystem') as mock_fs:
        mock_dest = Mock()
        mock_fs.return_value = mock_dest
        
        result = create_gcs_destination("gs://test-bucket/path")
        
        assert result == mock_dest
        mock_fs.assert_called_once_with(bucket_url="gs://test-bucket/path")


@patch('dlt_core.dlt')
@patch('dlt_core.create_gcs_destination')
@patch('dlt_core.create_sql_source')
def test_run_extraction_success(mock_create_source, mock_create_dest, mock_dlt):
    """Test successful extraction."""
    # Setup mocks
    mock_source = Mock()
    mock_dest = Mock()
    mock_pipeline = Mock()
    mock_load_info = Mock()
    
    mock_create_source.return_value = mock_source
    mock_create_dest.return_value = mock_dest
    mock_dlt.pipeline.return_value = mock_pipeline
    mock_pipeline.run.return_value = mock_load_info
    
    # Run extraction
    result = run_extraction(
        connection_string="test_conn",
        bucket_url="gs://test-bucket",
        schema="test_schema",
        table_names=["table1"],
        load_type="replace",
        loader_file_format="parquet",
        pipeline_name="test_pipeline"
    )
    
    # Verify result
    assert result["status"] == "success"
    assert "load_info" in result
    assert result["pipeline_name"] == "test_pipeline"
    assert result["bucket_url"] == "gs://test-bucket"
    assert result["file_format"] == "parquet"
    
    # Verify calls
    mock_create_source.assert_called_once_with(
        connection_string="test_conn",
        schema="test_schema", 
        table_names=["table1"]
    )
    mock_create_dest.assert_called_once_with("gs://test-bucket")
    mock_dlt.pipeline.assert_called_once_with(
        pipeline_name="test_pipeline",
        destination=mock_dest,
        dataset_name="azure_sql_data"
    )
    mock_pipeline.run.assert_called_once_with(
        mock_source,
        write_disposition="replace",
        loader_file_format="parquet"
    )


@patch('dlt_core.dlt')
@patch('dlt_core.create_gcs_destination')  
@patch('dlt_core.create_sql_source')
def test_run_extraction_failure(mock_create_source, mock_create_dest, mock_dlt):
    """Test extraction failure handling."""
    # Setup mocks to raise exception
    mock_dlt.pipeline.side_effect = Exception("Test error")
    
    # Run extraction
    result = run_extraction(
        connection_string="test_conn",
        bucket_url="gs://test-bucket"
    )
    
    # Verify error handling
    assert result["status"] == "error"
    assert "Test error" in result["error"]


@patch('dlt_core.run_extraction')
def test_run_single_table_extraction(mock_run_extraction):
    """Test single table extraction wrapper."""
    mock_result = {"status": "success"}
    mock_run_extraction.return_value = mock_result
    
    result = run_single_table_extraction(
        connection_string="test_conn",
        table_name="test_table",
        bucket_url="gs://test-bucket",
        schema="test_schema"
    )
    
    assert result == mock_result
    mock_run_extraction.assert_called_once_with(
        connection_string="test_conn",
        bucket_url="gs://test-bucket",
        schema="test_schema",
        table_names=["test_table"],
        load_type="replace",
        loader_file_format="parquet",
        pipeline_name="single_table_test_table"
    )


@patch('dlt_core.run_extraction')
def test_run_multi_table_extraction(mock_run_extraction):
    """Test multi table extraction wrapper."""
    mock_result = {"status": "success"}
    mock_run_extraction.return_value = mock_result
    
    result = run_multi_table_extraction(
        connection_string="test_conn",
        tables=["table1", "table2"],
        bucket_url="gs://test-bucket"
    )
    
    assert result == mock_result
    mock_run_extraction.assert_called_once_with(
        connection_string="test_conn",
        bucket_url="gs://test-bucket",
        schema="dbo",
        table_names=["table1", "table2"],
        load_type="replace",
        loader_file_format="parquet",
        pipeline_name="multi_table_2_tables"
    )


@patch('dlt_core.run_extraction')
def test_run_all_tables_extraction(mock_run_extraction):
    """Test all tables extraction wrapper."""
    mock_result = {"status": "success"}
    mock_run_extraction.return_value = mock_result
    
    result = run_all_tables_extraction(
        connection_string="test_conn",
        bucket_url="gs://test-bucket",
        schema="custom_schema"
    )
    
    assert result == mock_result
    mock_run_extraction.assert_called_once_with(
        connection_string="test_conn",
        bucket_url="gs://test-bucket",
        schema="custom_schema",
        table_names=None,
        load_type="replace",
        loader_file_format="parquet",
        pipeline_name="all_tables_custom_schema"
    )


def test_expand_env_vars():
    """Test environment variable expansion from main.py."""
    from main import expand_env_vars
    import os
    
    # Set test environment variable
    os.environ['TEST_VAR'] = 'test_value'
    
    # Test expansion
    result = expand_env_vars('prefix_${TEST_VAR}_suffix')
    assert result == 'prefix_test_value_suffix'
    
    # Test no variables
    result = expand_env_vars('no_variables_here')
    assert result == 'no_variables_here'
    
    # Clean up
    del os.environ['TEST_VAR']


@patch('main.yaml.safe_load')
@patch('builtins.open')
def test_load_config(mock_open, mock_yaml):
    """Test configuration loading from main.py."""
    from main import load_config
    
    # Setup mock
    mock_config = {"azure_sql": {"connection_string": "test"}}
    mock_yaml.return_value = mock_config
    mock_file = MagicMock()
    mock_open.return_value.__enter__.return_value = mock_file
    
    # Test loading
    result = load_config("test_config.yaml")
    
    assert result == mock_config
    mock_open.assert_called_once_with("test_config.yaml", "r")
    mock_yaml.assert_called_once_with(mock_file)
