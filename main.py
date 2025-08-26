#!/usr/bin/env python3
"""Simple entry point for Azure SQL to GCS data extraction.

Functional, DRY approach that keeps the logic simple and testable.
"""

import os
from typing import Dict, Any
import yaml

from dlt_core import run_all_tables_extraction


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def expand_env_vars(value: str) -> str:
    """Expand environment variables in string.
    
    Args:
        value: String that may contain ${VAR} patterns
        
    Returns:
        String with environment variables expanded
    """
    import re
    
    def replace_var(match):
        var_name = match.group(1)
        return os.getenv(var_name, match.group(0))
    
    return re.sub(r'\$\{([^}]+)\}', replace_var, value)


def main() -> None:
    """Run the data extraction pipeline."""
    try:
        # Load configuration
        config = load_config()
        
        # Expand environment variables
        connection_string = expand_env_vars(config["azure_sql"]["connection_string"])
        bucket_url = expand_env_vars(config["gcs"]["bucket_url"])
        
        print("🚀 Starting Azure SQL to GCS extraction...")
        
        # Run extraction
        result = run_all_tables_extraction(
            connection_string=connection_string,
            bucket_url=bucket_url,
            schema=config["azure_sql"]["schema"],
            load_type=config.get("load_type", "replace"),
            loader_file_format=config.get("file_format", "parquet")
        )
        
        # Check result
        if result["status"] == "success":
            print("✅ Extraction completed successfully!")
            print(f"📁 Data saved to: {bucket_url}")
            exit(0)
        else:
            print(f"❌ Extraction failed: {result['error']}")
            exit(1)
            
    except Exception as e:
        print(f"💥 Pipeline failed: {e}")
        exit(1)


if __name__ == "__main__":
    main()
