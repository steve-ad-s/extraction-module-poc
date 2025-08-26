# 🚀 **Azure SQL to GCS Extractor**

**A simple, functional data extraction pipeline using DLT**

> 👨‍💻 **Perfect for Junior Data Engineers** - Extract data from Azure SQL to Google Cloud Storage in minutes!

## 🎯 **What This Does**

This pipeline extracts data from **Azure SQL databases** and saves it as **Parquet files** in **Google Cloud Storage** using the powerful **DLT (Data Load Tool)** framework.

**Key Features:**
- ✅ **Simple & Functional** - Clean, testable code following DRY principles
- ✅ **Single YAML Config** - All settings in one simple file
- ✅ **Environment Security** - All secrets in `.envrc` file
- ✅ **Multiple Deployment Options** - Direct execution, Airflow DAG, or FastAPI service
- ✅ **Error Handling** - Clear error messages and proper exception handling
- ✅ **Unit Tested** - Comprehensive test coverage for reliability

## 🗂️ **Project Structure**

```
azure-sql-extractor/
├── .envrc                     # 🔐 Environment variables (secrets)
├── config.yaml                # 🔧 Simple configuration file
├── dlt_core.py                # 📦 Core extraction functions (testable)
├── main.py                    # 🚀 Main entry point
├── test_dlt_core.py           # 🧪 Unit tests
├── airflow_functional_dag.py  # 📅 Airflow scheduling
└── cloud_run_services/        # ☁️ FastAPI microservice
    └── azure_sql_extractor/
        └── app/               # FastAPI application
```

## ⚡ **Quick Start for Sandbox Testing**

### **Step 1: Setup Environment**

```bash
# 1. Clone and enter the project
cd extraction-module-poc

# 2. Install dependencies  
uv sync

# 3. Copy the environment template
cp .envrc.example .envrc

# 4. Edit .envrc with your credentials
nano .envrc
```

### **Step 2: Configure Your Data Sources**

Edit `.envrc` with your connection details:

```bash
# Database connections
export AZURE_SQL_SALES_CONN="mssql+pyodbc://user:pass@server/db?driver=ODBC+Driver+18+for+SQL+Server"
export AZURE_SQL_FINANCE_CONN="mssql+pyodbc://user:pass@server/db?driver=ODBC+Driver+18+for+SQL+Server"

# API credentials  
export ZENDESK_SUBDOMAIN="your-company"
export ZENDESK_EMAIL="your-email@company.com"
export ZENDESK_API_TOKEN="your-api-token"

# Destination
export GCS_BUCKET_URL="gs://your-data-lake/extracts/"
```

### **Step 3: Test Your Setup**

```bash
# Load environment variables
source .envrc

# Test the pipeline (dry run)
python main.py
```

**Expected Output:**
```
🔍 Validating source connections...
   ✅ sales_db (sql_database)
   ✅ zendesk_support (zendesk_api)
✅ All connections validated successfully!

🚀 Starting pipeline 'multi_source_extraction_pipeline' with 2 sources
✅ sales_db (sql_database) completed successfully  
✅ zendesk_support (zendesk_api) completed successfully

📊 Pipeline Summary: multi_source_extraction_pipeline
====================================================
Total Sources: 2
✅ Successful: 2
❌ Failed: 0
📈 Success Rate: 100.0%
⏱️ Total Duration: 45.2s
🎉 Pipeline completed successfully!
```

## 🔧 **Adding New Data Sources**

### **Adding a SQL Database**

Create `config/sources/new_database.yaml`:

```yaml
name: "inventory_db"
source_type: "sql_database"
connection_config:
  connection_string: "${INVENTORY_DB_CONN}"
load_type: "incremental"
file_format: "parquet"
schema: "dbo"
dataset_name: "inventory_data"
tables: ["products", "stock_levels"]  # Optional: specific tables
```

Add to `config/pipeline.yaml`:

```yaml
sources:
  # ... existing sources ...
  - name: "inventory_db"
    source_type: "sql_database"
    connection_config:
      connection_string: "${INVENTORY_DB_CONN}"
    load_type: "incremental"
    file_format: "parquet"
    schema: "dbo"
    dataset_name: "inventory_data"
```

### **Adding a New API Source (e.g., Slack)**

The modular architecture makes it easy! Just:

1. **Create the source implementation** in `core/sources/slack_api.py`
2. **Register it** with `@SourceRegistry.register("slack_api")`  
3. **Add YAML configuration** in `config/sources/slack.yaml`

Example `slack.yaml`:
```yaml
name: "slack_messages"
source_type: "slack_api"
connection_config:
  bot_token: "${SLACK_BOT_TOKEN}"
  channel_ids: ["general", "engineering"]
load_type: "incremental"
file_format: "parquet"
dataset_name: "slack_data"
```

## 📅 **Running in Different Environments**

### **Local Development**
```bash
python main.py
```

### **Airflow (Production)**
```bash
# Copy the DAG file to your Airflow DAGs folder
cp airflow_functional_dag.py $AIRFLOW_HOME/dags/

# The DAG will run daily at 2 AM
```

### **Cloud Run (API)**
```bash
# Deploy as a REST API for on-demand extractions
gcloud run deploy extraction-api --source .
```

## 🛠️ **Configuration Examples**

### **Incremental vs Full Loads**

```yaml
# Incremental (only new/changed data)
load_type: "incremental"

# Full refresh (all data every time)  
load_type: "full"
```

### **Different File Formats**

```yaml
file_format: "parquet"  # Recommended for analytics
file_format: "jsonl"    # For document stores  
file_format: "csv"      # For simple analysis
```

### **Destination Options**

```yaml
destination:
  type: "filesystem"
  bucket_url: "gs://my-bucket/data/"  # Google Cloud Storage
  
destination:
  type: "filesystem" 
  bucket_url: "s3://my-bucket/data/"  # AWS S3
  
destination:
  type: "bigquery"  # Direct to BigQuery
  
destination:
  type: "snowflake"  # Direct to Snowflake
```

## 🔍 **Troubleshooting**

### **Connection Issues**

```bash
# Check if environment variables are loaded
echo $AZURE_SQL_SALES_CONN

# Test individual source connections
python -c "
from core.pipeline import create_pipeline_from_config
pipeline = create_pipeline_from_config('config/pipeline.yaml')
pipeline.validate_all_connections()
"
```

### **Common Errors**

**❌ "Environment variable not found"**
- Check your `.envrc` file and run `source .envrc`

**❌ "Connection validation failed"**  
- Verify your connection strings and credentials
- Test database connectivity manually

**❌ "Unknown source type"**
- Make sure the source type is registered in `core/sources/`

### **Getting Help**

```bash
# List available source types
python -c "
from core.sources.base import SourceRegistry
SourceRegistry.list_sources()
"

# View pipeline configuration
python -c "
from core.config import ConfigLoader
config = ConfigLoader.load_pipeline_config('config/pipeline.yaml')
print(config)
"
```

## 📊 **Monitoring Your Data**

### **Check Extraction Results**

Your data will be organized like this in GCS:

```
gs://your-bucket/extracts/
├── sales_data/
│   ├── customers/
│   │   ├── load_123.file_001.parquet
│   │   └── load_124.file_001.parquet  
│   └── orders/
│       ├── load_123.file_001.parquet
│       └── load_124.file_001.parquet
└── zendesk_data/
    ├── tickets/
    └── users/
```

### **Query Your Data**

```sql
-- BigQuery example
SELECT * FROM `your-project.sales_data.customers` 
WHERE _dlt_load_id = (
  SELECT MAX(_dlt_load_id) FROM `your-project.sales_data.customers`
)
```

## 🎓 **Next Steps for Learning**

1. **Try adding a new SQL database** following the examples
2. **Experiment with different file formats** (parquet, jsonl, csv)
3. **Set up Airflow scheduling** for production runs
4. **Create a new API source** (Twitter, GitHub, etc.)
5. **Add data validation** and quality checks

## 💡 **Best Practices**

- ✅ **Always test connections** before running full extractions
- ✅ **Use incremental loads** for large datasets  
- ✅ **Keep credentials in `.envrc`** never in code
- ✅ **Monitor pipeline logs** for errors and performance
- ✅ **Start small** with a few tables, then expand
- ✅ **Use consistent naming** for datasets and tables

---

## 🚀 **Ready to Extract Data?**

1. Set up your `.envrc` file with credentials
2. Configure your sources in `config/pipeline.yaml`  
3. Run `python main.py`
4. Watch your data flow into GCS! 🎉

**Happy Data Engineering!** 🎯