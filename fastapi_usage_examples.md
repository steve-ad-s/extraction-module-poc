# FastAPI DLT Extractor - Usage Examples

The new FastAPI-based DLT extractor provides a clean, modular API for extracting Azure SQL data to Parquet files in GCS.

## 🏗️ **Modular Architecture**

```
cloud_run_services/azure_sql_extractor/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI app factory
│   ├── config.py            # Pydantic settings
│   ├── models.py            # Request/response models
│   ├── services.py          # Business logic layer
│   ├── dependencies.py      # Dependency injection
│   └── routers/
│       ├── __init__.py
│       ├── health.py        # Health check endpoints
│       └── extraction.py    # DLT extraction endpoints
└── main.py                  # Entry point
```

## 🚀 **API Endpoints**

### Health Check
```bash
curl http://localhost:8080/health
```

**Response:**
```json
{
  "status": "healthy",
  "service": "fastapi-azure-sql-dlt-parquet",
  "destination": "filesystem", 
  "file_format": "parquet",
  "version": "1.0.0"
}
```

### Single Table Extraction
```bash
curl -X POST "http://localhost:8080/extract/table" \
  -H "Content-Type: application/json" \
  -d '{
    "table": "customers",
    "load_type": "incremental",
    "schema": "dbo"
  }'
```

**Response:**
```json
{
  "table": "customers",
  "result": {
    "status": "success",
    "load_info": "Load info details...",
    "bucket_url": "gs://my-bucket/azure-sql-data/",
    "file_format": "parquet",
    "pipeline_name": "azure_sql_table_customers"
  }
}
```

### Multiple Tables Extraction
```bash
curl -X POST "http://localhost:8080/extract/tables" \
  -H "Content-Type: application/json" \
  -d '{
    "tables": ["customers", "orders", "products"],
    "load_type": "incremental", 
    "schema": "dbo"
  }'
```

### All Tables Extraction
```bash
curl -X POST "http://localhost:8080/extract/all" \
  -H "Content-Type: application/json" \
  -d '{
    "load_type": "full",
    "schema": "sales"
  }'
```

## 📋 **Request Models (Pydantic)**

### SingleTableRequest
```python
class SingleTableRequest(BaseModel):
    table: str                    # Required
    load_type: LoadType = "incremental"  # Optional, default "incremental" 
    schema: str = "dbo"          # Optional, default "dbo"
```

### MultiTableRequest
```python
class MultiTableRequest(BaseModel):
    tables: list[str]            # Required, min 1 table
    load_type: LoadType = "incremental"
    schema: str = "dbo"
```

### AllTablesRequest
```python
class AllTablesRequest(BaseModel):
    load_type: LoadType = "incremental"
    schema: str = "dbo"
```

## 🎯 **Key FastAPI Benefits**

### 1. **Automatic Documentation**
- Interactive docs at `/docs` (Swagger UI)
- Alternative docs at `/redoc`
- OpenAPI JSON at `/openapi.json`

### 2. **Request Validation**
```python
# Automatic validation with helpful error messages
{
  "detail": [
    {
      "type": "missing",
      "loc": ["body", "table"],
      "msg": "Field required"
    }
  ]
}
```

### 3. **Type Safety**
```python
# Full type hints and editor support
async def extract_single_table(
    request: SingleTableRequest,
    service: DLTExtractionService = Depends(get_dlt_service)
) -> SingleTableResponse:
```

### 4. **Dependency Injection**
```python
# Clean separation of concerns
@lru_cache()
def get_dlt_service(settings: Settings = Depends(get_settings)) -> DLTExtractionService:
    return DLTExtractionService(settings)
```

### 5. **Async Support**
```python
# Native async support for better performance
async def extract_single_table(request: SingleTableRequest) -> SingleTableResponse:
    # Non-blocking operations
    return await service.extract_single_table(request)
```

## 🔧 **Configuration (Environment Variables)**

```bash
# Required
export AZURE_SQL_CONNECTION_STRING="mssql+pyodbc://..."

# Optional (with defaults)
export GCS_BUCKET_URL="gs://my-bucket/azure-sql-data/"
export SQL_SCHEMA="dbo"
export LOADER_FILE_FORMAT="parquet"
export PORT=8080
export ENVIRONMENT="production"
```

## 🏃 **Running the Service**

### Development
```bash
cd cloud_run_services/azure_sql_extractor/
python main.py
```

### Production (Docker)
```bash
docker build -t fastapi-dlt-extractor .
docker run -p 8080:8080 \
  -e AZURE_SQL_CONNECTION_STRING="..." \
  -e GCS_BUCKET_URL="gs://my-bucket/" \
  fastapi-dlt-extractor
```

## 🎪 **Comparison: Flask vs FastAPI**

| Feature | Flask (Old) | FastAPI (New) |
|---------|-------------|---------------|
| **Validation** | Manual | Automatic with Pydantic |
| **Documentation** | Manual | Auto-generated |
| **Type Safety** | None | Full type hints |
| **Async Support** | Limited | Native |
| **Performance** | Standard | High performance |
| **Error Handling** | Manual | Structured |
| **Architecture** | Monolithic | Modular with DI |

## 🏆 **Result**

**FastAPI provides:**
- ✅ **Better Developer Experience** - Auto docs, validation, type safety
- ✅ **Modular Architecture** - Clean separation of concerns
- ✅ **Production Ready** - Async, performance, error handling
- ✅ **Maintainable** - Dependency injection, clear structure
- ✅ **Still Functional** - Uses the same DLT functional core!
