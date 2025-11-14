# Download Audit & Reporting System

This document explains how to populate the download audit database with dummy data for testing and demonstration purposes.

## Overview

The download audit system tracks all downloads from the MIMS catalog and provides reporting capabilities through:
- **Backend API** - `/download/logs`, `/download/stats`, `/download/export/csv`
- **Admin Dashboard** - `/admin/downloads` and `/admin/downloads/analytics`

## Populating the Database with Dummy Data

There are two methods to populate the `download_audit` table with test data:

### Method 1: Using Python Script (Recommended)

The Python script generates realistic, randomized dummy data with proper distribution.

**Requirements:**
```bash
pip install faker
```

**Usage:**

```bash
# Generate 100 records spread over 30 days (default)
python scripts/populate_download_audit.py

# Generate 500 records spread over 90 days
python scripts/populate_download_audit.py --count 500 --days 90

# Generate 1000 records for last 6 months
python scripts/populate_download_audit.py --count 1000 --days 180
```

**Features:**
- ✅ Realistic user names and email addresses
- ✅ Random selection from real South African organisations
- ✅ Proper IP address ranges (South African + local network)
- ✅ Random timestamps distributed across the date range
- ✅ Realistic file sizes (100KB-500MB for single records, 50MB-2GB for bundles)
- ✅ 95% success rate, 5% failure rate
- ✅ Browser user agents
- ✅ Complete metadata (organisation, download type, record count for bundles)
- ✅ Shows statistics after insertion

**Sample Output:**
```
Generating 100 dummy download audit records...
Date range: Last 30 days

Inserting 100 records into database...
✅ Successfully inserted 100 dummy records!

Sample data statistics:
  Total records: 100
  Single record downloads: 35
  ZIP bundle downloads: 65
  Successful: 95
  Failed: 5
  Unique users: 28
```

### Method 2: Using SQL Script

A pre-written SQL script with 20 manually crafted records for quick testing.

**Usage:**

```bash
# Connect to PostgreSQL and run the script
psql -U odp_user -d odp_db -h localhost -f sql/download/populate_dummy_data.sql

# Or from within psql:
\i sql/download/populate_dummy_data.sql
```

**Features:**
- ✅ 20 manually crafted realistic records
- ✅ Mix of single record and ZIP bundle downloads
- ✅ Spread across different dates
- ✅ Includes some failed downloads
- ✅ Real South African organisations
- ✅ Complete metadata

**Sample Output:**
```
Successfully inserted 20 download audit records
```

## Testing the Reporting System

### 1. Test the Backend API

After populating data, test the API endpoints:

```bash
# Get download logs
curl "http://localhost:8000/download/logs?page=1&size=10"

# Get statistics
curl "http://localhost:8000/download/stats"

# Export CSV
curl "http://localhost:8000/download/export/csv?start_date=2025-11-01&end_date=2025-11-14" \
  -o downloads.csv

# Filter by email
curl "http://localhost:8000/download/logs?email=john.smith@gmail.com"

# Filter by organisation
curl "http://localhost:8000/download/logs?organisation=University%20of%20Cape%20Town&page=1"

# Filter by download type
curl "http://localhost:8000/download/logs?download_type=zip_bundle&page=1"
```

### 2. Test the Admin Dashboard

Navigate to the admin interface:

1. **Download Logs Page**: `http://localhost:5000/admin/downloads`
   - View all downloads in a paginated table
   - Filter by date range, email, organisation, download type
   - Export results to CSV

2. **Analytics Dashboard**: `http://localhost:5000/admin/downloads/analytics`
   - View summary metrics (total downloads, unique users, data volume, success rate)
   - See breakdown by download type
   - Check success/failure rates

## Data Schema

The `download_audit` table structure:

```sql
CREATE TABLE download_audit (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    client_id VARCHAR NOT NULL,
    user_id VARCHAR NULL,
    download_url VARCHAR NULL,
    ip_address VARCHAR NULL,
    user_agent TEXT NULL,
    file_size BIGINT NULL,
    success BOOLEAN NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    meta JSONB NULL
);
```

### Meta Field Structure

The `meta` field stores JSON with the following structure:

```json
{
    "name": "John Smith",
    "email": "john.smith@gmail.com",
    "organisation": "University of Cape Town",
    "download_type": "single_record|zip_bundle",
    "source": "MIMS-UI|MIMS-UI-Detail-Page",

    // For zip_bundle downloads:
    "record_count": 5,
    "dois": ["10.15493/uuid-1", "10.15493/uuid-2", ...],
    "individual_urls": ["https://...", "https://...", ...]
}
```

## Sample Organisations in Dummy Data

- University of Cape Town
- CSIR
- Stellenbosch University
- SAEON
- South African Weather Service
- Department of Environmental Affairs
- Rhodes University
- Wits University
- Nelson Mandela University
- University of Pretoria
- Agricultural Research Council

## Clearing Dummy Data

To remove dummy data and start fresh:

```sql
DELETE FROM download_audit WHERE client_id = 'mims-client';
```

Or to clear all records:

```sql
TRUNCATE TABLE download_audit;
```

## Performance Notes

After populating data, make sure the database indexes are created:

```bash
# Run Alembic migration to add indexes
alembic upgrade head
```

This creates the following indexes for optimal query performance:
- `idx_download_audit_timestamp` - for date range queries
- `idx_download_audit_client_id` - for user tracking
- `idx_download_audit_meta_email` - for email filtering
- `idx_download_audit_meta_organisation` - for organisation filtering
- `idx_download_audit_meta_download_type` - for type filtering
- `idx_download_audit_meta` - full meta JSON index
- `idx_download_audit_timestamp_client_id` - for common filter combinations

## Next Steps

After populating and testing:

1. ✅ Verify all API endpoints respond correctly
2. ✅ Check admin dashboard displays data properly
3. ✅ Test filtering and pagination
4. ✅ Verify CSV export works
5. ✅ Check analytics calculations are correct
6. ✅ Clear dummy data when ready for production: `DELETE FROM download_audit WHERE client_id = 'mims-client';`

## API Endpoint Documentation

### GET /download/logs

Get paginated download logs with filtering.

**Query Parameters:**
- `start_date` (YYYY-MM-DD): Filter from date
- `end_date` (YYYY-MM-DD): Filter to date
- `email`: Filter by user email
- `organisation`: Filter by organisation
- `download_type`: Filter by type (single_record, zip_bundle)
- `page` (default: 1): Page number
- `size` (default: 50, max: 200): Items per page

**Response:**
```json
{
    "total": 100,
    "page": 1,
    "size": 50,
    "total_pages": 2,
    "items": [
        {
            "id": 1,
            "client_id": "mims-client",
            "timestamp": "2025-11-14T10:30:00+00:00",
            "email": "user@example.com",
            "name": "John Smith",
            "organisation": "University of Cape Town",
            "download_type": "zip_bundle",
            "file_size": 536870912,
            "success": true,
            "ip_address": "102.165.10.5"
        }
    ]
}
```

### GET /download/stats

Get aggregated download statistics.

**Query Parameters:**
- `start_date` (YYYY-MM-DD): Optional date filter
- `end_date` (YYYY-MM-DD): Optional date filter

**Response:**
```json
{
    "total_downloads": 100,
    "unique_users": 28,
    "total_data_volume": 45000000000,
    "successful_downloads": 95,
    "failed_downloads": 5,
    "downloads_by_type": {
        "single_record": 35,
        "zip_bundle": 65
    }
}
```

### GET /download/export/csv

Export download logs as CSV file.

**Query Parameters:** Same as `/logs` endpoint

**Response:** CSV file with columns:
- ID
- Timestamp
- Name
- Email
- Organisation
- Download Type
- File Size (bytes)
- Success
- IP Address
- Download URL
