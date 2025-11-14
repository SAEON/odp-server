# Populating Download Audit Table with Dummy Data

Since the database connection isn't directly available in this environment, here are the methods you can use to populate the download_audit table with test data:

## Option 1: Using Python Script (Recommended)

The simplest method that doesn't require external dependencies:

```bash
cd /path/to/odp-build/odp-server

# Make sure you're in the right environment with ODP dependencies
source .venv/bin/activate  # or your virtual environment

# Run the simplified script
python scripts/populate_simple.py
```

**What it does:**
- Generates 50 realistic dummy records
- Spreads them over the last 30 days
- Creates a mix of single record and ZIP bundle downloads
- Sets 95% success rate, 5% failures
- Shows statistics after insertion

**Requirements:**
- ODP dependencies already installed (no external packages needed)
- Database connection configured
- download_audit table already exists

## Option 2: Using SQL Script

For direct database access via psql:

```bash
cd /path/to/odp-build/odp-server

# Using psql (replace credentials as needed)
psql -U odp_user -d odp_db -h localhost -f sql/download/populate_dummy_data.sql

# Or set password in environment
PGPASSWORD=your_password psql -U odp_user -d odp_db -h localhost -f sql/download/populate_dummy_data.sql
```

**What it does:**
- Inserts 20 hand-crafted records directly via SQL
- Quick option for rapid testing
- No Python required

## Option 3: Using Python Script with Faker (Most Realistic)

If you want more varied data, install Faker first:

```bash
pip install faker

python scripts/populate_download_audit.py --count 100 --days 30
```

## Verification

After populating, verify the data was inserted:

```bash
# Using psql
psql -U odp_user -d odp_db -h localhost -c "SELECT COUNT(*) FROM download_audit;"

# Should return something like:
#  count
# -------
#     50
# (1 row)
```

## Checking the Admin Dashboard

After populating, navigate to:
- Download Logs: `http://localhost:5000/admin/downloads`
- Analytics: `http://localhost:5000/admin/downloads/analytics`

## Clearing Test Data

When ready to clear the test data:

```bash
# SQL command
psql -U odp_user -d odp_db -h localhost -c "DELETE FROM download_audit WHERE client_id = 'mims-client';"

# Or from psql prompt
DELETE FROM download_audit WHERE client_id = 'mims-client';
```

## Troubleshooting

### Database Connection Errors

If you get "connection refused":
1. Make sure PostgreSQL is running
2. Check the database host/port/credentials
3. Verify the database exists: `createdb -U odp_user odp_db`

### Table Doesn't Exist

If you get "relation 'download_audit' does not exist":
1. Run the Alembic migration: `alembic upgrade head`
2. This will create the table and indexes

### Python Import Errors

If you get "ModuleNotFoundError: No module named 'odp'":
1. Make sure you're in the right directory: `cd /path/to/odp-build/odp-server`
2. Make sure the virtual environment is activated
3. Make sure ODP is installed: `pip install -e .`

## Next Steps

Once data is populated:

1. **Test the API:**
   ```bash
   curl "http://localhost:8000/download/logs?page=1"
   curl "http://localhost:8000/download/stats"
   ```

2. **View in Admin Dashboard:**
   - Navigate to `http://localhost:5000/admin/downloads`
   - Try different filters
   - View analytics at `http://localhost:5000/admin/downloads/analytics`

3. **Export CSV:**
   ```bash
   curl "http://localhost:8000/download/export/csv" -o downloads.csv
   ```

## Sample Data Details

All dummy data includes:

- **Names**: Realistic first and last names
- **Emails**: Various domains (gmail.com, university domains, etc.)
- **Organisations**: Real South African institutions:
  - University of Cape Town
  - CSIR
  - Stellenbosch University
  - SAEON
  - South African Weather Service
  - And others

- **IP Addresses**:
  - South African ranges (102.165.x.x, 196.28.x.x, etc.)
  - Local network ranges

- **File Sizes**:
  - Single records: 100KB to 500MB
  - ZIP bundles: 50MB to 2GB

- **Timestamps**: Distributed over configurable date range (default 30 days)

- **Download Types**: Mix of single_record and zip_bundle

- **Success Rate**: 95% successful, 5% failures

## Running the Script Steps

### Step 1: Navigate to correct directory
```bash
cd /home/n.bingani/Documents/Work/odp-build/odp-server
```

### Step 2: Activate virtual environment (if needed)
```bash
# Typically one of:
source .venv/bin/activate
source venv/bin/activate
source env/bin/activate
```

### Step 3: Run the population script
```bash
python scripts/populate_simple.py
```

### Step 4: Check the output
Should show:
```
Generating 50 dummy download audit records...
Inserting 50 records into database...
✅ Successfully inserted 50 dummy records!

📊 Database Statistics:
  Total records: 50
  Single record downloads: 17
  ZIP bundle downloads: 33
  Successful: 48
  Failed: 2
```

---

The scripts are ready to use whenever you have database access!
