#!/usr/bin/env python3
"""
Script to populate the download_audit table with dummy data for testing.

Usage:
    python scripts/populate_download_audit.py [--count 100] [--days 30]

Options:
    --count: Number of dummy records to create (default: 100)
    --days: Days back to generate data for (default: 30)
"""
import argparse
import random
from datetime import datetime, timedelta, timezone
from faker import Faker

from odp.db import Session
from odp.db.models import DownloadAudit


# Sample data for realistic records
FIRST_NAMES = ['John', 'Jane', 'Robert', 'Mary', 'Michael', 'Patricia', 'David', 'Jennifer', 'James', 'Linda']
LAST_NAMES = ['Smith', 'Johnson', 'Williams', 'Brown', 'Jones', 'Garcia', 'Miller', 'Davis', 'Rodriguez', 'Martinez']
ORGANISATIONS = [
    'University of Cape Town',
    'CSIR',
    'Stellenbosch University',
    'SAEON',
    'South African Weather Service',
    'Department of Environmental Affairs',
    'Rhodes University',
    'Wits University',
    'Nelson Mandela University',
    'University of Pretoria',
    'Agricultural Research Council',
    'Council for Scientific and Industrial Research',
]

DOMAINS = [
    'gmail.com', 'yahoo.com', 'outlook.com', 'hotmail.com',
    'uct.ac.za', 'sun.ac.za', 'wits.ac.za', 'up.ac.za',
    'csir.co.za', 'saeon.ac.za', 'weather.gov.za',
]

DOWNLOAD_TYPES = ['single_record', 'zip_bundle', 'single_record', 'zip_bundle', 'zip_bundle']
IP_PREFIXES = [
    '102.165.', '196.28.', '196.44.', '197.97.',  # South African IPs
    '192.168.', '10.0.', '172.16.',  # Local network ranges
]


def generate_fake_data(count=100, days_back=30):
    """Generate dummy download audit records."""
    fake = Faker()
    records = []

    now = datetime.now(timezone.utc)
    start_date = now - timedelta(days=days_back)

    for _ in range(count):
        # Random timestamp within the range
        random_seconds = random.randint(0, int((now - start_date).total_seconds()))
        timestamp = start_date + timedelta(seconds=random_seconds)

        # Generate user info
        first_name = random.choice(FIRST_NAMES)
        last_name = random.choice(LAST_NAMES)
        name = f"{first_name} {last_name}"
        domain = random.choice(DOMAINS)
        email = f"{first_name.lower()}.{last_name.lower()}@{domain}"

        organisation = random.choice(ORGANISATIONS)

        # Generate download info
        download_type = random.choice(DOWNLOAD_TYPES)
        file_size = None
        if download_type == 'zip_bundle':
            # ZIP bundles typically larger: 50MB to 2GB
            file_size = random.randint(50 * 1024 * 1024, 2 * 1024 * 1024 * 1024)
        elif download_type == 'single_record':
            # Single records smaller: 100KB to 500MB
            if random.random() > 0.3:  # 70% have files
                file_size = random.randint(100 * 1024, 500 * 1024 * 1024)

        # Download URL
        if download_type == 'zip_bundle':
            download_url = 'client_generated_zip_bundle'
        else:
            download_url = f"https://example.com/data/{fake.uuid4()}/download"

        # IP address
        ip_prefix = random.choice(IP_PREFIXES)
        ip_address = f"{ip_prefix}{random.randint(1, 254)}.{random.randint(0, 255)}"

        # Success rate: 95% successful, 5% failed
        success = random.random() > 0.05

        # User agent (browser simulation)
        user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15',
        ]
        user_agent = random.choice(user_agents)

        # Metadata
        meta = {
            'name': name,
            'email': email,
            'organisation': organisation,
            'download_type': download_type,
        }

        # Add additional metadata for zip bundles
        if download_type == 'zip_bundle':
            record_count = random.randint(1, 50)
            meta['source'] = 'MIMS-UI'
            meta['record_count'] = record_count
            meta['dois'] = [f"10.15493/uuid-{i}" for i in range(record_count)]
        else:
            meta['source'] = 'MIMS-UI-Detail-Page'

        audit = DownloadAudit(
            client_id='mims-client',
            user_id=None,  # Anonymous
            download_url=download_url,
            ip_address=ip_address,
            user_agent=user_agent,
            file_size=file_size,
            success=success,
            timestamp=timestamp,
            meta=meta,
        )

        records.append(audit)

    return records


def main():
    parser = argparse.ArgumentParser(description='Populate download_audit table with dummy data')
    parser.add_argument('--count', type=int, default=100, help='Number of dummy records to create')
    parser.add_argument('--days', type=int, default=30, help='Days back to generate data for')
    args = parser.parse_args()

    print(f"Generating {args.count} dummy download audit records...")
    print(f"Date range: Last {args.days} days\n")

    # Generate records
    records = generate_fake_data(count=args.count, days_back=args.days)

    # Insert into database
    print(f"Inserting {len(records)} records into database...")
    with Session.begin():
        for record in records:
            Session.add(record)

    print(f"✅ Successfully inserted {len(records)} dummy records!")
    print("\nSample data statistics:")

    # Query to show sample statistics
    with Session() as session:
        total = session.query(DownloadAudit).count()
        single_records = session.query(DownloadAudit).filter(
            DownloadAudit.meta['download_type'].astext == 'single_record'
        ).count()
        zip_bundles = session.query(DownloadAudit).filter(
            DownloadAudit.meta['download_type'].astext == 'zip_bundle'
        ).count()
        successful = session.query(DownloadAudit).filter(DownloadAudit.success == True).count()
        failed = session.query(DownloadAudit).filter(DownloadAudit.success == False).count()

        print(f"  Total records: {total}")
        print(f"  Single record downloads: {single_records}")
        print(f"  ZIP bundle downloads: {zip_bundles}")
        print(f"  Successful: {successful}")
        print(f"  Failed: {failed}")

        # Get unique users
        from sqlalchemy import func
        unique_users = session.query(
            func.count(func.distinct(DownloadAudit.meta['email']))
        ).scalar()
        print(f"  Unique users: {unique_users}")


if __name__ == '__main__':
    main()
