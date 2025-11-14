#!/usr/bin/env python3
"""
Simplified script to populate download_audit with dummy data.
This version doesn't require the Faker library.

Usage:
    python scripts/populate_simple.py
"""
import random
from datetime import datetime, timedelta, timezone

from odp.db import Session
from odp.db.models import DownloadAudit


# Sample realistic data
NAMES = [
    ('John', 'Smith'), ('Jane', 'Doe'), ('Robert', 'Johnson'), ('Mary', 'Williams'),
    ('Michael', 'Brown'), ('Patricia', 'Garcia'), ('David', 'Miller'), ('Jennifer', 'Davis'),
    ('James', 'Rodriguez'), ('Linda', 'Martinez'), ('Barbara', 'Thompson'), ('Richard', 'Anderson'),
    ('Susan', 'Taylor'), ('Charles', 'White'), ('Karen', 'Thomas'), ('Peter', 'Jackson'),
    ('Nancy', 'Martin'), ('Joseph', 'Harris'), ('Margaret', 'Lee'), ('Christopher', 'Clark'),
]

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

DOMAINS = ['gmail.com', 'yahoo.com', 'outlook.com', 'hotmail.com', 'uct.ac.za', 'sun.ac.za', 'wits.ac.za']

DOWNLOAD_TYPES = ['single_record', 'zip_bundle']
IP_PREFIXES = ['102.165', '196.28', '196.44', '197.97', '192.168']


def generate_dummy_records(count=50):
    """Generate dummy download audit records without external dependencies."""
    records = []
    now = datetime.now(timezone.utc)

    for i in range(count):
        # Random timestamp within last 30 days
        days_ago = random.randint(0, 30)
        hours_ago = random.randint(0, 24)
        timestamp = now - timedelta(days=days_ago, hours=hours_ago)

        # User info
        first, last = random.choice(NAMES)
        email = f"{first.lower()}.{last.lower()}@{random.choice(DOMAINS)}"
        name = f"{first} {last}"
        organisation = random.choice(ORGANISATIONS)

        # Download info
        download_type = random.choice(DOWNLOAD_TYPES)
        file_size = None
        if download_type == 'zip_bundle':
            file_size = random.randint(50 * 1024 * 1024, 2 * 1024 * 1024 * 1024)
        elif random.random() > 0.3:  # 70% have files
            file_size = random.randint(100 * 1024, 500 * 1024 * 1024)

        # IP address
        ip_address = f"{random.choice(IP_PREFIXES)}.{random.randint(1, 254)}.{random.randint(0, 255)}"

        # Success: 95% success, 5% failure
        success = random.random() > 0.05

        # User agent
        user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64)',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)',
            'Mozilla/5.0 (X11; Linux x86_64)',
        ]
        user_agent = random.choice(user_agents)

        # Metadata
        meta = {
            'name': name,
            'email': email,
            'organisation': organisation,
            'download_type': download_type,
        }

        if download_type == 'zip_bundle':
            meta['source'] = 'MIMS-UI'
            meta['record_count'] = random.randint(1, 20)
        else:
            meta['source'] = 'MIMS-UI-Detail-Page'

        download_url = 'client_generated_zip_bundle' if download_type == 'zip_bundle' else f'https://example.com/data/{i}/download'

        audit = DownloadAudit(
            client_id='mims-client',
            user_id=None,
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
    print("Generating 50 dummy download audit records...")

    # Generate records
    records = generate_dummy_records(count=50)

    # Insert into database
    print(f"Inserting {len(records)} records into database...")
    try:
        with Session.begin():
            for record in records:
                Session.add(record)

        print(f"✅ Successfully inserted {len(records)} dummy records!")

        # Show statistics
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

            print("\n📊 Database Statistics:")
            print(f"  Total records: {total}")
            print(f"  Single record downloads: {single_records}")
            print(f"  ZIP bundle downloads: {zip_bundles}")
            print(f"  Successful: {successful}")
            print(f"  Failed: {failed}")

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        print("\nTroubleshooting:")
        print("1. Make sure the database is running")
        print("2. Check that the download_audit table exists")
        print("3. Run: alembic upgrade head")
        return 1

    return 0


if __name__ == '__main__':
    exit(main())
