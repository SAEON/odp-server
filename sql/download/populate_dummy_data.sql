-- SQL script to populate download_audit table with dummy data
-- This script generates 100 realistic test records spread over the last 30 days

-- Sample data for testing the download audit reporting features
INSERT INTO download_audit (
    client_id, user_id, download_url, ip_address, user_agent,
    file_size, success, timestamp, meta
) VALUES

-- Single record downloads
('mims-client', NULL, 'https://example.com/data/uuid-001/download', '102.165.10.5', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)', 52428800, true, NOW() - INTERVAL '29 days',
 jsonb_build_object('name', 'John Smith', 'email', 'john.smith@gmail.com', 'organisation', 'University of Cape Town', 'download_type', 'single_record', 'source', 'MIMS-UI-Detail-Page')),

('mims-client', NULL, 'https://example.com/data/uuid-002/download', '196.28.15.12', 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)', 157286400, true, NOW() - INTERVAL '28 days',
 jsonb_build_object('name', 'Jane Doe', 'email', 'jane.doe@uct.ac.za', 'organisation', 'University of Cape Town', 'download_type', 'single_record', 'source', 'MIMS-UI-Detail-Page')),

('mims-client', NULL, 'https://example.com/data/uuid-003/download', '196.44.22.8', 'Mozilla/5.0 (X11; Linux x86_64)', 314572800, false, NOW() - INTERVAL '27 days',
 jsonb_build_object('name', 'Robert Johnson', 'email', 'robert.johnson@csir.co.za', 'organisation', 'CSIR', 'download_type', 'single_record', 'source', 'MIMS-UI-Detail-Page')),

('mims-client', NULL, 'https://example.com/data/uuid-004/download', '197.97.50.3', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0)', 104857600, true, NOW() - INTERVAL '26 days',
 jsonb_build_object('name', 'Mary Williams', 'email', 'mary.williams@sun.ac.za', 'organisation', 'Stellenbosch University', 'download_type', 'single_record', 'source', 'MIMS-UI-Detail-Page')),

('mims-client', NULL, 'https://example.com/data/uuid-005/download', '102.165.20.45', 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)', 209715200, true, NOW() - INTERVAL '25 days',
 jsonb_build_object('name', 'Michael Brown', 'email', 'michael.brown@outlook.com', 'organisation', 'South African Weather Service', 'download_type', 'single_record', 'source', 'MIMS-UI-Detail-Page')),

-- ZIP bundle downloads
('mims-client', NULL, 'client_generated_zip_bundle', '196.28.35.67', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)', 536870912, true, NOW() - INTERVAL '24 days',
 jsonb_build_object('name', 'Patricia Garcia', 'email', 'patricia.garcia@saeon.ac.za', 'organisation', 'SAEON', 'download_type', 'zip_bundle', 'source', 'MIMS-UI', 'record_count', 5, 'dois', jsonb_build_array('10.15493/uuid-1', '10.15493/uuid-2', '10.15493/uuid-3', '10.15493/uuid-4', '10.15493/uuid-5'))),

('mims-client', NULL, 'client_generated_zip_bundle', '196.44.88.22', 'Mozilla/5.0 (X11; Linux x86_64)', 1073741824, true, NOW() - INTERVAL '23 days',
 jsonb_build_object('name', 'David Miller', 'email', 'david.miller@weather.gov.za', 'organisation', 'South African Weather Service', 'download_type', 'zip_bundle', 'source', 'MIMS-UI', 'record_count', 12, 'dois', jsonb_build_array('10.15493/uuid-6', '10.15493/uuid-7', '10.15493/uuid-8', '10.15493/uuid-9', '10.15493/uuid-10', '10.15493/uuid-11', '10.15493/uuid-12', '10.15493/uuid-13', '10.15493/uuid-14', '10.15493/uuid-15', '10.15493/uuid-16', '10.15493/uuid-17'))),

('mims-client', NULL, 'client_generated_zip_bundle', '197.97.60.100', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)', 2147483648, true, NOW() - INTERVAL '22 days',
 jsonb_build_object('name', 'Jennifer Davis', 'email', 'jennifer.davis@wits.ac.za', 'organisation', 'Wits University', 'download_type', 'zip_bundle', 'source', 'MIMS-UI', 'record_count', 25, 'dois', jsonb_build_array('10.15493/uuid-18', '10.15493/uuid-19', '10.15493/uuid-20', '10.15493/uuid-21', '10.15493/uuid-22', '10.15493/uuid-23', '10.15493/uuid-24', '10.15493/uuid-25', '10.15493/uuid-26', '10.15493/uuid-27', '10.15493/uuid-28', '10.15493/uuid-29', '10.15493/uuid-30', '10.15493/uuid-31', '10.15493/uuid-32', '10.15493/uuid-33', '10.15493/uuid-34', '10.15493/uuid-35', '10.15493/uuid-36', '10.15493/uuid-37', '10.15493/uuid-38', '10.15493/uuid-39', '10.15493/uuid-40', '10.15493/uuid-41', '10.15493/uuid-42'))),

('mims-client', NULL, 'client_generated_zip_bundle', '102.165.75.88', 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)', 805306368, false, NOW() - INTERVAL '21 days',
 jsonb_build_object('name', 'James Rodriguez', 'email', 'james.rodriguez@up.ac.za', 'organisation', 'University of Pretoria', 'download_type', 'zip_bundle', 'source', 'MIMS-UI', 'record_count', 8, 'dois', jsonb_build_array('10.15493/uuid-43', '10.15493/uuid-44', '10.15493/uuid-45', '10.15493/uuid-46', '10.15493/uuid-47', '10.15493/uuid-48', '10.15493/uuid-49', '10.15493/uuid-50'))),

('mims-client', NULL, 'https://example.com/data/uuid-051/download', '196.28.100.150', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)', 629145600, true, NOW() - INTERVAL '20 days',
 jsonb_build_object('name', 'Linda Martinez', 'email', 'linda.martinez@gmail.com', 'organisation', 'Agricultural Research Council', 'download_type', 'single_record', 'source', 'MIMS-UI-Detail-Page')),

('mims-client', NULL, 'client_generated_zip_bundle', '196.44.45.99', 'Mozilla/5.0 (X11; Linux x86_64)', 419430400, true, NOW() - INTERVAL '19 days',
 jsonb_build_object('name', 'Barbara Thompson', 'email', 'barbara.thompson@yahoo.com', 'organisation', 'Rhodes University', 'download_type', 'zip_bundle', 'source', 'MIMS-UI', 'record_count', 3, 'dois', jsonb_build_array('10.15493/uuid-52', '10.15493/uuid-53', '10.15493/uuid-54'))),

('mims-client', NULL, 'https://example.com/data/uuid-055/download', '197.97.12.45', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0)', 314572800, true, NOW() - INTERVAL '18 days',
 jsonb_build_object('name', 'Richard Anderson', 'email', 'richard.anderson@hotmail.com', 'organisation', 'Nelson Mandela University', 'download_type', 'single_record', 'source', 'MIMS-UI-Detail-Page')),

('mims-client', NULL, 'client_generated_zip_bundle', '102.165.50.22', 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)', 1610612736, true, NOW() - INTERVAL '17 days',
 jsonb_build_object('name', 'Susan Taylor', 'email', 'susan.taylor@uct.ac.za', 'organisation', 'University of Cape Town', 'download_type', 'zip_bundle', 'source', 'MIMS-UI', 'record_count', 15, 'dois', jsonb_build_array('10.15493/uuid-56', '10.15493/uuid-57', '10.15493/uuid-58', '10.15493/uuid-59', '10.15493/uuid-60', '10.15493/uuid-61', '10.15493/uuid-62', '10.15493/uuid-63', '10.15493/uuid-64', '10.15493/uuid-65', '10.15493/uuid-66', '10.15493/uuid-67', '10.15493/uuid-68', '10.15493/uuid-69', '10.15493/uuid-70'))),

('mims-client', NULL, 'https://example.com/data/uuid-071/download', '196.28.65.78', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)', 209715200, true, NOW() - INTERVAL '16 days',
 jsonb_build_object('name', 'Charles White', 'email', 'charles.white@csir.co.za', 'organisation', 'CSIR', 'download_type', 'single_record', 'source', 'MIMS-UI-Detail-Page')),

('mims-client', NULL, 'https://example.com/data/uuid-072/download', '196.44.33.44', 'Mozilla/5.0 (X11; Linux x86_64)', 104857600, false, NOW() - INTERVAL '15 days',
 jsonb_build_object('name', 'Karen Thomas', 'email', 'karen.thomas@sun.ac.za', 'organisation', 'Stellenbosch University', 'download_type', 'single_record', 'source', 'MIMS-UI-Detail-Page')),

('mims-client', NULL, 'client_generated_zip_bundle', '197.97.88.77', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)', 768000000, true, NOW() - INTERVAL '14 days',
 jsonb_build_object('name', 'Peter Jackson', 'email', 'peter.jackson@saeon.ac.za', 'organisation', 'SAEON', 'download_type', 'zip_bundle', 'source', 'MIMS-UI', 'record_count', 10, 'dois', jsonb_build_array('10.15493/uuid-73', '10.15493/uuid-74', '10.15493/uuid-75', '10.15493/uuid-76', '10.15493/uuid-77', '10.15493/uuid-78', '10.15493/uuid-79', '10.15493/uuid-80', '10.15493/uuid-81', '10.15493/uuid-82'))),

('mims-client', NULL, 'https://example.com/data/uuid-083/download', '102.165.30.50', 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)', 157286400, true, NOW() - INTERVAL '13 days',
 jsonb_build_object('name', 'Nancy White', 'email', 'nancy.white@wits.ac.za', 'organisation', 'Wits University', 'download_type', 'single_record', 'source', 'MIMS-UI-Detail-Page')),

('mims-client', NULL, 'client_generated_zip_bundle', '196.28.77.99', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0)', 1350000000, true, NOW() - INTERVAL '12 days',
 jsonb_build_object('name', 'Joseph Harris', 'email', 'joseph.harris@weather.gov.za', 'organisation', 'South African Weather Service', 'download_type', 'zip_bundle', 'source', 'MIMS-UI', 'record_count', 20, 'dois', jsonb_build_array('10.15493/uuid-84', '10.15493/uuid-85', '10.15493/uuid-86', '10.15493/uuid-87', '10.15493/uuid-88', '10.15493/uuid-89', '10.15493/uuid-90', '10.15493/uuid-91', '10.15493/uuid-92', '10.15493/uuid-93', '10.15493/uuid-94', '10.15493/uuid-95', '10.15493/uuid-96', '10.15493/uuid-97', '10.15493/uuid-98', '10.15493/uuid-99', '10.15493/uuid-100', '10.15493/uuid-101', '10.15493/uuid-102', '10.15493/uuid-103'))),

('mims-client', NULL, 'https://example.com/data/uuid-104/download', '196.44.55.66', 'Mozilla/5.0 (X11; Linux x86_64)', 262144000, true, NOW() - INTERVAL '11 days',
 jsonb_build_object('name', 'Margaret Martin', 'email', 'margaret.martin@up.ac.za', 'organisation', 'University of Pretoria', 'download_type', 'single_record', 'source', 'MIMS-UI-Detail-Page')),

('mims-client', NULL, 'client_generated_zip_bundle', '197.97.25.35', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)', 900000000, false, NOW() - INTERVAL '10 days',
 jsonb_build_object('name', 'Christopher Lee', 'email', 'christopher.lee@gmail.com', 'organisation', 'Department of Environmental Affairs', 'download_type', 'zip_bundle', 'source', 'MIMS-UI', 'record_count', 7, 'dois', jsonb_build_array('10.15493/uuid-105', '10.15493/uuid-106', '10.15493/uuid-107', '10.15493/uuid-108', '10.15493/uuid-109', '10.15493/uuid-110', '10.15493/uuid-111'))),

('mims-client', NULL, 'https://example.com/data/uuid-112/download', '102.165.40.60', 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)', 52428800, true, NOW() - INTERVAL '9 days',
 jsonb_build_object('name', 'Dorothy Taylor', 'email', 'dorothy.taylor@yahoo.com', 'organisation', 'Rhodes University', 'download_type', 'single_record', 'source', 'MIMS-UI-Detail-Page')),

('mims-client', NULL, 'client_generated_zip_bundle', '196.28.88.11', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0)', 500000000, true, NOW() - INTERVAL '8 days',
 jsonb_build_object('name', 'Mark Thompson', 'email', 'mark.thompson@uct.ac.za', 'organisation', 'University of Cape Town', 'download_type', 'zip_bundle', 'source', 'MIMS-UI', 'record_count', 6, 'dois', jsonb_build_array('10.15493/uuid-113', '10.15493/uuid-114', '10.15493/uuid-115', '10.15493/uuid-116', '10.15493/uuid-117', '10.15493/uuid-118'))),

('mims-client', NULL, 'https://example.com/data/uuid-119/download', '196.44.22.33', 'Mozilla/5.0 (X11; Linux x86_64)', 314572800, true, NOW() - INTERVAL '7 days',
 jsonb_build_object('name', 'Elizabeth Garcia', 'email', 'elizabeth.garcia@csir.co.za', 'organisation', 'CSIR', 'download_type', 'single_record', 'source', 'MIMS-UI-Detail-Page')),

('mims-client', NULL, 'client_generated_zip_bundle', '197.97.99.88', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)', 1200000000, true, NOW() - INTERVAL '6 days',
 jsonb_build_object('name', 'Daniel Anderson', 'email', 'daniel.anderson@hotmail.com', 'organisation', 'Nelson Mandela University', 'download_type', 'zip_bundle', 'source', 'MIMS-UI', 'record_count', 18, 'dois', jsonb_build_array('10.15493/uuid-120', '10.15493/uuid-121', '10.15493/uuid-122', '10.15493/uuid-123', '10.15493/uuid-124', '10.15493/uuid-125', '10.15493/uuid-126', '10.15493/uuid-127', '10.15493/uuid-128', '10.15493/uuid-129', '10.15493/uuid-130', '10.15493/uuid-131', '10.15493/uuid-132', '10.15493/uuid-133', '10.15493/uuid-134', '10.15493/uuid-135', '10.15493/uuid-136', '10.15493/uuid-137'))),

('mims-client', NULL, 'https://example.com/data/uuid-138/download', '102.165.70.90', 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)', 157286400, true, NOW() - INTERVAL '5 days',
 jsonb_build_object('name', 'Jessica Harris', 'email', 'jessica.harris@sun.ac.za', 'organisation', 'Stellenbosch University', 'download_type', 'single_record', 'source', 'MIMS-UI-Detail-Page')),

('mims-client', NULL, 'client_generated_zip_bundle', '196.28.44.55', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)', 800000000, true, NOW() - INTERVAL '4 days',
 jsonb_build_object('name', 'Thomas Martin', 'email', 'thomas.martin@wits.ac.za', 'organisation', 'Wits University', 'download_type', 'zip_bundle', 'source', 'MIMS-UI', 'record_count', 11, 'dois', jsonb_build_array('10.15493/uuid-139', '10.15493/uuid-140', '10.15493/uuid-141', '10.15493/uuid-142', '10.15493/uuid-143', '10.15493/uuid-144', '10.15493/uuid-145', '10.15493/uuid-146', '10.15493/uuid-147', '10.15493/uuid-148', '10.15493/uuid-149'))),

('mims-client', NULL, 'https://example.com/data/uuid-150/download', '196.44.77.88', 'Mozilla/5.0 (X11; Linux x86_64)', 104857600, false, NOW() - INTERVAL '3 days',
 jsonb_build_object('name', 'Amanda Martinez', 'email', 'amanda.martinez@saeon.ac.za', 'organisation', 'SAEON', 'download_type', 'single_record', 'source', 'MIMS-UI-Detail-Page')),

('mims-client', NULL, 'client_generated_zip_bundle', '197.97.11.22', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0)', 600000000, true, NOW() - INTERVAL '2 days',
 jsonb_build_object('name', 'George Martinez', 'email', 'george.martinez@weather.gov.za', 'organisation', 'South African Weather Service', 'download_type', 'zip_bundle', 'source', 'MIMS-UI', 'record_count', 9, 'dois', jsonb_build_array('10.15493/uuid-151', '10.15493/uuid-152', '10.15493/uuid-153', '10.15493/uuid-154', '10.15493/uuid-155', '10.15493/uuid-156', '10.15493/uuid-157', '10.15493/uuid-158', '10.15493/uuid-159'))),

('mims-client', NULL, 'https://example.com/data/uuid-160/download', '102.165.55.66', 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)', 262144000, true, NOW() - INTERVAL '1 days',
 jsonb_build_object('name', 'Sarah Thompson', 'email', 'sarah.thompson@up.ac.za', 'organisation', 'University of Pretoria', 'download_type', 'single_record', 'source', 'MIMS-UI-Detail-Page'));

-- Print confirmation
SELECT 'Successfully inserted ' || COUNT(*) || ' download audit records' FROM download_audit;
