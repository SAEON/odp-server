-- Add Keyword facets for MIMS
INSERT INTO catalog_record_facet (catalog_id, record_id, facet, value)
SELECT DISTINCT
  cr.catalog_id,
  cr.record_id,
  'Keyword' as facet,
  kw as value
FROM (
  SELECT 
    catalog_id,
    record_id,
    unnest(keywords) as kw
  FROM catalog_record
  WHERE catalog_id = 'MIMS'
    AND keywords IS NOT NULL
    AND array_length(keywords, 1) > 0
) cr
WHERE kw NOT LIKE 'EOV:%' 
  AND kw NOT LIKE 'EBV:%'
  AND kw NOT LIKE 'SDG%'
ON CONFLICT DO NOTHING;

-- Add EOV facets for MIMS
INSERT INTO catalog_record_facet (catalog_id, record_id, facet, value)
SELECT DISTINCT
  cr.catalog_id,
  cr.record_id,
  'EOV' as facet,
  TRIM(SUBSTRING(kw, 6)) as value
FROM (
  SELECT 
    catalog_id,
    record_id,
    unnest(keywords) as kw
  FROM catalog_record
  WHERE catalog_id = 'MIMS'
    AND keywords IS NOT NULL
    AND array_length(keywords, 1) > 0
) cr
WHERE kw LIKE 'EOV:%'
ON CONFLICT DO NOTHING;

-- Add EBV facets for MIMS
INSERT INTO catalog_record_facet (catalog_id, record_id, facet, value)
SELECT DISTINCT
  cr.catalog_id,
  cr.record_id,
  'EBV' as facet,
  TRIM(SUBSTRING(kw, 6)) as value
FROM (
  SELECT 
    catalog_id,
    record_id,
    unnest(keywords) as kw
  FROM catalog_record
  WHERE catalog_id = 'MIMS'
    AND keywords IS NOT NULL
    AND array_length(keywords, 1) > 0
) cr
WHERE kw LIKE 'EBV:%'
ON CONFLICT DO NOTHING;

-- Add SDG facets for MIMS
INSERT INTO catalog_record_facet (catalog_id, record_id, facet, value)
SELECT DISTINCT
  cr.catalog_id,
  cr.record_id,
  'SDG' as facet,
  kw as value
FROM (
  SELECT 
    catalog_id,
    record_id,
    unnest(keywords) as kw
  FROM catalog_record
  WHERE catalog_id = 'MIMS'
    AND keywords IS NOT NULL
    AND array_length(keywords, 1) > 0
) cr
WHERE kw LIKE 'SDG%'
ON CONFLICT DO NOTHING;

-- Show counts by facet
SELECT 
  facet,
  COUNT(*) as count
FROM catalog_record_facet
WHERE catalog_id = 'MIMS'
GROUP BY facet
ORDER BY facet;