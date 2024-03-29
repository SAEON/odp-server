/* search_records()
 */
-- total count
EXPLAIN
SELECT count(*) AS count_1
FROM (SELECT 1
      FROM catalog_record
      WHERE catalog_record.catalog_id = :catalog_id_1
        AND catalog_record.published
        AND catalog_record.searchable) AS anon_1;

-- result list
EXPLAIN
SELECT 1
FROM catalog_record
WHERE catalog_record.catalog_id = :catalog_id_1
  AND catalog_record.published
  AND catalog_record.searchable
ORDER BY catalog_record.timestamp DESC
LIMIT :param_1 OFFSET :param_2;

-- result facets
EXPLAIN
SELECT anon_1.facet, anon_1.value, count(*) AS count_1
FROM (SELECT catalog_record.catalog_id AS catalog_id,
             catalog_record.record_id  AS record_id
      FROM catalog_record
      WHERE catalog_record.catalog_id = :catalog_id_1
        AND catalog_record.published
        AND catalog_record.searchable) AS anon_2
         JOIN (SELECT *
               FROM catalog_record_facet) AS anon_1 ON anon_2.catalog_id = anon_1.catalog_id AND anon_2.record_id = anon_1.record_id
GROUP BY anon_1.facet, anon_1.value;

/* search_records(text_query)
   :plainto_tsquery_1 = 'english'
   :plainto_tsquery_2 = text query
   :ts_rank_cd_1 = 'full_text'
   :ts_rank_cd_2 = 'query'
   :ts_rank_cd_3  = 1 | 4
 */
-- total count
EXPLAIN
SELECT count(*) AS count_1
FROM (SELECT 1
      FROM catalog_record,
           plainto_tsquery(:plainto_tsquery_1, :plainto_tsquery_2) AS query
      WHERE catalog_record.catalog_id = :catalog_id_1
        AND catalog_record.published
        AND catalog_record.searchable
        AND full_text @@ query) AS anon_1;

-- result list, sorted by timestamp
EXPLAIN
SELECT 1
FROM catalog_record,
     plainto_tsquery(:plainto_tsquery_1, :plainto_tsquery_2) AS query
WHERE catalog_record.catalog_id = :catalog_id_1
  AND catalog_record.published
  AND catalog_record.searchable
  AND full_text @@ query
ORDER BY catalog_record.timestamp DESC
LIMIT :param_1 OFFSET :param_2;

-- result list, sorted by rank
EXPLAIN
SELECT ts_rank_cd(:ts_rank_cd_1, :ts_rank_cd_2, :ts_rank_cd_3) AS rank
FROM catalog_record,
     plainto_tsquery(:plainto_tsquery_1, :plainto_tsquery_2) AS query
WHERE catalog_record.catalog_id = :catalog_id_1
  AND catalog_record.published
  AND catalog_record.searchable
  AND full_text @@ query
ORDER BY rank DESC
LIMIT :param_1 OFFSET :param_2;

-- result facets
EXPLAIN
SELECT anon_1.facet, anon_1.value, count(*) AS count_1
FROM (SELECT catalog_record.catalog_id AS catalog_id,
             catalog_record.record_id  AS record_id
      FROM catalog_record,
           plainto_tsquery(:plainto_tsquery_1, :plainto_tsquery_2) AS query
      WHERE catalog_record.catalog_id = :catalog_id_1
        AND catalog_record.published
        AND catalog_record.searchable
        AND full_text @@ query) AS anon_2
         JOIN (SELECT *
               FROM catalog_record_facet) AS anon_1 ON anon_2.catalog_id = anon_1.catalog_id AND anon_2.record_id = anon_1.record_id
GROUP BY anon_1.facet, anon_1.value;

/* search_records(facet_query(location, instrument))
 */
-- total count
EXPLAIN
SELECT count(*) AS count_1
FROM (SELECT 1
      FROM catalog_record
               JOIN catalog_record_facet AS "crfLocation"
                    ON catalog_record.catalog_id = "crfLocation".catalog_id AND catalog_record.record_id = "crfLocation".record_id
               JOIN catalog_record_facet AS "crfInstrument"
                    ON catalog_record.catalog_id = "crfInstrument".catalog_id AND catalog_record.record_id = "crfInstrument".record_id
      WHERE catalog_record.catalog_id = :catalog_id_1
        AND catalog_record.published
        AND catalog_record.searchable
        AND "crfLocation".facet = :facet_1
        AND "crfLocation".value = :value_1
        AND "crfInstrument".facet = :facet_2
        AND "crfInstrument".value = :value_2) AS anon_1;

-- result list
EXPLAIN
SELECT 1
FROM catalog_record
         JOIN catalog_record_facet AS "crfLocation"
              ON catalog_record.catalog_id = "crfLocation".catalog_id AND catalog_record.record_id = "crfLocation".record_id
         JOIN catalog_record_facet AS "crfInstrument"
              ON catalog_record.catalog_id = "crfInstrument".catalog_id AND catalog_record.record_id = "crfInstrument".record_id
WHERE catalog_record.catalog_id = :catalog_id_1
  AND catalog_record.published
  AND catalog_record.searchable
  AND "crfLocation".facet = :facet_1
  AND "crfLocation".value = :value_1
  AND "crfInstrument".facet = :facet_2
  AND "crfInstrument".value = :value_2
ORDER BY catalog_record.timestamp DESC
LIMIT :param_1 OFFSET :param_2;

-- result facets
EXPLAIN
SELECT anon_1.facet, anon_1.value, count(*) AS count_1
FROM (SELECT catalog_record.catalog_id AS catalog_id,
             catalog_record.record_id  AS record_id
      FROM catalog_record
               JOIN catalog_record_facet AS "crfLocation"
                    ON catalog_record.catalog_id = "crfLocation".catalog_id AND catalog_record.record_id = "crfLocation".record_id
               JOIN catalog_record_facet AS "crfInstrument"
                    ON catalog_record.catalog_id = "crfInstrument".catalog_id AND catalog_record.record_id = "crfInstrument".record_id
      WHERE catalog_record.catalog_id = :catalog_id_1
        AND catalog_record.published
        AND catalog_record.searchable
        AND "crfLocation".facet = :facet_1
        AND "crfLocation".value = :value_1
        AND "crfInstrument".facet = :facet_2
        AND "crfInstrument".value = :value_2) AS anon_2
         JOIN (SELECT *
               FROM catalog_record_facet) AS anon_1 ON anon_2.catalog_id = anon_1.catalog_id AND anon_2.record_id = anon_1.record_id
GROUP BY anon_1.facet, anon_1.value;

/* search_records(n, s, e, w)
   Note that in this non-exclusive region case, the param names produced by
   SQLA are mismatched with the API params; i.e. :spatial_south_1 takes the
   north_bound API param, etc
 */
-- total count
EXPLAIN
SELECT count(*) AS count_1
FROM (SELECT 1
      FROM catalog_record
      WHERE catalog_record.catalog_id = :catalog_id_1
        AND catalog_record.published
        AND catalog_record.searchable
        AND catalog_record.spatial_south <= :spatial_south_1
        AND catalog_record.spatial_north >= :spatial_north_1
        AND catalog_record.spatial_west <= :spatial_west_1
        AND catalog_record.spatial_east >= :spatial_east_1) AS anon_1;

-- result list
EXPLAIN
SELECT 1
FROM catalog_record
WHERE catalog_record.catalog_id = :catalog_id_1
  AND catalog_record.published
  AND catalog_record.searchable
  AND catalog_record.spatial_south <= :spatial_south_1
  AND catalog_record.spatial_north >= :spatial_north_1
  AND catalog_record.spatial_west <= :spatial_west_1
  AND catalog_record.spatial_east >= :spatial_east_1
ORDER BY catalog_record.timestamp DESC
LIMIT :param_1 OFFSET :param_2;

-- result facets
EXPLAIN
SELECT anon_1.facet, anon_1.value, count(*) AS count_1
FROM (SELECT catalog_record.catalog_id AS catalog_id,
             catalog_record.record_id  AS record_id
      FROM catalog_record
      WHERE catalog_record.catalog_id = :catalog_id_1
        AND catalog_record.published
        AND catalog_record.searchable
        AND catalog_record.spatial_south <= :spatial_south_1
        AND catalog_record.spatial_north >= :spatial_north_1
        AND catalog_record.spatial_west <= :spatial_west_1
        AND catalog_record.spatial_east >= :spatial_east_1) AS anon_2
         JOIN (SELECT *
               FROM catalog_record_facet) AS anon_1 ON anon_2.catalog_id = anon_1.catalog_id AND anon_2.record_id = anon_1.record_id
GROUP BY anon_1.facet, anon_1.value;

/* search_records(n, s, e, w, exclusive_region)
   :spatial_south_1 takes the south_bound API param, etc
 */
-- total count
EXPLAIN
SELECT count(*) AS count_1
FROM (SELECT 1
      FROM catalog_record
      WHERE catalog_record.catalog_id = :catalog_id_1
        AND catalog_record.published
        AND catalog_record.searchable
        AND catalog_record.spatial_north <= :spatial_north_1
        AND catalog_record.spatial_south >= :spatial_south_1
        AND catalog_record.spatial_east <= :spatial_east_1
        AND catalog_record.spatial_west >= :spatial_west_1) AS anon_1;

-- result list
EXPLAIN
SELECT 1
FROM catalog_record
WHERE catalog_record.catalog_id = :catalog_id_1
  AND catalog_record.published
  AND catalog_record.searchable
  AND catalog_record.spatial_north <= :spatial_north_1
  AND catalog_record.spatial_south >= :spatial_south_1
  AND catalog_record.spatial_east <= :spatial_east_1
  AND catalog_record.spatial_west >= :spatial_west_1
ORDER BY catalog_record.timestamp DESC
LIMIT :param_1 OFFSET :param_2;

-- result facets
EXPLAIN
SELECT anon_1.facet, anon_1.value, count(*) AS count_1
FROM (SELECT catalog_record.catalog_id AS catalog_id,
             catalog_record.record_id  AS record_id
      FROM catalog_record
      WHERE catalog_record.catalog_id = :catalog_id_1
        AND catalog_record.published
        AND catalog_record.searchable
        AND catalog_record.spatial_north <= :spatial_north_1
        AND catalog_record.spatial_south >= :spatial_south_1
        AND catalog_record.spatial_east <= :spatial_east_1
        AND catalog_record.spatial_west >= :spatial_west_1) AS anon_2
         JOIN (SELECT *
               FROM catalog_record_facet) AS anon_1 ON anon_2.catalog_id = anon_1.catalog_id AND anon_2.record_id = anon_1.record_id
GROUP BY anon_1.facet, anon_1.value;

/* search_records(start, end)
   Note that in this non-exclusive interval case, the param names produced by
   SQLA are mismatched with the API params; i.e.
   :temporal_end_1 takes the start_date API param
   :temporal_start_1 takes the end_date API param
 */
-- total count
EXPLAIN
SELECT count(*) AS count_1
FROM (SELECT 1
      FROM catalog_record
      WHERE catalog_record.catalog_id = :catalog_id_1
        AND catalog_record.published
        AND catalog_record.searchable
        AND catalog_record.temporal_end >= :temporal_end_1
        AND catalog_record.temporal_start <= :temporal_start_1) AS anon_1;

-- result list
EXPLAIN
SELECT 1
FROM catalog_record
WHERE catalog_record.catalog_id = :catalog_id_1
  AND catalog_record.published
  AND catalog_record.searchable
  AND catalog_record.temporal_end >= :temporal_end_1
  AND catalog_record.temporal_start <= :temporal_start_1
ORDER BY catalog_record.timestamp DESC
LIMIT :param_1 OFFSET :param_2;

-- result facets
EXPLAIN
SELECT anon_1.facet, anon_1.value, count(*) AS count_1
FROM (SELECT catalog_record.catalog_id AS catalog_id,
             catalog_record.record_id  AS record_id
      FROM catalog_record
      WHERE catalog_record.catalog_id = :catalog_id_1
        AND catalog_record.published
        AND catalog_record.searchable
        AND catalog_record.temporal_end >= :temporal_end_1
        AND catalog_record.temporal_start <= :temporal_start_1) AS anon_2
         JOIN (SELECT *
               FROM catalog_record_facet) AS anon_1 ON anon_2.catalog_id = anon_1.catalog_id AND anon_2.record_id = anon_1.record_id
GROUP BY anon_1.facet, anon_1.value;

/* search_records(start, end, exclusive_interval)
   :temporal_start_1 takes the start_date API param
   :temporal_end_1 takes the end_date API param
 */
-- total count
EXPLAIN
SELECT count(*) AS count_1
FROM (SELECT 1
      FROM catalog_record
      WHERE catalog_record.catalog_id = :catalog_id_1
        AND catalog_record.published
        AND catalog_record.searchable
        AND catalog_record.temporal_start >= :temporal_start_1
        AND catalog_record.temporal_end <= :temporal_end_1) AS anon_1;

-- result list
EXPLAIN
SELECT 1
FROM catalog_record
WHERE catalog_record.catalog_id = :catalog_id_1
  AND catalog_record.published
  AND catalog_record.searchable
  AND catalog_record.temporal_start >= :temporal_start_1
  AND catalog_record.temporal_end <= :temporal_end_1
ORDER BY catalog_record.timestamp DESC
LIMIT :param_1 OFFSET :param_2;

-- result facets
EXPLAIN
SELECT anon_1.facet, anon_1.value, count(*) AS count_1
FROM (SELECT catalog_record.catalog_id AS catalog_id,
             catalog_record.record_id  AS record_id
      FROM catalog_record
      WHERE catalog_record.catalog_id = :catalog_id_1
        AND catalog_record.published
        AND catalog_record.searchable
        AND catalog_record.temporal_start >= :temporal_start_1
        AND catalog_record.temporal_end <= :temporal_end_1) AS anon_2
         JOIN (SELECT *
               FROM catalog_record_facet) AS anon_1 ON anon_2.catalog_id = anon_1.catalog_id AND anon_2.record_id = anon_1.record_id
GROUP BY anon_1.facet, anon_1.value;
