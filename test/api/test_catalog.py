import os
from copy import copy, deepcopy
from datetime import datetime
from itertools import product
from random import randint

import pytest
from sqlalchemy import select

import migrate.systemdata
from odp.catalog.mims import MIMSCatalog
from odp.catalog.saeon import SAEONCatalog
from odp.const import ODPScope
from odp.db.models import Catalog, Tag
from test import TestSession, datacite4_example, isequal, iso19115_example, ris_example, ranked_metadata_example
from test.api.assertions import assert_forbidden, assert_new_timestamp, assert_not_found, assert_redirect
from test.factories import CatalogFactory, CollectionTagFactory, FactorySession, RecordFactory, RecordTagFactory


@pytest.fixture
def catalog_batch():
    """Create and commit a batch of Catalog instances."""
    return [CatalogFactory() for _ in range(randint(3, 5))]


@pytest.fixture(params=[None, True])
def tag_collection_published(request):
    return request.param


@pytest.fixture(params=[None, 'MIMS', 'FOO'])
def tag_collection_infrastructure(request):
    return request.param


@pytest.fixture(params=[None, False, True])
def tag_record_qc(request):
    return request.param


@pytest.fixture(params=[None, True])
def tag_record_retracted(request):
    return request.param


@pytest.fixture
def static_publishing_data():
    os.environ['SAEON_CATALOG_URL'] = 'http://odp.catalog/saeon'
    os.environ['MIMS_CATALOG_URL'] = 'http://odp.catalog/mims'
    os.environ['DATACITE_CATALOG_URL'] = 'http://odp.catalog/datacite'
    migrate.systemdata.init_schemas()
    migrate.systemdata.init_vocabularies()
    migrate.systemdata.init_tags()
    migrate.systemdata.init_catalogs()
    migrate.systemdata.Session.commit()


@pytest.mark.require_scope(ODPScope.CATALOG_SEARCH)
def test_search_ranking(
        api,
        scopes,
        static_publishing_data
):
    metadata_data = ranked_metadata_example()

    title_variations = ['no_harmonic_distance', 'harmonic_distance', 'no_search_terms']
    abstract_length_variations = ['short', 'long']
    abstract_search_terms_variations = ['many_search_terms', 'few_search_terms']
    abstract_harmonic_distance_variations = ['no_harmonic_distance', 'harmonic_distance']
    keywords_variations = ['all_search_terms', 'no_search_terms']

    metadata_permutations = product(
        title_variations, abstract_length_variations, abstract_search_terms_variations,
        abstract_harmonic_distance_variations, keywords_variations
    )

    for index, (title, abstract_length, abstract_search_terms, abstract_harmonic, keywords) in enumerate(
            list(metadata_permutations)):
        metadata_example = metadata_data["example"]
        metadata_example["title"] = metadata_data['titles'][title]
        metadata_example["abstract"] = metadata_data['abstracts'][
            f'{abstract_length}-{abstract_search_terms}-{abstract_harmonic}']
        metadata_example["descriptiveKeywords"] = metadata_data['keywords'][keywords]
        metadata_example["doi"] = f'{index}/example_metadata_record'

        create_example_record(
            tag_collection_published=True,
            tag_collection_infrastructure='SAEON',
            tag_record_qc=True,
            tag_record_retracted=None,
            schema_id='SAEON.ISO19115',
            metadata_=metadata_example
        )

    ranked_doi_order = metadata_data['generated_doi_ranked_order']

    authorized = ODPScope.CATALOG_SEARCH in scopes
    catalog_id = 'SAEON'

    r = api(scopes).get(f'catalog/{catalog_id}/search', params=dict(
        text_query='Lorem ipsum',
        sort='rank desc',
    ))

    if authorized:
        for index, item in enumerate(r.json()['items']):
            current_doi_number = item['metadata_records'][0]['metadata']['doi'].split('/')[0]
            assert int(current_doi_number) == int(ranked_doi_order[index])
    else:
        assert_forbidden(r)


@pytest.fixture(params=['SAEON', 'MIMS'])
def catalog_id(request):
    return request.param


@pytest.fixture(params=['list', 'get'])
def endpoint(request):
    return request.param


def create_example_record(
        tag_collection_published,
        tag_collection_infrastructure,
        tag_record_qc,
        tag_record_retracted,
        schema_id=None,
        metadata_=None,
):
    """Create and return a single record instance,
    with valid (example) metadata, optionally with collection and/or
    record tags, and evaluated for publishing."""
    kwargs = dict(use_example_metadata=True)
    if schema_id:
        kwargs |= dict(schema_id=schema_id)

    if metadata_:
        kwargs |= dict(metadata_=metadata_)

    record = RecordFactory(**kwargs)

    if tag_collection_published is not None:
        CollectionTagFactory.create(
            tag=FactorySession.get(Tag, ('Collection.Published', 'collection')),
            collection=record.collection,
        )
    if tag_collection_infrastructure is not None:
        CollectionTagFactory.create(
            tag=FactorySession.get(Tag, ('Collection.Infrastructure', 'collection')),
            collection=record.collection,
            data={'infrastructure': tag_collection_infrastructure}
        )
    if tag_record_qc is not None:
        RecordTagFactory.create(
            tag=FactorySession.get(Tag, ('Record.QC', 'record')),
            record=record,
            data={'pass_': tag_record_qc}
        )
    if tag_record_retracted is not None:
        RecordTagFactory.create(
            tag=FactorySession.get(Tag, ('Record.Retracted', 'record')),
            record=record,
        )

    catalog_classes = {
        'SAEON': SAEONCatalog,
        'MIMS': MIMSCatalog,
    }
    for catalog_id, catalog_cls in catalog_classes.items():
        catalog_cls(catalog_id).publish()

    return record


def assert_db_state(catalogs):
    """Verify that the DB catalog table contains the given catalog batch."""
    result = TestSession.execute(select(Catalog)).scalars().all()
    result.sort(key=lambda r: r.id)
    catalogs.sort(key=lambda c: c.id)
    for n, row in enumerate(result):
        assert row.id == catalogs[n].id
        assert row.url == catalogs[n].url
        assert row.data == catalogs[n].data
        assert row.timestamp == catalogs[n].timestamp


def assert_json_result(response, json, catalog, withdata):
    """Verify that the API result matches the given catalog object."""
    assert response.status_code == 200
    assert json['id'] == catalog.id
    assert json['url'] == catalog.url
    if withdata:
        assert json['data'] == catalog.data
        assert datetime.fromisoformat(json['timestamp']) == catalog.timestamp


def assert_json_results(response, json, catalogs):
    """Verify that the API result list matches the given catalog batch."""
    items = json['items']
    assert json['total'] == len(items) == len(catalogs)
    items.sort(key=lambda i: i['id'])
    catalogs.sort(key=lambda c: c.id)
    for n, catalog in enumerate(catalogs):
        assert_json_result(response, items[n], catalog, withdata=False)


@pytest.mark.require_scope(ODPScope.CATALOG_READ)
def test_list_catalogs(api, catalog_batch, scopes):
    authorized = ODPScope.CATALOG_READ in scopes
    r = api(scopes).get('/catalog/')
    if authorized:
        assert_json_results(r, r.json(), catalog_batch)
    else:
        assert_forbidden(r)
    assert_db_state(catalog_batch)


@pytest.mark.require_scope(ODPScope.CATALOG_READ)
def test_get_catalog(api, catalog_batch, scopes):
    authorized = ODPScope.CATALOG_READ in scopes
    r = api(scopes).get(f'/catalog/{catalog_batch[2].id}')
    if authorized:
        assert_json_result(r, r.json(), catalog_batch[2], withdata=True)
    else:
        assert_forbidden(r)
    assert_db_state(catalog_batch)


def test_get_catalog_not_found(api, catalog_batch):
    scopes = [ODPScope.CATALOG_READ]
    r = api(scopes).get('/catalog/foo')
    assert_not_found(r)
    assert_db_state(catalog_batch)


def test_redirect_to(
        api,
        static_publishing_data, catalog_id,
        tag_collection_published,
):
    catalog = FactorySession.get(Catalog, catalog_id)
    example_record = create_example_record(
        tag_collection_published,
        tag_collection_infrastructure='MIMS',
        tag_record_qc=True,
        tag_record_retracted=None,
    )

    route = f'/catalog/{catalog_id}/go/'
    target = f'{catalog.url}/'
    if example_record.doi:
        route += example_record.doi.swapcase()
        target += example_record.doi
    else:
        route += example_record.id
        target += example_record.id

    r = api([]).get(route, follow_redirects=False)

    if tag_collection_published:
        assert_redirect(r, target)
    else:
        assert_not_found(r)


schema_uris = {
    'SAEON.DataCite4': 'https://odp.saeon.ac.za/schema/metadata/saeon/datacite4',
    'SAEON.ISO19115': 'https://odp.saeon.ac.za/schema/metadata/saeon/iso19115',
    'SchemaOrg.Dataset': 'https://odp.saeon.ac.za/schema/metadata/schema.org/dataset',
    'RIS.Citation': 'https://odp.saeon.ac.za/schema/metadata/ris/citation'
}
metadata_examples = {
    'SAEON.DataCite4': datacite4_example(),
    'SAEON.ISO19115': iso19115_example(),
    'SchemaOrg.Dataset': {
        '@context': 'https://schema.org/',
        '@type': 'Dataset',
        # '@id': dynamic,
        'name': 'Example Metadata Record: ISO19115 - SAEON Profile',
        'description': "Concerning things that'd like to be under the sea, in an octopus's garden, in the shade.",
        'license': 'https://creativecommons.org/licenses/by/4.0/',
        # 'identifier': dynamic,
        # 'keywords': dynamic,
        # 'url': dynamic,
        'spatialCoverage': {
            '@type': 'Place',
            'geo': {
                '@type': 'GeoShape',
                'box': '-34.4 17.9 -34.0 18.3',
            }
        },
        'temporalCoverage': '2019-11-01T00:00:00+02:00/2019-12-01T00:00:00+02:00',
    },
    'RIS.Citation': ris_example()
}

iso_keywords = [
    dk['keyword'] for dk in metadata_examples['SAEON.ISO19115']['descriptiveKeywords']
    if dk['keywordType'] in ('general', 'place', 'stratum')
]

generic_keywords = [
    s['subject'] for s in metadata_examples['SAEON.DataCite4']['subjects']
]


@pytest.mark.require_scope(ODPScope.CATALOG_READ)
def test_get_published_record(
        api, scopes,
        static_publishing_data, catalog_id, endpoint,
        tag_collection_published, tag_collection_infrastructure,
        tag_record_qc, tag_record_retracted,
):
    def assert_metadata_record(schema_id):
        # select the actual metadata record from the API result
        metadata_record = next(filter(
            lambda m: m['schema_id'] == schema_id, result['metadata_records']
        ))
        assert metadata_record['schema_uri'] == schema_uris[schema_id]

        # construct the expected metadata
        expected_metadata = deepcopy(metadata_examples[schema_id])
        if schema_id == 'SchemaOrg.Dataset':
            expected_metadata['identifier'] = f'doi:{example_record.doi}' if example_record.doi else None
            expected_metadata['keywords'] = iso_keywords if has_iso19115 else generic_keywords
            expected_metadata['@id'] = expected_metadata['url'] = (
                'http://odp.catalog/mims/'
                f'{example_record.doi if example_record.doi else example_record.id}'
            )
        elif schema_id == 'RIS.Citation':
            expected_metadata['ris'] = expected_metadata['ris'].split('\n')
            metadata_record['metadata']['ris'] = metadata_record['metadata']['ris'].split('\n')

            url = (
                'http://odp.catalog/mims/'
                f'{example_record.doi if example_record.doi else example_record.id}'
            )

            for item in copy(expected_metadata['ris']):
                if "DO  - " in item:
                    expected_metadata['ris'].remove(item)
                    if example_record.doi:
                        expected_metadata['ris'].append(f"DO  - {example_record.doi}")

                if "UR  - " in item:
                    expected_metadata['ris'].remove(item)
                    expected_metadata['ris'].append(f"UR  - {url}")

            expected_metadata['ris'] = [element for element in expected_metadata['ris'] if "KW  -" not in element]
            replacement_keywords = iso_keywords if has_iso19115 else generic_keywords
            for keyword in replacement_keywords:
                expected_metadata['ris'].append(f"KW  - {keyword}")

            expected_metadata['ris'] = '\n'.join(sorted(expected_metadata['ris']))
            metadata_record['metadata']['ris'] = '\n'.join(sorted(metadata_record['metadata']['ris']))
        else:
            if example_record.doi:
                expected_metadata |= {'doi': example_record.doi}
            else:
                expected_metadata.pop('doi', None)

        # deep-compare actual vs expected
        assert isequal(metadata_record['metadata'], expected_metadata)

    authorized = ODPScope.CATALOG_READ in scopes
    published = (
            tag_collection_published is True and
            tag_record_qc is True and
            tag_record_retracted is None
    )
    if catalog_id == 'MIMS':
        published = published and tag_collection_infrastructure == 'MIMS'

    example_record = create_example_record(
        tag_collection_published,
        tag_collection_infrastructure,
        tag_record_qc,
        tag_record_retracted,
    )

    route = f'/catalog/{catalog_id}/records'
    resp_code = 200
    if endpoint == 'get':
        route += f'/{example_record.doi.swapcase()}' if example_record.doi else f'/{example_record.id}'
        resp_code = 200 if published else 404

    r = api(scopes).get(route)

    if not authorized:
        assert_forbidden(r)
        return

    assert r.status_code == resp_code

    if endpoint == 'list':
        json = r.json()
        items = json['items']
        assert json['total'] == len(items) == published

    if not published:
        return

    result = items[0] if endpoint == 'list' else r.json()

    assert result['id'] == example_record.id
    assert result['doi'] == example_record.doi
    assert result['sid'] == example_record.sid
    assert result['collection_key'] == example_record.collection.key
    assert result['collection_name'] == example_record.collection.name
    assert result['provider_key'] == example_record.collection.provider.key
    assert result['provider_name'] == example_record.collection.provider.name
    assert result['published'] is True
    assert result['searchable'] is True
    assert_new_timestamp(datetime.fromisoformat(result['timestamp']))

    has_datacite = True
    has_iso19115 = example_record.schema_id == 'SAEON.ISO19115'
    has_jsonld = catalog_id == 'MIMS'
    has_ris = catalog_id == 'MIMS'

    assert len(result['metadata_records']) == has_datacite + has_iso19115 + has_jsonld + has_ris

    if has_datacite:
        assert_metadata_record('SAEON.DataCite4')

    if has_iso19115:
        assert_metadata_record('SAEON.ISO19115')

    if has_jsonld:
        assert_metadata_record('SchemaOrg.Dataset')

    if has_ris:
        assert_metadata_record('RIS.Citation')


@pytest.mark.parametrize('schema_id, json_pointer, expected_value', [
    ('SAEON.DataCite4', '/titles/0/title', 'Example Metadata Record: ISO19115 - SAEON Profile'),
    ('SAEON.ISO19115', '/title', 'Example Metadata Record: ISO19115 - SAEON Profile'),
    ('SAEON.DataCite4', '/creators/0/nameIdentifiers/0', {
        "nameIdentifier": "https://orcid.org/0000-0001-2345-6789",
        "nameIdentifierScheme": "https://orcid.org",
        "schemeURI": "https://orcid.org"
    }),
    ('SAEON.ISO19115', '/responsibleParties/1/contactInfo', 'Intertidal Zone, Seashore Business Park'),
    ('SAEON.DataCite4', '/geoLocations/0/geoLocationPolygons/1/polygonPoints/3/pointLatitude', -34.3),
    ('SAEON.ISO19115', '/extent/geographicElements/0/boundingPolygon/0/polygon/2', {
        "longitude": 18.24, "latitude": -34.18
    }),
])
@pytest.mark.require_scope(ODPScope.CATALOG_READ)
def test_get_published_metadata_value(
        api, scopes,
        static_publishing_data, catalog_id,
        schema_id, json_pointer, expected_value,
):
    authorized = ODPScope.CATALOG_READ in scopes
    example_record = create_example_record(
        tag_collection_published=True,
        tag_collection_infrastructure='MIMS',
        tag_record_qc=True,
        tag_record_retracted=None,
        schema_id=schema_id,
    )

    route = f'/catalog/{catalog_id}/getvalue/'
    route += example_record.doi.swapcase() if example_record.doi else example_record.id

    r = api(scopes).get(route, params=dict(
        schema_id=schema_id,
        json_pointer=json_pointer,
    ))

    if not authorized:
        assert_forbidden(r)
        return

    assert r.status_code == 200
    assert r.json() == expected_value


@pytest.mark.parametrize('schema_id', ['SAEON.DataCite4', 'SAEON.ISO19115'])
@pytest.mark.require_scope(ODPScope.CATALOG_READ)
def test_get_published_metadata_document(
        api, scopes,
        static_publishing_data, catalog_id,
        schema_id,
):
    authorized = ODPScope.CATALOG_READ in scopes
    example_record = create_example_record(
        tag_collection_published=True,
        tag_collection_infrastructure='MIMS',
        tag_record_qc=True,
        tag_record_retracted=None,
        schema_id=schema_id,
    )

    route = f'/catalog/{catalog_id}/getvalue/'
    route += example_record.doi.swapcase() if example_record.doi else example_record.id

    r = api(scopes).get(route, params=dict(
        schema_id=schema_id,
        # json_pointer='',  # default
    ))

    if not authorized:
        assert_forbidden(r)
        return

    expected_document = datacite4_example() if schema_id == 'SAEON.DataCite4' else iso19115_example()
    if example_record.doi:
        expected_document |= dict(doi=example_record.doi)
    else:
        expected_document.pop('doi')

    assert r.status_code == 200
    assert r.json() == expected_document
