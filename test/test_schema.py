import pytest
from jschon import JSON, JSONPatch, JSONSchemaError, URI

from odp.lib.schema import schema_catalog as catalog
from test.factories import VocabularyFactory


def test_validity():
    with catalog.cache() as cacheid:
        input_schema = catalog.get_schema(URI('https://odp.saeon.ac.za/schema/metadata/saeon/iso19115'),
                                          cacheid=cacheid)
        input_json = catalog.load_json(URI('https://odp.saeon.ac.za/schema/metadata/saeon/iso19115-example'))
        output_schema = catalog.get_schema(URI('https://odp.saeon.ac.za/schema/metadata/saeon/datacite4'),
                                           cacheid=cacheid)
        output_json = catalog.load_json(
            URI('https://odp.saeon.ac.za/schema/metadata/saeon/datacite4-example-translated'))

        assert input_schema.validate().valid
        assert input_schema.evaluate(JSON(input_json)).valid
        assert output_schema.validate().valid
        assert output_schema.evaluate(JSON(output_json)).valid

def test_validity_eml():
    with catalog.cache() as cacheid:
        eml_schema = catalog.get_schema(
            URI('https://odp.saeon.ac.za/schema/metadata/saeon/eml'),
            cacheid=cacheid,
        )
        eml_json = catalog.load_json(
            URI('https://odp.saeon.ac.za/schema/metadata/saeon/eml-example')
        )

        datacite_schema = catalog.get_schema(
            URI('https://odp.saeon.ac.za/schema/metadata/saeon/datacite4'),
            cacheid=cacheid,
        )
        datacite_json_from_eml = catalog.load_json(
            URI('https://odp.saeon.ac.za/schema/metadata/saeon/datacite4-example-from-eml')
        )

        assert eml_schema.validate().valid
        assert eml_schema.evaluate(JSON(eml_json)).valid
        assert datacite_schema.evaluate(JSON(datacite_json_from_eml)).valid

def test_translate_iso19115_to_datacite():
    with catalog.cache() as cacheid:
        input_schema = catalog.get_schema(URI('https://odp.saeon.ac.za/schema/metadata/saeon/iso19115'),
                                          cacheid=cacheid)
        input_json = catalog.load_json(URI('https://odp.saeon.ac.za/schema/metadata/saeon/iso19115-example'))
        output_json = catalog.load_json(
            URI('https://odp.saeon.ac.za/schema/metadata/saeon/datacite4-example-translated'))

        result = input_schema.evaluate(JSON(input_json))
        patch = result.output('translation-patch', scheme='saeon/datacite4')
        translation = result.output('translation', scheme='saeon/datacite4')

        assert JSONPatch(*patch).evaluate(None) == translation

        translation = result.output('translation', scheme='saeon/datacite4', clear_empties=True)

        assert translation == output_json


def test_translate_data_submission_to_datacite():
    with catalog.cache() as cacheid:
        input_schema = catalog.get_schema(URI('https://odp.saeon.ac.za/schema/metadata/saeon/data-submission'), cacheid=cacheid)
        input_json = catalog.load_json(URI('https://odp.saeon.ac.za/schema/metadata/saeon/data-submission-example'))
        output_json = catalog.load_json(URI('https://odp.saeon.ac.za/schema/metadata/saeon/datacite4-example-translated-from-data-submission'))

        result = input_schema.evaluate(JSON(input_json))
        patch = result.output('translation-patch', scheme='saeon/datacite4')
        translation = result.output('translation', scheme='saeon/datacite4')

        assert JSONPatch(*patch).evaluate(None) == translation

        translation = result.output('translation', scheme='saeon/datacite4', clear_empties=True)

        assert translation == output_json

def test_translate_eml_to_datacite():
    with catalog.cache() as cacheid:
        input_schema = catalog.get_schema(
            URI('https://odp.saeon.ac.za/schema/metadata/saeon/eml'),
            cacheid=cacheid,
        )

        input_json = catalog.load_json(
            URI('https://odp.saeon.ac.za/schema/metadata/saeon/eml-example')
        )

        output_json = catalog.load_json(
            URI('https://odp.saeon.ac.za/schema/metadata/saeon/datacite4-example-from-eml')
        )

        result = input_schema.evaluate(JSON(input_json))

        patch = result.output(
            'translation-patch',
            scheme='saeon/datacite4'
        )

        translation = result.output(
            'translation',
            scheme='saeon/datacite4'
        )

        assert JSONPatch(*patch).evaluate(None) == translation

        translation = result.output(
            'translation',
            scheme='saeon/datacite4',
            clear_empties=True
        )

        assert translation == output_json
        
@pytest.mark.parametrize('vocab_id', ['Project', 'Infrastructure'])
def test_vocabulary_keyword_valid_term(vocab_id):
    with catalog.cache() as cacheid:
        vocab_key = vocab_id.lower()
        vocab = VocabularyFactory(id=vocab_id)
        tag_schema = catalog.get_schema(URI(f'https://odp.saeon.ac.za/schema/tag/collection/{vocab_key}'),
                                        cacheid=cacheid)
        tag_json = JSON({
            vocab_key: vocab.terms[0].term_id
        })
        assert (result := tag_schema.evaluate(tag_json)).valid
        annotation = next(
            ann for ann in result.output('basic')['annotations']
            if ann['instanceLocation'] == f'/{vocab_key}'
        )
        assert annotation['annotation'] == vocab_id


@pytest.mark.parametrize('vocab_id', ['Project', 'Infrastructure'])
def test_vocabulary_keyword_invalid_term(vocab_id):
    with catalog.cache() as cacheid:
        vocab_key = vocab_id.lower()
        vocab = VocabularyFactory(id=vocab_id)
        tag_schema = catalog.get_schema(URI(f'https://odp.saeon.ac.za/schema/tag/collection/{vocab_key}'),
                                        cacheid=cacheid)
        tag_json = JSON({
            vocab_key: 'foo'
        })
        assert not (result := tag_schema.evaluate(tag_json)).valid
        error = next(
            err for err in result.output('basic')['errors']
            if err['instanceLocation'] == f'/{vocab_key}'
        )
        assert error['error'] == f"Vocabulary '{vocab_id}' does not contain the term 'foo'"


@pytest.mark.parametrize('vocab_id', ['Project', 'Infrastructure'])
def test_vocabulary_keyword_unknown_vocab(vocab_id):
    with catalog.cache() as cacheid:
        vocab_key = vocab_id.lower()
        tag_schema = catalog.get_schema(URI(f'https://odp.saeon.ac.za/schema/tag/collection/{vocab_key}'),
                                        cacheid=cacheid)
        tag_json = JSON({
            vocab_key: 'foo'
        })
        with pytest.raises(JSONSchemaError) as excinfo:
            tag_schema.evaluate(tag_json)
        assert str(excinfo.value) == f'Unknown vocabulary {vocab_id!r}'
