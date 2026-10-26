from datetime import datetime, timezone
from functools import partial

from fastapi import APIRouter, Depends, HTTPException
from jschon import JSON, JSONSchema, URI
from sqlalchemy import select
from starlette.status import HTTP_404_NOT_FOUND, HTTP_409_CONFLICT, HTTP_422_UNPROCESSABLE_ENTITY

from odp.api.lib.auth import Authorize, Authorized, VocabularyAuthorize
from odp.api.lib.paging import Page, Paginator
from odp.api.lib.schema import get_vocabulary_schema
from odp.api.models import VocabularyModel, VocabularyTermAuditModel, VocabularyTermModel, VocabularyTermModelIn
from odp.const import ODPScope
from odp.const.db import AuditCommand
from odp.db import Session
from odp.db.models import Vocabulary, VocabularyTerm, VocabularyTermAudit
from odp.lib.schema import schema_catalog

router = APIRouter()


def output_vocabulary_model(vocabulary: Vocabulary) -> VocabularyModel:
    return VocabularyModel(
        id=vocabulary.id,
        scope_id=vocabulary.scope_id,
        schema_id=vocabulary.schema_id,
        schema_uri=vocabulary.schema.uri,
        schema_=schema_catalog.get_schema(URI(vocabulary.schema.uri)).value,
        static=vocabulary.static,
        terms=[VocabularyTermModel(
            id=term.term_id,
            data=term.data,
        ) for term in sorted(vocabulary.terms, key=lambda vt: vt.term_id)]
    )


def output_audit_model(audit: VocabularyTermAudit) -> VocabularyTermAuditModel:
    return VocabularyTermAuditModel(
        table='vocabulary_term',
        tag_id=None,
        audit_id=audit.id,
        client_id=audit.client_id,
        user_id=audit.user_id,
        user_name=audit.user_name,
        command=audit.command,
        timestamp=audit.timestamp.isoformat(),
        vocabulary_id=audit._vocabulary_id,
        term_id=audit._term_id,
        data=audit._data,
    )


def create_audit_record(auth: Authorized, term: VocabularyTerm, command: AuditCommand) -> None:
    VocabularyTermAudit(
        client_id=auth.client_id,
        user_id=auth.user_id,
        user_name=auth.user_name,
        command=command,
        timestamp=datetime.now(timezone.utc),
        _vocabulary_id=term.vocabulary_id,
        _term_id=term.term_id,
        _data=term.data,
    ).save()


@router.get(
    '/',
    response_model=Page[VocabularyModel],
    dependencies=[Depends(Authorize(ODPScope.VOCABULARY_READ))],
)
async def list_vocabularies(
        paginator: Paginator = Depends(),
):
    return paginator.paginate(
        select(Vocabulary),
        lambda row: output_vocabulary_model(row.Vocabulary),
    )


@router.get(
    '/{vocabulary_id}',
    response_model=VocabularyModel,
    dependencies=[Depends(Authorize(ODPScope.VOCABULARY_READ))],
)
async def get_vocabulary(
        vocabulary_id: str,
):
    if not (vocabulary := Session.get(Vocabulary, vocabulary_id)):
        raise HTTPException(HTTP_404_NOT_FOUND)

    return output_vocabulary_model(vocabulary)


@router.post(
    '/{vocabulary_id}/term',
)
async def create_term(
        vocabulary_id: str,
        term_in: VocabularyTermModelIn,
        term_schema: JSONSchema = Depends(get_vocabulary_schema),
        auth: Authorized = Depends(VocabularyAuthorize()),
):
    if not (vocabulary := Session.get(Vocabulary, vocabulary_id)):
        raise HTTPException(HTTP_404_NOT_FOUND)

    if vocabulary.static:
        raise HTTPException(HTTP_422_UNPROCESSABLE_ENTITY, 'Static vocabulary cannot be modified')

    if Session.get(VocabularyTerm, (vocabulary_id, term_in.id)):
        raise HTTPException(HTTP_409_CONFLICT, 'Term already exists in vocabulary')

    # the id is validated by the term schema
    term_in.data['id'] = term_in.id

    validity = term_schema.evaluate(JSON(term_in.data)).output('detailed')
    if not validity['valid']:
        raise HTTPException(HTTP_422_UNPROCESSABLE_ENTITY, validity)

    term = VocabularyTerm(
        vocabulary_id=vocabulary_id,
        term_id=term_in.id,
        data=term_in.data,
    )
    term.save()
    create_audit_record(auth, term, AuditCommand.insert)


@router.put(
    '/{vocabulary_id}/term',
)
async def update_term(
        vocabulary_id: str,
        term_in: VocabularyTermModelIn,
        term_schema: JSONSchema = Depends(get_vocabulary_schema),
        auth: Authorized = Depends(VocabularyAuthorize()),
):
    if not (vocabulary := Session.get(Vocabulary, vocabulary_id)):
        raise HTTPException(HTTP_404_NOT_FOUND)

    if vocabulary.static:
        raise HTTPException(HTTP_422_UNPROCESSABLE_ENTITY, 'Static vocabulary cannot be modified')

    if not (term := Session.get(VocabularyTerm, (vocabulary_id, term_in.id))):
        raise HTTPException(HTTP_404_NOT_FOUND)

    # the id in the data cannot be changed
    term_in.data['id'] = term_in.id

    if term.data != term_in.data:
        validity = term_schema.evaluate(JSON(term_in.data)).output('detailed')
        if not validity['valid']:
            raise HTTPException(HTTP_422_UNPROCESSABLE_ENTITY, validity)

        term.data = term_in.data
        term.save()
        create_audit_record(auth, term, AuditCommand.update)


@router.delete(
    '/{vocabulary_id}/term/{term_id}',
)
async def delete_term(
        vocabulary_id: str,
        term_id: str,
        auth: Authorized = Depends(VocabularyAuthorize()),
):
    if not (vocabulary := Session.get(Vocabulary, vocabulary_id)):
        raise HTTPException(HTTP_404_NOT_FOUND)

    if vocabulary.static:
        raise HTTPException(HTTP_422_UNPROCESSABLE_ENTITY, 'Static vocabulary cannot be modified')

    if not (term := Session.get(VocabularyTerm, (vocabulary_id, term_id))):
        raise HTTPException(HTTP_404_NOT_FOUND)

    term.delete()
    create_audit_record(auth, term, AuditCommand.delete)


@router.get(
    '/{vocabulary_id}/audit',
    response_model=Page[VocabularyTermAuditModel],
    dependencies=[Depends(Authorize(ODPScope.VOCABULARY_READ))],
)
async def get_vocabulary_audit_log(
        vocabulary_id: str,
        paginator: Paginator = Depends(partial(Paginator, sort='timestamp')),
):
    stmt = (
        select(VocabularyTermAudit).
        where(VocabularyTermAudit._vocabulary_id == vocabulary_id)
    )

    return paginator.paginate(
        stmt,
        lambda row: output_audit_model(row.VocabularyTermAudit),
    )


@router.get(
    '/{vocabulary_id}/audit/{audit_id}',
    response_model=VocabularyTermAuditModel,
    dependencies=[Depends(Authorize(ODPScope.VOCABULARY_READ))],
)
async def get_vocabulary_audit_detail(
        vocabulary_id: str,
        audit_id: int,
):
    if not (audit := Session.execute(
            select(VocabularyTermAudit).
            where(VocabularyTermAudit._vocabulary_id == vocabulary_id).
            where(VocabularyTermAudit.id == audit_id)
    ).scalar_one_or_none()):
        raise HTTPException(HTTP_404_NOT_FOUND)

    return output_audit_model(audit)
