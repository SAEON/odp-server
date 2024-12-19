from fastapi import APIRouter, Depends, HTTPException
from jschon import URI
from sqlalchemy import select
from starlette.status import HTTP_404_NOT_FOUND

from odp.api.lib.auth import Authorize
from odp.api.lib.paging import Paginator
from odp.api.models import Page, VocabularyDetailModel, VocabularyModel
from odp.api.routers.keyword import RecurseMode, output_keyword_model
from odp.const import ODPScope
from odp.const.db import KeywordStatus
from odp.db import Session
from odp.db.models import Vocabulary
from odp.lib.schema import schema_catalog

router = APIRouter()


def output_vocabulary_model(
        vocabulary: Vocabulary,
        detail=False,
) -> VocabularyModel | VocabularyDetailModel:
    cls = VocabularyDetailModel if detail else VocabularyModel
    kwargs = dict(
        id=vocabulary.id,
        uri=vocabulary.uri,
        schema_id=vocabulary.schema_id,
        schema_uri=vocabulary.schema.uri,
        static=vocabulary.static,
    )
    if detail:
        kwargs |= dict(
            schema_=schema_catalog.get_schema(URI(vocabulary.schema.uri)).value,
            keywords=[
                output_keyword_model(keyword, recurse=RecurseMode.APPROVED)
                for keyword in vocabulary.keywords
                if keyword.parent_id is None and keyword.status == KeywordStatus.approved
            ]
        )

    return cls(**kwargs)


@router.get(
    '/',
    dependencies=[Depends(Authorize(ODPScope.VOCABULARY_READ))],
)
async def list_vocabularies(
        paginator: Paginator = Depends(),
) -> Page[VocabularyModel]:
    """
    List all vocabularies. Requires scope `odp.vocabulary:read`.
    """
    return paginator.paginate(
        select(Vocabulary),
        lambda row: output_vocabulary_model(row.Vocabulary),
    )


@router.get(
    '/{vocabulary_id}',
    dependencies=[Depends(Authorize(ODPScope.VOCABULARY_READ))],
)
async def get_vocabulary(
        vocabulary_id: str,
) -> VocabularyDetailModel:
    """
    Get a vocabulary with its (approved) keywords, nested if hierarchical.
    Requires scope `odp.vocabulary:read`.
    """
    if not (vocabulary := Session.get(Vocabulary, vocabulary_id)):
        raise HTTPException(HTTP_404_NOT_FOUND)

    return output_vocabulary_model(vocabulary, detail=True)
