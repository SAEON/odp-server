from sqlalchemy import select, text

from odp.const.db import KeywordStatus
from odp.db import Session
from odp.db.models import Keyword


def legacy_terms():
    stmt = text(
        "select vocabulary_id, term_id, data "
        "from vocabulary_term "
        "where vocabulary_id in ('Infrastructure', 'Project')"
    )
    for vocabulary_id, term_id, data in Session.execute(stmt).all():
        if Session.execute(
                select(Keyword).
                where(Keyword.vocabulary_id == vocabulary_id).
                where(Keyword.key == term_id)
        ).first() is not None:
            continue

        data['key'] = data.pop('id')
        kw = Keyword(
            vocabulary_id=vocabulary_id,
            key=term_id,
            data=data,
            status=KeywordStatus.approved,
        )
        kw.save()

    Session.commit()


def institutions():
    def add(key, abbr, ror=None):
        if not Session.execute(
                select(Keyword).where(Keyword.vocabulary_id == 'Institution').where(Keyword.key == key)
        ).scalar_one_or_none():
            Keyword(
                vocabulary_id='Institution',
                key=key,
                data=dict(key=key, abbr=abbr, ror=ror),
                status=KeywordStatus.approved,
            ).save()

    add('South African Environmental Observation Network', 'SAEON', 'https://ror.org/041j42q70')
    add('Department of Forestry, Fisheries and the Environment', 'DFFE', 'https://ror.org/01vm69c87')
    add('University of Cape Town', 'UCT', 'https://ror.org/03p74gp79')
    add('Department of Agriculture, Fisheries and Forestry', 'DAFF')
    add('South African Institute for Aquatic Biodiversity', 'SAIAB', 'https://ror.org/00bfgxv06')
    add('South African Weather Service', 'SAWS', 'https://ror.org/03mef6610')
    add('Department of Science and Innovation', 'DSI', 'https://ror.org/048kh7197')
    add('Laboratoire de Météorologie Dynamique', 'LMD', 'https://ror.org/000ehr937')
    add('Atlantic Oceanographic and Meteorological Laboratory', 'AOML', 'https://ror.org/042r9xb17')

    Session.commit()
