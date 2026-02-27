from sqlalchemy import Enum, Column, ForeignKey, Identity, Integer, \
    String, TIMESTAMP
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship

from odp.const.db import SubmissionStatus
from odp.db import Base


class Submission(Base):
    """Represents a public catalog providing access to published
    digital object records."""

    __tablename__ = 'submission'

    id = Column(Integer, Identity(), primary_key=True)
    doi = Column(String, unique=True)
    user_id = Column(String, nullable=False)
    data = Column(JSONB, nullable=False)
    status = Column(Enum(SubmissionStatus), nullable=False)
    dataset_file_name = Column(String, nullable=True)
    timestamp = Column(TIMESTAMP(timezone=True))
    collection_id = Column(String, ForeignKey('collection.id'), nullable=True)
    schema_id = Column(String, nullable=True)
    record_id = Column(String, ForeignKey('record.id'), nullable=True)

    collection = relationship('Collection')
    record = relationship('Record')

    _repr_ = 'id', 'status'
