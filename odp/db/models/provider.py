import uuid

from sqlalchemy import Column, Enum, Identity, Integer, String, TIMESTAMP
from sqlalchemy.orm import relationship

from odp.const.db import AuditCommand
from odp.db import Base


class Provider(Base):
    """A data provider.

    This model represents the person, group or organization considered
    to be the originating party of a digital object identified by an
    ODP record.
    """

    __tablename__ = 'provider'

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    key = Column(String, unique=True, nullable=False)
    name = Column(String, nullable=False)
    timestamp = Column(TIMESTAMP(timezone=True), nullable=False)

    # view of associated collections (one-to-many)
    collections = relationship('Collection', viewonly=True)

    _repr_ = 'id', 'key', 'name'


class ProviderAudit(Base):
    """Provider audit log."""

    __tablename__ = 'provider_audit'

    id = Column(Integer, Identity(), primary_key=True)
    client_id = Column(String, nullable=False)
    user_id = Column(String)
    user_name = Column(String)
    command = Column(Enum(AuditCommand), nullable=False)
    timestamp = Column(TIMESTAMP(timezone=True), nullable=False)

    _id = Column(String, nullable=False)
    _key = Column(String, nullable=False)
    _name = Column(String, nullable=False)
