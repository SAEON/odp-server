import uuid

from sqlalchemy import Column, Enum, Identity, Integer, String, TIMESTAMP
from sqlalchemy.ext.associationproxy import association_proxy
from sqlalchemy.orm import relationship

from odp.const.db import AuditCommand
from odp.db import Base


class Provider(Base):
    """A data provider.

    An originating party for data (resources) and metadata (packages).
    """

    __tablename__ = 'provider'

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    key = Column(String, unique=True, nullable=False)
    name = Column(String, nullable=False)
    timestamp = Column(TIMESTAMP(timezone=True), nullable=False)

    # view of associated users via many-to-many user_provider relation
    provider_users = relationship('UserProvider', viewonly=True)
    users = association_proxy('provider_users', 'user')

    # view of associated collections (one-to-many)
    collections = relationship('Collection', viewonly=True)

    # view of associated clients (one-to-many)
    clients = relationship('Client', viewonly=True)

    _repr_ = 'id', 'key', 'name'


class ProviderAudit(Base):
    """Provider audit log."""

    __tablename__ = 'provider_audit'

    id = Column(Integer, Identity(), primary_key=True)
    client_id = Column(String, nullable=False)
    user_id = Column(String)
    command = Column(Enum(AuditCommand), nullable=False)
    timestamp = Column(TIMESTAMP(timezone=True), nullable=False)

    _id = Column(String, nullable=False)
    _key = Column(String, nullable=False)
    _name = Column(String, nullable=False)
