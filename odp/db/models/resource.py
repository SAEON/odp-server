import uuid

from sqlalchemy import BigInteger, CheckConstraint, Column, Enum, ForeignKey, String, TIMESTAMP, UniqueConstraint
from sqlalchemy.ext.associationproxy import association_proxy
from sqlalchemy.orm import relationship

from odp.const.db import HashAlgorithm, ResourceStatus
from odp.db import Base


class Resource(Base):
    """A resource comprises the metadata for an individual file,
    folder or dataset."""

    __tablename__ = 'resource'

    __table_args__ = (
        UniqueConstraint('package_id', 'path'),
        CheckConstraint(
            'hash is null or hash_algorithm is not null',
            name='resource_hash_algorithm_check',
        ),
    )

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    path = Column(String, nullable=False)
    mimetype = Column(String)
    size = Column(BigInteger)
    hash = Column(String)
    hash_algorithm = Column(Enum(HashAlgorithm))
    title = Column(String)
    description = Column(String)
    status = Column(Enum(ResourceStatus), nullable=False)
    timestamp = Column(TIMESTAMP(timezone=True), nullable=False)

    package_id = Column(String, ForeignKey('package.id', ondelete='RESTRICT'), nullable=False)
    package = relationship('Package')

    # view of associated archives via many-to-many archive_resource relation
    archive_resources = relationship('ArchiveResource', viewonly=True)
    archives = association_proxy('archive_resources', 'archive')

    _repr_ = 'id', 'path', 'mimetype', 'size', 'hash', 'package_id', 'status'
