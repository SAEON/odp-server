import uuid
from datetime import datetime, timezone

from sqlalchemy import ARRAY, CheckConstraint, Column, Enum, ForeignKey, ForeignKeyConstraint, Identity, Index, Integer, String, TIMESTAMP, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.associationproxy import association_proxy
from sqlalchemy.orm import relationship

from odp.const.db import AuditCommand, SchemaType, TagType
from odp.db import Base


class Record(Base):
    """An ODP record.

    This model represents a uniquely identifiable digital object
    and its associated metadata.
    """

    __tablename__ = 'record'

    __table_args__ = (
        Index(
            'ix_record_doi', text('lower(doi)'),
            unique=True,
        ),
        ForeignKeyConstraint(
            ('schema_id', 'schema_type'), ('schema.id', 'schema.type'),
            name='record_schema_fkey', ondelete='RESTRICT',
        ),
        CheckConstraint(
            f"schema_type = '{SchemaType.metadata}'",
            name='record_schema_type_check',
        ),
        CheckConstraint(
            'doi IS NOT NULL OR sid IS NOT NULL',
            name='record_doi_sid_check',
        ),
    )

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    doi = Column(String, unique=True)
    sid = Column(String, unique=True)
    metadata_ = Column(JSONB, nullable=False)
    validity = Column(JSONB, nullable=False)
    timestamp = Column(TIMESTAMP(timezone=True), nullable=False)

    collection_id = Column(String, ForeignKey('collection.id', ondelete='RESTRICT'), nullable=False)
    collection = relationship('Collection')

    schema_id = Column(String, nullable=False)
    schema_type = Column(Enum(SchemaType), nullable=False)
    schema = relationship('Schema')

    # parent-child relationship for HasPart/IsPartOf related identifiers
    parent_id = Column(String, ForeignKey('record.id', ondelete='RESTRICT'))
    parent = relationship('Record', remote_side=id)
    children = relationship('Record', viewonly=True)

    # one-to-many record_package entities are persisted by
    # assigning/removing Package instances to/from packages
    record_packages = relationship('RecordPackage', cascade='all, delete-orphan', passive_deletes=True)
    packages = association_proxy('record_packages', 'package', creator=lambda p: RecordPackage(package=p))

    # view of associated tags (one-to-many)
    tags = relationship('RecordTag', viewonly=True)

    # view of associated catalog records (one-to-many)
    catalog_records = relationship('CatalogRecord', viewonly=True)

    _repr_ = 'id', 'doi', 'sid', 'collection_id', 'schema_id', 'parent_id'


class RecordAudit(Base):
    """Record audit log."""

    __tablename__ = 'record_audit'

    id = Column(Integer, Identity(), primary_key=True)
    client_id = Column(String, nullable=False)
    user_id = Column(String)
    command = Column(Enum(AuditCommand), nullable=False)
    timestamp = Column(TIMESTAMP(timezone=True), nullable=False)

    _id = Column(String, nullable=False)
    _doi = Column(String)
    _sid = Column(String)
    _metadata = Column(JSONB, nullable=False)
    _collection_id = Column(String, nullable=False)
    _schema_id = Column(String, nullable=False)
    _parent_id = Column(String)
    _packages = Column(ARRAY(String))


class RecordPackage(Base):
    """One-to-many record-package association.

    This association table allows the record-package linkage to be
    controlled from the record side while enforcing a one-to-many
    relationship.
    """
    __tablename__ = 'record_package'

    record_id = Column(String, ForeignKey('record.id', ondelete='CASCADE'), primary_key=True)
    package_id = Column(String, ForeignKey('package.id', ondelete='RESTRICT'), primary_key=True, unique=True)

    record = relationship('Record', viewonly=True)
    package = relationship('Package')


class RecordTag(Base):
    """Tag instance model, representing a tag attached to a record."""

    __tablename__ = 'record_tag'

    __table_args__ = (
        ForeignKeyConstraint(
            ('tag_id', 'tag_type'), ('tag.id', 'tag.type'),
            name='record_tag_tag_fkey', ondelete='CASCADE',
        ),
        CheckConstraint(
            f"tag_type = '{TagType.record}'",
            name='record_tag_tag_type_check',
        ),
        ForeignKeyConstraint(
            ('vocabulary_id', 'keyword_id'), ('keyword.vocabulary_id', 'keyword.id'),
            name='record_tag_keyword_fkey', ondelete='RESTRICT',
        ),
    )

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    record_id = Column(String, ForeignKey('record.id', ondelete='CASCADE'), nullable=False)
    tag_id = Column(String, nullable=False)
    tag_type = Column(Enum(TagType), nullable=False)
    user_id = Column(String, ForeignKey('user.id', ondelete='RESTRICT'))

    vocabulary_id = Column(String)
    keyword_id = Column(Integer)

    data = Column(JSONB, nullable=False)
    timestamp = Column(TIMESTAMP(timezone=True), nullable=False)

    record = relationship('Record')
    tag = relationship('Tag')
    user = relationship('User')
    keyword = relationship('Keyword')


class RecordTagAudit(Base):
    """Record tag audit log."""

    __tablename__ = 'record_tag_audit'

    id = Column(Integer, Identity(), primary_key=True)
    client_id = Column(String, nullable=False)
    user_id = Column(String)
    command = Column(Enum(AuditCommand), nullable=False)
    timestamp = Column(TIMESTAMP(timezone=True), nullable=False)

    _id = Column(String, nullable=False)
    _record_id = Column(String, nullable=False)
    _tag_id = Column(String, nullable=False)
    _user_id = Column(String)
    _data = Column(JSONB, nullable=False)
    _keyword_id = Column(Integer)


def _doi_published_timestamp(context):
    if context.get_current_parameters()['doi'] is not None:
        return datetime.now(timezone.utc)


class PublishedRecord(Base):
    """This table preserves all record ids and DOIs that have ever
    been published, and prevents associated records from being deleted
    or having their DOIs changed or removed."""

    __tablename__ = 'published_record'

    __table_args__ = (
        Index(
            'ix_published_record_doi', text('lower(doi)'),
            unique=True,
        ),
    )

    id = Column(String, ForeignKey('record.id', ondelete='RESTRICT', onupdate='RESTRICT'), primary_key=True)
    doi = Column(String, ForeignKey('record.doi', ondelete='RESTRICT', onupdate='RESTRICT'), unique=True)
    id_published = Column(TIMESTAMP(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
    doi_published = Column(TIMESTAMP(timezone=True), default=_doi_published_timestamp, onupdate=_doi_published_timestamp)
