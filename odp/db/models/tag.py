from sqlalchemy import Boolean, CheckConstraint, Column, Enum, ForeignKey, ForeignKeyConstraint, String
from sqlalchemy.orm import relationship

from odp.const.db import SchemaType, TagCardinality, TagType
from odp.db import Base


class Tag(Base):
    """A tag definition."""

    __tablename__ = 'tag'

    __table_args__ = (
        ForeignKeyConstraint(
            ('schema_id', 'schema_type'), ('schema.id', 'schema.type'),
            name='tag_schema_fkey', ondelete='RESTRICT',
        ),
        CheckConstraint(
            f"schema_type = '{SchemaType.tag}'",
            name='tag_schema_type_check',
        ),
    )

    id = Column(String, unique=True, primary_key=True)
    type = Column(Enum(TagType), primary_key=True)
    cardinality = Column(Enum(TagCardinality), nullable=False)
    public = Column(Boolean, nullable=False)

    schema_id = Column(String, nullable=False)
    schema_type = Column(Enum(SchemaType), nullable=False)
    schema = relationship('Schema')

    scope_id = Column(String, nullable=False)

    vocabulary_id = Column(String, ForeignKey('vocabulary.id', ondelete='RESTRICT'))
    vocabulary = relationship('Vocabulary')

    _repr_ = 'id', 'type', 'cardinality', 'public', 'schema_id', 'scope_id', 'vocabulary_id'
