from sqlalchemy import Column, ForeignKey, String
from sqlalchemy.orm import relationship

from odp.db import Base


class UserRole(Base):
    """A user-role assignment."""

    __tablename__ = 'user_role'

    user_id = Column(String, ForeignKey('user.id', ondelete='CASCADE'), primary_key=True)
    role_id = Column(String, ForeignKey('role.id', ondelete='CASCADE'), primary_key=True)

    user = relationship('User', viewonly=True)
    role = relationship('Role')
