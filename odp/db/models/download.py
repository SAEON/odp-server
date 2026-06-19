from datetime import datetime, timezone

from sqlalchemy import Column, Integer, String, TIMESTAMP, Boolean, BigInteger, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship

from odp.db import Base


class DownloadAudit(Base):
    """Download audit log."""

    __tablename__ = 'download_audit'

    id = Column(Integer, primary_key=True, autoincrement=True)
    client_id = Column(String, nullable=False)
    user_id = Column(String, nullable=True)
    download_url = Column(String, nullable=True)
    ip_address = Column(String, nullable=True)
    user_agent = Column(Text, nullable=True)
    file_size = Column(BigInteger, nullable=True)
    success = Column(Boolean, nullable=False)
    timestamp = Column(TIMESTAMP(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
    meta = Column(JSONB, nullable=True)