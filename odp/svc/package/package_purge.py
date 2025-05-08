import logging

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from odp.const.db import PackageStatus
from odp.db import Session
from odp.db.models import Package
from odp.svc import ServiceModule

logger = logging.getLogger(__name__)


class PackagePurgeModule(ServiceModule):

    def exec(self):
        packages_to_delete = Session.execute(
            select(Package).where(Package.status == PackageStatus.delete_pending)
        ).scalars().all()

        for package in packages_to_delete:
            # Delete package only after resources have been deleted by the archival service.
            if not package.resources:
                try:
                    package.delete()
                    logger.info(f'{package.key} deleted')

                except IntegrityError as e:
                    logger.exception(f'{package.key} delete failed: {e!r}')
