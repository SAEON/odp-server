import asyncio
import logging

from sqlalchemy import select

from odp.archive import ArchiveModule
from odp.const.db import ResourceStatus
from odp.db import Session
from odp.db.models import ArchiveResource
from odp.lib.archive import ArchiveAdapter, ArchiveError

logger = logging.getLogger(__name__)


class FilePurgeModule(ArchiveModule):

    def exec(self):
        archive_resources_to_delete = Session.execute(
            select(ArchiveResource).where(ArchiveResource.status == ResourceStatus.delete_pending)
        ).scalars().all()

        for ar in archive_resources_to_delete:
            archive_adapter = ArchiveAdapter.get_instance(ar.archive)
            try:
                asyncio.run(archive_adapter.delete(ar.path))
                logger.info(f'Deleted {ar.path} in {ar.archive_id}')

            except ArchiveError as e:
                if e.status_code == 404:
                    logger.info(f'Delete {ar.path} in {ar.archive_id}: already gone')
                else:
                    logger.exception(f'{e.status_code}: {e.error_detail}')
                    continue

            except NotImplementedError:
                pass

            ar.delete()
            Session.commit()
