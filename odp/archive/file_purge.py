import asyncio
import logging

from sqlalchemy import select

from odp.archive import ArchiveModule
from odp.const.db import ResourceStatus
from odp.db import Session
from odp.db.models import Resource
from odp.lib.archive import ArchiveAdapter, ArchiveError

logger = logging.getLogger(__name__)


class FilePurgeModule(ArchiveModule):

    def exec(self):
        resources_to_delete = Session.execute(
            select(Resource).where(Resource.status == ResourceStatus.delete_pending)
        ).scalars().all()

        for resource in resources_to_delete:
            for ar in resource.archive_resources:
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

            # Delete resource only if there are no archive_resources left.
            if not resource.archive_resources:
                resource.delete()
                Session.commit()
