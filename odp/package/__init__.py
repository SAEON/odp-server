import logging

from odp.db import Session

logger = logging.getLogger(__name__)


class PackageModule:

    def execute(self):
        try:
            self._internal_execute()
            Session.commit()
        except Exception as e:
            Session.rollback()
            logger.exception(f'PACKAGING EXECUTION FAILED: {str(e)}')

    def _internal_execute(self):
        raise NotImplementedError


def run_all():
    from .date_range import DateRangeInc

    DateRangeInc().execute()

