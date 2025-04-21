import logging
from importlib import import_module
from pathlib import Path
from pkgutil import iter_modules
from typing import final

from odp.db import Session

logger = logging.getLogger(__name__)


class ArchiveModule:

    @final
    def run(self):
        modname = self.__class__.__name__
        try:
            logger.info(f'{modname} started')
            self.exec()
            Session.commit()
            logger.info(f'{modname} completed')
        except Exception as e:
            Session.rollback()
            logger.exception(f'{modname} failed: {e!r}')

    def exec(self):
        raise NotImplementedError


def run_all():
    archive_modules = []
    archive_dir = str(Path(__file__).parent)
    for mod_info in iter_modules([archive_dir]):
        mod = import_module(f'odp.archive.{mod_info.name}')
        for cls in mod.__dict__.values():
            if isinstance(cls, type) and issubclass(cls, ArchiveModule) and cls is not ArchiveModule:
                archive_modules += [cls()]

    for archive_module in archive_modules:
        archive_module.run()
