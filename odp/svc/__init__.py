import logging
from importlib import import_module
from pathlib import Path
from pkgutil import iter_modules
from typing import final

from odp.db import Session

logger = logging.getLogger(__name__)


class ServiceModule:
    """Abstract base class for a background service module."""

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


def run_service(name: str):
    """Run all service modules within `name` directory."""
    modules = []
    dir_ = str(Path(__file__).parent / name)
    for mod_info in iter_modules([dir_]):
        mod = import_module(f'odp.svc.{name}.{mod_info.name}')
        for cls in mod.__dict__.values():
            if isinstance(cls, type) and issubclass(cls, ServiceModule) and cls is not ServiceModule:
                modules += [cls()]

    for module in modules:
        module.run()
