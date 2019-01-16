import pkgutil
import logging
import itertools

from . import utils
from . import common
from . import modules


_KNOWN_PACKAGES = set()
_CLSMAP_CACHE = None


def _enum_packages(parent):
    yield parent
    for _, pkgname, ispkg in pkgutil.iter_modules(parent.__path__):
        yield utils.import_module(f"{parent.__name__}.{pkgname}")


def _is_module_class(item):
    return isinstance(item, type) and issubclass(item, common.Module) \
           and item.name is not None


def _enum_module_classes(package):
    for item in package.__dict__.values():
        if _is_module_class(item):
            yield item


def _resolve_package_argument(package=None):
    if package is None:
        pkgs = _KNOWN_PACKAGES
    else:
        if isinstance(package, str):
            package = utils.import_module(package)
        pkgs = [package]
    return pkgs


def register_packages(parent):
    _KNOWN_PACKAGES.update(set(_enum_packages(parent)))


def get_module_classes(package=None):
    packages = _resolve_package_argument(package)
    modlst = list(map(list, map(_enum_module_classes, packages)))
    return list(itertools.chain(*modlst))


def get_module_names(package=None):
    classes = get_module_classes(package)
    return list(map(lambda x: x.name, classes))


def get_module_dict(package=None):
    return {cls.name: cls for cls in get_module_classes(package)}


def resolve(name, package=None):
    clsmap = get_module_dict(package)
    if name not in clsmap:
        logging.error(f"{name} is not a recognized module. "
                      f"Did you add the parent package to 'manager.py'?")
    return clsmap.get(name)


register_packages(modules)
