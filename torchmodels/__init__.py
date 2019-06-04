__version__ = "0.2.8"

from . import utils
from . import common
from . import manager
from .common import Module
from .common import Sequential
from .common import Linear
from .common import Identity
from .common import ModuleList
from .common import MultiModule
from .common import Parameter
from .manager import register_packages
from .manager import get_module_dict
from .manager import get_module_classes
from .manager import get_module_names


def create_model_cls(package=None, model_path=None, name=None, modargs=None):
    """
    Create a model-initializing function that accepts positional arguments.
    :param package:
        the package to search for the model. If none given, all
        known packages will be searched.
    :param model_path:
        yaml file path that contains keyword-only arguments.
    :param name:
        model name to search for. If no model path is specified, this
        option will be used.
    :param modargs:
        keyword-only module arguments to initialize the function.
    :return:
        function
    """
    if model_path is None:
        if name is None:
            classes = manager.get_module_classes(package)
            assert len(classes) > 0, \
                f"no modules found in package " \
                f"'{package if package is not None else 'all'}"
            name = classes[0].name
            modargs = get_optarg_template(classes[0])
        if modargs is None:
            modargs = dict()
    else:
        opts = utils.load_yaml(model_path)
        name, modargs = opts.get("type"), opts.get("vargs")
    namemap = manager.get_module_dict(package)
    assert name in namemap, \
        f"module name '{name}' does not exist. available names: " \
        f"{list(namemap.keys())}"
    model_cls = namemap[name]
    caster = common.get_caster(model_cls)
    return caster({
        "type": model_cls.name,
        "vargs": modargs
    })


def get_optarg_template(cls: common.Module):
    def get_value_template(optarg: common.OptionalArgument):
        if optarg.islist:
            sample = optarg.default[0]
        else:
            sample = optarg.default
        if common.is_module_cls(sample):
            pkg = sample.get_package()
            classes = manager.get_module_classes(pkg)
            assert classes, \
                f"no available modules found for package '{pkg}'"
            cls = classes[0]
            val = {"type": cls.name}
            args = get_optarg_template(cls)
            if args:
                val["vargs"] = args
        else:
            val = sample
        if optarg.islist:
            val = [val]
        return val
    return {
        name: get_value_template(optarg)
        for name, optarg in cls.get_optargs().items()
    }
