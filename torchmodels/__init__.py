from . import utils
from . import common
from . import manager
from .common import Module
from .common import Sequential
from .common import Linear
from .common import Identity
from .common import ModuleList
from .common import AmbidextrousModule
from .common import Parameter
from .manager import register_packages
from .manager import get_module_dict
from .manager import get_module_classes
from .manager import get_module_names



def create_model_cls(package, model_path=None):
    if model_path is None:
        model_cls = manager.get_module_classes(package)[0]
        modargs = get_optarg_template(model_cls)
    else:
        namemap = manager.get_module_dict(package)
        opts = utils.load_yaml(model_path)
        name, modargs = opts.get("type"), opts.get("vargs")
        assert name in namemap, \
            f"module name does not exist: '{name}'"
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
            cls = manager.get_module_classes(pkg)[0]
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