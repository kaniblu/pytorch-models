import json
import argparse

import torchmodels
from . import utils
from . import manager


parser = argparse.ArgumentParser()
parser.add_argument("--package", required=True)
parser.add_argument("--module-name", type=str, default=None)
parser.add_argument("--save-path", required=True)
parser.add_argument("--format", type=str, default="yaml",
                    choices=["yaml", "json"])


def save_template(args):
    pkg = utils.import_module(args.package)
    if args.module_name is not None:
        clsmap = manager.get_module_dict(pkg)
        cls = clsmap.get(args.module_name)
    else:
        cls = manager.get_module_classes(pkg)[0]
    template = {
        "type": cls.name,
        "vargs": torchmodels.get_optarg_template(cls)
    }
    dump = utils.map_val(args.format, {
        "yaml": utils.dump_yaml,
        "json": json.dump
    }, "template format")
    with open(args.save_path, "w") as f:
        dump(template, args.save_path)


if __name__ == "__main__":
    args, _ = parser.parse_known_args()
    save_template(args)