import sys
import yaml
import logging
import importlib

import torch


def import_module(pkg_name, relative_to=None):
    logging.debug(f"importing '{pkg_name}' under '{relative_to}'")
    if relative_to is not None:
        full_name = f"{relative_to.__name__}.pkg_name"
    else:
        full_name = pkg_name
    if full_name not in sys.modules:
        return importlib.import_module(pkg_name, package=relative_to)
    else:
        return sys.modules[full_name]


def resolve_obj(module, name):
    items = module.__dict__
    assert name in items, \
        f"Unrecognized attribute '{name}' in module '{module}'"
    return items[name]


def mask(lens, max_len=None):
    if max_len is None:
        max_len = lens.max().item()
    enum = torch.arange(max_len).long().to(lens)
    return lens.unsqueeze(1) > enum.unsqueeze(0)


def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def unkval_error(name=None, val=None, choices=None):
    if name is None:
        name = "value"
    if val is None:
        val = ""
    else:
        val = f": {val}"
    if choices is not None:
        choices = ", ".join(f"\"{c}\"" for c in choices)
        choices = f"; possible choices: {choices}"
    else:
        choices = ""
    return ValueError(f"Unrecognized {name}{val}{choices}")


def map_val(key, maps: dict, name=None, ignore_err=False, fallback=None):
    if not ignore_err and key not in maps:
        raise unkval_error(name, key, list(maps.keys()))
    return maps.get(key, fallback)


def dump_yaml(obj, stream=None):
    def dump(obj, stream=None):
        return yaml.dump(
            data=obj,
            stream=stream,
            allow_unicode=True,
            default_flow_style=False,
            indent=2
        )

    if stream is None:
        return dump(obj)
    elif isinstance(stream, str):
        with open(stream, "w") as f:
            dump(obj, f)
    else:
        dump(obj, stream)


def assert_oneof(item, candidates, name=None):
    assert item in candidates, \
        (f"'{name}' " if name is not None else "") + \
        f"is not one of '{candidates}'. item given: {item}"
