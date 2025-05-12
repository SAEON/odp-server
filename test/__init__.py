import json
import pathlib
from copy import deepcopy

import odp.schema

schema_dir = pathlib.Path(odp.schema.__file__).parent


def datacite4_example():
    example_file = schema_dir / 'metadata' / 'saeon' / 'datacite4-example-translated.json'
    with open(example_file) as f:
        return json.load(f)


def iso19115_example():
    example_file = schema_dir / 'metadata' / 'saeon' / 'iso19115-example.json'
    with open(example_file) as f:
        return json.load(f)

def eml_example():
    example_file = schema_dir / 'metadata' / 'saeon' / 'eml-example.json'
    with open(example_file) as f:
        return json.load(f)

def ris_example():
    example_file = schema_dir / 'metadata' / 'ris' / 'citation-example.json'
    with open(example_file) as f:
        return json.load(f)


def isequal(x, y):
    """Perform a deep comparison, useful for comparing metadata records.
    Lists are compared as if they were sets."""
    if type(x) is not type(y):
        return False  # avoid `0 == False`, etc

    if isinstance(x, dict):
        return x.keys() == y.keys() and all(isequal(x[k], y[k]) for k in x)

    if isinstance(x, list):
        if len(x) != len(y):
            return False
        x_ = deepcopy(x)
        y_ = deepcopy(y)
        while x_:
            xi = x_.pop()
            found = False
            for j, yj in enumerate(y_):
                if isequal(xi, yj):
                    y_.pop(j)
                    found = True
                    break
            if not found:
                return False
        return True

    return x == y
