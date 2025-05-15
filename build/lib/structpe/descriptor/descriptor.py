"""
descriptor.py: Provides reflection-based JSON serialization and deserialization 
so we can store/retrieve entire dataset objects.
"""

import json
import importlib

def _serialize_object(obj):
    """
    Recursively serialize any Python object into JSON-friendly structure.
    If it's a built-in (int, float, bool, str, None), return as is.
    If it's a list/tuple, recurse.
    If it's a dict, recurse.
    Otherwise, assume custom object, store __class__ and __attributes__.
    """
    if isinstance(obj, (int, float, bool, str, type(None))):
        return obj
    if isinstance(obj, (list, tuple)):
        return [_serialize_object(i) for i in obj]
    if isinstance(obj, dict):
        return {k: _serialize_object(v) for k, v in obj.items()}

    data = {
        "__class__": f"{obj.__class__.__module__}.{obj.__class__.__name__}",
        "__attributes__": {}
    }
    for attr_name, attr_value in vars(obj).items():
        data["__attributes__"][attr_name] = _serialize_object(attr_value)
    return data

def _deserialize_object(data):
    """
    Reverse of _serialize_object. Rebuilds custom objects from 
    their __class__ path and __attributes__ dict, or reconstructs
    basic containers if no __class__ key present.
    """
    if not isinstance(data, dict) or "__class__" not in data:
        # Might be a primitive, list, or dict
        if isinstance(data, list):
            return [_deserialize_object(i) for i in data]
        elif isinstance(data, dict):
            return {k: _deserialize_object(v) for k, v in data.items()}
        else:
            return data

    # We do have a custom class signature
    class_path = data["__class__"]
    attributes = data["__attributes__"]

    module_name, class_name = class_path.rsplit(".", 1)
    mod = importlib.import_module(module_name)
    cls = getattr(mod, class_name)

    # Create the instance without calling __init__
    obj = cls.__new__(cls)

    for attr_name, attr_val in attributes.items():
        setattr(obj, attr_name, _deserialize_object(attr_val))

    return obj

class DatasetDescriptor:
    """
    The main entry point for dataset serialization/deserialization.
    """
    def serialize(self, dataset_obj) -> str:
        """
        Convert a dataset object to a JSON string.
        """
        return json.dumps(_serialize_object(dataset_obj))

    def deserialize(self, json_str: str):
        """
        Convert a JSON string back into a dataset object.
        """
        data = json.loads(json_str)
        return _deserialize_object(data)
