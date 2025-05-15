"""
descriptor.py: Provides reflection-based JSON serialization and deserialization 
so we can store/retrieve entire dataset objects.
"""

import json
import importlib
import enum

def _serialize_object(obj):
    """
    Recursively serialize Python objects to a JSON-friendly structure.
    """

    # 1) Check basic built-in types:
    if isinstance(obj, (int, float, bool, str, type(None))):
        return obj

    # 2) Check if it's an Enum
    if isinstance(obj, enum.Enum):
        # Return something like the enum's value or name
        return {
            "__enum__": f"{obj.__class__.__module__}.{obj.__class__.__name__}.{obj.name}"
        }
        # Or if you prefer: return obj.value or just a string.

    # 3) Check list/tuple
    if isinstance(obj, (list, tuple)):
        return [_serialize_object(i) for i in obj]

    # 4) Check dict
    if isinstance(obj, dict):
        return {k: _serialize_object(v) for k, v in obj.items()}

    # 5) Otherwise assume a custom class
    data = {
        "__class__": f"{obj.__class__.__module__}.{obj.__class__.__name__}",
        "__attributes__": {}
    }
    # This line fails for objects that don't have __dict__. But custom classes typically do.
    for attr_name, attr_value in vars(obj).items():
        data["__attributes__"][attr_name] = _serialize_object(attr_value)
    return data

def _deserialize_object(data):
    """
    Opposite of _serialize_object. If we detect __enum__, we reconstruct the enum.
    """
    if isinstance(data, list):
        return [_deserialize_object(i) for i in data]
    if isinstance(data, dict) and "__enum__" in data:
        # Rebuild the enum from the string
        enum_path = data["__enum__"]  # e.g. "structpe._types.SentimentLabel.POSITIVE"
        module_name, class_name, member_name = enum_path.rsplit(".", 2)
        mod = importlib.import_module(module_name)
        enum_cls = getattr(mod, class_name)
        return getattr(enum_cls, member_name)

    # If we see __class__, do the reflection logic
    if isinstance(data, dict) and "__class__" in data:
        class_path = data["__class__"]
        attributes = data["__attributes__"]
        module_name, class_name = class_path.rsplit(".", 1)
        mod = importlib.import_module(module_name)
        cls = getattr(mod, class_name)
        obj = cls.__new__(cls)
        for attr_name, attr_val in attributes.items():
            setattr(obj, attr_name, _deserialize_object(attr_val))
        return obj

    # Otherwise it's a primitive
    return data


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
