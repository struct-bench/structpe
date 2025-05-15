"""
registry.py

Keeps a dictionary of dataset classes keyed by their string name.
"""

DATASET_CLASSES = {}

def register_dataset(name: str, dataset_cls):
    """
    Register a dataset class under a given name (e.g. 'sentiment').
    """
    DATASET_CLASSES[name] = dataset_cls

def get_dataset_class(name: str):
    """
    Retrieve a dataset class by name, or raise an error if not found.
    """
    if name not in DATASET_CLASSES:
        raise ValueError(f"Dataset class '{name}' not registered.")
    return DATASET_CLASSES[name]
