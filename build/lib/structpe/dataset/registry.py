"""
Simple registry to keep track of dataset classes by name,
so we can dynamically retrieve them in the CLI.
"""

DATASET_CLASSES = {}

def register_dataset(name: str, dataset_cls):
    DATASET_CLASSES[name] = dataset_cls

def get_dataset_class(name: str):
    if name not in DATASET_CLASSES:
        raise ValueError(f"Dataset class '{name}' not registered.")
    return DATASET_CLASSES[name]
