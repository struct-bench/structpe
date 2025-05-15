# structpe/dataset/decorators.py

def dataset_metric(level="dataset"):
    """
    Decorator to mark a function as a dataset metric,
    specifying whether 'dataset' or 'sample'.
    """
    def wrapper(fn):
        fn.dataset_metric_level = level
        return fn
    return wrapper
