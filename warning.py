import warnings
def warn():
    warnings.filterwarnings(
        "ignore",
        message=".*pkg_resources.*",
        category=DeprecationWarning )# Or specify UserWarning, RuntimeWarning, etc.
                                    # DeprecationWarning is a common category for such warnings.
                                    # You might need to experiment with the category if it persists.
    
    warnings.filterwarnings(
        "ignore",
        message="Please use pkg_resources.resource_filename instead",
        category=DeprecationWarning # Try this specific message as well
    )