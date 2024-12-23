class ForbiddenModelError(Exception):
    """
    Custom exception raised when a forbidden model is passed to a function.

    This exception is raised when an anomaly detection model that is not suitable
    for one-class classification is provided to a function. Forbidden models are
    models that do not meet the criteria for use in the given context.

    Attributes:
        message (str): The error message that describes the exception.
    """

    def __init__(self, message="Forbidden model"):
        self.message = message
        super().__init__(self.message)
