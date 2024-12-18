class ForbiddenModelError(Exception):
    """
    Custom exception raised when a forbidden model is passed to a function.
    A forbidden model is unsuitable for one-class classification.
    """

    def __init__(self, message="Forbidden model"):
        self.message = message
        super().__init__(self.message)
