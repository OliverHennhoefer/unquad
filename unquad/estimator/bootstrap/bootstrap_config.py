import warnings


class BootstrapConfiguration:

    def __init__(
        self,
        n: int,
        b: int = None,
        m: float = None,
        c: int = None,
        enforce_c: bool = False,
    ):

        self._sanity_check(n, b, m, c, enforce_c)

        self._n = n  # size of training data
        self._b = b  # number of bootstraps
        self._m = m  # train bootstrap proportion

        self._c = c  # calibration set size
        self._enforce_c = enforce_c

        self._configure()

    @staticmethod
    def _sanity_check(
        n: int, b: int = None, m: float = None, c: int = None, enforce_c: bool = False
    ):
        """
        Ensure parameter 'n', and exactly two of parameters 'b', 'm' and 'c' are given.
        """

        provided_count = sum(x is not None for x in (b, m, c))

        # Ensure exactly two of b, m, c are provided
        if provided_count != 2:
            raise ValueError(
                "Exactly two of parameters 'b', 'm', or 'c' must be provided."
            )

        if b is not None and c is not None and enforce_c is False:
            warnings.warn(
                "Calibration set size will not be enforced during computation."
            )

        if n is None:
            raise ValueError("Parameter 'n' must be provided.")

    def _configure(self):

        if self.b is not None and self._m is not None:
            self._c = self.calculate_c(n=self._n, b=self.b, m=self._m)
        elif self.b is not None and self._c is not None:
            self._m = self.calculate_m(n=self._n, b=self.b, c=self._c)
        elif self._m is not None and self._c is not None:
            self._b = self.calculate_b(n=self._n, m=self._m, c=self._c)

    @staticmethod
    def calculate_c(n: int, b: int, m: float) -> int:
        return round(b * n * m)

    @staticmethod
    def calculate_b(n: int, c: int, m: float) -> int:
        return round(c / (n * m))

    @staticmethod
    def calculate_m(n: int, b: int, c: int) -> float:
        return c / (b * n)

    @property
    def b(self):
        return self._b

    @property
    def c(self):
        return self._c

    @property
    def m(self):
        return self._m

    @property
    def n(self):
        return self._n

    @property
    def enforce_c(self):
        return self._enforce_c
