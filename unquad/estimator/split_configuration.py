import math


class SplitConfiguration:
    def __init__(
        self, n_split: float = None, n_bootstraps: int = None, n_calib: int = None
    ):
        self.n_split = n_split
        self.n_bootstraps = n_bootstraps
        self.n_calib = n_calib

        self.n_train = None
        self.n_params = sum(
            x is not None for x in (self.n_bootstraps, self.n_split, self.n_calib)
        )

    def configure(self, n_train: int):
        self.n_train = n_train
        self._sanity_check()

        if self.n_bootstraps is not None and self.n_split is not None:
            self.n_calib = self.calculate_n_calib(
                n_train=self.n_train,
                n_bootstraps=self.n_bootstraps,
                n_split=self.n_split,
            )
        elif self.n_bootstraps is not None and self.n_calib is not None:
            self.n_split = self.calculate_n_split(
                n_train=self.n_train,
                n_bootstraps=self.n_bootstraps,
                n_calib=self.n_calib,
            )
        elif self.n_split is not None and self.n_calib is not None:
            self.n_bootstraps = self.calculate_n_bootstraps(
                n_train=self.n_train, n_split=self.n_split, n_calib=self.n_calib
            )

    @staticmethod
    def calculate_n_calib(n_train: int, n_bootstraps: int, n_split: float) -> int:
        return math.ceil(n_bootstraps * n_train * n_split)

    @staticmethod
    def calculate_n_bootstraps(n_train: int, n_calib: int, n_split: float) -> int:
        return math.ceil(n_calib / (n_train * (1 - n_split)))

    @staticmethod
    def calculate_n_split(n_train: int, n_bootstraps: int, n_calib: int) -> float:
        return 1 - (n_calib / (n_bootstraps * n_train))

    def _sanity_check(
        self,
    ) -> None:
        if self.n_split is not None and self.n_split >= self.n_train:
            raise ValueError(
                """
                Parameter 'n_split' can't be larger than
                available training data.
                """
            )

        if self.n_params == 1 and self.n_split is not None:
            if self.n_split > 0.0:
                return
            else:
                raise ValueError(
                    """
                    Parameter 'n_split' can't be negative.
                    """
                )
        elif self.n_params != 2:
            raise ValueError(
                """
                Two of parameters 'n_split', 'n_bootstraps', or 'n_calib'
                must be provided for Jackknife[+]-after-Bootstrap.
                """
            )
