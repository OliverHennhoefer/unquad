import functools
import random
import numpy as np

from sys import float_info
from pyod.models.iforest import IForest

from unquad.data.loader import DataLoader
from unquad.estimator.configuration import DetectorConfig
from unquad.estimator.detector import ConformalDetector
from unquad.utils.enums import Dataset, Strategy

if __name__ == "__main__":

    random.seed(1)
    np.random.seed(1)

    datasets = [member for member in Dataset]
    strategy = [member for member in Strategy]

    models = [
        IForest(behaviour="new", contamination=float_info.min),
        # LOF(contamination=float_info.min),
        # PCA(n_components=3, contamination=float_info.min),
    ]

    dc = DetectorConfig()

    L, J = 100, 100
    fdr, power = [], []

    for i, dataset in enumerate(datasets):
        dl = DataLoader(dataset=dataset)
        df = dl.df

        for j, strategy in enumerate(strategy):
            if strategy in [Strategy.J, Strategy.J_PLUS] and dl.rows > 1_000:
                continue

            x_train, x_test, y_test = dl.get_example_setup(int(str(i) + str(j)))

            for model in models:

                ce = ConformalDetector(detector=model, strategy=strategy, config=dc)
                
                j = range(J)
                func = functools.partial(ce.fit(x_train))