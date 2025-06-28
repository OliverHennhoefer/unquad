import unittest

import pandas as pd
from pandas._testing import assert_frame_equal, assert_series_equal

from unquad.utils.load import load_fraud, load_wbc


class TestDatasetSetup(unittest.TestCase):
    def test_dataset_setup_wbc(self):
        x_train, x_test, y_test = load_wbc(setup=True)

        data = {
            "V1": [3.0, 3.0, 1.0],
            "V2": [2.0, 1.0, 3.0],
            "V3": [2.0, 1.0, 1.0],
            "V4": [1.0, 1.0, 1.0],
            "V5": [4.0, 2.0, 2.0],
            "V6": [3.0, 1.0, 1.0],
            "V7": [2.0, 2.0, 2.0],
            "V8": [1.0, 1.0, 2.0],
            "V9": [1.0, 1.0, 1.0],
        }

        index = [113, 25, 143]
        x_train_ref = pd.DataFrame(data, index=index)
        assert_frame_equal(x_train.iloc[[3, 50, 100], :], x_train_ref)

        data = {
            "V1": [1.0, 3.0, 5.0],
            "V2": [1.0, 1.0, 8.0],
            "V3": [1.0, 1.0, 9.0],
            "V4": [1.0, 1.0, 4.0],
            "V5": [1.0, 2.0, 3.0],
            "V6": [1.0, 1.0, 10.0],
            "V7": [1.0, 2.0, 7.0],
            "V8": [1.0, 2.0, 1.0],
            "V9": [1.0, 1.0, 1.0],
        }

        index = [3, 22, 33]
        x_test_ref = pd.DataFrame(data, index=index)
        assert_frame_equal(x_test.iloc[[3, 22, 33], :], x_test_ref)

        data = [0.0, 1.0, 1.0, 1.0]
        index = [31, 32, 33, 34]
        y_test_ref = pd.Series(data, index=index, dtype="float64", name="Class")
        assert_series_equal(y_test.iloc[31:], y_test_ref)

    def test_dataset_setup_fraud(self):
        x_train, x_test, y_test = load_fraud(setup=True)

        data = {
            "V2": [0.773279, 0.770999, 0.766325, 0.778388],
            "V3": [0.873998, 0.798806, 0.813703, 0.866914],
            "V4": [0.271307, 0.291438, 0.259473, 0.253920],
            "V5": [0.770581, 0.769176, 0.766805, 0.764134],
        }
        index = [97968, 237057, 239093, 128297]
        x_train_ref = pd.DataFrame(data, index=index)
        assert_frame_equal(
            x_train.iloc[[44, 444, 100_000, 142156], [1, 2, 3, 4]], x_train_ref
        )

        data = {
            "V7": [0.255491, 0.268542, 0.258861, 0.179342],
            "V8": [0.796720, 0.790594, 0.786722, 0.905110],
            "V9": [0.549090, 0.440766, 0.454853, 0.321558],
            "V10": [0.500931, 0.490720, 0.528230, 0.317905],
        }
        index = [3, 33, 333, 999]
        x_test_ref = pd.DataFrame(data, index=index)
        assert_frame_equal(x_test.iloc[[3, 33, 333, 999], [6, 7, 8, 9]], x_test_ref)

        data = [0.0, 0.0, 0.0, 1.0, 1.0]
        index = [897, 898, 899, 900, 901]
        y_test_ref = pd.Series(data, index=index, dtype="float64", name="Class")
        assert_series_equal(y_test.iloc[897:902], y_test_ref)


if __name__ == "__main__":
    unittest.main()
