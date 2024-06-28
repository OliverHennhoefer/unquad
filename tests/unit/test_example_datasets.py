import unittest

from unquad.datasets.loader import DataLoader
from unquad.enums.dataset import Dataset


class TestExampleDatasets(unittest.TestCase):
    def test_dataloader_breastw(self):
        df = DataLoader(dataset=Dataset.BREASTW).df
        self.assertEqual(len(df.index), 683)

    def test_dataloader_fraud(self):
        df = DataLoader(dataset=Dataset.FRAUD).df
        self.assertEqual(len(df.index), 284807)

    def test_dataloader_ionosphere(self):
        df = DataLoader(dataset=Dataset.IONOSPHERE).df
        self.assertEqual(len(df.index), 351)

    def test_dataloader_mammography(self):
        df = DataLoader(dataset=Dataset.MAMMOGRAPHY).df
        self.assertEqual(len(df.index), 11183)

    def test_dataloader_musk(self):
        df = DataLoader(dataset=Dataset.MUSK).df
        self.assertEqual(len(df.index), 3062)

    def test_dataloader_shuttle(self):
        df = DataLoader(dataset=Dataset.SHUTTLE).df
        self.assertEqual(len(df.index), 49097)

    def test_dataloader_thyroid(self):
        df = DataLoader(dataset=Dataset.THYROID).df
        self.assertEqual(len(df.index), 3772)

    def test_dataloader_wbc(self):
        df = DataLoader(dataset=Dataset.WBC).df
        self.assertEqual(len(df.index), 223)
