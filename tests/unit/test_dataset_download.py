import unittest
import io
from unittest.mock import patch
from urllib.error import URLError

from unquad.utils import load


class TestDatasetDownload(unittest.TestCase):
    """Test dataset in-memory download functionality."""
    
    def setUp(self):
        """Set up test environment."""
        # Clear memory cache before each test
        load.clear_memory_cache()
    
    def tearDown(self):
        """Clean up after tests."""
        # Clear memory cache after each test
        load.clear_memory_cache()
    
    def test_download_verify_delete(self):
        """Test downloading a dataset to memory, verify it exists, then delete it."""
        dataset_filename = "breast.parquet.gz"
        
        # 1. Download the dataset to memory
        try:
            compressed_stream = load._download_dataset(dataset_filename, show_progress=False)
        except URLError as e:
            self.skipTest(f"Network error, skipping test: {e}")
        
        # 2. Verify it's there
        self.assertIsInstance(compressed_stream, io.BytesIO, "Should return BytesIO object")
        self.assertIn(dataset_filename, load._DATASET_CACHE, "Dataset should be in memory cache")
        self.assertGreater(len(load._DATASET_CACHE[dataset_filename]), 0, "Cached data should not be empty")
        
        # 3. Verify we can read it again from cache (should be instant)
        cached_stream = load._download_dataset(dataset_filename, show_progress=False)
        self.assertIsInstance(cached_stream, io.BytesIO, "Cached retrieval should also return BytesIO")
        
        # 4. Delete it from memory
        load.clear_memory_cache("breast")
        
        # 5. Verify it's gone
        self.assertNotIn(dataset_filename, load._DATASET_CACHE, "Dataset should be removed from memory cache")
    
    def test_memory_cache_functions(self):
        """Test memory cache management functions."""
        # Initially should be empty
        self.assertEqual(load.list_cached_datasets(), [])
        
        # Download a dataset
        try:
            load._download_dataset("breast.parquet.gz", show_progress=False)
        except URLError as e:
            self.skipTest(f"Network error, skipping test: {e}")
        
        # Should now list the dataset
        cached = load.list_cached_datasets()
        self.assertIn("breast", cached)
        self.assertEqual(len(cached), 1)
        
        # Get cache info
        info = load.get_memory_cache_info()
        self.assertEqual(info["cache_type"], "in-memory")
        self.assertEqual(len(info["datasets"]), 1)
        self.assertEqual(info["datasets"][0]["name"], "breast")
        self.assertGreater(info["datasets"][0]["size_bytes"], 0)
        
        # Clear specific dataset
        load.clear_memory_cache("breast")
        self.assertEqual(load.list_cached_datasets(), [])
    
    def test_load_function_with_memory_download(self):
        """Test that load functions trigger memory download when needed."""
        # Ensure no cached data
        load.clear_memory_cache()
        self.assertEqual(len(load._DATASET_CACHE), 0)
        
        # Try to load dataset (should trigger download to memory)
        try:
            df = load.load_breast()
        except URLError as e:
            self.skipTest(f"Network error, skipping test: {e}")
        except ImportError:
            self.skipTest("pyarrow not installed, skipping test")
        
        # Verify dataset was downloaded to memory and loaded
        self.assertIn("breast.parquet.gz", load._DATASET_CACHE)
        self.assertIsNotNone(df)
        self.assertGreater(len(df), 0)
        
        # Load again - should use cached version (no additional download)
        df2 = load.load_breast()
        self.assertEqual(len(df), len(df2))


if __name__ == '__main__':
    unittest.main()