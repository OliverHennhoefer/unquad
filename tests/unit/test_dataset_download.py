import unittest
import io
import unittest.mock as mock
from urllib.error import URLError

from unquad.utils.data import load


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
            compressed_stream = load._download_dataset(
                dataset_filename, show_progress=False
            )
        except URLError as e:
            self.skipTest(f"Network error, skipping test: {e}")

        # 2. Verify it's there
        self.assertIsInstance(
            compressed_stream, io.BytesIO, "Should return BytesIO object"
        )
        self.assertIn(
            dataset_filename, load._DATASET_CACHE, "Dataset should be in memory cache"
        )
        self.assertGreater(
            len(load._DATASET_CACHE[dataset_filename]),
            0,
            "Cached data should not be empty",
        )

        # 3. Verify we can read it again from cache (should be instant)
        cached_stream = load._download_dataset(dataset_filename, show_progress=False)
        self.assertIsInstance(
            cached_stream, io.BytesIO, "Cached retrieval should also return BytesIO"
        )

        # 4. Delete it from memory
        load.clear_memory_cache("breast")

        # 5. Verify it's gone
        self.assertNotIn(
            dataset_filename,
            load._DATASET_CACHE,
            "Dataset should be removed from memory cache",
        )

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

    def test_cache_persistence_across_calls(self):
        """Test that cache persists across multiple function calls."""
        # Clear cache
        load.clear_memory_cache()

        # Mock only the network call, not the entire _download_dataset function
        mock_data = b"fake_parquet_data"
        with mock.patch("unquad.utils.data.load.urlopen") as mock_urlopen:
            # Mock the response object
            mock_response = mock.MagicMock()
            mock_response.read.return_value = mock_data
            mock_response.headers = {"Content-Length": str(len(mock_data))}
            mock_response.__enter__.return_value = mock_response
            mock_urlopen.return_value = mock_response

            with mock.patch("unquad.utils.data.load.Request"):
                # First call should trigger download
                stream1 = load._download_dataset("test.parquet.gz", show_progress=False)
                self.assertEqual(mock_urlopen.call_count, 1)

                # Second call should use cache (no network call)
                stream2 = load._download_dataset("test.parquet.gz", show_progress=False)
                self.assertEqual(mock_urlopen.call_count, 1)  # Should not increase

                # Verify cache content
                self.assertIn("test.parquet.gz", load._DATASET_CACHE)
                self.assertEqual(load._DATASET_CACHE["test.parquet.gz"], mock_data)

    def test_cache_clear_all_vs_specific(self):
        """Test clearing all cache vs specific dataset."""
        # Add mock data to cache
        load._DATASET_CACHE["dataset1.parquet.gz"] = b"data1"
        load._DATASET_CACHE["dataset2.parquet.gz"] = b"data2"

        # Verify both are cached
        self.assertEqual(len(load._DATASET_CACHE), 2)
        self.assertIn("dataset1.parquet.gz", load._DATASET_CACHE)
        self.assertIn("dataset2.parquet.gz", load._DATASET_CACHE)

        # Clear specific dataset
        load.clear_memory_cache("dataset1")
        self.assertEqual(len(load._DATASET_CACHE), 1)
        self.assertNotIn("dataset1.parquet.gz", load._DATASET_CACHE)
        self.assertIn("dataset2.parquet.gz", load._DATASET_CACHE)

        # Clear all
        load.clear_memory_cache()
        self.assertEqual(len(load._DATASET_CACHE), 0)

    def test_cache_info_accuracy(self):
        """Test that cache info functions return accurate information."""
        # Clear and add test data
        load.clear_memory_cache()
        test_data1 = b"x" * 1000  # 1000 bytes
        test_data2 = b"y" * 2000  # 2000 bytes

        load._DATASET_CACHE["small.parquet.gz"] = test_data1
        load._DATASET_CACHE["large.parquet.gz"] = test_data2

        # Test list_cached_datasets
        cached_list = load.list_cached_datasets()
        self.assertIn("small", cached_list)
        self.assertIn("large", cached_list)
        self.assertEqual(len(cached_list), 2)

        # Test get_memory_cache_info
        info = load.get_memory_cache_info()
        self.assertEqual(info["cache_type"], "in-memory")
        self.assertEqual(len(info["datasets"]), 2)
        # Use round to handle floating point precision
        expected_kb = round((1000 + 2000) / 1024, 1)
        self.assertEqual(info["total_size_kb"], expected_kb)

        # Check individual dataset info
        dataset_names = [d["name"] for d in info["datasets"]]
        self.assertIn("small", dataset_names)
        self.assertIn("large", dataset_names)

        # Find small dataset info
        small_info = next(d for d in info["datasets"] if d["name"] == "small")
        self.assertEqual(small_info["size_bytes"], 1000)
        self.assertEqual(small_info["size_kb"], 1.0)

    def test_cache_robustness(self):
        """Test cache behavior in edge cases."""
        # Test clearing non-existent dataset
        load.clear_memory_cache("nonexistent")  # Should not raise error

        # Test with empty cache
        load.clear_memory_cache()
        self.assertEqual(load.list_cached_datasets(), [])

        info = load.get_memory_cache_info()
        self.assertEqual(info["total_size_kb"], 0)
        self.assertEqual(info["total_size_mb"], 0)
        self.assertEqual(len(info["datasets"]), 0)

    def test_cache_memory_efficiency(self):
        """Test that cache doesn't duplicate data unnecessarily."""
        # Create large test data
        large_data = b"x" * 10000  # 10KB
        load._DATASET_CACHE["large.parquet.gz"] = large_data

        # Get multiple streams from same cached data
        stream1 = load._download_dataset("large.parquet.gz", show_progress=False)
        stream2 = load._download_dataset("large.parquet.gz", show_progress=False)

        # Verify both streams work
        self.assertEqual(len(stream1.read()), 10000)
        self.assertEqual(len(stream2.read()), 10000)

        # Verify cache still contains original data
        self.assertEqual(len(load._DATASET_CACHE["large.parquet.gz"]), 10000)

    def test_cache_state_isolation(self):
        """Test that cache state is properly isolated between operations."""
        # Clear cache
        load.clear_memory_cache()

        # Add data and verify initial state
        load._DATASET_CACHE["test.parquet.gz"] = b"test_data"
        self.assertEqual(len(load._DATASET_CACHE), 1)

        # Get cache info - should not modify cache
        info = load.get_memory_cache_info()
        self.assertEqual(len(load._DATASET_CACHE), 1)
        expected_kb = round(len(b"test_data") / 1024, 1)
        self.assertEqual(info["total_size_kb"], expected_kb)

        # List cached datasets - should not modify cache
        cached_list = load.list_cached_datasets()
        self.assertEqual(len(load._DATASET_CACHE), 1)
        self.assertEqual(cached_list, ["test"])

        # Clear specific dataset
        load.clear_memory_cache("test")
        self.assertEqual(len(load._DATASET_CACHE), 0)

    def test_bytesio_stream_reuse(self):
        """Test that BytesIO streams from cache can be used multiple times."""
        # Clear cache
        load.clear_memory_cache()

        # Add test data to cache
        test_data = b"test_parquet_data" * 100
        load._DATASET_CACHE["reuse_test.parquet.gz"] = test_data

        # Get multiple streams from the same cached data
        stream1 = load._download_dataset("reuse_test.parquet.gz", show_progress=False)
        stream2 = load._download_dataset("reuse_test.parquet.gz", show_progress=False)
        stream3 = load._download_dataset("reuse_test.parquet.gz", show_progress=False)

        # Each stream should contain the full data
        data1 = stream1.read()
        data2 = stream2.read()
        data3 = stream3.read()

        self.assertEqual(
            len(data1), len(test_data), "Stream 1 should contain full data"
        )
        self.assertEqual(
            len(data2), len(test_data), "Stream 2 should contain full data"
        )
        self.assertEqual(
            len(data3), len(test_data), "Stream 3 should contain full data"
        )

        self.assertEqual(data1, test_data, "Stream 1 data should match original")
        self.assertEqual(data2, test_data, "Stream 2 data should match original")
        self.assertEqual(data3, test_data, "Stream 3 data should match original")

        # Cache should still contain the original data
        self.assertEqual(
            len(load._DATASET_CACHE["reuse_test.parquet.gz"]), len(test_data)
        )

    def test_download_caching_behavior(self):
        """Test that _download_dataset properly caches and avoids re-downloads."""
        # Clear cache
        load.clear_memory_cache()

        # Mock the network layer, not the entire download function
        test_data = b"cached_test_data"
        with mock.patch("unquad.utils.data.load.urlopen") as mock_urlopen:
            # Mock the response object
            mock_response = mock.MagicMock()
            mock_response.read.return_value = test_data
            mock_response.headers = {"Content-Length": str(len(test_data))}
            mock_response.__enter__.return_value = mock_response
            mock_urlopen.return_value = mock_response

            with mock.patch("unquad.utils.data.load.Request"):
                # First call should trigger download
                stream1 = load._download_dataset(
                    "cache_test.parquet.gz", show_progress=False
                )
                self.assertEqual(
                    mock_urlopen.call_count, 1, "First call should trigger download"
                )

                # Second call should use cache (no additional download)
                stream2 = load._download_dataset(
                    "cache_test.parquet.gz", show_progress=False
                )
                self.assertEqual(
                    mock_urlopen.call_count,
                    1,
                    "Second call should not trigger download",
                )

                # Third call should also use cache
                stream3 = load._download_dataset(
                    "cache_test.parquet.gz", show_progress=False
                )
                self.assertEqual(
                    mock_urlopen.call_count, 1, "Third call should not trigger download"
                )

                # All streams should contain the same data
                data1 = stream1.read()
                data2 = stream2.read()
                data3 = stream3.read()

                self.assertEqual(data1, data2, "Stream 1 and 2 should have same data")
                self.assertEqual(data2, data3, "Stream 2 and 3 should have same data")


if __name__ == "__main__":
    unittest.main()
