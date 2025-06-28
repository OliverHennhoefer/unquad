import unittest
import shutil
from pathlib import Path
from unittest.mock import patch
from urllib.error import URLError

from unquad.utils import load


class TestDatasetDownload(unittest.TestCase):
    """Test dataset download functionality."""
    
    def setUp(self):
        """Set up test environment."""
        # Use a temporary cache directory for testing
        self.test_cache_dir = Path("./test_unquad_cache")
        
        # Patch the CACHE_DIR to use our test directory
        self.patcher = patch.object(load, 'CACHE_DIR', self.test_cache_dir)
        self.patcher.start()
        
        # Ensure test cache directory is clean
        if self.test_cache_dir.exists():
            shutil.rmtree(self.test_cache_dir)
    
    def tearDown(self):
        """Clean up after tests."""
        self.patcher.stop()
        
        # Remove test cache directory
        if self.test_cache_dir.exists():
            shutil.rmtree(self.test_cache_dir)
    
    def test_download_verify_delete(self):
        """Test downloading a dataset, verify it exists, then delete it."""
        dataset_filename = "breast.parquet.gz"
        expected_path = self.test_cache_dir / dataset_filename
        
        # 1. Download the dataset
        try:
            cached_path = load._download_dataset(dataset_filename, show_progress=False)
        except URLError as e:
            self.skipTest(f"Network error, skipping test: {e}")
        
        # 2. Verify it's there
        self.assertTrue(cached_path.exists(), "Downloaded file should exist")
        self.assertEqual(cached_path, expected_path, "Path should be in test cache dir")
        
        # 3. Delete it
        cached_path.unlink()
        
        # 4. Verify it's gone
        self.assertFalse(cached_path.exists(), "File should be deleted")


if __name__ == '__main__':
    unittest.main()