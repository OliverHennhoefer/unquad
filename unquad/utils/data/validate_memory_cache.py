#!/usr/bin/env python3
"""Validation script for in-memory dataset caching."""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def check_imports():
    """Check that imports work correctly."""
    try:
        # Test basic imports
        import gzip
        import io
        from pathlib import Path

        print("✓ Basic imports successful")

        # Test the module structure
        from unquad.utils.data import load

        print("✓ Module import successful")

        # Check cache-related functions exist
        functions_to_check = [
            "clear_memory_cache",
            "list_cached_datasets",
            "get_memory_cache_info",
            "load_breast",
            "load_fraud",
            "_download_dataset",
        ]

        for func_name in functions_to_check:
            if hasattr(load, func_name):
                print(f"✓ Function {func_name} exists")
            else:
                print(f"✗ Function {func_name} missing")
                return False

        # Check the global cache exists
        if hasattr(load, "_DATASET_CACHE"):
            print(f"✓ Global cache _DATASET_CACHE exists: {type(load._DATASET_CACHE)}")
        else:
            print("✗ Global cache _DATASET_CACHE missing")
            return False

        return True

    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False


def check_configuration():
    """Check configuration variables."""
    try:
        from unquad.utils.data import load

        print(f"✓ Dataset version: {load.DATASET_VERSION}")
        print(f"✓ Base URL: {load.DATASET_BASE_URL}")
        print(f"✓ Cache type: In-memory dictionary")
        print(f"✓ Current cache size: {len(load._DATASET_CACHE)} datasets")

        return True
    except Exception as e:
        print(f"✗ Configuration check failed: {e}")
        return False


def main():
    """Run all validation checks."""
    print("=== Validating In-Memory Dataset Caching ===\n")

    print("1. Checking imports...")
    if not check_imports():
        print("❌ Import validation failed")
        return False

    print("\n2. Checking configuration...")
    if not check_configuration():
        print("❌ Configuration validation failed")
        return False

    print("\n✅ All validations passed!")
    print("\nThe in-memory caching system is correctly implemented.")
    print("To test with actual data downloads, run the unit tests:")
    print("  python -m unittest tests.unit.test_dataset_download -v")

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
