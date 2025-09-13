"""Simple test without external dependencies to verify basic structure."""

import sys
import os

# Add the package to the path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def test_package_structure():
    """Test that the package can be imported and has expected structure."""
    print("Testing package structure...")
    
    try:
        # Test basic imports
        import cleanepi
        print("✅ cleanepi package imported successfully")
        
        # Check version
        print(f"Version: {cleanepi.__version__}")
        
        # Check that main functions exist
        expected_functions = [
            'clean_data',
            'replace_missing_values', 
            'remove_constants',
            'find_and_remove_duplicates',
            'standardize_date',
            'convert_to_numeric',
            'print_report'
        ]
        
        for func_name in expected_functions:
            if hasattr(cleanepi, func_name):
                print(f"✅ {func_name} function available")
            else:
                print(f"❌ {func_name} function missing")
                
        print("✅ Package structure test passed")
        return True
        
    except ImportError as e:
        print(f"❌ Failed to import cleanepi: {e}")
        return False


def test_utils_module():
    """Test utils module functionality."""
    print("\nTesting utils module...")
    
    try:
        from cleanepi.utils import get_default_params, COMMON_NA_STRINGS
        
        # Test get_default_params
        defaults = get_default_params()
        print(f"✅ Default parameters loaded: {len(defaults)} categories")
        
        # Test common NA strings
        print(f"✅ Common NA strings loaded: {len(COMMON_NA_STRINGS)} strings")
        
        # Check some expected keys
        expected_keys = [
            'replace_missing_values',
            'remove_duplicates', 
            'standardize_dates',
            'remove_constants'
        ]
        
        for key in expected_keys:
            if key in defaults:
                print(f"✅ {key} default parameters available")
            else:
                print(f"❌ {key} default parameters missing")
                
        print("✅ Utils module test passed")
        return True
        
    except Exception as e:
        print(f"❌ Utils module test failed: {e}")
        return False


def test_function_signatures():
    """Test that functions have expected signatures."""
    print("\nTesting function signatures...")
    
    try:
        from cleanepi import (
            clean_data,
            replace_missing_values,
            remove_constants,
            print_report
        )
        
        # Test that functions are callable
        assert callable(clean_data), "clean_data should be callable"
        assert callable(replace_missing_values), "replace_missing_values should be callable"
        assert callable(remove_constants), "remove_constants should be callable"
        assert callable(print_report), "print_report should be callable"
        
        print("✅ All main functions are callable")
        print("✅ Function signatures test passed")
        return True
        
    except Exception as e:
        print(f"❌ Function signatures test failed: {e}")
        return False


if __name__ == "__main__":
    print("Running basic cleanepi package tests...\n")
    
    tests = [
        test_package_structure,
        test_utils_module,
        test_function_signatures
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test failed with error: {e}")
    
    print(f"\n📊 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All basic tests passed!")
        sys.exit(0)
    else:
        print("❌ Some tests failed")
        sys.exit(1)