#!/usr/bin/env python3
"""
Test runner for SVG Find and Replace project.

This script runs all unit and integration tests for the SVG Find and Replace project.
"""

import unittest
import sys
import os
import argparse

# Add the project root directory to the Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

def create_test_suite():
    """Create a test suite containing all tests."""
    loader = unittest.TestLoader()

    # Discover all tests in the tests directory
    start_dir = os.path.join(project_root, 'tests')
    suite = loader.discover(start_dir, pattern='test_*.py', top_level_dir=project_root)

    return suite

def run_tests(verbosity=2):
    """Run all tests."""
    suite = create_test_suite()
    
    runner = unittest.TextTestRunner(
        verbosity=verbosity,
        stream=sys.stdout,
        buffer=True  # Capture stdout/stderr during tests
    )
    
    result = runner.run(suite)
    
    # Return True if all tests passed, False otherwise
    return result.wasSuccessful()

def run_specific_test_module(module_name):
    """Run tests from a specific module."""
    loader = unittest.TestLoader()

    # Import the specific test module
    project_root = os.path.dirname(os.path.abspath(__file__))
    if module_name == 'unit':
        start_dir = os.path.join(project_root, 'tests', 'unit')
        suite = loader.discover(start_dir, pattern='test_*.py', top_level_dir=project_root)
    elif module_name == 'integration':
        start_dir = os.path.join(project_root, 'tests', 'integration')
        suite = loader.discover(start_dir, pattern='test_*.py', top_level_dir=project_root)
    else:
        # Try to import as a specific test file
        try:
            suite = loader.loadTestsFromName(f'tests.{module_name}')
        except AttributeError:
            print(f"Test module '{module_name}' not found.")
            return False

    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout, buffer=True)
    result = runner.run(suite)

    return result.wasSuccessful()

def main():
    """Main function to run the test suite."""
    parser = argparse.ArgumentParser(description='Run tests for SVG Find and Replace project')
    parser.add_argument(
        '--module', 
        '-m', 
        choices=['unit', 'integration'],
        help='Run tests from a specific module (unit or integration)'
    )
    parser.add_argument(
        '--verbose', 
        '-v', 
        action='store_true',
        help='Run tests with verbose output'
    )
    
    args = parser.parse_args()
    
    print("SVG Find and Replace - Test Runner")
    print("=" * 40)
    
    if args.module:
        success = run_specific_test_module(args.module)
    else:
        verbosity = 2 if args.verbose else 1
        success = run_tests(verbosity=verbosity)
    
    print("\n" + "=" * 40)
    if success:
        print("All tests passed! ✓")
        sys.exit(0)
    else:
        print("Some tests failed! ✗")
        sys.exit(1)

if __name__ == '__main__':
    main()