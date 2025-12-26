# SVG Group Replacer - Project Documentation

## Overview

SVG Group Replacer is a Python tool for finding and replacing groups of elements in SVG files based on visual similarity matching. The project provides functionality to process SVG files and replace groups of elements according to a lookup mapping. It uses visual comparison to identify matching groups between input and lookup SVG files, then replaces them with corresponding replacement content.

This tool is particularly useful for batch processing SVG files where specific groups need to be replaced with alternative content.

## Project Structure

- `svg_replacer.py` - Main Python script for processing SVG files
- `svg_element_matcher.py` - Functions for matching SVG elements based on visual content
- `svg_element_replacer.py` - Functions for replacing matched SVG elements
- `svg_bounding_box.py` - Advanced bounding box calculation with transform support
- `svg_replacer_ui.py` - GUI interface for the SVG replacer (using tkinter)
- `svg_replacer_qtside.py` - Qt-based GUI interface (alternative to tkinter)
- `mainwindow.ui` - Qt Designer UI file
- `ui_mainwindow.py` - UI components for the main window (generated from mainwindow.ui)
- `input_find_003.txt` - Sample input file
- `README.md` - Project documentation
- `.gitignore` - Git ignore configuration

## Key Features

- Visual similarity matching of SVG groups using advanced comparison algorithms
- Support for rotated groups (compensates for rotation differences)
- Position-aware replacement with transform preservation
- Automatic removal of matched groups and overlapping elements
- Support for complex SVG elements (paths, rectangles, circles, ellipses, lines, polylines, polygons)
- Precise bounding box calculations for accurate positioning
- Advanced coordinate normalization and path data processing
- GUI interface for easy usage (both tkinter and Qt versions)
- Matrix transformation support for complex positioning

## Usage

### Command Line Usage
```bash
python svg_replacer.py <input.svg> <lookup.svg> <output.svg>
```

### GUI Usage
```bash
python svg_replacer_ui.py
```

The lookup SVG file should contain groups with IDs starting with:
- `find_XXX` - Groups to match in the input SVG
- `replace_XXX` - Corresponding replacement groups (where XXX is the same number as in the find group)

## Dependencies

### Core Dependencies
- Python 3.x
- xml.etree.ElementTree (standard library)
- re (standard library)
- math (standard library)
- os (standard library)
- sys (standard library)
- copy (standard library)
- typing (standard library)

### Additional Dependencies
- svgpathtools
- numpy
- PySide6 (for Qt GUI)

Install dependencies with:
```bash
pip install svgpathtools numpy PySide6
```

## How It Works

1. Parses input and lookup SVG files using XML parsing
2. Identifies groups in both files (excluding lookup-specific groups)
3. Normalizes SVG content by removing non-essential attributes and standardizing formats
4. Matches input groups with lookup groups based on visual similarity using advanced comparison algorithms
5. Calculates precise bounding boxes and centers for accurate positioning
6. Applies rotation compensation to handle rotated elements
7. Calculates appropriate transformations to position replacements correctly
8. Removes matched groups and overlapping elements
9. Adds replacement groups with appropriate transforms and positioning
10. Outputs the modified SVG with all replacements applied

## Advanced Features

### Coordinate Normalization
- Converts relative path commands to absolute coordinates
- Normalizes color formats (RGB, hex, named colors) to consistent representations
- Handles floating-point precision differences
- Standardizes path data formatting

### Transform Processing
- Handles complex matrix transformations
- Applies rotation, scaling, and translation transformations
- Preserves original element transforms while applying positioning
- Supports nested transforms in group hierarchies

### Bounding Box Calculation
- Precise bounding box calculation with all transforms applied
- Handles stroke widths and other visual properties
- Supports all SVG element types (rect, circle, ellipse, path, polygon, polyline, line)
- Accounts for complex path shapes using svgpathtools

## Development Conventions

The codebase follows Python best practices with:
- Clear, descriptive function names
- Comprehensive docstrings
- Type hints where appropriate
- Modular design with separate modules for different functionality
- Proper error handling and logging

## Building and Running

### Prerequisites
- Python 3.x installed
- Required packages installed via pip

### Running the Tool
1. Command line: `python svg_replacer.py input.svg lookup.svg output.svg`
2. GUI (tkinter): `python svg_replacer_ui.py`
3. GUI (Qt): `python svg_replacer_qtside.py`

## Testing

The project doesn't appear to have formal unit tests, but the functionality can be verified by:
1. Running the tool with sample SVG files
2. Verifying that matching groups are properly identified and replaced
3. Checking that transformations are correctly applied
4. Ensuring output SVG files are valid and properly formatted

## Architecture

The project is organized into several modules:

- `svg_element_matcher.py`: Contains functions for extracting and matching SVG elements based on visual content
- `svg_element_replacer.py`: Contains functions for replacing matched elements with appropriate transformations
- `svg_bounding_box.py`: Contains advanced bounding box calculation functions with transform support
- `svg_replacer.py`: Main entry point that coordinates the matching and replacement process
- `svg_replacer_ui.py`: GUI interface using tkinter
- `svg_replacer_qtside.py`: GUI interface using Qt (PySide6)
- `ui_mainwindow.py`: Generated UI code from Qt Designer file
- `mainwindow.ui`: Qt Designer UI definition file

Each module has a clear responsibility and can be tested independently.

## Testing

The project includes a comprehensive test suite with:

- Unit tests for individual functions in each module
- Integration tests for end-to-end functionality
- Test fixtures with sample SVG files
- A test runner script for easy execution

The tests are organized in the `tests/` directory with the following structure:
- `tests/unit/` - Unit tests for individual functions
- `tests/integration/` - Integration tests for full workflows
- `tests/fixtures/` - Sample SVG files for testing

To run all tests, use: `python run_tests.py`
To run only unit tests: `python run_tests.py --module unit`
To run only integration tests: `python run_tests.py --module integration`