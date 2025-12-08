# SVG Group Replacer

A Python tool for finding and replacing groups of elements in SVG files based on visual similarity matching.

## Overview

This project provides functionality to process SVG files and replace groups of elements according to a lookup mapping. The tool uses visual comparison to identify matching groups between input and lookup SVG files, then replaces them with corresponding replacement content. It's particularly useful for batch processing SVG files where specific groups need to be replaced with alternative content.

## Features

- Visual similarity matching of SVG groups using advanced comparison algorithms
- Support for rotated groups (compensates for rotation differences)
- Position-aware replacement with transform preservation
- Automatic removal of matched groups and overlapping elements
- Support for complex SVG elements (paths, rectangles, circles, ellipses, lines, polylines, polygons)
- Precise bounding box calculations for accurate positioning
- Advanced coordinate normalization and path data processing
- GUI interface for easy usage
- Matrix transformation support for complex positioning

## Files

- `svg_replacer.py` - Main Python script for processing SVG files
- `svg_element_matcher.py` - Functions for matching SVG elements based on visual content
- `svg_element_replacer.py` - Functions for replacing matched SVG elements
- `svg_bounding_box.py` - Advanced bounding box calculation with transform support
- `svg_replacer_ui.py` - GUI interface for the SVG replacer
- `ui_mainwindow.py` - UI components for the main window
- `mainwindow.ui` - Qt Designer UI file
- `input.svg` - Sample input SVG file to be processed
- `lookup.svg` - SVG file containing replacement mappings (with `find_*` and `replace_*` groups)
- `BEAU-TX-CO-Plans BEFORE.svg` - Sample SVG file (possibly before processing)
- `test.txt` - Test file

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

## Requirements

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
- PySide6 (for GUI)

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

## License

[Specify license type if applicable]
