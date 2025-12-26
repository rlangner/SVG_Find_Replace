# SVG Group Replacer Project

## Project Overview

This is a Python-based SVG processing tool that enables finding and replacing groups of elements in SVG files based on visual similarity matching. The tool uses advanced comparison algorithms to identify matching SVG groups between input and lookup files, then replaces them with corresponding replacement content.

The project consists of multiple Python modules that work together to provide both command-line and GUI interfaces for batch processing SVG files where specific groups need to be replaced with alternative content.

## Key Features

- **Visual similarity matching** of SVG groups using advanced comparison algorithms
- **Support for rotated groups** with compensation for rotation differences
- **Position-aware replacement** with transform preservation
- **Automatic removal** of matched groups and overlapping elements
- **Support for complex SVG elements** (paths, rectangles, circles, ellipses, lines, polylines, polygons)
- **Precise bounding box calculations** for accurate positioning
- **Advanced coordinate normalization** and path data processing
- **GUI interface** for easy usage
- **Matrix transformation support** for complex positioning

## Project Structure

```
SVG_Find_Replace/
├── svg_replacer.py           # Main script for processing SVG files
├── svg_element_matcher.py    # Functions for matching SVG elements based on visual content
├── svg_element_replacer.py   # Functions for replacing matched SVG elements
├── svg_bounding_box.py       # Advanced bounding box calculation with transform support
├── svg_replacer_ui.py        # GUI interface for the SVG replacer
├── ui_mainwindow.py          # UI components for the main window
├── mainwindow.ui             # Qt Designer UI file
├── input.svg                 # Sample input SVG file to be processed
├── lookup.svg                # SVG file containing replacement mappings
└── README.md                 # Project documentation
```

## Core Components

### 1. `svg_replacer.py`
Main entry point that orchestrates the SVG processing workflow by importing and calling functions from other modules.

### 2. `svg_element_matcher.py`
Contains sophisticated algorithms for matching SVG elements based on:
- Path element comparison
- Color normalization
- Coordinate normalization
- Transform processing
- Shape signature creation
- Structure matching

### 3. `svg_element_replacer.py`
Handles the replacement process including:
- Position calculation for replacements
- Transform matrix calculations
- Rotation compensation
- Group positioning and transformation

### 4. `svg_bounding_box.py`
Provides precise bounding box calculations for SVG elements with:
- Full transform support
- Stroke width accounting
- Matrix multiplication for nested transforms
- Support for all SVG element types

## Building and Running

### Prerequisites
- Python 3.x
- Required Python packages:
  - svgpathtools
  - numpy
  - tkinter (for GUI)

### Installation
```bash
pip install svgpathtools numpy
```

### Command Line Usage
```bash
python svg_replacer.py <input.svg> <lookup.svg> <output.svg>
```

### GUI Usage
```bash
python svg_replacer_ui.py
```

## How It Works

1. **Parsing**: Parses input and lookup SVG files using XML parsing
2. **Group Identification**: Identifies groups in both files (excluding lookup-specific groups)
3. **Normalization**: Normalizes SVG content by removing non-essential attributes and standardizing formats
4. **Matching**: Matches input groups with lookup groups based on visual similarity using advanced comparison algorithms
5. **Positioning**: Calculates precise bounding boxes and centers for accurate positioning
6. **Rotation Compensation**: Applies rotation compensation to handle rotated elements
7. **Transformation**: Calculates appropriate transformations to position replacements correctly
8. **Replacement**: Removes matched groups and overlapping elements, adds replacement groups with appropriate transforms
9. **Output**: Outputs the modified SVG with all replacements applied

## Lookup SVG Format

The lookup SVG file should contain groups with IDs following specific patterns:
- `find_XXX` - Groups to match in the input SVG
- `replace_XXX` - Corresponding replacement groups (where XXX is the same number as in the find group)

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
- Comprehensive error handling
- Detailed documentation in docstrings
- Modular design with clear separation of concerns
- Type hints for function parameters and return values
- Consistent naming conventions
- Proper handling of SVG namespaces

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
- tkinter (for GUI)

## Usage Examples

### Basic Command Line Usage
```bash
python svg_replacer.py input.svg lookup.svg output.svg
```

### GUI Usage
1. Run `python svg_replacer_ui.py`
2. Select input SVG file
3. Select lookup SVG file
4. Select output SVG file
5. Click "Run Find and Replace"

## File Processing Workflow

The tool processes SVG files in the following order:
1. Loads input and lookup SVG files
2. Extracts groups from both files
3. Matches groups based on visual similarity
4. Calculates positioning and transformations
5. Applies replacements with proper positioning
6. Writes the output SVG file

## Error Handling

The tool includes robust error handling for:
- Missing files
- Invalid SVG formats
- Malformed transform attributes
- Coordinate parsing errors
- Matrix calculation errors
- File I/O operations