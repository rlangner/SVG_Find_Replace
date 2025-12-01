# SVG Group Replacer

A Python tool for finding and replacing groups of elements in SVG files based on visual similarity matching.

## Overview

This project provides functionality to process SVG files and replace groups of elements according to a lookup mapping. The tool uses visual comparison to identify matching groups between input and lookup SVG files, then replaces them with corresponding replacement content. It's particularly useful for batch processing SVG files where specific groups need to be replaced with alternative content.

## Features

- Visual similarity matching of SVG groups using bitmap comparison
- Support for rotated groups (compensates for 0°, 90°, 180°, 270° rotations)
- Position-aware replacement with transform preservation
- Automatic removal of groups at the same position as matched groups
- Support for complex SVG elements (paths, rectangles, circles, ellipses, lines, polylines, polygons)

## Files

- `svg_group_replacer.py` - Main Python script for processing SVG files
- `input.svg` - Sample input SVG file to be processed
- `lookup.svg` - SVG file containing replacement mappings (with `find_*` and `replace_*` groups)
- `BEAU-TX-CO-Plans BEFORE.svg` - Sample SVG file (possibly before processing)
- `test.txt` - Test file

## Usage

```bash
python svg_group_replacer.py <input.svg> <lookup.svg> <output.svg>
```

The lookup SVG file should contain groups with IDs starting with:
- `find_XXX` - Groups to match in the input SVG
- `replace_XXX` - Corresponding replacement groups (where XXX is the same number as in the find group)

## Requirements

- Python 3.x
- xml.etree.ElementTree (standard library)
- re (standard library)
- cairo
- math (standard library)
- PIL (Pillow)
- numpy
- os (standard library)
- sys (standard library)
- xml.dom (standard library)

Install dependencies with:
```bash
pip install pycairo Pillow numpy
```

## How It Works

1. Parses input and lookup SVG files
2. Identifies groups in both files (excluding lookup-specific groups)
3. Converts SVG groups to bitmaps for visual comparison
4. Matches input groups with lookup groups based on visual similarity
5. Applies rotation compensation to handle rotated elements
6. Removes matched groups and groups at the same position
7. Adds replacement groups with appropriate transforms
8. Outputs the modified SVG

## License

[Specify license type if applicable]
