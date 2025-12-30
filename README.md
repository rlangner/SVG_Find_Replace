# SVG Pattern Matcher & Replacer

A Python tool for finding and replacing patterns of SVG elements based on element type, order, style, and color matching.

## Overview

This tool processes SVG files to find and replace specific patterns of elements. Unlike simple ID-based replacement, it uses **pattern matching** to identify sequences of elements based on their visual characteristics (tag type, fill color, stroke color, stroke width) and replaces them with corresponding replacement groups from a lookup file.

The tool is particularly useful for:
- Batch processing architectural or technical drawings
- Replacing repeated patterns with updated designs
- Standardizing symbols across multiple SVG files
- Converting legacy SVG elements to new formats

## Key Features

### Pattern Matching
- **Element-based pattern matching**: Matches sequences of elements by tag type, fill, stroke, and stroke-width
- **Color-aware matching**: Distinguishes patterns based on fill and stroke colors (supports rgb(), hex, and named colors)
- **Style attribute parsing**: Extracts colors from both element attributes and CSS-style attributes
- **Duplicate detection**: Finds ALL instances of a pattern, not just the first occurrence
- **Overlap prevention**: Ensures matched elements aren't reused in multiple replacements

### Transform Handling
- **Scale compensation**: Maintains original replacement size regardless of parent transforms
- **Rotation handling**: Calculates net rotation between find and replace patterns
- **Position preservation**: Places replacements at the exact center of matched elements
- **Local coordinate systems**: Works correctly within nested group hierarchies
- **Matrix transformations**: Full support for complex SVG transformation matrices

### Accurate Positioning
- **Bounding box calculations**: Precise bbox calculation with all transforms applied
- **Center-based positioning**: Aligns replacement center with matched elements center
- **Parent-relative coordinates**: Calculates positions in parent's local coordinate system
- **Transform chain support**: Handles nested transforms correctly

## Project Structure

### Core Modules (304 + 212 = 516 lines total)

- **`svg_replacer.py`** - Main entry point script (33 lines)
- **`svg_element_replacer.py`** - Core replacement logic (304 lines)
  - `get_transform_rotation_angle()` - Extracts rotation from transform matrices
  - `replace_groups_in_svg()` - Main replacement function with scale/rotation/position handling
- **`svg_pattern_matcher.py`** - Pattern matching engine (212 lines)
  - `normalize_color()` - Normalizes color formats for comparison
  - `get_element_detailed_signature()` - Creates element signatures with style parsing
  - `match_all_patterns()` - Finds all occurrences of a pattern
  - `find_pattern_matches()` - Main pattern matching function

### Supporting Modules

- **`svg_element_matcher.py`** - Group extraction and basic matching utilities
- **`svg_bounding_box.py`** - Bounding box calculation with transform support
- **`svg_replacer_ui.py`** - GUI interface (optional)

### Sample Files

- **`input.svg`** - Sample input SVG file to be processed
- **`lookup.svg`** - Lookup file with `find_*` and `replace_*` pattern groups
- **`output.svg`** - Expected output (for testing)

## Installation

### Requirements

- Python 3.x
- Standard library modules: `xml.etree.ElementTree`, `re`, `math`, `copy`, `uuid`, `sys`
- External dependencies:
  - `svgpathtools` - For advanced path calculations
  - `numpy` - For numerical operations

### Install Dependencies

```bash
pip install svgpathtools numpy
```
## Usage

### Command Line

```bash
python svg_replacer.py <input.svg> <lookup.svg> <output.svg>
```

**Example:**
```bash
python svg_replacer.py input.svg lookup.svg output_test.svg
```

### GUI (Optional)

```bash
python svg_replacer_ui.py
```

### Lookup File Format

The lookup SVG file must contain pattern groups with specific ID naming:

- **`find_XXX`** - Pattern to search for in the input SVG
  - Contains a sequence of child elements that define the pattern
  - Elements are matched by tag type, fill, stroke, and stroke-width
  - Can have transforms (rotation is accounted for)

- **`replace_XXX`** - Replacement content (where XXX matches the find group number)
  - Will be inserted at the position of matched elements
  - Original size and rotation are preserved
  - Can have transforms that will be applied relative to the find pattern

**Example lookup.svg structure:**
```xml
<svg>
  <g id="find_001">
    <path fill="rgb(255,0,0)" d="..."/>
    <path fill="rgb(0,255,0)" d="..."/>
  </g>
  <g id="replace_001">
    <rect fill="blue" width="100" height="100"/>
  </g>
</svg>
```

## How It Works

### 1. Pattern Matching Phase

The tool scans the input SVG and creates **element signatures** for each child element:

```python
signature = {
    'tag': 'path',                    # Element type
    'fill': 'rgb(255,0,0)',          # Fill color (normalized)
    'stroke': 'rgb(0,0,0)',          # Stroke color (normalized)
    'stroke_width': '2',             # Stroke width
    'id_base': 'HATCH'               # ID without trailing numbers
}
```

It then looks for **contiguous sequences** of elements that match the pattern defined in each `find_XXX` group.

**Key features:**
- Matches by element order and characteristics
- Distinguishes patterns by color differences
- Finds ALL occurrences (handles duplicates)
- Prevents overlapping matches

### 2. Replacement Phase

For each matched pattern, the tool:

1. **Calculates matched elements center** in parent's local coordinate system
2. **Extracts parent's scale** from its transform matrix
3. **Calculates compensation scale** to maintain original replacement size
4. **Extracts rotations** from both find and replace patterns
5. **Calculates net rotation** (replace_rotation - find_rotation)
6. **Builds final transform matrix** with:
   - Scale: `original_scale × compensation_scale`
   - Rotation: `net_rotation`
   - Translation: Positions replacement center at matched elements center
7. **Removes matched elements** from parent group
8. **Inserts replacement** with calculated transform

### 3. Transform Calculation

The final transform matrix is calculated as:

```python
# Scale with rotation
total_scale = original_scale * compensation_scale
final_a = total_scale * cos(net_rotation)
final_b = total_scale * sin(net_rotation)
final_c = total_scale * (-sin(net_rotation))
final_d = total_scale * cos(net_rotation)

# Translation to position center correctly
final_tx = target_x - final_a * center_x - final_c * center_y
final_ty = target_y - final_b * center_x - final_d * center_y

# Result: matrix(a, b, c, d, tx, ty)
```

This ensures:
- ✅ Replacement maintains its original size from lookup.svg
- ✅ Replacement is positioned at the matched elements' center
- ✅ Rotation is correctly applied relative to the find pattern
- ✅ Works correctly within nested group hierarchies

## Algorithm Details

### Color Normalization

Colors are normalized to `rgb(r,g,b)` format for consistent matching:

- `#FF0000` → `rgb(255,0,0)`
- `#F00` → `rgb(255,0,0)`
- `rgb(255, 0, 0)` → `rgb(255,0,0)`
- Named colors → lowercase

### Style Attribute Parsing

The tool parses CSS-style attributes to extract colors:

```python
style = "fill:rgb(255,0,0);stroke:none;stroke-width:2"
# Extracts: fill='rgb(255,0,0)', stroke='none', stroke_width='2'
```

This is crucial because many SVG editors (like Inkscape) store colors in the style attribute rather than as element attributes.

### Duplicate Handling

When multiple instances of a pattern are found, each gets a unique ID:

- First instance: `replace_003_1`
- Second instance: `replace_003_2`
- Third instance: `replace_003_3`

### Overlap Prevention

The algorithm tracks which elements have been matched and prevents them from being matched again:

```python
used_groups = set()
for match in matches:
    if any(elem in used_groups for elem in matched_elements):
        skip_match()  # Already used
    else:
        process_match()
        used_groups.update(matched_elements)
```

## Example Output

Given an input SVG with patterns and a lookup file with find/replace definitions:

```
Input: 6 pattern matches found
- find_001: 1 match  → replace_001_1
- find_003: 2 matches → replace_003_1, replace_003_2
- find_004: 1 match  → replace_004_1
- find_006: 1 match  → replace_006_1
- find_007: 1 match  → replace_007_1

Output: 6 replacements made
```

## Troubleshooting

### Pattern not matching?

1. Check that element types match (path, rect, circle, etc.)
2. Verify colors match (use normalized rgb() format)
3. Check element order - patterns must be contiguous sequences
4. Ensure stroke-width matches when specified

### Replacement in wrong position?

1. Verify parent group has correct transform
2. Check that find and replace patterns have consistent structure
3. Look for nested transforms that might affect positioning

### Replacement wrong size?

1. Check parent group's scale in transform matrix
2. Verify replacement group's original transform in lookup.svg
3. Ensure compensation scale is being calculated correctly

## Performance

- **Pattern matching**: O(n × m) where n = input elements, m = pattern length
- **Duplicate detection**: O(n) with set-based tracking
- **Transform calculations**: O(1) per replacement
- **Typical processing time**: < 1 second for files with hundreds of groups

## License

MIT License (or specify your license)
