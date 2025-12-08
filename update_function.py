#!/usr/bin/env python3
import re

# Read the file
with open('/workspace/svg_replacer.py', 'r') as f:
    content = f.read()

# Define the new function
new_function = '''def create_shape_signature(path_contents):
    """
    Create a signature of shapes by counting types and attributes.
    This helps match paths that have different coordinate representations but same structure.
    """
    signature = {}
    for content in path_contents:
        # Count different element types
        if 'path' in content:
            key = 'path'
        elif 'polygon' in content:
            key = 'polygon'
        elif 'polyline' in content:
            key = 'polyline'
        elif 'text' in content:
            key = 'text'
            # Extract text content for signature
            text_match = re.search(r'>([^<]*)<', content)
            if text_match:
                text_content = text_match.group(1).strip()
                if text_content:
                    signature[f"text_content_{text_content}"] = signature.get(f"text_content_{text_content}", 0) + 1
        else:
            key = 'other'

        # Count by element type and stroke/fill attributes
        stroke_match = re.search(r'stroke="([^\"]*)"', content)
        fill_match = re.search(r'fill="([^\"]*)"', content)

        stroke = stroke_match.group(1) if stroke_match else 'none'
        fill = fill_match.group(1) if fill_match else 'none'

        # Normalize colors for comparison
        stroke = normalize_color(stroke)
        fill = normalize_color(fill)

        attr_key = f"{key}_{stroke}_{fill}"
        signature[attr_key] = signature.get(attr_key, 0) + 1

    return signature'''

# Replace the function in the content
pattern = r'def create_shape_signature\(path_contents\):\s*"""\s*[^\n]*\s*[^\n]*\s*[^\n]*\s*[^\n]*\s*[^\n]*\s*signature = \{\}\s*for content in path_contents:\s*# Count different element types\s*if \'path\' in content:\s*key = \'path\'\s*elif \'polygon\' in content:\s*key = \'polygon\'\s*elif \'polyline\' in content:\s*key = \'polyline\'\s*else:\s*key = \'other\'\s*\s*# Count by element type and stroke/fill attributes\s*stroke_match = re.search\(r\'stroke="([^"]*)"\'\, content\)\s*fill_match = re.search\(r\'fill="([^"]*)"\'\, content\)\s*\s*stroke = stroke_match\.group\(1\) if stroke_match else \'none\'\s*fill = fill_match\.group\(1\) if fill_match else \'none\'\s*\s*# Normalize colors for comparison\s*stroke = normalize_color\(stroke\)\s*fill = normalize_color\(fill\)\s*\s*attr_key = f"[^"]*"\s*signature\[attr_key\] = signature\.get\(attr_key, 0\) \+ 1\s*\s*return signature'
updated_content = re.sub(pattern, new_function, content, flags=re.MULTILINE)

# Write the updated content back to the file
with open('/workspace/svg_replacer.py', 'w') as f:
    f.write(updated_content)

print('Function updated successfully!')