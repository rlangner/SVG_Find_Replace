import xml.etree.ElementTree as ET
import cairo
import math
from PIL import Image
import numpy as np
import os
import sys
from xml.dom import minidom


def svg_to_bitmap(svg_content, width=64, height=64):
    """Convert SVG content to a bitmap image."""
    # Create a Cairo surface
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, width, height)
    ctx = cairo.Context(surface)
    
    # Set background to transparent
    ctx.set_source_rgba(0, 0, 0, 0)
    ctx.paint()
    
    # Parse the SVG content
    root = ET.fromstring(svg_content)
    
    # Render the SVG content to the context
    # This is a simplified renderer - for full SVG support, a library like CairoSVG would be better
    render_svg_element(ctx, root, width, height)
    
    # Convert to PIL Image
    buf = surface.get_data()
    img_array = np.ndarray(shape=(height, width, 4), dtype=np.uint8, buffer=buf)
    img = Image.fromarray(img_array, 'RGBA')
    
    return img


def render_svg_element(ctx, element, width, height, transform=None):
    """Render an SVG element to the Cairo context."""
    tag = element.tag
    if tag.startswith('{http://www.w3.org/2000/svg}'):
        tag = tag[36:]  # Remove namespace
    
    # Handle transforms
    saved_matrix = ctx.get_matrix()
    if 'transform' in element.attrib:
        # Parse transform attribute and apply
        apply_transform(ctx, element.attrib['transform'])
    
    # Handle different SVG elements
    if tag == 'g':
        # Group element - render children
        for child in element:
            render_svg_element(ctx, child, width, height)
    elif tag in ['path', 'rect', 'circle', 'ellipse', 'line', 'polyline', 'polygon']:
        # Basic shape rendering
        draw_shape(ctx, element, tag)
    
    # Restore matrix
    ctx.set_matrix(saved_matrix)


def apply_transform(ctx, transform_str):
    """Apply a transform string to the Cairo context."""
    # This is a simplified transform parser
    # In a real implementation, you'd want more robust parsing
    if 'translate' in transform_str:
        import re
        match = re.search(r'translate\(([^)]+)\)', transform_str)
        if match:
            values = [float(x.strip()) for x in match.group(1).split(',')]
            if len(values) == 2:
                ctx.translate(values[0], values[1])
    if 'rotate' in transform_str:
        import re
        match = re.search(r'rotate\(([^)]+)\)', transform_str)
        if match:
            values = [float(x.strip()) for x in match.group(1).split(',')]
            angle = math.radians(values[0])
            if len(values) == 3:  # rotate(angle cx cy)
                cx, cy = values[1], values[2]
                ctx.translate(cx, cy)
                ctx.rotate(angle)
                ctx.translate(-cx, -cy)
            else:  # rotate(angle)
                ctx.rotate(angle)


def draw_shape(ctx, element, tag):
    """Draw a basic SVG shape."""
    if tag == 'path':
        d = element.attrib.get('d', '')
        # Simplified path drawing
        parse_path(ctx, d)
    elif tag == 'rect':
        x = float(element.attrib.get('x', 0))
        y = float(element.attrib.get('y', 0))
        width = float(element.attrib.get('width', 0))
        height = float(element.attrib.get('height', 0))
        rx = float(element.attrib.get('rx', 0))
        ry = float(element.attrib.get('ry', 0))
        
        if rx == 0 and ry == 0:
            ctx.rectangle(x, y, width, height)
        else:
            # Rounded rectangle
            ctx.move_to(x + rx, y)
            ctx.arc(x + width - rx, y + ry, ry, -math.pi/2, 0)
            ctx.arc(x + width - rx, y + height - ry, ry, 0, math.pi/2)
            ctx.arc(x + rx, y + height - ry, ry, math.pi/2, math.pi)
            ctx.arc(x + rx, y + ry, ry, math.pi, 3*math.pi/2)
            ctx.close_path()
    elif tag == 'circle':
        cx = float(element.attrib.get('cx', 0))
        cy = float(element.attrib.get('cy', 0))
        r = float(element.attrib.get('r', 0))
        ctx.arc(cx, cy, r, 0, 2 * math.pi)
    elif tag == 'ellipse':
        cx = float(element.attrib.get('cx', 0))
        cy = float(element.attrib.get('cy', 0))
        rx = float(element.attrib.get('rx', 0))
        ry = float(element.attrib.get('ry', 0))
        ctx.save()
        ctx.translate(cx, cy)
        ctx.scale(1, ry/rx)
        ctx.arc(0, 0, rx, 0, 2 * math.pi)
        ctx.restore()
    elif tag == 'line':
        x1 = float(element.attrib.get('x1', 0))
        y1 = float(element.attrib.get('y1', 0))
        x2 = float(element.attrib.get('x2', 0))
        y2 = float(element.attrib.get('y2', 0))
        ctx.move_to(x1, y1)
        ctx.line_to(x2, y2)
    elif tag == 'polyline' or tag == 'polygon':
        points = element.attrib.get('points', '')
        coords = [float(x) for x in points.replace(',', ' ').split()]
        if len(coords) >= 2:
            ctx.move_to(coords[0], coords[1])
            for i in range(2, len(coords), 2):
                if i + 1 < len(coords):
                    ctx.line_to(coords[i], coords[i+1])
            if tag == 'polygon':
                ctx.close_path()
    
    # Apply basic styling
    fill = element.attrib.get('fill', 'black')
    stroke = element.attrib.get('stroke', 'none')
    stroke_width = float(element.attrib.get('stroke-width', 1))
    
    if fill != 'none':
        if fill.startswith('#'):
            r, g, b = int(fill[1:3], 16)/255.0, int(fill[3:5], 16)/255.0, int(fill[5:7], 16)/255.0
            ctx.set_source_rgb(r, g, b)
        else:
            ctx.set_source_rgb(0, 0, 0)  # default to black
        if stroke != 'none':
            ctx.fill_preserve()
        else:
            ctx.fill()
    
    if stroke != 'none':
        ctx.set_line_width(stroke_width)
        if stroke.startswith('#'):
            r, g, b = int(stroke[1:3], 16)/255.0, int(stroke[3:5], 16)/255.0, int(stroke[5:7], 16)/255.0
            ctx.set_source_rgb(r, g, b)
        else:
            ctx.set_source_rgb(0, 0, 0)  # default to black
        ctx.stroke()


def parse_path(ctx, path_data):
    """Parse and draw a path element."""
    import re
    commands = re.findall(r'([MmLlHhVvCcSsQqTtAaZz])([^MmLlHhVvCcSsQqTtAaZz]*)', path_data)
    
    for cmd, args in commands:
        values = [float(x) for x in re.findall(r'-?\d*\.?\d+', args)]
        if cmd == 'M':
            if len(values) >= 2:
                ctx.move_to(values[0], values[1])
        elif cmd == 'L':
            for i in range(0, len(values), 2):
                if i + 1 < len(values):
                    ctx.line_to(values[i], values[i+1])
        elif cmd == 'm':
            if len(values) >= 2:
                ctx.move_to(values[0], values[1])
        elif cmd == 'l':
            for i in range(0, len(values), 2):
                if i + 1 < len(values):
                    ctx.line_to(values[i], values[i+1])
        elif cmd == 'H':
            for val in values:
                ctx.line_to(val, ctx.get_current_point()[1])
        elif cmd == 'V':
            for val in values:
                ctx.line_to(ctx.get_current_point()[0], val)
        elif cmd == 'h':
            for val in values:
                x, y = ctx.get_current_point()
                ctx.line_to(x + val, y)
        elif cmd == 'v':
            for val in values:
                x, y = ctx.get_current_point()
                ctx.line_to(x, y + val)
        elif cmd == 'Z' or cmd == 'z':
            ctx.close_path()


def compare_images(img1, img2, threshold=0.9):
    """Compare two images and return True if they are visually similar."""
    # Convert to grayscale for comparison
    gray1 = img1.convert('L')
    gray2 = img2.convert('L')
    
    # Convert to numpy arrays
    arr1 = np.array(gray1)
    arr2 = np.array(gray2)
    
    # Calculate similarity using normalized cross-correlation
    # First normalize the arrays
    arr1 = arr1.astype(np.float64) / 255.0
    arr2 = arr2.astype(np.float64) / 255.0
    
    # Calculate correlation coefficient
    mean1 = np.mean(arr1)
    mean2 = np.mean(arr2)
    var1 = np.var(arr1)
    var2 = np.var(arr2)
    
    if var1 == 0 and var2 == 0:
        return 1.0  # Both images are uniform
    if var1 == 0 or var2 == 0:
        return 0.0  # One image is uniform, the other is not
    
    numerator = np.sum((arr1 - mean1) * (arr2 - mean2))
    denominator = np.sqrt(var1 * var2 * arr1.size)
    
    if denominator == 0:
        return 0.0
    
    correlation = numerator / denominator
    return correlation


def rotate_image(img, angle):
    """Rotate an image by the given angle in degrees."""
    return img.rotate(angle, expand=False, fillcolor=(0, 0, 0, 0))


def get_element_transform(element):
    """Extract transform information from an SVG element."""
    transform = element.attrib.get('transform', '')
    # Parse transform to extract rotation, translation, etc.
    # This is a simplified version
    import re
    
    rotation = 0
    translation_x = 0
    translation_y = 0
    
    # Extract translation
    translate_match = re.search(r'translate\(([^)]+)\)', transform)
    if translate_match:
        values = [float(x.strip()) for x in translate_match.group(1).split(',')]
        if len(values) >= 2:
            translation_x, translation_y = values[0], values[1]
        elif len(values) == 1:
            translation_x = values[0]
    
    # Extract rotation
    rotate_match = re.search(r'rotate\(([^)]+)\)', transform)
    if rotate_match:
        values = [float(x.strip()) for x in rotate_match.group(1).split(',')]
        rotation = values[0]  # degrees
    
    return rotation, translation_x, translation_y


def set_element_transform(element, rotation, tx, ty):
    """Set transform information for an SVG element."""
    # Create transform string
    transforms = []
    if tx != 0 or ty != 0:
        transforms.append(f"translate({tx},{ty})")
    if rotation != 0:
        transforms.append(f"rotate({rotation})")
    
    if transforms:
        element.attrib['transform'] = ' '.join(transforms)
    elif 'transform' in element.attrib:
        del element.attrib['transform']


def find_matching_groups(input_groups, lookup_groups, replace_groups):
    """Find matching groups between input and lookup SVGs."""
    matches = []
    used_lookup_indices = set()  # Track which lookup groups have been used to avoid duplicate matches
    
    for input_idx, input_group in enumerate(input_groups):
        # Get transform info for the input group
        input_rotation, input_tx, input_ty = get_element_transform(input_group)
        
        # Create a temporary group without transform for visual matching
        temp_input_group = ET.fromstring(ET.tostring(input_group))
        if 'transform' in temp_input_group.attrib:
            del temp_input_group.attrib['transform']
        
        input_svg_str = ET.tostring(temp_input_group, encoding='unicode')
        input_bitmap = svg_to_bitmap(input_svg_str)
        
        best_match = None
        best_score = 0
        best_rotation = 0
        
        # Try different rotations for the lookup groups to match rotated input
        for angle in [0, 90, 180, 270]:
            if angle != 0:
                rotated_input_bitmap = rotate_image(input_bitmap, angle)
            else:
                rotated_input_bitmap = input_bitmap
            
            for lookup_idx, (lookup_id, lookup_group) in enumerate(lookup_groups):
                if lookup_idx in used_lookup_indices:  # Skip already used lookup groups
                    continue
                
                # Create a temporary lookup group without transform for visual matching
                temp_lookup_group = ET.fromstring(ET.tostring(lookup_group))
                if 'transform' in temp_lookup_group.attrib:
                    del temp_lookup_group.attrib['transform']
                
                lookup_svg_str = ET.tostring(temp_lookup_group, encoding='unicode')
                lookup_bitmap = svg_to_bitmap(lookup_svg_str)
                
                score = compare_images(rotated_input_bitmap, lookup_bitmap)
                
                if score > best_score and score > 0.8:  # Threshold for visual match
                    best_score = score
                    best_match = lookup_idx
                    best_rotation = -angle  # Negative because we're compensating for rotation
        
        if best_match is not None:
            # Mark this lookup group as used to prevent duplicate matches
            used_lookup_indices.add(best_match)
            
            # Find corresponding replace group
            lookup_id = lookup_groups[best_match][0]  # The ID of the matched find_ group
            find_num = lookup_id.split('_')[-1]  # Extract the 3-digit number
            
            # Find the corresponding replace group
            replace_group = None
            replace_id = f"replace_{find_num}"
            for rep_id, rep_group in replace_groups:
                if rep_id == replace_id:
                    replace_group = rep_group
                    break
            
            if replace_group is not None:
                # Apply compensation for the rotation that was detected
                final_rotation = input_rotation + best_rotation
                
                matches.append({
                    'input_idx': input_idx,
                    'lookup_idx': best_match,
                    'replace_group': replace_group,
                    'rotation': final_rotation,
                    'tx': input_tx,
                    'ty': input_ty
                })
    
    return matches


def remove_groups_at_same_position(input_root, input_groups, matched_indices):
    """Remove only the matched groups from the input root."""
    # Remove the matched groups themselves (in reverse order to maintain indices)
    for idx in sorted(matched_indices, reverse=True):
        input_root.remove(input_groups[idx])


def main(input_svg_path, lookup_svg_path, output_svg_path):
    """Main function to process SVG files."""
    # Parse input SVG
    input_tree = ET.parse(input_svg_path)
    input_root = input_tree.getroot()
    
    # Find all <g> groups in input SVG (excluding find_ and replace_ groups from lookup)
    # Only get direct children of the root to avoid nested groups
    input_groups = []
    for elem in input_root:
        if elem.tag.endswith('g') and not (elem.get('id') and elem.get('id').startswith(('find_', 'replace_'))):
            input_groups.append(elem)
    
    # Parse lookup SVG
    lookup_tree = ET.parse(lookup_svg_path)
    lookup_root = lookup_tree.getroot()
    
    # Find all find_ and replace_ groups in lookup SVG
    lookup_groups = []
    replace_groups = []
    
    for elem in lookup_root:
        elem_id = elem.get('id', '')
        if elem_id.startswith('find_'):
            lookup_groups.append((elem_id, elem))
        elif elem_id.startswith('replace_'):
            replace_groups.append((elem_id, elem))
    
    print(f"Found {len(input_groups)} input groups, {len(lookup_groups)} lookup groups, {len(replace_groups)} replace groups")
    
    # Find matches
    matches = find_matching_groups(input_groups, lookup_groups, replace_groups)
    
    print(f"Found {len(matches)} matches")
    
    # Get indices of matched input groups
    matched_indices = [match['input_idx'] for match in matches]
    
    # Remove groups at same position as matched groups
    # We need to work with the actual elements, not indices, because removal changes the list
    remove_groups_at_same_position(input_root, input_groups, matched_indices)
    
    # After removal, we need to re-identify the matched groups by their position or other unique characteristics
    # Store the position and transform of matched groups before removal
    matched_groups_info = []
    for match in matches:
        original_group = input_groups[match['input_idx']]
        matched_groups_info.append({
            'original_group': original_group,
            'replace_group': match['replace_group'],
            'rotation': match['rotation'],
            'tx': match['tx'],
            'ty': match['ty']
        })
    
    # Remove the original matched groups
    for info in matched_groups_info:
        try:
            input_root.remove(info['original_group'])
        except ValueError:
            # Element may have already been removed by remove_groups_at_same_position
            pass
    
    # Add the replacement groups
    for info in matched_groups_info:
        # Create a copy of the replacement group
        new_group = ET.fromstring(ET.tostring(info['replace_group']))
        
        # Set the position and rotation
        set_element_transform(new_group, info['rotation'], info['tx'], info['ty'])
        
        # Add the new group to the root
        input_root.append(new_group)
    
    # Write the output SVG
    # Pretty print the XML
    rough_string = ET.tostring(input_root, encoding='unicode')
    
    # Fix namespace issue
    import re
    # Remove namespace prefixes but avoid creating duplicate attributes
    rough_string = re.sub(r'ns\d+:([a-zA-Z]+)', r'\1', rough_string)
    # Remove xmlns:ns0="..." and similar attributes
    rough_string = re.sub(r' xmlns:ns\d+="[^"]*"', '', rough_string)
    # Add proper xmlns attribute if not present
    if ' xmlns=' not in rough_string:
        rough_string = rough_string.replace('<svg', '<svg xmlns="http://www.w3.org/2000/svg"', 1)
    
    # Function to remove duplicate attributes from tags
    import re
    def remove_duplicate_attrs(match):
        full_match = match.group(0)
        tag_name = match.group(1)
        attrs_part = match.group(2) if match.group(2) else ""
        
        # Check if the tag is self-closing (ends with />)
        is_self_closing = full_match.endswith('/>')
        
        # Find all attributes with regex
        attr_pattern = r"([a-zA-Z_:][a-zA-Z0-9_:.-]*)\s*=\s*(?:\"([^\"]*)\"|'([^']*)')"
        found_attrs = []
        seen_attr_names = set()
        
        for attr_match in re.finditer(attr_pattern, attrs_part):
            attr_name = attr_match.group(1)
            attr_value = attr_match.group(2) or attr_match.group(3)  # Get value from double or single quotes
            if attr_name not in seen_attr_names:
                if attr_match.group(2):  # If it was in double quotes
                    found_attrs.append(f'{attr_name}="{attr_value}"')
                else:  # If it was in single quotes
                    found_attrs.append(f"{attr_name}='{attr_value}'")
                seen_attr_names.add(attr_name)
        
        # Return the tag with proper closing
        attrs_str = " ".join(found_attrs)
        if is_self_closing:
            return f'<{tag_name} {attrs_str}/>'
        else:
            return f'<{tag_name} {attrs_str}>'
    
    # Apply to all tags (including self-closing ones) - use a pattern that handles both <tag> and <tag/>
    rough_string = re.sub(r'<(\w+)([^>]*)/?>', remove_duplicate_attrs, rough_string)
    
    # Simple approach to output the SVG without complex processing that causes issues
    try:
        # Pretty print using minidom but without complex regex processing that causes malformed XML
        reparsed = minidom.parseString(rough_string.encode('utf-8'))
        pretty_xml = reparsed.toprettyxml(indent="  ")
        
        # Remove extra blank lines and fix potential issues
        lines = [line for line in pretty_xml.split('\n') if line.strip()]
        pretty_xml = '\n'.join(lines)
        
        # Ensure proper SVG closing tag
        if not pretty_xml.strip().endswith('</svg>'):
            if '</svg>' not in pretty_xml:
                # Add the closing tag if it's missing
                pretty_xml = pretty_xml.rstrip() + '\n</svg>'
    except Exception as e:
        print(f"Error processing XML: {e}")
        # If there's an error, just clean the rough string
        lines = [line for line in rough_string.split('\n') if line.strip()]
        pretty_xml = '\n'.join(lines)
        # Ensure proper closing tag
        if not pretty_xml.strip().endswith('</svg>'):
            if '</svg>' not in pretty_xml:
                pretty_xml = pretty_xml.rstrip() + '\n</svg>'
    
    with open(output_svg_path, 'w') as f:
        f.write(pretty_xml)
    
    print(f"Output written to {output_svg_path}")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python svg_group_replacer.py <input.svg> <lookup.svg> <output.svg>")
        sys.exit(1)
    
    input_svg = sys.argv[1]
    lookup_svg = sys.argv[2]
    output_svg = sys.argv[3]
    
    main(input_svg, lookup_svg, output_svg)