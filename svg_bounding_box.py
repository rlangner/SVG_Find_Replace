import xml.etree.ElementTree as ET
import math
import re
from svgpathtools import parse_path, Line, Arc, CubicBezier, QuadraticBezier
import numpy as np

def multiply_matrices(m1, m2):
    """Multiply two 3x3 transformation matrices represented as 6-element tuples [a, b, c, d, e, f]
    where the matrix is:
    [a c e]
    [b d f]
    [0 0 1]
    """
    a1, b1, c1, d1, e1, f1 = m1
    a2, b2, c2, d2, e2, f2 = m2
    
    # Calculate the resulting matrix
    a = a1 * a2 + c1 * b2
    b = b1 * a2 + d1 * b2
    c = a1 * c2 + c1 * d2
    d = b1 * c2 + d1 * d2
    e = a1 * e2 + c1 * f2 + e1
    f = b1 * e2 + d1 * f2 + f1
    
    return [a, b, c, d, e, f]

def parse_transform(transform_str):
    """Parse SVG transform string into a transformation matrix."""
    if not transform_str:
        return [1, 0, 0, 1, 0, 0]  # identity matrix
    
    # Start with identity matrix
    matrix = [1, 0, 0, 1, 0, 0]
    
    # Find all transform functions
    transforms = re.findall(r'(\w+)\s*\(\s*([^)]+)\s*\)', transform_str)
    
    for func, params_str in transforms:
        params = [float(p.strip()) for p in params_str.split(',')]
        
        if func == 'matrix':
            # matrix(a, b, c, d, e, f)
            new_matrix = params
        elif func == 'translate':
            tx = params[0]
            ty = params[1] if len(params) > 1 else 0
            new_matrix = [1, 0, 0, 1, tx, ty]
        elif func == 'scale':
            sx = params[0]
            sy = params[1] if len(params) > 1 else sx
            new_matrix = [sx, 0, 0, sy, 0, 0]
        elif func == 'rotate':
            angle = math.radians(params[0])
            cos_a = math.cos(angle)
            sin_a = math.sin(angle)
            cx, cy = 0, 0
            if len(params) > 2:
                cx, cy = params[1], params[2]
                # Translate to origin, rotate, then translate back
                translate_to_origin = [1, 0, 0, 1, -cx, -cy]
                rotate_matrix = [cos_a, sin_a, -sin_a, cos_a, 0, 0]
                translate_back = [1, 0, 0, 1, cx, cy]
                new_matrix = multiply_matrices(translate_back, 
                                             multiply_matrices(rotate_matrix, 
                                                             translate_to_origin))
            else:
                new_matrix = [cos_a, sin_a, -sin_a, cos_a, 0, 0]
        elif func == 'skewX':
            tan_a = math.tan(math.radians(params[0]))
            new_matrix = [1, 0, tan_a, 1, 0, 0]
        elif func == 'skewY':
            tan_a = math.tan(math.radians(params[0]))
            new_matrix = [1, tan_a, 0, 1, 0, 0]
        else:
            continue  # Unknown transform function
        
        matrix = multiply_matrices(matrix, new_matrix)
    
    return matrix

def apply_transform_to_point(point, matrix):
    """Apply a transformation matrix to a point (x, y)."""
    x, y = point
    a, b, c, d, e, f = matrix
    
    new_x = a * x + c * y + e
    new_y = b * x + d * y + f
    
    return (new_x, new_y)

def get_element_bbox(element):
    """Get the bounding box of an SVG element."""
    if element.tag.endswith('}rect'):
        x = float(element.get('x', 0))
        y = float(element.get('y', 0))
        width = float(element.get('width', 0))
        height = float(element.get('height', 0))
        
        # Account for stroke width if present
        stroke_width = element.get('stroke-width')
        stroke_offset = 0
        if stroke_width:
            try:
                stroke_offset = float(stroke_width) / 2.0
            except ValueError:
                stroke_offset = 0
        
        return [(x - stroke_offset, y - stroke_offset), 
                (x + width + stroke_offset, y - stroke_offset), 
                (x + width + stroke_offset, y + height + stroke_offset), 
                (x - stroke_offset, y + height + stroke_offset)]
    
    elif element.tag.endswith('}circle'):
        cx = float(element.get('cx', 0))
        cy = float(element.get('cy', 0))
        r = float(element.get('r', 0))
        
        # Account for stroke width if present
        stroke_width = element.get('stroke-width')
        stroke_offset = 0
        if stroke_width:
            try:
                stroke_offset = float(stroke_width) / 2.0
            except ValueError:
                stroke_offset = 0
        
        # Add stroke offset to radius
        r_with_stroke = r + stroke_offset
        return [(cx - r_with_stroke, cy - r_with_stroke), 
                (cx + r_with_stroke, cy - r_with_stroke), 
                (cx + r_with_stroke, cy + r_with_stroke), 
                (cx - r_with_stroke, cy + r_with_stroke)]
    
    elif element.tag.endswith('}ellipse'):
        cx = float(element.get('cx', 0))
        cy = float(element.get('cy', 0))
        rx = float(element.get('rx', 0))
        ry = float(element.get('ry', 0))
        
        # Account for stroke width if present
        stroke_width = element.get('stroke-width')
        stroke_offset = 0
        if stroke_width:
            try:
                stroke_offset = float(stroke_width) / 2.0
            except ValueError:
                stroke_offset = 0
        
        # Add stroke offset to radii
        rx_with_stroke = rx + stroke_offset
        ry_with_stroke = ry + stroke_offset
        return [(cx - rx_with_stroke, cy - ry_with_stroke), 
                (cx + rx_with_stroke, cy - ry_with_stroke), 
                (cx + rx_with_stroke, cy + ry_with_stroke), 
                (cx - rx_with_stroke, cy + ry_with_stroke)]
    
    elif element.tag.endswith('}polygon') or element.tag.endswith('}polyline'):
        points_str = element.get('points', '')
        points = []
        for coord_pair in points_str.split():
            x, y = coord_pair.split(',')
            points.append((float(x.strip()), float(y.strip())))
        return points
    
    elif element.tag.endswith('}path'):
        # Parse path 'd' attribute using svgpathtools
        d = element.get('d', '')
        if not d:
            return []
        
        # Parse the path using svgpathtools
        path = parse_path(d)
        
        # Get the exact bounding box of the path using svgpathtools
        # This is more accurate than sampling points
        if len(path) > 0:
            # Get the bounding box of the path
            xmin, xmax, ymin, ymax = path.bbox()
            
            # Account for stroke width if present
            stroke_width = element.get('stroke-width')
            if stroke_width:
                try:
                    stroke_width = float(stroke_width)
                    # Add half the stroke width to each side
                    stroke_offset = stroke_width / 2.0
                    xmin -= stroke_offset
                    xmax += stroke_offset
                    ymin -= stroke_offset
                    ymax += stroke_offset
                except ValueError:
                    # If stroke-width is not a number (e.g., "auto"), ignore it
                    pass
            
            # Return the 4 corners of the bounding box (including stroke if applicable)
            return [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)]
        else:
            return []
    
    elif element.tag.endswith('}line'):
        x1 = float(element.get('x1', 0))
        y1 = float(element.get('y1', 0))
        x2 = float(element.get('x2', 0))
        y2 = float(element.get('y2', 0))
        
        # Account for stroke width if present
        stroke_width = element.get('stroke-width')
        stroke_offset = 0
        if stroke_width:
            try:
                stroke_offset = float(stroke_width) / 2.0
            except ValueError:
                stroke_offset = 0
        
        # Return bounding box that includes stroke width
        min_x = min(x1, x2) - stroke_offset
        max_x = max(x1, x2) + stroke_offset
        min_y = min(y1, y2) - stroke_offset
        max_y = max(y1, y2) + stroke_offset
        return [(min_x, min_y), (max_x, min_y), (max_x, max_y), (min_x, max_y)]
    
    else:
        # For groups and other elements, we'll return None and handle children separately
        return None

def calculate_group_bbox(svg_root, group_id):
    """Calculate the bounding box of a group with all transforms applied."""
    # Find the group element by ID
    target_group = None
    for elem in svg_root.iter():
        if elem.get('id') == group_id:
            target_group = elem
            break
    
    if target_group is None:
        return None
    
    # Collect all points from the group's content WITH ALL TRANSFORMS applied
    all_points = []
    
    def collect_points_with_transforms(elem, accumulated_matrix):
        """Recursively collect all points from elements within the group with accumulated transforms."""
        # Get element's own transform
        elem_transform = elem.get('transform')
        elem_matrix = parse_transform(elem_transform) if elem_transform else [1, 0, 0, 1, 0, 0]
        
        # Accumulate the current element's transform
        current_matrix = multiply_matrices(accumulated_matrix, elem_matrix)
        
        # Get the element's bounding box points
        elem_bbox = get_element_bbox(elem)
        
        if elem_bbox is not None:
            # Transform each point by the accumulated matrix (includes all parent transforms)
            for point in elem_bbox:
                transformed_point = apply_transform_to_point(point, current_matrix)
                all_points.append(transformed_point)
        
        # Recursively process child elements with the accumulated matrix
        for child in elem:
            collect_points_with_transforms(child, current_matrix)
    
    # Start with identity matrix for the target group
    identity_matrix = [1, 0, 0, 1, 0, 0]
    collect_points_with_transforms(target_group, identity_matrix)
    
    if not all_points:
        return None
    
    # Calculate the final bounding box from all transformed points
    min_x = min(point[0] for point in all_points)
    max_x = max(point[0] for point in all_points)
    min_y = min(point[1] for point in all_points)
    max_y = max(point[1] for point in all_points)
    
    width = max_x - min_x
    height = max_y - min_y
    
    return {'width': width, 'height': height, 'min_x': min_x, 'min_y': min_y, 'max_x': max_x, 'max_y': max_y}

def get_svg_element_sizes(svg_path):
    """Main function to return the bounding box sizes for find_ and replace_ groups."""
    tree = ET.parse(svg_path)
    root = tree.getroot()
    
    results = {}
    
    # Find all groups that start with "find_" or "replace_"
    for elem in root.iter():
        elem_id = elem.get('id')
        if elem_id and (elem_id.startswith('find_') or elem_id.startswith('replace_')):
            bbox = calculate_group_bbox(root, elem_id)
            if bbox:
                results[elem_id] = {
                    'width': round(bbox['width'], 3),
                    'height': round(bbox['height'], 3)
                }
    
    return results

# Test the function with the lookup.svg file
if __name__ == "__main__":
    svg_path = "lookup.svg"
    results = get_svg_element_sizes(svg_path)
    
    print("Bounding box sizes for find_ and replace_ groups:")
    for group_id, size in results.items():
        print(f"{group_id}: W:{size['width']} H:{size['height']}")
    
    # Print expected vs actual for the replace groups mentioned in the problem
    print("\nComparison with expected measurements:")
    expected = {
        'replace_001': {'width': 35.323, 'height': 40.320},
        'replace_002': {'width': 13.539, 'height': 16.365},
        'replace_003': {'width': 48.583, 'height': 37.098}
    }
    
    for group_id, exp_size in expected.items():
        if group_id in results:
            actual = results[group_id]
            print(f"{group_id}:")
            print(f"  Expected: W:{exp_size['width']} H:{exp_size['height']}")
            print(f"  Actual:   W:{actual['width']} H:{actual['height']}")
            print(f"  Diff:     W:{abs(exp_size['width'] - actual['width']):.3f} H:{abs(exp_size['height'] - actual['height']):.3f}")
