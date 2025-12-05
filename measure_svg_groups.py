#!/usr/bin/env python3

import xml.etree.ElementTree as ET
from xml.dom import minidom
import math
import re

def parse_transform(transform_str):
    """Parse SVG transform string and return transformation matrix"""
    if not transform_str:
        return [1, 0, 0, 1, 0, 0]  # identity matrix
    
    # Parse different transform types
    matrix_match = re.search(r'matrix\(([^)]+)\)', transform_str)
    translate_match = re.search(r'translate\(([^)]+)\)', transform_str)
    scale_match = re.search(r'scale\(([^)]+)\)', transform_str)
    rotate_match = re.search(r'rotate\(([^)]+)\)', transform_str)
    
    # Start with identity matrix
    matrix = [1, 0, 0, 1, 0, 0]
    
    # Process each transform in order
    transforms = re.findall(r'(matrix|translate|scale|rotate)\(([^)]+)\)', transform_str)
    
    for transform_type, params_str in transforms:
        params = [float(x.strip()) for x in params_str.split(',')]
        
        if transform_type == 'matrix':
            # matrix(a, b, c, d, e, f) where e, f are translation
            new_matrix = params
            matrix = multiply_matrices(matrix, new_matrix)
        elif transform_type == 'translate':
            tx = params[0]
            ty = params[1] if len(params) > 1 else 0
            translate_matrix = [1, 0, 0, 1, tx, ty]
            matrix = multiply_matrices(matrix, translate_matrix)
        elif transform_type == 'scale':
            sx = params[0]
            sy = params[1] if len(params) > 1 else sx
            scale_matrix = [sx, 0, 0, sy, 0, 0]
            matrix = multiply_matrices(matrix, scale_matrix)
        elif transform_type == 'rotate':
            angle = math.radians(params[0])
            cos_a = math.cos(angle)
            sin_a = math.sin(angle)
            cx, cy = 0, 0
            if len(params) > 2:
                cx, cy = params[1], params[2]
            
            # Rotate around point (cx, cy)
            translate_to_origin = [1, 0, 0, 1, -cx, -cy]
            rotate_matrix = [cos_a, sin_a, -sin_a, cos_a, 0, 0]
            translate_back = [1, 0, 0, 1, cx, cy]
            
            matrix = multiply_matrices(matrix, translate_to_origin)
            matrix = multiply_matrices(matrix, rotate_matrix)
            matrix = multiply_matrices(matrix, translate_back)
    
    return matrix

def multiply_matrices(m1, m2):
    """Multiply two 2x3 transformation matrices"""
    a1, b1, c1, d1, e1, f1 = m1
    a2, b2, c2, d2, e2, f2 = m2
    
    # [a1 c1 e1]   [a2 c2 e2]   [a1*a2+c1*b2 a1*c2+c1*d2 a1*e2+c1*f2+e1]
    # [b1 d1 f1] * [b2 d2 f2] = [b1*a2+d1*b2 b1*c2+d1*d2 b1*e2+d1*f2+f1]
    # [0  0  1 ]   [0  0  1 ]   [0          0          1           ]
    
    a = a1 * a2 + c1 * b2
    b = b1 * a2 + d1 * b2
    c = a1 * c2 + c1 * d2
    d = b1 * c2 + d1 * d2
    e = a1 * e2 + c1 * f2 + e1
    f = b1 * e2 + d1 * f2 + f1
    
    return [a, b, c, d, e, f]

def apply_transform(x, y, matrix):
    """Apply transformation matrix to point (x, y)"""
    a, b, c, d, e, f = matrix
    new_x = a * x + c * y + e
    new_y = b * x + d * y + f
    return new_x, new_y

def get_element_bbox(element, parent_matrix=[1, 0, 0, 1, 0, 0]):
    """Get bounding box of an SVG element considering transformations"""
    # Parse the element's transform
    transform = element.get('transform', '')
    element_matrix = parse_transform(transform)
    
    # Combine with parent matrix
    current_matrix = multiply_matrices(parent_matrix, element_matrix)
    
    # Handle different element types
    if element.tag.endswith('rect'):
        x = float(element.get('x', 0))
        y = float(element.get('y', 0))
        width = float(element.get('width', 0))
        height = float(element.get('height', 0))
        
        # Get all 4 corners of the rectangle
        corners = [
            (x, y),
            (x + width, y),
            (x + width, y + height),
            (x, y + height)
        ]
        
        # Apply transformation to each corner
        transformed_corners = [apply_transform(cx, cy, current_matrix) for cx, cy in corners]
        
        # Find min and max coordinates
        min_x = min(cx for cx, cy in transformed_corners)
        max_x = max(cx for cx, cy in transformed_corners)
        min_y = min(cy for cx, cy in transformed_corners)
        max_y = max(cy for cx, cy in transformed_corners)
        
        return min_x, min_y, max_x, max_y
    
    elif element.tag.endswith('circle'):
        cx = float(element.get('cx', 0))
        cy = float(element.get('cy', 0))
        r = float(element.get('r', 0))
        
        # Circle bounds before transformation
        x1, y1 = cx - r, cy - r
        x2, y2 = cx + r, cy + r
        
        # Apply transformation to the corners of the bounding box
        corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
        transformed_corners = [apply_transform(x, y, current_matrix) for x, y in corners]
        
        # Find min and max coordinates
        min_x = min(cx for cx, cy in transformed_corners)
        max_x = max(cx for cx, cy in transformed_corners)
        min_y = min(cy for cx, cy in transformed_corners)
        max_y = max(cy for cx, cy in transformed_corners)
        
        return min_x, min_y, max_x, max_y
    
    elif element.tag.endswith('ellipse'):
        cx = float(element.get('cx', 0))
        cy = float(element.get('cy', 0))
        rx = float(element.get('rx', 0))
        ry = float(element.get('ry', 0))
        
        # Ellipse bounds before transformation
        x1, y1 = cx - rx, cy - ry
        x2, y2 = cx + rx, cy + ry
        
        # Apply transformation to the corners of the bounding box
        corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
        transformed_corners = [apply_transform(x, y, current_matrix) for x, y in corners]
        
        # Find min and max coordinates
        min_x = min(cx for cx, cy in transformed_corners)
        max_x = max(cx for cx, cy in transformed_corners)
        min_y = min(cy for cx, cy in transformed_corners)
        max_y = max(cy for cx, cy in transformed_corners)
        
        return min_x, min_y, max_x, max_y
    
    elif element.tag.endswith('path'):
        d = element.get('d', '')
        # Extract coordinates from path data (simplified approach)
        # This is a basic parser - for full SVG path parsing, a more complex solution is needed
        bbox = parse_path_data(d, current_matrix)
        return bbox
    
    elif element.tag.endswith('polygon') or element.tag.endswith('polyline'):
        points_str = element.get('points', '')
        points = []
        for point in points_str.split():
            x, y = map(float, point.split(','))
            points.append((x, y))
        
        # Apply transformation to each point
        transformed_points = [apply_transform(x, y, current_matrix) for x, y in points]
        
        # Find min and max coordinates
        min_x = min(px for px, py in transformed_points)
        max_x = max(px for px, py in transformed_points)
        min_y = min(py for px, py in transformed_points)
        max_y = max(py for px, py in transformed_points)
        
        return min_x, min_y, max_x, max_y
    
    # For groups, process all child elements
    elif element.tag.endswith('g'):
        min_x, min_y, max_x, max_y = float('inf'), float('inf'), float('-inf'), float('-inf')
        has_content = False
        
        for child in element:
            child_bbox = get_element_bbox(child, current_matrix)
            if child_bbox:
                has_content = True
                cx1, cy1, cx2, cy2 = child_bbox
                min_x = min(min_x, cx1)
                min_y = min(min_y, cy1)
                max_x = max(max_x, cx2)
                max_y = max(max_y, cy2)
        
        if has_content:
            return min_x, min_y, max_x, max_y
    
    return None

def parse_path_data(d, matrix):
    """Parse SVG path data and compute bounding box"""
    # This is a simplified path parser that handles basic commands
    import re
    
    # Extract all coordinate values from path string
    numbers = re.findall(r'[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?', d)
    numbers = [float(n) for n in numbers]
    
    if not numbers:
        return None
    
    # Simplified approach: just get the points mentioned in the path
    # This is not a complete SVG path parser, but will work for many cases
    points = []
    
    # Basic path command parsing (M, L, C, S, Q, T, A, H, V)
    # For simplicity, we'll extract all coordinate pairs
    i = 0
    while i < len(numbers):
        # Look for coordinate pairs (x, y)
        if i + 1 < len(numbers):
            x, y = numbers[i], numbers[i + 1]
            points.append((x, y))
            i += 2
        else:
            i += 1
    
    if not points:
        return None
    
    # Apply transformation to each point
    transformed_points = [apply_transform(x, y, matrix) for x, y in points]
    
    # Find min and max coordinates
    min_x = min(px for px, py in transformed_points)
    max_x = max(px for px, py in transformed_points)
    min_y = min(py for px, py in transformed_points)
    max_y = max(py for px, py in transformed_points)
    
    return min_x, min_y, max_x, max_y

def find_group_bbox(svg_file, group_id):
    """Find the bounding box of a specific group in an SVG file"""
    tree = ET.parse(svg_file)
    root = tree.getroot()
    
    # Define namespace map to handle SVG namespace
    ns = {'svg': 'http://www.w3.org/2000/svg', 
          'sodipodi': 'http://sodipodi.sourceforge.net/DTD/sodipodi-0.dtd',
          'inkscape': 'http://www.inkscape.org/namespaces/inkscape',
          'serif': 'http://www.serif.com/'}
    
    # Try to find the element with the given ID
    elements = []
    for ns_prefix, ns_uri in ns.items():
        elements.extend(root.findall(f'.//{{{ns_uri}}}g[@id="{group_id}"]'))
    
    # Also try without namespace
    elements.extend(root.findall(f'.//{{http://www.w3.org/2000/svg}}g[@id="{group_id}"]'))
    elements.extend(root.findall(f'.//g[@id="{group_id}"]'))
    
    if not elements:
        return None
    
    group = elements[0]
    bbox = get_element_bbox(group)
    
    if bbox:
        min_x, min_y, max_x, max_y = bbox
        width = max_x - min_x
        height = max_y - min_y
        return width, height, min_x, min_y, max_x, max_y
    
    return None

def main():
    svg_file = '/workspace/lookup.svg'
    
    # Find bounding boxes for replace groups
    groups = ['replace_001', 'replace_002', 'replace_003']
    
    for group_id in groups:
        result = find_group_bbox(svg_file, group_id)
        if result:
            width, height, min_x, min_y, max_x, max_y = result
            print(f"Group {group_id}: height={height:.3f}, width={width:.3f}")
        else:
            print(f"Group {group_id}: not found or has no content")

if __name__ == "__main__":
    main()