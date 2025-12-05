#!/usr/bin/env python3

import xml.etree.ElementTree as ET
import re
import math

def parse_transform(transform_str):
    """Parse SVG transform string and return transformation matrix"""
    if not transform_str:
        return [1, 0, 0, 1, 0, 0]  # identity matrix
    
    # Process each transform in order
    transforms = re.findall(r'(matrix|translate|scale|rotate)\(([^)]+)\)', transform_str)
    
    # Start with identity matrix
    matrix = [1, 0, 0, 1, 0, 0]
    
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

def get_element_bbox_simple(element):
    """Get bounding box of an SVG element without complex transforms"""
    # Handle different element types
    if element.tag.endswith('rect'):
        x = float(element.get('x', 0))
        y = float(element.get('y', 0))
        width = float(element.get('width', 0))
        height = float(element.get('height', 0))
        return x, y, x + width, y + height
    
    elif element.tag.endswith('circle'):
        cx = float(element.get('cx', 0))
        cy = float(element.get('cy', 0))
        r = float(element.get('r', 0))
        return cx - r, cy - r, cx + r, cy + r
    
    elif element.tag.endswith('ellipse'):
        cx = float(element.get('cx', 0))
        cy = float(element.get('cy', 0))
        rx = float(element.get('rx', 0))
        ry = float(element.get('ry', 0))
        return cx - rx, cy - ry, cx + rx, cy + ry
    
    elif element.tag.endswith('polygon') or element.tag.endswith('polyline'):
        points_str = element.get('points', '')
        points = []
        for point in points_str.split():
            x, y = map(float, point.split(','))
            points.append((x, y))
        
        min_x = min(x for x, y in points)
        max_x = max(x for x, y in points)
        min_y = min(y for x, y in points)
        max_y = max(y for x, y in points)
        return min_x, min_y, max_x, max_y
    
    elif element.tag.endswith('path'):
        # For path elements, we need a more complex parsing
        # This is a simplified approach that just extracts numbers
        d = element.get('d', '')
        numbers = re.findall(r'[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?', d)
        numbers = [float(n) for n in numbers]
        
        if len(numbers) >= 2:
            # Group into coordinate pairs
            coords = [(numbers[i], numbers[i+1]) for i in range(0, len(numbers)-1, 2) if i+1 < len(numbers)]
            if coords:
                min_x = min(x for x, y in coords)
                max_x = max(x for x, y in coords)
                min_y = min(y for x, y in coords)
                max_y = max(y for x, y in coords)
                return min_x, min_y, max_x, max_y
    
    return None

def find_group_bbox_simple(svg_file, group_id):
    """Find the bounding box of a specific group in an SVG file"""
    tree = ET.parse(svg_file)
    root = tree.getroot()
    
    # Find the group element
    group = None
    for g in root.iter():
        if g.get('id') == group_id:
            group = g
            break
    
    if group is None:
        return None
    
    # Get the transform of the group
    group_transform = group.get('transform', '')
    group_matrix = parse_transform(group_transform)
    
    min_x, min_y, max_x, max_y = float('inf'), float('inf'), float('-inf'), float('-inf')
    has_content = False
    
    # Process all child elements recursively
    def process_element(element, parent_matrix):
        nonlocal min_x, min_y, max_x, max_y, has_content
        
        # Get element's transform
        element_transform = element.get('transform', '')
        element_matrix = parse_transform(element_transform)
        
        # Combine matrices
        current_matrix = multiply_matrices(parent_matrix, element_matrix)
        
        # Get bounding box of the element
        bbox = get_element_bbox_simple(element)
        if bbox:
            ex1, ey1, ex2, ey2 = bbox
            
            # Apply the transformation matrix to the bounding box corners
            corners = [
                (ex1, ey1),
                (ex2, ey1),
                (ex2, ey2),
                (ex1, ey2)
            ]
            
            transformed_corners = [apply_transform(x, y, current_matrix) for x, y in corners]
            
            # Update global bounding box
            for tx, ty in transformed_corners:
                min_x = min(min_x, tx)
                max_x = max(max_x, tx)
                min_y = min(min_y, ty)
                max_y = max(max_y, ty)
            
            has_content = True
        
        # Process child elements
        for child in element:
            process_element(child, current_matrix)
    
    # Process all children of the group
    for child in group:
        process_element(child, group_matrix)
    
    if has_content:
        width = max_x - min_x
        height = max_y - min_y
        return width, height, min_x, min_y, max_x, max_y
    
    return None

def main():
    svg_file = '/workspace/lookup.svg'
    
    # Find bounding boxes for replace groups
    groups = ['replace_001', 'replace_002', 'replace_003']
    
    print("Group dimensions based on transformed bounding boxes:")
    for group_id in groups:
        result = find_group_bbox_simple(svg_file, group_id)
        if result:
            width, height, min_x, min_y, max_x, max_y = result
            print(f"Group {group_id}: height={height:.3f}, width={width:.3f}")
        else:
            print(f"Group {group_id}: not found or has no content")

if __name__ == "__main__":
    main()