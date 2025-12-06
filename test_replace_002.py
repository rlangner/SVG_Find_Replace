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
        return [(x, y), (x + width, y), (x + width, y + height), (x, y + height)]
    
    elif element.tag.endswith('}circle'):
        cx = float(element.get('cx', 0))
        cy = float(element.get('cy', 0))
        r = float(element.get('r', 0))
        return [(cx - r, cy - r), (cx + r, cy - r), (cx + r, cy + r), (cx - r, cy + r)]
    
    elif element.tag.endswith('}ellipse'):
        cx = float(element.get('cx', 0))
        cy = float(element.get('cy', 0))
        rx = float(element.get('rx', 0))
        ry = float(element.get('ry', 0))
        return [(cx - rx, cy - ry), (cx + rx, cy - ry), (cx + rx, cy + ry), (cx - rx, cy + ry)]
    
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
        
        # Sample points along the path to get bounding box
        # We'll sample at regular intervals to approximate the path shape
        points = []
        for segment in path:
            # Sample points along each segment
            for t in np.linspace(0, 1, 20):  # 20 points per segment
                point = segment.point(t)
                points.append((point.real, point.imag))
        
        return points
    
    elif element.tag.endswith('}line'):
        x1 = float(element.get('x1', 0))
        y1 = float(element.get('y1', 0))
        x2 = float(element.get('x2', 0))
        y2 = float(element.get('y2', 0))
        return [(x1, y1), (x2, y2)]
    
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
    
    # Calculate the cumulative transformation matrix for the group
    def get_cumulative_transform(element):
        """Get the cumulative transformation matrix from the element up to the root."""
        matrices = []
        
        current = element
        while current is not None and current != svg_root:
            transform_attr = current.get('transform')
            if transform_attr:
                matrix = parse_transform(transform_attr)
                matrices.append(matrix)
            current = current.getparent() if hasattr(current, 'getparent') else None
        
        # Multiply all matrices in reverse order (from root to element)
        result_matrix = [1, 0, 0, 1, 0, 0]  # identity
        for matrix in reversed(matrices):
            result_matrix = multiply_matrices(result_matrix, matrix)
        
        return result_matrix
    
    cumulative_matrix = get_cumulative_transform(target_group)
    
    # Find all leaf elements within the group and calculate their transformed bounding boxes
    all_points = []
    
    def collect_points_from_element(elem, parent_matrix):
        """Recursively collect all points from elements within the group."""
        # Get element's own transform
        elem_transform = elem.get('transform')
        elem_matrix = parse_transform(elem_transform) if elem_transform else [1, 0, 0, 1, 0, 0]
        
        # Combine parent and element transforms
        combined_matrix = multiply_matrices(parent_matrix, elem_matrix)
        
        # Get the element's bounding box points
        elem_bbox = get_element_bbox(elem)
        
        if elem_bbox is not None:
            # Transform each point by the combined matrix
            for point in elem_bbox:
                transformed_point = apply_transform_to_point(point, combined_matrix)
                all_points.append(transformed_point)
        
        # Recursively process child elements
        for child in elem:
            collect_points_from_element(child, combined_matrix)
    
    collect_points_from_element(target_group, cumulative_matrix)
    
    if not all_points:
        return None
    
    # Calculate the bounding box from all transformed points
    min_x = min(point[0] for point in all_points)
    max_x = max(point[0] for point in all_points)
    min_y = min(point[1] for point in all_points)
    max_y = max(point[1] for point in all_points)
    
    width = max_x - min_x
    height = max_y - min_y
    
    return {'width': width, 'height': height, 'min_x': min_x, 'min_y': min_y, 'max_x': max_x, 'max_y': max_y}

# Test just the replace_002 group
svg_path = "/workspace/lookup.svg"
tree = ET.parse(svg_path)
root = tree.getroot()

# Calculate bbox for replace_002
bbox = calculate_group_bbox(root, 'replace_002')
if bbox:
    print(f"replace_002: W:{bbox['width']:.3f} H:{bbox['height']:.3f}")
    print(f"  Coordinates: ({bbox['min_x']:.3f}, {bbox['min_y']:.3f}) to ({bbox['max_x']:.3f}, {bbox['max_y']:.3f})")

# Let's also look at the transform for replace_002 specifically
for elem in root.iter():
    if elem.get('id') == 'replace_002':
        transform = elem.get('transform')
        print(f"\nTransform for replace_002: {transform}")
        matrix = parse_transform(transform)
        print(f"Matrix: {matrix}")
        break