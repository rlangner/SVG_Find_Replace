#!/usr/bin/env python3

import xml.etree.ElementTree as ET
import re
import math

def apply_transform(x, y, transform):
    """Apply a transform to a point (x, y)"""
    if transform.startswith('matrix('):
        values = [float(x) for x in re.findall(r'[-+]?(?:\d+\.?\d*)', transform)]
        if len(values) >= 6:
            a, b, c, d, e, f = values
            new_x = a * x + c * y + e
            new_y = b * x + d * y + f
            return new_x, new_y
    elif transform.startswith('translate('):
        values = [float(x) for x in re.findall(r'[-+]?(?:\d+\.?\d*)', transform)]
        if len(values) >= 2:
            tx, ty = values[0], values[1]
            return x + tx, y + ty
    
    return x, y

def parse_points(points_str):
    """Parse points string into list of (x, y) tuples"""
    coords = [float(x) for x in re.findall(r'[-+]?(?:\d+\.?\d*)', points_str)]
    return [(coords[i], coords[i+1]) for i in range(0, len(coords), 2)]

def calculate_bbox_for_group(group_elem):
    """Calculate the bounding box for a group by collecting all coordinates and applying transforms"""
    all_coords = []
    
    # Get the group's transform
    group_transform = group_elem.get('transform', '')
    
    # Process all nested elements
    for child in group_elem.iter():
        # Skip the group itself
        if child == group_elem:
            continue
        
        # Check for points attribute (polygons, polylines)
        points = child.get('points')
        if points:
            coords = parse_points(points)
            # Get transform for this element
            elem_transform = child.get('transform', '')
            
            # Apply nested transforms: first element transform, then group transform
            for x, y in coords:
                # Apply element transform
                if elem_transform:
                    x, y = apply_transform(x, y, elem_transform)
                # Apply group transform
                if group_transform:
                    x, y = apply_transform(x, y, group_transform)
                all_coords.append((x, y))
        
        # Check for path data (d attribute)
        d = child.get('d')
        if d:
            # Extract coordinates from path data
            path_coords = [float(x) for x in re.findall(r'[-+]?(?:\d+\.?\d*)', d)]
            # Process path coordinates (they alternate between x and y)
            for i in range(0, len(path_coords), 2):
                if i + 1 < len(path_coords):
                    x, y = path_coords[i], path_coords[i+1]
                    # Get transform for this element
                    elem_transform = child.get('transform', '')
                    
                    # Apply nested transforms: first element transform, then group transform
                    # Apply element transform
                    if elem_transform:
                        x, y = apply_transform(x, y, elem_transform)
                    # Apply group transform
                    if group_transform:
                        x, y = apply_transform(x, y, group_transform)
                    all_coords.append((x, y))
    
    if all_coords:
        min_x = min(coord[0] for coord in all_coords)
        max_x = max(coord[0] for coord in all_coords)
        min_y = min(coord[1] for coord in all_coords)
        max_y = max(coord[1] for coord in all_coords)
        
        width = max_x - min_x
        height = max_y - min_y
        
        return width, height
    else:
        return None, None

def main():
    # Parse the SVG file
    tree = ET.parse('/workspace/lookup.svg')
    root = tree.getroot()
    
    # Find all groups with IDs matching the pattern
    for elem in root.iter():
        if elem.tag.endswith('g'):
            group_id = elem.get('id')
            if group_id and ('find_' in group_id or 'replace_' in group_id):
                width, height = calculate_bbox_for_group(elem)
                
                if width is not None and height is not None:
                    print(f"Group {group_id}: height={height}, width={width}")
                else:
                    print(f"Group {group_id}: No coordinates found")

if __name__ == "__main__":
    main()