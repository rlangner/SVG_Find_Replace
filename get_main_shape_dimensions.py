#!/usr/bin/env python3

import xml.etree.ElementTree as ET
import re

def parse_points(points_str):
    """Parse points string into list of (x, y) tuples"""
    coords = [float(x) for x in re.findall(r'[-+]?(?:\d+\.?\d*)', points_str.replace(',', ' '))]
    return [(coords[i], coords[i+1]) for i in range(0, len(coords), 2)]

def calculate_dimensions_from_points(points_list):
    """Calculate width and height from a list of points"""
    if not points_list:
        return None, None
    
    min_x = min(x for x, y in points_list)
    max_x = max(x for x, y in points_list)
    min_y = min(y for x, y in points_list)
    max_y = max(y for x, y in points_list)
    
    width = max_x - min_x
    height = max_y - min_y
    return width, height

def get_main_shape_dimensions(group_elem):
    """Get the dimensions of the main shape in a group"""
    shape_dimensions = []
    
    # Process all nested elements to find shapes
    for child in group_elem.iter():
        # Skip the group itself
        if child == group_elem:
            continue
        
        # Check for points attribute (polygons, polylines)
        points = child.get('points')
        if points:
            coords = parse_points(points)
            width, height = calculate_dimensions_from_points(coords)
            if width is not None and height is not None:
                # Store the dimensions along with the element to track it
                shape_dimensions.append((width, height, child.get('id', 'unknown')))
        
        # Check for path data (d attribute) for simple shapes
        d = child.get('d')
        if d:
            # Extract coordinates from path data
            path_coords = [float(x) for x in re.findall(r'[-+]?(?:\d+\.?\d*)', d)]
            # Process path coordinates (they alternate between x and y)
            if len(path_coords) >= 4:
                x_coords = [path_coords[i] for i in range(0, len(path_coords), 2)]
                y_coords = [path_coords[i+1] for i in range(0, len(path_coords), 2) if i+1 < len(path_coords)]
                
                if x_coords and y_coords:
                    width = max(x_coords) - min(x_coords)
                    height = max(y_coords) - min(y_coords)
                    shape_dimensions.append((width, height, f"path-{child.get('id', 'unknown')}"))
    
    if shape_dimensions:
        # Return the dimensions of the largest shape (by area)
        largest = max(shape_dimensions, key=lambda x: x[0] * x[1])
        return largest[0], largest[1]
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
                width, height = get_main_shape_dimensions(elem)
                
                if width is not None and height is not None:
                    print(f"Group {group_id}: height={height}, width={width}")
                else:
                    print(f"Group {group_id}: No coordinates found")

if __name__ == "__main__":
    main()