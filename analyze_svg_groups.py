#!/usr/bin/env python3

import xml.etree.ElementTree as ET
import re

def parse_points(points_str):
    """Parse points string into list of (x, y) tuples"""
    coords = [float(x) for x in re.findall(r'[-+]?(?:\d+\.?\d*)', points_str)]
    return [(coords[i], coords[i+1]) for i in range(0, len(coords), 2)]

def calculate_dimensions_from_points(points_list):
    """Calculate width and height from a list of points"""
    if not points_list:
        return None, None
    
    min_x = min(x for x, y in points_list)
    max_x = max(x for x, y in points_list)
    min_y = min(y for x, y in points_list)
    max_y = max(y for x, y in points_list)
    
    return max_x - min_x, max_y - min_y

def main():
    # Parse the SVG file
    tree = ET.parse('/workspace/lookup.svg')
    root = tree.getroot()
    
    # Find all groups with IDs matching the pattern
    for elem in root.iter():
        if elem.tag.endswith('g'):
            group_id = elem.get('id')
            if group_id and ('find_' in group_id or 'replace_' in group_id):
                # Collect all coordinates in the group before applying transforms
                all_coords = []
                
                # Process all nested elements to find the original shape dimensions
                for child in elem.iter():
                    # Skip the group itself
                    if child == elem:
                        continue
                    
                    # Check for points attribute (polygons, polylines)
                    points = child.get('points')
                    if points:
                        coords = parse_points(points)
                        all_coords.extend(coords)
                    
                    # Check for path data (d attribute) for simple shapes
                    d = child.get('d')
                    if d:
                        # Extract coordinates from path data
                        path_coords = [float(x) for x in re.findall(r'[-+]?(?:\d+\.?\d*)', d)]
                        # Process path coordinates (they alternate between x and y)
                        for i in range(0, len(path_coords), 2):
                            if i + 1 < len(path_coords):
                                x, y = path_coords[i], path_coords[i+1]
                                all_coords.append((x, y))
                
                if all_coords:
                    width, height = calculate_dimensions_from_points(all_coords)
                    if width is not None and height is not None:
                        print(f"Group {group_id}: height={height}, width={width}")
                    else:
                        print(f"Group {group_id}: Could not calculate dimensions")
                else:
                    print(f"Group {group_id}: No coordinates found")

if __name__ == "__main__":
    main()