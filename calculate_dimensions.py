#!/usr/bin/env python3

import xml.etree.ElementTree as ET
import re
import math

def calculate_transformed_bbox(x, y, width, height, transform):
    """Calculate the bounding box after applying a transform"""
    # Parse transform matrix
    if transform.startswith('matrix('):
        values = [float(x) for x in re.findall(r'[-+]?(?:\d+\.?\d*)', transform)]
        if len(values) >= 6:
            a, b, c, d, e, f = values
            # Transform the four corners of the rectangle
            corners = [
                (x, y),                    # top-left
                (x + width, y),            # top-right
                (x + width, y + height),   # bottom-right
                (x, y + height)            # bottom-left
            ]
            transformed_corners = []
            for px, py in corners:
                new_x = a * px + c * py + e
                new_y = b * px + d * py + f
                transformed_corners.append((new_x, new_y))
            
            # Find the bounding box of transformed corners
            min_x = min(c[0] for c in transformed_corners)
            max_x = max(c[0] for c in transformed_corners)
            min_y = min(c[1] for c in transformed_corners)
            max_y = max(c[1] for c in transformed_corners)
            
            return max_x - min_x, max_y - min_y
    elif transform.startswith('translate('):
        values = [float(x) for x in re.findall(r'[-+]?(?:\d+\.?\d*)', transform)]
        if len(values) >= 2:
            # Translation doesn't change dimensions
            return width, height
    elif transform.startswith('rotate('):
        # For rotation, calculate the bounding box
        values = [float(x) for x in re.findall(r'[-+]?(?:\d+\.?\d*)', transform)]
        if len(values) >= 1:
            angle = math.radians(values[0])
            cos_a = abs(math.cos(angle))
            sin_a = abs(math.sin(angle))
            new_width = width * cos_a + height * sin_a
            new_height = width * sin_a + height * cos_a
            return new_width, new_height
    
    # If no transformation or unknown transformation, return original dimensions
    return width, height

def get_direct_children_points(elem):
    """Get points from direct children of the element (not nested)"""
    min_x, max_x, min_y, max_y = float('inf'), float('-inf'), float('inf'), float('-inf')
    found_coords = False
    
    # Only check direct children, not all descendants
    for child in elem:
        # Check for points attribute (polygons, polylines)
        points = child.get('points')
        if points:
            found_coords = True
            coords = [float(x) for x in re.findall(r'[-+]?(?:\d+\.?\d*)', points)]
            for i in range(0, len(coords), 2):
                x, y = coords[i], coords[i+1]
                min_x = min(min_x, x)
                max_x = max(max_x, x)
                min_y = min(min_y, y)
                max_y = max(max_y, y)
        
        # Check for path data (d attribute)
        d = child.get('d')
        if d:
            found_coords = True
            # Extract coordinates from path data
            path_coords = [float(x) for x in re.findall(r'[-+]?(?:\d+\.?\d*)', d)]
            # Process path coordinates (they alternate between x and y)
            for i in range(0, len(path_coords), 2):
                if i + 1 < len(path_coords):
                    x, y = path_coords[i], path_coords[i+1]
                    min_x = min(min_x, x)
                    max_x = max(max_x, x)
                    min_y = min(min_y, y)
                    max_y = max(max_y, y)
    
    if found_coords:
        return max_x - min_x, max_y - min_y
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
                # Get dimensions from direct children
                width, height = get_direct_children_points(elem)
                
                if width is not None and height is not None:
                    # Apply the group's transformation
                    transform = elem.get('transform', '')
                    if transform:
                        # For the transform, we need the original min_x, min_y values
                        # We'll need to recalculate them for the transform
                        min_x, max_x, min_y, max_y = float('inf'), float('-inf'), float('inf'), float('-inf')
                        found_coords = False
                        
                        for child in elem:
                            points = child.get('points')
                            if points:
                                found_coords = True
                                coords = [float(x) for x in re.findall(r'[-+]?(?:\d+\.?\d*)', points)]
                                for i in range(0, len(coords), 2):
                                    x, y = coords[i], coords[i+1]
                                    min_x = min(min_x, x)
                                    max_x = max(max_x, x)
                                    min_y = min(min_y, y)
                                    max_y = max(max_y, y)
                            
                            d = child.get('d')
                            if d:
                                found_coords = True
                                path_coords = [float(x) for x in re.findall(r'[-+]?(?:\d+\.?\d*)', d)]
                                for i in range(0, len(path_coords), 2):
                                    if i + 1 < len(path_coords):
                                        x, y = path_coords[i], path_coords[i+1]
                                        min_x = min(min_x, x)
                                        max_x = max(max_x, x)
                                        min_y = min(min_y, y)
                                        max_y = max(max_y, y)
                        
                        if found_coords:
                            width, height = calculate_transformed_bbox(min_x, min_y, width, height, transform)
                    
                    print(f"Group {group_id}: height={height}, width={width}")
                else:
                    print(f"Group {group_id}: No coordinates found")

if __name__ == "__main__":
    main()