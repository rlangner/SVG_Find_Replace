#!/usr/bin/env python3

import xml.etree.ElementTree as ET
import re

def parse_path_data(path_d):
    """Parse SVG path data to extract coordinates."""
    # This is a simplified parser that handles basic path commands
    # Extract all coordinate values from the path data
    numbers = re.findall(r'-?\d+\.?\d*', path_d)
    numbers = [float(n) for n in numbers if n]
    
    # Group numbers into coordinate pairs (x, y)
    coords = [(numbers[i], numbers[i+1]) for i in range(0, len(numbers)-1, 2)]
    return coords

def get_group_dimensions(svg_file):
    tree = ET.parse(svg_file)
    root = tree.getroot()
    
    # Define namespaces
    namespaces = {
        'svg': 'http://www.w3.org/2000/svg',
        'inkscape': 'http://www.inkscape.org/namespaces/inkscape',
        'sodipodi': 'http://sodipodi.sourceforge.net/DTD/sodipodi-0.dtd',
        'serif': 'http://www.serif.com/'
    }
    
    # Register default namespace
    ET.register_namespace('', 'http://www.w3.org/2000/svg')
    
    # Find all groups with IDs containing "find" or "replace"
    groups = root.findall('.//svg:g[@id]', namespaces)
    
    for group in groups:
        group_id = group.get('id')
        all_points = []
        
        if 'find' in group_id:
            # Look for polygon or path elements that define the bounding box
            polygon = group.find('.//svg:polygon[@points]', namespaces)
            if polygon is not None:
                points_str = polygon.get('points')
                points = [tuple(map(float, point.split(','))) for point in points_str.split()]
                
                # Calculate min and max coordinates
                x_coords = [p[0] for p in points]
                y_coords = [p[1] for p in points]
                
                width = max(x_coords) - min(x_coords)
                height = max(y_coords) - min(y_coords)
                
                print(f"Group {group_id}: height={height}, width={width}")
        
        elif 'replace' in group_id:
            # For replace groups, we need to find all child elements to determine bounding box
            # Get all path elements with 'd' attribute
            paths = group.findall('.//svg:path[@d]', namespaces)
            for path in paths:
                d = path.get('d')
                path_coords = parse_path_data(d)
                all_points.extend(path_coords)
            
            # Get all polygon elements
            polygons = group.findall('.//svg:polygon[@points]', namespaces)
            for poly in polygons:
                points_str = poly.get('points')
                points = [tuple(map(float, point.split(','))) for point in points_str.split()]
                all_points.extend(points)
            
            # Get all polyline elements
            polylines = group.findall('.//svg:polyline[@points]', namespaces)
            for polyline in polylines:
                points_str = polyline.get('points')
                points = [tuple(map(float, point.split(','))) for point in points_str.split()]
                all_points.extend(points)
            
            # Get all rect elements
            rects = group.findall('.//svg:rect', namespaces)
            for rect in rects:
                x = float(rect.get('x', 0))
                y = float(rect.get('y', 0))
                width = float(rect.get('width', 0))
                height = float(rect.get('height', 0))
                
                # Add all 4 corners of the rectangle
                all_points.extend([
                    (x, y),
                    (x + width, y),
                    (x, y + height),
                    (x + width, y + height)
                ])
            
            # Get all circle elements
            circles = group.findall('.//svg:circle', namespaces)
            for circle in circles:
                cx = float(circle.get('cx', 0))
                cy = float(circle.get('cy', 0))
                r = float(circle.get('r', 0))
                
                # Add bounding box of the circle
                all_points.extend([
                    (cx - r, cy - r),
                    (cx + r, cy - r),
                    (cx - r, cy + r),
                    (cx + r, cy + r)
                ])
            
            # Get all ellipse elements
            ellipses = group.findall('.//svg:ellipse', namespaces)
            for ellipse in ellipses:
                cx = float(ellipse.get('cx', 0))
                cy = float(ellipse.get('cy', 0))
                rx = float(ellipse.get('rx', 0))
                ry = float(ellipse.get('ry', 0))
                
                # Add bounding box of the ellipse
                all_points.extend([
                    (cx - rx, cy - ry),
                    (cx + rx, cy - ry),
                    (cx - rx, cy + ry),
                    (cx + rx, cy + ry)
                ])
            
            if all_points:
                x_coords = [p[0] for p in all_points]
                y_coords = [p[1] for p in all_points]
                
                width = max(x_coords) - min(x_coords)
                height = max(y_coords) - min(y_coords)
                
                print(f"Group {group_id}: height={height}, width={width}")
            else:
                # If no explicit points found, try to get dimensions from transform or other attributes
                print(f"Group {group_id}: height=unknown, width=unknown (no explicit dimensions found)")

def main():
    svg_file = '/workspace/lookup.svg'
    print("Dimensions of find and replace groups in lookup.svg:")
    get_group_dimensions(svg_file)

if __name__ == "__main__":
    main()