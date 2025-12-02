#!/usr/bin/env python3
"""
SVG Group Replacer - Debug Version

This script searches through an input.svg file, parses out every <g> group,
compares them to known .svg <g> groups found in lookup.svg, and replaces
visually matching groups with corresponding replacements.

This debug version writes PNG files of each rasterized group for debugging.
"""

import argparse
import xml.etree.ElementTree as ET
import math
import os
import sys
from xml.dom import minidom
import re
from PIL import Image


def save_bitmap_as_png(bitmap, filename):
    """Save a bitmap grid as a PNG file."""
    if not bitmap:
        return
    
    height = len(bitmap)
    width = len(bitmap[0]) if height > 0 else 0
    
    if width == 0:
        return
    
    # Create a new image
    img = Image.new('RGB', (width, height), color='white')
    pixels = img.load()
    
    # Set pixels based on the bitmap
    for y in range(height):
        for x in range(width):
            if bitmap[y][x] == 1:
                pixels[x, y] = (0, 0, 0)  # Black for drawn pixels
            else:
                pixels[x, y] = (255, 255, 255)  # White for background
    
    # Save the image
    img.save(filename)


def svg_to_bitmap(svg_content, width=64, height=64):
    """Convert SVG content to a bitmap image for visual comparison."""
    try:
        # Import here to handle cases where it's not available
        import xml.etree.ElementTree as ET
        import math
        
        # Create a simple rendering system using a 2D grid
        # We'll use a 64x64 grid to represent the image
        grid = [[0 for _ in range(width)] for _ in range(height)]
        
        # If the svg_content is an element, convert it to string
        if isinstance(svg_content, ET.Element):
            # Get the outer group element and all its children
            group_element = svg_content
            # Create a temporary SVG with just this group
            temp_svg = f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 8666 8666">{ET.tostring(group_element, encoding="unicode")}</svg>'
        else:
            temp_svg = svg_content
            if not temp_svg.strip().startswith('<svg'):
                temp_svg = f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 8666 8666">{temp_svg}</svg>'
        
        # Parse the SVG content
        root = ET.fromstring(temp_svg)
        
        # Find all drawable elements within the SVG
        for elem in root.iter():
            tag = elem.tag.split('}')[-1] if '}' in elem.tag else elem.tag  # Remove namespace
            
            if tag in ['path', 'line', 'rect', 'circle', 'ellipse', 'polygon', 'polyline']:
                # Render the element to the grid
                render_element_to_grid(elem, grid, width, height)
        
        return grid
    
    except Exception as e:
        print(f"Error converting SVG to bitmap: {e}")
        # Return a simple hash-based signature as fallback
        return generate_geometric_signature(svg_content)


def render_element_to_grid(element, grid, width, height):
    """Render an SVG element to the grid."""
    tag = element.tag.split('}')[-1] if '}' in element.tag else element.tag  # Remove namespace
    
    if tag == 'polygon' or tag == 'polyline':
        points = element.get('points', '')
        coords = extract_coords_from_points(points)
        if len(coords) >= 2:
            # Draw lines between consecutive points
            for i in range(len(coords) - 1):
                x1, y1 = coords[i]
                x2, y2 = coords[i + 1]
                draw_line_on_grid(grid, x1, y1, x2, y2, width, height)
            
            # For polygon, also draw line from last to first point
            if tag == 'polygon' and len(coords) > 2:
                x1, y1 = coords[-1]
                x2, y2 = coords[0]
                draw_line_on_grid(grid, x1, y1, x2, y2, width, height)
                
    elif tag == 'path':
        d = element.get('d', '')
        # For now, just extract coordinates and draw basic shapes
        path_commands = parse_path_commands(d)
        coords = extract_coords_from_path(d)
        # Draw lines between coordinates
        for i in range(len(coords) - 1):
            x1, y1 = coords[i]
            x2, y2 = coords[i + 1]
            draw_line_on_grid(grid, x1, y1, x2, y2, width, height)
            
    elif tag == 'line':
        x1 = float(element.get('x1', 0))
        y1 = float(element.get('y1', 0))
        x2 = float(element.get('x2', 0))
        y2 = float(element.get('y2', 0))
        draw_line_on_grid(grid, x1, y1, x2, y2, width, height)
        
    elif tag == 'rect':
        x = float(element.get('x', 0))
        y = float(element.get('y', 0))
        width_attr = float(element.get('width', 0))
        height_attr = float(element.get('height', 0))
        # Draw the rectangle outline
        x2, y2 = x + width_attr, y + height_attr
        draw_line_on_grid(grid, x, y, x2, y, width, height)  # top
        draw_line_on_grid(grid, x2, y, x2, y2, width, height)  # right
        draw_line_on_grid(grid, x2, y2, x, y2, width, height)  # bottom
        draw_line_on_grid(grid, x, y2, x, y, width, height)  # left
        
    elif tag == 'circle':
        cx = float(element.get('cx', 0))
        cy = float(element.get('cy', 0))
        r = float(element.get('r', 0))
        draw_circle_on_grid(grid, cx, cy, r, width, height)
        
    elif tag == 'ellipse':
        cx = float(element.get('cx', 0))
        cy = float(element.get('cy', 0))
        rx = float(element.get('rx', 0))
        ry = float(element.get('ry', 0))
        draw_ellipse_on_grid(grid, cx, cy, rx, ry, width, height)


def parse_path_commands(path_data):
    """Parse SVG path data into commands and coordinates."""
    import re
    # This is a simplified parser for basic path commands
    commands = []
    tokens = re.findall(r'[MmLlHhVvZzCcSsQqTtAa]|[+-]?\d*\.?\d+', path_data)
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if token in 'MmLlHhVvZzCcSsQqTtAa':
            commands.append((token, []))
            i += 1
            # Collect coordinates for this command
            while i < len(tokens) and tokens[i] not in 'MmLlHhVvZzCcSsQqTtAa':
                try:
                    commands[-1][1].append(float(tokens[i]))
                except ValueError:
                    pass
                i += 1
        else:
            i += 1
    return commands


def draw_line_on_grid(grid, x1, y1, x2, y2, grid_width, grid_height):
    """Draw a line on the grid using Bresenham's algorithm."""
    # Map coordinates from SVG space to grid space
    # Assuming SVG viewBox is 0,0 to 8666,8666
    scale_x = grid_width / 8666.0
    scale_y = grid_height / 8666.0
    
    x1 = int(x1 * scale_x)
    y1 = int(y1 * scale_y)
    x2 = int(x2 * scale_x)
    y2 = int(y2 * scale_y)
    
    # Clamp to grid bounds
    x1 = max(0, min(grid_width - 1, x1))
    y1 = max(0, min(grid_height - 1, y1))
    x2 = max(0, min(grid_width - 1, x2))
    y2 = max(0, min(grid_height - 1, y2))
    
    # Bresenham's line algorithm
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    sx = 1 if x1 < x2 else -1
    sy = 1 if y1 < y2 else -1
    err = dx - dy
    
    x, y = x1, y1
    
    while True:
        if 0 <= x < grid_width and 0 <= y < grid_height:
            grid[y][x] = 1  # Mark pixel as drawn
        
        if x == x2 and y == y2:
            break
            
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += sx
        if e2 < dx:
            err += dx
            y += sy


def draw_circle_on_grid(grid, cx, cy, r, grid_width, grid_height):
    """Draw a circle on the grid."""
    scale_x = grid_width / 8666.0
    scale_y = grid_height / 8666.0
    
    cx = int(cx * scale_x)
    cy = int(cy * scale_y)
    r = int(r * min(scale_x, scale_y))
    
    # Midpoint circle algorithm
    x = 0
    y = r
    d = 3 - 2 * r
    
    while x <= y:
        # Draw 8 symmetric points
        draw_circle_points(grid, cx, cy, x, y, grid_width, grid_height)
        if d < 0:
            d = d + 4 * x + 6
        else:
            d = d + 4 * (x - y) + 10
            y -= 1
        x += 1


def draw_ellipse_on_grid(grid, cx, cy, rx, ry, grid_width, grid_height):
    """Draw an ellipse on the grid."""
    scale_x = grid_width / 8666.0
    scale_y = grid_height / 8666.0
    
    cx = int(cx * scale_x)
    cy = int(cy * scale_y)
    rx = int(rx * scale_x)
    ry = int(ry * scale_y)
    
    # Draw ellipse using parametric equations
    for angle in range(0, 360, 5):  # Every 5 degrees
        rad = math.radians(angle)
        x = int(cx + rx * math.cos(rad))
        y = int(cy + ry * math.sin(rad))
        if 0 <= x < grid_width and 0 <= y < grid_height:
            grid[y][x] = 1


def draw_circle_points(grid, cx, cy, x, y, grid_width, grid_height):
    """Draw 8 symmetric points of a circle."""
    points = [
        (cx + x, cy + y), (cx - x, cy + y), (cx + x, cy - y), (cx - x, cy - y),
        (cx + y, cy + x), (cx - y, cy + x), (cx + y, cy - x), (cx - y, cy - x)
    ]
    
    for px, py in points:
        if 0 <= px < grid_width and 0 <= py < grid_height:
            grid[py][px] = 1


def compare_bitmaps(bitmap1, bitmap2, tolerance=0.1):
    """Compare two bitmap grids for similarity."""
    if not bitmap1 or not bitmap2:
        return False
    
    if len(bitmap1) != len(bitmap2) or len(bitmap1[0]) != len(bitmap2[0]):
        return False
    
    height = len(bitmap1)
    width = len(bitmap1[0])
    
    # Count matching pixels
    matches = 0
    total_pixels = 0
    
    for y in range(height):
        for x in range(width):
            total_pixels += 1
            if bitmap1[y][x] == bitmap2[y][x]:
                matches += 1
    
    similarity = matches / total_pixels if total_pixels > 0 else 0
    return similarity >= (1 - tolerance)


def get_simple_visual_signature(group_element):
    """Generate a simple visual signature that's more efficient for matching."""
    # Count different types of elements and their properties
    signature_parts = []
    
    for child in group_element:
        tag = child.tag.split('}')[-1] if '}' in child.tag else child.tag  # Remove namespace
        
        if tag in ['path', 'line', 'rect', 'circle', 'ellipse', 'polygon', 'polyline']:
            # Create a signature based on element type and simple properties
            if tag == 'polygon' or tag == 'polyline':
                points = child.get('points', '')
                coords = extract_coords_from_points(points)
                if coords:
                    # Use bounding box as signature
                    xs = [x for x, y in coords]
                    ys = [y for x, y in coords]
                    min_x, max_x = min(xs), max(xs)
                    min_y, max_y = min(ys), max(ys)
                    width = max_x - min_x
                    height = max_y - min_y
                    center_x = (min_x + max_x) / 2
                    center_y = (min_y + max_y) / 2
                    signature_parts.append((tag, round(width, 1), round(height, 1), len(coords), round(center_x, 1), round(center_y, 1)))
            elif tag == 'path':
                d = child.get('d', '')
                # Count path commands as signature
                path_commands = re.findall(r'[MmLlHhVvZzCcSsQqTtAa]', d)
                command_counts = {}
                for cmd in path_commands:
                    command_counts[cmd] = command_counts.get(cmd, 0) + 1
                signature_parts.append((tag, tuple(sorted(command_counts.items())), len(d)))
            elif tag == 'line':
                x1 = round(float(child.get('x1', 0)), 1)
                y1 = round(float(child.get('y1', 0)), 1)
                x2 = round(float(child.get('x2', 0)), 1)
                y2 = round(float(child.get('y2', 0)), 1)
                length = round(((x2-x1)**2 + (y2-y1)**2)**0.5, 1)
                signature_parts.append((tag, length, x1, y1, x2, y2))
            elif tag == 'rect':
                x = round(float(child.get('x', 0)), 1)
                y = round(float(child.get('y', 0)), 1)
                width = round(float(child.get('width', 0)), 1)
                height = round(float(child.get('height', 0)), 1)
                signature_parts.append((tag, width, height, x, y))
            elif tag == 'circle':
                cx = round(float(child.get('cx', 0)), 1)
                cy = round(float(child.get('cy', 0)), 1)
                r = round(float(child.get('r', 0)), 1)
                signature_parts.append((tag, r, cx, cy))
            elif tag == 'ellipse':
                cx = round(float(child.get('cx', 0)), 1)
                cy = round(float(child.get('cy', 0)), 1)
                rx = round(float(child.get('rx', 0)), 1)
                ry = round(float(child.get('ry', 0)), 1)
                signature_parts.append((tag, rx, ry, cx, cy))
    
    # Sort to make signature order-independent
    signature_parts.sort()
    return tuple(signature_parts)


def extract_coords_from_path(path_data):
    """Extract coordinates from SVG path data."""
    coords = []
    current_x, current_y = 0, 0
    
    # Simple approach: just extract all numbers from path
    # This is a simplified approach that doesn't handle all SVG path commands perfectly
    # but should be sufficient for signature generation
    import re
    numbers = re.findall(r'-?\d+\.?\d+', path_data)
    numbers = [float(n) for n in numbers]
    
    # Process numbers in pairs as coordinates
    for i in range(0, len(numbers)-1, 2):
        x, y = numbers[i], numbers[i+1]
        coords.append((x, y))
    
    return coords


def extract_coords_from_points(points_data):
    """Extract coordinates from SVG points data."""
    coords = []
    pairs = points_data.strip().split()
    for pair in pairs:
        x, y = pair.split(',')
        coords.append((float(x), float(y)))
    return coords


def get_element_bbox(element):
    """Get the bounding box of an SVG element by analyzing its child elements."""
    min_x, min_y = float('inf'), float('inf')
    max_x, max_y = float('-inf'), float('-inf')
    has_significant_coords = False
    
    for child in element:
        tag = child.tag.split('}')[-1] if '}' in child.tag else child.tag  # Remove namespace
        
        if tag == 'polygon' or tag == 'polyline':
            points = child.get('points', '')
            coords = extract_coords_from_points(points)
            for x, y in coords:
                # Only consider coordinates that are significantly away from origin
                if abs(x) > 1 or abs(y) > 1:
                    min_x, max_x = min(min_x, x), max(max_x, x)
                    min_y, max_y = min(min_y, y), max(max_y, y)
                    has_significant_coords = True
        elif tag == 'path':
            d = child.get('d', '')
            coords = extract_coords_from_path(d)
            for x, y in coords:
                # Only consider coordinates that are significantly away from origin
                if abs(x) > 1 or abs(y) > 1:
                    min_x, max_x = min(min_x, x), max(max_x, x)
                    min_y, max_y = min(min_y, y), max(max_y, y)
                    has_significant_coords = True
        elif tag == 'rect':
            x = float(child.get('x', 0))
            y = float(child.get('y', 0))
            width = float(child.get('width', 0))
            height = float(child.get('height', 0))
            # Only consider if the rect is significantly away from origin
            if abs(x) > 1 or abs(y) > 1 or width > 1 or height > 1:
                min_x, max_x = min(min_x, x), max(max_x, x + width)
                min_y, max_y = min(min_y, y), max(max_y, y + height)
                has_significant_coords = True
        elif tag == 'circle':
            cx = float(child.get('cx', 0))
            cy = float(child.get('cy', 0))
            r = float(child.get('r', 0))
            # Only consider if the circle is significantly away from origin or has significant radius
            if abs(cx) > 1 or abs(cy) > 1 or r > 1:
                min_x, max_x = min(min_x, cx - r), max(max_x, cx + r)
                min_y, max_y = min(min_y, cy - r), max(max_y, cy + r)
                has_significant_coords = True
        elif tag == 'ellipse':
            cx = float(child.get('cx', 0))
            cy = float(child.get('cy', 0))
            rx = float(child.get('rx', 0))
            ry = float(child.get('ry', 0))
            # Only consider if the ellipse is significantly away from origin or has significant radius
            if abs(cx) > 1 or abs(cy) > 1 or rx > 1 or ry > 1:
                min_x, max_x = min(min_x, cx - rx), max(max_x, cx + rx)
                min_y, max_y = min(min_y, cy - ry), max(max_y, cy + ry)
                has_significant_coords = True
        elif tag in ['line']:
            x1 = float(child.get('x1', 0))
            y1 = float(child.get('y1', 0))
            x2 = float(child.get('x2', 0))
            y2 = float(child.get('y2', 0))
            # Only consider if the line is significantly away from origin or has significant length
            if abs(x1) > 1 or abs(y1) > 1 or abs(x2) > 1 or abs(y2) > 1 or \
               abs(x2-x1) > 1 or abs(y2-y1) > 1:
                min_x, max_x = min(min_x, x1, x2), max(max_x, x1, x2)
                min_y, max_y = min(min_y, y1, y2), max(max_y, y1, y2)
                has_significant_coords = True
    
    # If no significant coordinates were found, return a default bounding box
    if not has_significant_coords:
        # Try to get the transform values instead
        rotation, tx, ty = get_element_transform(element)
        if tx != 0 or ty != 0:
            # Use the transform position as the bounding box
            return (tx, ty, tx + 1, ty + 1)
        else:
            return (0, 0, 0, 0)
    
    return (min_x, min_y, max_x, max_y)


def generate_geometric_signature(svg_content):
    """Generate a geometric signature of an SVG element based on its visual properties."""
    # Parse the SVG content if it's a string
    if isinstance(svg_content, str):
        try:
            # Wrap in SVG tag if it's just a group
            if not svg_content.strip().startswith('<svg'):
                svg_content = f'<svg xmlns="http://www.w3.org/2000/svg">{svg_content}</svg>'
            root = ET.fromstring(svg_content)
        except ET.ParseError:
            # If parsing fails, return a hash of the string
            return hash(svg_content)
    elif isinstance(svg_content, ET.Element):
        root = svg_content
    else:
        return hash(str(svg_content))
    
    # Create a signature based on geometric properties
    signature = []
    
    for elem in root.iter():
        tag = elem.tag.split('}')[-1] if '}' in elem.tag else elem.tag  # Remove namespace
        
        if tag in ['path', 'line', 'rect', 'circle', 'ellipse', 'polygon', 'polyline']:
            if tag == 'path':
                d = elem.get('d', '')
                # For path elements, create a more robust signature by analyzing the path structure
                # Count path elements and analyze the structure rather than just coordinates
                path_commands = re.findall(r'[MmLlHhVvZzCcSsQqTtAa]', d)
                command_counts = {}
                for cmd in path_commands:
                    command_counts[cmd] = command_counts.get(cmd, 0) + 1
                
                # Also include some geometric properties
                coords = extract_coords_from_path(d)
                if coords:
                    # Filter out coordinates that are near the origin (0,0) - these are often placeholders
                    significant_coords = [(x, y) for x, y in coords if abs(x) > 1 or abs(y) > 1]
                    
                    if significant_coords:
                        xs = [x for x, y in significant_coords]
                        ys = [y for x, y in significant_coords]
                        if xs and ys:
                            min_x, max_x = min(xs), max(xs)
                            min_y, max_y = min(ys), max(ys)
                            width = max_x - min_x
                            height = max_y - min_y
                            signature.append(('path', tuple(sorted(command_counts.items())), round(width, 1), round(height, 1), len(significant_coords)))
                    else:
                        # If no significant coords, use all coords but mark as potentially problematic
                        xs = [x for x, y in coords]
                        ys = [y for x, y in coords]
                        if xs and ys:
                            min_x, max_x = min(xs), max(xs)
                            min_y, max_y = min(ys), max(ys)
                            width = max_x - min_x
                            height = max_y - min_y
                            signature.append(('path', tuple(sorted(command_counts.items())), round(width, 1), round(height, 1), len(coords)))
            elif tag == 'line':
                x1 = round(float(elem.get('x1', 0)), 1)
                y1 = round(float(elem.get('y1', 0)), 1)
                x2 = round(float(elem.get('x2', 0)), 1)
                y2 = round(float(elem.get('y2', 0)), 1)
                length = round(((x2-x1)**2 + (y2-y1)**2)**0.5, 1)
                signature.append((tag, length))
            elif tag == 'rect':
                x = round(float(elem.get('x', 0)), 1)
                y = round(float(elem.get('y', 0)), 1)
                width = round(float(elem.get('width', 0)), 1)
                height = round(float(elem.get('height', 0)), 1)
                signature.append((tag, round(width, 1), round(height, 1)))
            elif tag == 'circle':
                cx = round(float(elem.get('cx', 0)), 1)
                cy = round(float(elem.get('cy', 0)), 1)
                r = round(float(elem.get('r', 0)), 1)
                signature.append((tag, round(r, 1)))
            elif tag == 'ellipse':
                cx = round(float(elem.get('cx', 0)), 1)
                cy = round(float(elem.get('cy', 0)), 1)
                rx = round(float(elem.get('rx', 0)), 1)
                ry = round(float(elem.get('ry', 0)), 1)
                signature.append((tag, round(rx, 1), round(ry, 1)))
            elif tag in ['polygon', 'polyline']:
                points = elem.get('points', '')
                coords = extract_coords_from_points(points)
                # Round coordinates to reduce precision differences
                rounded_coords = [(round(x, 1), round(y, 1)) for x, y in coords]
                
                # Filter out coordinates that are near the origin (0,0) - these are often placeholders
                significant_coords = [(x, y) for x, y in rounded_coords if abs(x) > 1 or abs(y) > 1]
                
                if significant_coords:
                    # Calculate bounding box
                    xs = [x for x, y in significant_coords]
                    ys = [y for x, y in significant_coords]
                    min_x, max_x = min(xs), max(xs)
                    min_y, max_y = min(ys), max(ys)
                    width = max_x - min_x
                    height = max_y - min_y
                    signature.append(('shape', round(width, 1), round(height, 1), len(significant_coords)))
                else:
                    # If no significant coords, use all coords but mark as potentially problematic
                    if rounded_coords:
                        xs = [x for x, y in rounded_coords]
                        ys = [y for x, y in rounded_coords]
                        min_x, max_x = min(xs), max(xs)
                        min_y, max_y = min(ys), max(ys)
                        width = max_x - min_x
                        height = max_y - min_y
                        signature.append(('shape', round(width, 1), round(height, 1), len(rounded_coords)))
    
    # Sort signature to make it order-independent
    signature.sort()
    return tuple(signature)


def extract_groups(svg_tree):
    """Extract all <g> groups from an SVG tree."""
    groups = {}
    root = svg_tree.getroot()
    
    # Find all groups in the SVG
    for i, elem in enumerate(root.iter()):
        if elem.tag.endswith('}g') or elem.tag == 'g':  # Handle namespace
            group_id = elem.get('id', '')
            if group_id:  # If group has an ID, use it
                groups[group_id] = elem
            else:  # If no ID, create a unique ID based on its content signature
                signature = generate_geometric_signature(ET.tostring(elem, encoding='unicode'))
                unique_id = f"group_{i}_{abs(hash(signature)) % 10000}"
                groups[unique_id] = elem
    
    return groups


def find_subgroups(group_element):
    """Find all subgroups within a group element."""
    subgroups = []
    for child in group_element:
        if child.tag.endswith('}g') or child.tag == 'g':
            subgroups.append(child)
    return subgroups


def compare_groups_visual(group1, group2, tolerance=0.1):
    """Compare two SVG groups visually by comparing their bitmap representations."""
    # Generate bitmaps for both groups
    bitmap1 = svg_to_bitmap(group1)
    bitmap2 = svg_to_bitmap(group2)
    
    # Compare the bitmaps for similarity
    return compare_bitmaps(bitmap1, bitmap2, tolerance)


def compare_signatures(sig1, sig2, tolerance=0.1):
    """Compare two signatures for similarity."""
    if sig1 == sig2:
        return True
    
    # For more complex comparison, calculate similarity based on common elements
    if isinstance(sig1, tuple) and isinstance(sig2, tuple):
        # Calculate how many elements are in common
        set1 = set(sig1) if sig1 else set()
        set2 = set(sig2) if sig2 else set()
        
        if len(set1) == 0 and len(set2) == 0:
            return True
        if len(set1) == 0 or len(set2) == 0:
            return False
            
        # Calculate Jaccard similarity
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        if union == 0:
            return True  # Both empty
            
        similarity = intersection / union
        return similarity >= (1 - tolerance)
    
    return False


def get_element_transform(element):
    """Extract transform information from an SVG element."""
    transform = element.attrib.get('transform', '')
    # Parse transform to extract rotation, translation, etc.
    
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
        element.set('transform', ' '.join(transforms))
    elif 'transform' in element.attrib:
        del element.attrib['transform']


def find_matching_groups(input_groups, lookup_groups, replace_groups):
    """Find matching groups between input and lookup SVGs."""
    matches = []
    used_input_ids = set()  # Track which input groups have been used to avoid duplicate matches
    
    # First, calculate the bounding boxes for all input groups to help with spatial matching
    input_bboxes = {}
    for input_id, input_group in input_groups.items():
        bbox = get_element_bbox(input_group)
        input_bboxes[input_id] = bbox
    
    # Create output directory for debug PNGs
    os.makedirs("debug_pngs", exist_ok=True)
    
    # Process input groups
    for input_id, input_group in input_groups.items():
        input_bitmap = svg_to_bitmap(input_group)
        if input_bitmap:
            # Save PNG for input group
            png_filename = f"debug_pngs/input_{input_id}.png"
            save_bitmap_as_png(input_bitmap, png_filename)
            print(f"Saved input group {input_id} as {png_filename}")
    
    # Process lookup groups
    for lookup_id, lookup_group in lookup_groups.items():
        if not lookup_id.startswith('find_'):
            continue  # Only process find_ groups
            
        lookup_bitmap = svg_to_bitmap(lookup_group)
        if lookup_bitmap:
            # Save PNG for lookup group
            png_filename = f"debug_pngs/lookup_{lookup_id}.png"
            save_bitmap_as_png(lookup_bitmap, png_filename)
            print(f"Saved lookup group {lookup_id} as {png_filename}")
    
    for lookup_id, lookup_group in lookup_groups.items():
        if not lookup_id.startswith('find_'):
            continue  # Only process find_ groups
            
        # Extract the number part from lookup_id (e.g., "001" from "find_001")
        num_part = lookup_id.split('_')[1]
        replace_id = f"replace_{num_part}"
        
        if replace_id not in replace_groups:
            print(f"Warning: No corresponding replace group for {lookup_id}")
            continue
        
        # Get the subgroups to match from the find group
        target_subgroups = find_subgroups(lookup_group)
        
        if not target_subgroups:
            # If no subgroups, treat the entire lookup group as the target
            print(f"Looking for full group matching {lookup_id}")
            
            # Create bitmap for the entire lookup group
            target_bitmap = svg_to_bitmap(lookup_group)
            print(f"  Target bitmap generated with dimensions: {len(target_bitmap) if target_bitmap else 0}x{len(target_bitmap[0]) if target_bitmap and target_bitmap[0] else 0}")
            
            # Look for input groups that match this bitmap
            matched_input_groups = []
            for input_id, input_group in input_groups.items():
                if input_id in used_input_ids:
                    continue
                
                input_bitmap = svg_to_bitmap(input_group)
                
                # Check if this input group matches the target
                if compare_bitmaps(input_bitmap, target_bitmap):
                    matched_input_groups.append((input_id, input_group))
                    print(f"  Input {input_id} matches target bitmap")
            
            if matched_input_groups:
                # Get transform from the first matched group as reference
                first_input_id, first_input_group = matched_input_groups[0]
                input_rotation, input_tx, input_ty = get_element_transform(first_input_group)
                
                for input_id, input_group in matched_input_groups:
                    matches.append({
                        'input_groups': [input_id],  # Single input group
                        'lookup_id': lookup_id,
                        'replace_group': replace_groups[replace_id],
                        'rotation': input_rotation,
                        'tx': input_tx,
                        'ty': input_ty
                    })
                    
                    # Mark this input group as used
                    used_input_ids.add(input_id)
                    
                    print(f"Found match: {input_id} matches {lookup_id}")
            else:
                print(f"  No input groups matched target bitmap")
                # We have subgroups to match - match based on visual signatures and spatial arrangement
                print(f"Looking for {len(target_subgroups)} subgroups matching {lookup_id}")
                
                # Create simple visual signatures for all target subgroups
                target_signatures = []
                target_bboxes = []
                for i, target_subgroup in enumerate(target_subgroups):
                    sig = get_simple_visual_signature(target_subgroup)
                    target_signatures.append(sig)
                    
                    # Get the bounding box of the target subgroup
                    bbox = get_element_bbox(target_subgroup)
                    # If bbox is invalid (all zeros or extreme values), try to get a more meaningful position
                    if bbox == (0, 0, 0, 0) or (bbox[2] - bbox[0] == 0 and bbox[3] - bbox[1] == 0):
                        # Calculate position based on transform or first element
                        rotation, tx, ty = get_element_transform(target_subgroup)
                        if tx != 0 or ty != 0:
                            bbox = (tx, ty, tx + 1, ty + 1)  # Create a small bbox at the transform position
                        else:
                            # Calculate position from first child element
                            for child in target_subgroup:
                                tag = child.tag.split('}')[-1] if '}' in child.tag else child.tag  # Remove namespace
                                if tag in ['path', 'line', 'rect', 'circle', 'ellipse', 'polygon', 'polyline']:
                                    points = child.get('points', '')
                                    if points:
                                        coords = extract_coords_from_points(points)
                                        if coords:
                                            xs = [x for x, y in coords]
                                            ys = [y for x, y in coords]
                                            min_x, max_x = min(xs), max(xs)
                                            min_y, max_y = min(ys), max(ys)
                                            bbox = (min_x, min_y, max_x, max_y)
                                            break
                                    elif tag == 'path':
                                        d = child.get('d', '')
                                        if d:
                                            coords = extract_coords_from_path(d)
                                            if coords:
                                                xs = [x for x, y in coords]
                                                ys = [y for x, y in coords]
                                                if xs and ys:
                                                    min_x, max_x = min(xs), max(xs)
                                                    min_y, max_y = min(ys), max(ys)
                                                    bbox = (min_x, min_y, max_x, max_y)
                                                    break
                                    elif tag == 'rect':
                                        x = float(child.get('x', 0))
                                        y = float(child.get('y', 0))
                                        width = float(child.get('width', 0))
                                        height = float(child.get('height', 0))
                                        bbox = (x, y, x + width, y + height)
                                        break
                                    elif tag == 'circle':
                                        cx = float(child.get('cx', 0))
                                        cy = float(child.get('cy', 0))
                                        r = float(child.get('r', 0))
                                        bbox = (cx - r, cy - r, cx + r, cy + r)
                                        break
                                    elif tag == 'ellipse':
                                        cx = float(child.get('cx', 0))
                                        cy = float(child.get('cy', 0))
                                        rx = float(child.get('rx', 0))
                                        ry = float(child.get('ry', 0))
                                        bbox = (cx - rx, cy - ry, cx + rx, cy + ry)
                                        break
                                    elif tag in ['line']:
                                        x1 = float(child.get('x1', 0))
                                        y1 = float(child.get('y1', 0))
                                        x2 = float(child.get('x2', 0))
                                        y2 = float(child.get('y2', 0))
                                        min_x, max_x = min(x1, x2), max(x1, x2)
                                        min_y, max_y = min(y1, y2), max(y1, y2)
                                        bbox = (min_x, min_y, max_x, max_y)
                                        break

                    target_bboxes.append(bbox)

                    print(f"  Target {i} signature: {sig}, bbox: {bbox}")

                # Calculate relative positions between target subgroups (for spatial matching)
                target_relative_positions = []
                if len(target_bboxes) > 1:
                    # Use the first target as the reference point
                    ref_x = (target_bboxes[0][0] + target_bboxes[0][2]) / 2  # center x of first target
                    ref_y = (target_bboxes[0][1] + target_bboxes[0][3]) / 2  # center y of first target

                    for bbox in target_bboxes:
                        center_x = (bbox[0] + bbox[2]) / 2
                        center_y = (bbox[1] + bbox[3]) / 2
                        target_relative_positions.append((center_x - ref_x, center_y - ref_y))
                else:
                    target_relative_positions.append((0, 0))

                print(f"  Target relative positions: {target_relative_positions}")

                # Find all input groups that match any of the target signatures
                input_to_targets = {}
                for input_id, input_group in input_groups.items():
                    if input_id in used_input_ids:
                        continue
                    input_sig = get_simple_visual_signature(input_group)

                    # Check if this input group matches any of the target subgroups
                    matching_targets = []
                    for i, target_sig in enumerate(target_signatures):
                        if input_sig == target_sig:  # Direct signature comparison for efficiency
                            matching_targets.append(i)

                    if matching_targets:
                        input_to_targets[input_id] = (input_group, matching_targets)
                        print(f"  Input {input_id} matches targets: {matching_targets}")

                print(f"  Found {len(input_to_targets)} potential input matches")

                # Instead of checking all combinations, find clusters of input groups that match
                # all target subgroups and have similar spatial relationships
                from itertools import combinations

                potential_clusters = []

                # Try to form clusters of input groups that could match the target subgroups
                available_input_ids = list(input_to_targets.keys())

                if len(available_input_ids) >= len(target_subgroups):
                    # Limit the number of combinations to avoid exponential growth
                    max_combinations = 100  # Limit to prevent long computation
                    count = 0
                    for combo in combinations(available_input_ids, len(target_subgroups)):
                        if count >= max_combinations:
                            print(f"  Reached maximum combination limit ({max_combinations}), stopping search")
                            break
                        count += 1

                        # Check if this combination covers all target subgroups
                        combo_targets = set()
                        for input_id in combo:
                            _, possible_targets = input_to_targets[input_id]
                            combo_targets.update(possible_targets)

                        # Check if this combination covers all target subgroups
                        if len(combo_targets) == len(target_subgroups):
                            # Calculate relative positions for this cluster
                            cluster_bboxes = []
                            for input_id in combo:
                                bbox = input_bboxes[input_id]
                                cluster_bboxes.append(bbox)

                            # Calculate relative positions of this cluster
                            cluster_relative_positions = []
                            if len(cluster_bboxes) > 1:
                                # Use the first input as the reference point
                                ref_x = (cluster_bboxes[0][0] + cluster_bboxes[0][2]) / 2
                                ref_y = (cluster_bboxes[0][1] + cluster_bboxes[0][3]) / 2

                                for bbox in cluster_bboxes:
                                    center_x = (bbox[0] + bbox[2]) / 2
                                    center_y = (bbox[1] + bbox[3]) / 2
                                    cluster_relative_positions.append((center_x - ref_x, center_y - ref_y))
                            else:
                                cluster_relative_positions.append((0, 0))

                            # Check if relative positions are similar (within tolerance)
                            position_match = True
                            if len(target_relative_positions) > 1:
                                # Calculate distances between points for both target and cluster
                                target_distances = []
                                cluster_distances = []

                                for i in range(len(target_relative_positions)):
                                    for j in range(i + 1, len(target_relative_positions)):
                                        t_dist = ((target_relative_positions[i][0] - target_relative_positions[j][0])**2 +
                                                 (target_relative_positions[i][1] - target_relative_positions[j][1])**2)**0.5
                                        c_dist = ((cluster_relative_positions[i][0] - cluster_relative_positions[j][0])**2 +
                                                 (cluster_relative_positions[i][1] - cluster_relative_positions[j][1])**2)**0.5

                                        target_distances.append(t_dist)
                                        cluster_distances.append(c_dist)

                                # Compare distances (allowing for some scaling)
                                if target_distances and cluster_distances:
                                    # Calculate scaling factor
                                    avg_target_dist = sum(target_distances) / len(target_distances) if target_distances else 1
                                    avg_cluster_dist = sum(cluster_distances) / len(cluster_distances) if cluster_distances else 1

                                    if avg_target_dist > 0:
                                        scale_factor = avg_cluster_dist / avg_target_dist
                                    else:
                                        scale_factor = 1

                                    # Check if distances match after scaling
                                    for t_dist, c_dist in zip(target_distances, cluster_distances):
                                        expected_c_dist = t_dist * scale_factor
                                        if expected_c_dist > 0 and abs(c_dist - expected_c_dist) / expected_c_dist > 0.5:  # 50% tolerance
                                            position_match = False
                                            break
                                        elif expected_c_dist == 0 and abs(c_dist) > 10:  # If target distance is 0, allow small cluster distance
                                            position_match = False
                                            break
                            else:
                                # Single element case - always matches position
                                pass

                            if position_match:
                                potential_clusters.append(combo)

                # Check each potential cluster for a complete match
                for cluster in potential_clusters:
                    # Create a bipartite matching for this cluster
                    cluster_input_to_targets = {id: input_to_targets[id] for id in cluster}

                    # Try to assign each input in the cluster to a unique target
                    cluster_list = list(cluster)
                    target_indices = list(range(len(target_subgroups)))

                    # Try all permutations to find a valid assignment
                    from itertools import permutations
                    for assignment in permutations(cluster_list, len(target_subgroups)):
                        valid_assignment = True
                        matched_cluster = []

                        for i, input_id in enumerate(assignment):
                            target_idx = i
                            input_group, possible_targets = input_to_targets[input_id]

                            # Check if this input can match the required target
                            if target_idx not in possible_targets:
                                valid_assignment = False
                                break

                            matched_cluster.append((input_id, input_group))

                        if valid_assignment and len(matched_cluster) == len(target_subgroups):
                            # Get transform from the first matched group as reference
                            first_input_id, first_input_group = matched_cluster[0]
                            input_rotation, input_tx, input_ty = get_element_transform(first_input_group)

                            matches.append({
                                'input_groups': [item[0] for item in matched_cluster],  # List of input group IDs
                                'lookup_id': lookup_id,
                                'replace_group': replace_groups[replace_id],
                                'rotation': input_rotation,
                                'tx': input_tx,
                                'ty': input_ty
                            })

                            # Mark these input groups as used
                            for input_id, _ in matched_cluster:
                                used_input_ids.add(input_id)

                            print(f"Found match: {[item[0] for item in matched_cluster]} matches {lookup_id}")
                            break  # Found a valid assignment for this lookup, continue to next lookup

    return matches


def main(input_svg_path, lookup_svg_path, output_svg_path):
    """Main function to process SVG files."""
    # Parse input SVG
    input_tree = ET.parse(input_svg_path)
    input_root = input_tree.getroot()

    # Parse lookup SVG
    lookup_tree = ET.parse(lookup_svg_path)
    lookup_root = lookup_tree.getroot()

    # Extract all groups from both SVGs
    input_groups = extract_groups(input_tree)
    lookup_groups = extract_groups(lookup_tree)

    # Separate find_ and replace_ groups from lookup
    find_groups = {}
    replace_groups = {}

    for group_id, group in lookup_groups.items():
        if group_id.startswith('find_'):
            find_groups[group_id] = group
        elif group_id.startswith('replace_'):
            replace_groups[group_id] = group

    print(f"Found {len(input_groups)} input groups, {len(find_groups)} find groups, {len(replace_groups)} replace groups")

    # Find matches
    matches = find_matching_groups(input_groups, find_groups, replace_groups)

    print(f"Found {len(matches)} matches")

    # For each match, remove the matched input groups and add the replacement group
    # while preserving the position of the entire cluster of matched input groups
    for match in matches:
        input_group_ids = match['input_groups']
        replace_group = match['replace_group']

        # Get the position of the first matched input group to preserve the original location
        first_input_id = input_group_ids[0]
        first_input_group = input_groups[first_input_id]

        # Get the transform of the first input group to preserve the original location
        input_rotation, input_tx, input_ty = get_element_transform(first_input_group)

        # If the first group has no explicit transform, we need to look for its position in its content
        # Looking at the elements, we need to find the actual coordinates from the geometry
        if input_tx == 0 and input_ty == 0:
            # Calculate the bounding box of the first input group to get its position
            bbox = get_element_bbox(first_input_group)
            if bbox[0] != 0 or bbox[1] != 0:  # If bbox has meaningful values
                input_tx = bbox[0]  # Use the left edge as x position
                input_ty = bbox[1]  # Use the top edge as y position
            else:
                # If bbox is also zero, we'll need to get the position differently
                # For now, use a default approach by looking at the first element's coordinates
                for child in first_input_group:
                    tag = child.tag.split('}')[-1] if '}' in child.tag else child.tag  # Remove namespace
                    if tag == 'polygon' or tag == 'polyline':
                        points = child.get('points', '')
                        if points:
                            coord_pairs = points.strip().split()
                            if coord_pairs:
                                first_pair = coord_pairs[0].split(',')
                                if len(first_pair) >= 2:
                                    input_tx = float(first_pair[0])
                                    input_ty = float(first_pair[1])
                                    break
                    elif tag == 'path':
                        d = child.get('d', '')
                        if d and d.startswith('M'):
                            # Extract first coordinates after M command
                            import re
                            coords = re.findall(r'-?\d+\.?\d+', d)
                            if len(coords) >= 2:
                                input_tx = float(coords[0])
                                input_ty = float(coords[1])
                                break
                    elif tag == 'rect':
                        input_tx = float(child.get('x', 0))
                        input_ty = float(child.get('y', 0))
                        break
                    elif tag == 'circle' or tag == 'ellipse':
                        input_tx = float(child.get('cx', 0))
                        input_ty = float(child.get('cy', 0))
                        break

        # Create a copy of the replacement group
        new_group = ET.fromstring(ET.tostring(replace_group, encoding='unicode'))

        # Get the original transform from the replacement group
        original_transform = replace_group.get('transform', '')

        # Apply the position translation while preserving the original transform
        if original_transform:
            # If there's an original transform, combine it with the translation
            new_transform = f'translate({input_tx},{input_ty}) {original_transform}'
        else:
            # If no original transform, just use the translation
            new_transform = f'translate({input_tx},{input_ty})'

        new_group.set('transform', new_transform)

        # Add the new group to the root
        input_root.append(new_group)

        print(f"Replacing groups {input_group_ids} with {replace_group.get('id', 'unknown')}")

    # Remove matched input groups from the SVG
    for input_id in input_groups:
        if any(input_id in match['input_groups'] for match in matches):
            # Find and remove the original group from its parent
            original_group = input_groups[input_id]
            for parent in input_root.iter():
                for child in list(parent):
                    if child == original_group:
                        parent.remove(child)
                        break


    # Write the output SVG with proper formatting
    # Convert to string and fix common formatting issues
    rough_string = ET.tostring(input_root, encoding='unicode')

    # Fix namespace issue and ensure proper formatting
    if ' xmlns=' not in rough_string:
        rough_string = rough_string.replace('<svg', '<svg xmlns="http://www.w3.org/2000/svg"', 1)

    # Pretty print using minidom
    try:
        reparsed = minidom.parseString(rough_string.encode('utf-8'))
        pretty_xml = reparsed.toprettyxml(indent="  ")

        # Remove extra blank lines
        lines = [line for line in pretty_xml.split('\n') if line.strip()]
        pretty_xml = '\n'.join(lines)
    except:
        # If there's an error, just use the rough string
        lines = [line for line in rough_string.split('\n') if line.strip()]
        pretty_xml = '\n'.join(lines)

    with open(output_svg_path, 'w') as f:
        f.write(pretty_xml)

    print(f"Output written to {output_svg_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Replace SVG groups based on visual matching')
    parser.add_argument('input_svg', help='Input SVG file path')
    parser.add_argument('lookup_svg', help='Lookup SVG file path')
    parser.add_argument('output_svg', help='Output SVG file path')

    args = parser.parse_args()

    main(args.input_svg, args.lookup_svg, args.output_svg)