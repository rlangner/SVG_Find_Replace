#!/usr/bin/env python3
"""
SVG Element Matcher

This module contains functions for matching SVG elements based on visual content.
"""

import xml.etree.ElementTree as ET
from xml.etree.ElementTree import Element
import re
import copy
from typing import Dict, List, Tuple, Optional

# Import the improved bounding box calculation functions
from svg_bounding_box import calculate_group_bbox


def normalize_color(color):
    """
    Normalize color values to handle different formats (rgb, hex, named colors).
    """
    if not color:
        return color
    
    # Handle rgb(r,g,b) format
    rgb_match = re.match(r'rgb\s*\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)', color)
    if rgb_match:
        r, g, b = map(int, rgb_match.groups())
        return f"rgb({r},{g},{b})"
    
    # Handle hex format
    if color.startswith('#'):
        # Convert to rgb for comparison
        hex_color = color[1:]
        if len(hex_color) == 3:
            hex_color = ''.join([c*2 for c in hex_color])  # Expand shorthand
        if len(hex_color) == 6:
            try:
                r = int(hex_color[0:2], 16)
                g = int(hex_color[2:4], 16)
                b = int(hex_color[4:6], 16)
                return f"rgb({r},{g},{b})"
            except ValueError:
                pass
    
    return color.lower()  # Normalize to lowercase

def normalize_transform(transform):
    """
    Normalize transform values to handle different formats and precision.
    """
    if not transform:
        return transform
    
    # Parse transform string and normalize numbers
    parts = re.split(r'(\w+\s*\([^)]*\))', transform)
    normalized_parts = []
    
    for part in parts:
        if part.strip():
            # Match transform functions and normalize their numeric values
            func_match = re.match(r'(\w+)\s*\(([^)]*)\)', part.strip())
            if func_match:
                func_name = func_match.group(1)
                params_str = func_match.group(2)
                # Split parameters by comma or space and normalize numbers
                params = re.split(r'[, ]+', params_str.strip())
                normalized_params = []
                for param in params:
                    param = param.strip()
                    if param:
                        try:
                            # Round to 4 decimal places for consistency
                            normalized_params.append(f"{float(param):.4f}")
                        except ValueError:
                            normalized_params.append(param)
                normalized_parts.append(f"{func_name}({','.join(normalized_params)})")
            else:
                normalized_parts.append(part)
    
    return ' '.join(normalized_parts)

def normalize_path_data(path_d):
    """
    Normalize path data by converting all commands to absolute coordinates
    and standardizing the format for comparison.
    """
    if not path_d:
        return ""
    
    # Clean up the path string
    path_d = re.sub(r'\s+', ' ', path_d.strip())
    
    # Split into commands and coordinates
    segments = []
    current_pos = 0
    i = 0
    while i < len(path_d):
        char = path_d[i]
        if char.isalpha():
            # Found a command
            command = char
            i += 1
            # Skip any whitespace after the command
            while i < len(path_d) and path_d[i].isspace():
                i += 1
            # Collect coordinates for this command
            coords = []
            while i < len(path_d):
                if path_d[i].isspace():
                    i += 1
                    continue
                elif path_d[i].isalpha() and path_d[i] not in 'eE':
                    # New command
                    break
                else:
                    # Collect the number
                    num_str = ''
                    while i < len(path_d) and (path_d[i].isdigit() or path_d[i] in '.-+eE'):
                        num_str += path_d[i]
                        i += 1
                    if num_str:
                        try:
                            num = float(num_str)
                            # Round to 4 decimal places for consistency
                            coords.append(f"{num:.4f}")
                        except ValueError:
                            coords.append(num_str)
                    else:
                        i += 1
                    # Skip whitespace after number
                    while i < len(path_d) and path_d[i].isspace():
                        i += 1
            
            segments.append(command)
            segments.extend(coords)
        else:
            i += 1
    
    # Convert relative commands to absolute (simplified approach)
    result = []
    current_x, current_y = 0, 0
    start_x, start_y = 0, 0
    i = 0
    
    while i < len(segments):
        cmd = segments[i]
        if cmd in 'Mm':
            i += 1
            try:
                x = float(segments[i])
            except (ValueError, IndexError):
                i += 1
                continue
            i += 1
            try:
                y = float(segments[i])
            except (ValueError, IndexError):
                i += 1
                continue
            if cmd.islower():
                x += current_x
                y += current_y
            result.extend(['M', str(x), str(y)])
            current_x, current_y = x, y
            start_x, start_y = x, y
            i += 1
        elif cmd in 'LlTt':
            i += 1
            try:
                x = float(segments[i])
            except (ValueError, IndexError):
                i += 1
                continue
            i += 1
            try:
                y = float(segments[i])
            except (ValueError, IndexError):
                i += 1
                continue
            if cmd.islower():
                x += current_x
                y += current_y
            result.extend(['L', str(x), str(y)])
            current_x, current_y = x, y
            i += 1
        elif cmd in 'Hh':
            i += 1
            try:
                x = float(segments[i])
            except (ValueError, IndexError):
                i += 1
                continue
            if cmd.islower():
                x += current_x
            y = current_y
            result.extend(['L', str(x), str(y)])
            current_x = x
            i += 1
        elif cmd in 'Vv':
            i += 1
            try:
                y = float(segments[i])
            except (ValueError, IndexError):
                i += 1
                continue
            if cmd.islower():
                y += current_y
            x = current_x
            result.extend(['L', str(x), str(y)])
            current_y = y
            i += 1
        elif cmd in 'Zz':
            result.append('Z')
            current_x, current_y = start_x, start_y
            i += 1
        elif cmd in 'Cc':
            # Cubic Bezier - just add as is for now
            result.append(cmd.upper())
            i += 1
            for j in range(6):  # 6 coordinates for cubic Bezier
                if i < len(segments):
                    result.append(segments[i])
                    i += 1
        elif cmd in 'Ss':
            # Smooth cubic Bezier - just add as is for now
            result.append(cmd.upper())
            i += 1
            for j in range(4):  # 4 coordinates for smooth cubic Bezier
                if i < len(segments):
                    result.append(segments[i])
                    i += 1
        elif cmd in 'Qq':
            # Quadratic Bezier - just add as is for now
            result.append(cmd.upper())
            i += 1
            for j in range(4):  # 4 coordinates for quadratic Bezier
                if i < len(segments):
                    result.append(segments[i])
                    i += 1
        elif cmd in 'Aa':
            # Arc - just add as is for now
            result.append(cmd.upper())
            i += 1
            for j in range(7):  # 7 parameters for arc
                if i < len(segments):
                    result.append(segments[i])
                    i += 1
        else:
            # Unknown command, just add it
            result.append(cmd)
            i += 1
    
    return ' '.join(result)


def extract_path_elements(element: Element) -> List[Element]:
    """
    Extract all path elements from an element and its children.
    """
    paths = []
    for child in element.iter():
        if child.tag.endswith('path') or child.tag.endswith('polygon') or child.tag.endswith('polyline'):
            paths.append(child)
    return paths


def normalize_path_content(path_element: Element) -> str:
    """
    Normalize a path element for visual comparison by removing non-essential attributes
    and normalizing coordinates.
    """
    # Create a deep copy to avoid modifying the original
    elem_copy = copy.deepcopy(path_element)
    
    # Remove attributes that shouldn't affect visual matching
    for attr in ['id', 'class', 'style', 'transform']:
        if attr in elem_copy.attrib:
            del elem_copy.attrib[attr]
    
    # Convert to string with consistent namespace handling
    import xml.etree.ElementTree as ET
    
    # Use a custom method to serialize without namespace prefixes
    def strip_namespaces(elem):
        """Recursively strip namespace from element tags"""
        # Remove namespace from tag
        if elem.tag.startswith('{'):
            elem.tag = elem.tag.split('}')[1]
        
        # Process children recursively
        for child in elem:
            strip_namespaces(child)
    
    # Make a copy to modify for serialization
    serial_copy = copy.deepcopy(elem_copy)
    strip_namespaces(serial_copy)
    
    # Serialize to string without namespace prefixes
    content = ET.tostring(serial_copy, encoding='unicode')
    
    # Normalize clip-path references to remove the suffix numbers (like -2, -23, -3)
    content = re.sub(r'clipId\\\d+\\\.\d*-\d+', 'clipId', content)
    content = re.sub(r'clipId\\\d+\\\.\d*', 'clipId', content)
    
    # Normalize color formats: convert both rgb(r,g,b) and hex to rgb for consistency
    def normalize_color_to_rgb(match):
        full_match = match.group(0)
        if full_match.startswith('rgb'):
            # Parse rgb(r,g,b) format
            import re
            rgb_match = re.search(r'rgb\((\d+),(\d+),(\d+)\)', full_match)
            if rgb_match:
                r, g, b = map(int, rgb_match.groups())
                return f"rgb({r},{g},{b})"
        elif full_match.startswith('#'):
            # Convert hex to rgb format for consistency
            hex_color = full_match[1:]
            if len(hex_color) == 3:
                hex_color = ''.join([c*2 for c in hex_color])  # Expand shorthand
            if len(hex_color) == 6:
                try:
                    r = int(hex_color[0:2], 16)
                    g = int(hex_color[2:4], 16)
                    b = int(hex_color[4:6], 16)
                    return f"rgb({r},{g},{b})"
                except ValueError:
                    pass
        return full_match
    
    # Find and replace rgb/hex colors with a consistent format (rgb)
    content = re.sub(r'rgb\(\d+,\d+,\d+\)', normalize_color_to_rgb, content)
    content = re.sub(r'#[0-9a-fA-F]{3,6}', normalize_color_to_rgb, content)
    
    # Extract path coordinates and normalize them to canonical representation
    def extract_and_sort_path_coords(path_d):
        # This function converts path data to a canonical form by extracting coordinates
        import re
        
        # Convert relative path commands to absolute for comparison
        path_d = convert_relative_to_absolute(path_d)
        
        # Extract all coordinate pairs from the path data
        coords = re.findall(r'[-+]?\d*\.?\d+', path_d)
        
        # Convert to float and group into coordinate pairs
        coord_pairs = []
        for i in range(0, len(coords), 2):
            if i + 1 < len(coords):
                try:
                    x, y = float(coords[i]), float(coords[i + 1])
                    coord_pairs.append((x, y))
                except ValueError:
                    continue
        
        # Sort coordinate pairs to create canonical representation
        coord_pairs.sort()
        
        # Create a canonical string representation
        canonical_coords = ' '.join([f"{x:.3f},{y:.3f}" for x, y in coord_pairs])
        return canonical_coords
    
    # Convert relative path commands to absolute coordinates
    def convert_relative_to_absolute(path_d):
        # Parse path data and convert relative commands to absolute
        import re
        
        # Split path into commands and coordinates
        result = []
        i = 0
        current_x, current_y = 0, 0
        start_x, start_y = 0, 0
        
        # Clean up the path string
        path_d = re.sub(r'\s+', ' ', path_d.strip())
        
        while i < len(path_d):
            char = path_d[i]
            if char.isspace():
                i += 1
                continue
            elif char.isalpha():
                # Found a command
                command = char
                i += 1
                # Collect coordinates for this command
                coords = []
                while i < len(path_d):
                    if path_d[i].isspace():
                        i += 1
                        continue
                    elif path_d[i].isalpha() and path_d[i] not in 'eE':
                        # New command
                        break
                    else:
                        # Collect the number
                        num_str = ''
                        while i < len(path_d) and (path_d[i].isdigit() or path_d[i] in '.-+eE'):
                            num_str += path_d[i]
                            i += 1
                        if num_str:
                            try:
                                coords.append(float(num_str))
                            except ValueError:
                                pass
                        else:
                            i += 1
                        # Skip whitespace after number
                        while i < len(path_d) and path_d[i].isspace():
                            i += 1
                
                # Process the command and coordinates
                if command.upper() == 'M':
                    if command.isupper():  # Absolute
                        current_x, current_y = coords[0], coords[1]
                        start_x, start_y = current_x, current_y
                        result.append(f"M {current_x:.3f} {current_y:.3f}")
                    else:  # Relative
                        current_x += coords[0]
                        current_y += coords[1]
                        start_x, start_y = current_x, current_y
                        result.append(f"M {current_x:.3f} {current_y:.3f}")
                    if len(coords) > 2:  # Multiple coordinates in one M command
                        for j in range(2, len(coords), 2):
                            if j + 1 < len(coords):
                                if command.islower():
                                    current_x += coords[j]
                                    current_y += coords[j+1]
                                else:
                                    current_x, current_y = coords[j], coords[j+1]
                                result.append(f"L {current_x:.3f} {current_y:.3f}")
                elif command.upper() == 'L':
                    for j in range(0, len(coords), 2):
                        if j + 1 < len(coords):
                            if command.islower():
                                current_x += coords[j]
                                current_y += coords[j+1]
                            else:
                                current_x, current_y = coords[j], coords[j+1]
                            result.append(f"L {current_x:.3f} {current_y:.3f}")
                elif command.upper() == 'H':
                    for j in range(len(coords)):
                        if command.islower():
                            current_x += coords[j]
                        else:
                            current_x = coords[j]
                        result.append(f"L {current_x:.3f} {current_y:.3f}")
                elif command.upper() == 'V':
                    for j in range(len(coords)):
                        if command.islower():
                            current_y += coords[j]
                        else:
                            current_y = coords[j]
                        result.append(f"L {current_x:.3f} {current_y:.3f}")
                elif command.upper() == 'Z':
                    current_x, current_y = start_x, start_y
                    result.append('Z')
                elif command.upper() == 'C':
                    for j in range(0, len(coords), 6):
                        if j + 5 < len(coords):
                            if command.islower():
                                x1, y1 = current_x + coords[j], current_y + coords[j+1]
                                x2, y2 = current_x + coords[j+2], current_y + coords[j+3]
                                x, y = current_x + coords[j+4], current_y + coords[j+5]
                            else:
                                x1, y1 = coords[j], coords[j+1]
                                x2, y2 = coords[j+2], coords[j+3]
                                x, y = coords[j+4], coords[j+5]
                            current_x, current_y = x, y
                            result.append(f"C {x1:.3f} {y1:.3f} {x2:.3f} {y2:.3f} {current_x:.3f} {current_y:.3f}")
                elif command.upper() == 'S':
                    for j in range(0, len(coords), 4):
                        if j + 3 < len(coords):
                            if command.islower():
                                x2, y2 = current_x + coords[j], current_y + coords[j+1]
                                x, y = current_x + coords[j+2], current_y + coords[j+3]
                            else:
                                x2, y2 = coords[j], coords[j+1]
                                x, y = coords[j+2], coords[j+3]
                            current_x, current_y = x, y
                            result.append(f"S {x2:.3f} {y2:.3f} {current_x:.3f} {current_y:.3f}")
                elif command.upper() == 'Q':
                    for j in range(0, len(coords), 4):
                        if j + 3 < len(coords):
                            if command.islower():
                                x1, y1 = current_x + coords[j], current_y + coords[j+1]
                                x, y = current_x + coords[j+2], current_y + coords[j+3]
                            else:
                                x1, y1 = coords[j], coords[j+1]
                                x, y = coords[j+2], coords[j+3]
                            current_x, current_y = x, y
                            result.append(f"Q {x1:.3f} {y1:.3f} {current_x:.3f} {current_y:.3f}")
                elif command.upper() == 'T':
                    for j in range(0, len(coords), 2):
                        if j + 1 < len(coords):
                            if command.islower():
                                x, y = current_x + coords[j], current_y + coords[j+1]
                            else:
                                x, y = coords[j], coords[j+1]
                            current_x, current_y = x, y
                            result.append(f"T {current_x:.3f} {current_y:.3f}")
                elif command.upper() == 'A':
                    for j in range(0, len(coords), 7):
                        if j + 6 < len(coords):
                            if command.islower():
                                rx, ry = coords[j], coords[j+1]
                                x_axis_rotation = coords[j+2]
                                large_arc_flag = coords[j+3]
                                sweep_flag = coords[j+4]
                                x = current_x + coords[j+5]
                                y = current_y + coords[j+6]
                            else:
                                rx, ry = coords[j], coords[j+1]
                                x_axis_rotation = coords[j+2]
                                large_arc_flag = coords[j+3]
                                sweep_flag = coords[j+4]
                                x, y = coords[j+5], coords[j+6]
                            current_x, current_y = x, y
                            result.append(f"A {rx:.3f} {ry:.3f} {x_axis_rotation:.3f} {large_arc_flag:.3f} {sweep_flag:.3f} {current_x:.3f} {current_y:.3f}")
                else:  # Other commands like 'm', 'l', etc. treated as line to
                    if command.lower() == 'm':
                        for j in range(0, len(coords), 2):
                            if j + 1 < len(coords):
                                if command.islower():
                                    current_x += coords[j]
                                    current_y += coords[j+1]
                                else:
                                    current_x, current_y = coords[j], coords[j+1]
                                if j == 0:
                                    result.append(f"M {current_x:.3f} {current_y:.3f}")
                                else:
                                    result.append(f"L {current_x:.3f} {current_y:.3f}")
                                if j == 0:
                                    start_x, start_y = current_x, current_y
                    elif command.lower() == 'l' or command.lower() == 't':
                        for j in range(0, len(coords), 2):
                            if j + 1 < len(coords):
                                if command.islower():
                                    current_x += coords[j]
                                    current_y += coords[j+1]
                                else:
                                    current_x, current_y = coords[j], coords[j+1]
                                result.append(f"L {current_x:.3f} {current_y:.3f}")
                    elif command.lower() == 'h':
                        for j in range(len(coords)):
                            if command.islower():
                                current_x += coords[j]
                            else:
                                current_x = coords[j]
                            result.append(f"L {current_x:.3f} {current_y:.3f}")
                    elif command.lower() == 'v':
                        for j in range(len(coords)):
                            if command.islower():
                                current_y += coords[j]
                            else:
                                current_y = coords[j]
                            result.append(f"L {current_x:.3f} {current_y:.3f}")
                    elif command.lower() == 'z':
                        current_x, current_y = start_x, start_y
                        result.append('Z')
        
        return ' '.join(result)
    
    # Apply coordinate normalization to path data
    def normalize_path_coords(match):
        path_d = match.group(1)
        canonical_coords = extract_and_sort_path_coords(path_d)
        return f'd="{canonical_coords}"'
    
    content = re.sub(r'd="([^"]*)"', normalize_path_coords, content)
    
    # Normalize whitespace
    content = re.sub(r'\s+', ' ', content)
    
    # Normalize numbers (round to 3 decimal places to handle floating point differences)
    def round_numbers(match):
        try:
            num = float(match.group())
            return f"{num:.3f}"
        except ValueError:
            return match.group()  # Return original if conversion fails

    content = re.sub(r'\d+\.?\d*', round_numbers, content)
    
    return content.strip()


def normalize_svg_content(element: Element) -> str:
    """
    Normalize SVG element content for comparison by removing IDs, 
    whitespace, and other non-essential differences.
    """
    # Create a deep copy to avoid modifying the original
    elem_copy = copy.deepcopy(element)
    
    # Remove attributes that shouldn't affect visual matching
    for attr in ['id', 'class', 'style', 'transform']:
        if attr in elem_copy.attrib:
            del elem_copy.attrib[attr]
    
    # Remove IDs from all child elements too
    for child in elem_copy.iter():
        for attr in ['id', 'class', 'style', 'transform']:
            if attr in child.attrib:
                del child.attrib[attr]
    
    # Convert to string with consistent namespace handling
    # Register a default namespace to avoid inconsistent prefixes
    import xml.etree.ElementTree as ET
    
    # Use a custom method to serialize without namespace prefixes
    def strip_namespaces(elem):
        """Recursively strip namespace from element tags"""
        # Remove namespace from tag
        if elem.tag.startswith('{'):
            elem.tag = elem.tag.split('}')[1]
        
        # Process children recursively
        for child in elem:
            strip_namespaces(child)
    
    # Make a copy to modify for serialization
    serial_copy = copy.deepcopy(elem_copy)
    strip_namespaces(serial_copy)
    
    # Serialize to string without namespace prefixes
    content = ET.tostring(serial_copy, encoding='unicode')
    
    # Normalize clip-path references to remove the suffix numbers (like -2, -23, -3)
    content = re.sub(r'clipId\d+\.\d*-\d+', 'clipId', content)
    content = re.sub(r'clipId\d+\.\d*', 'clipId', content)
    
    # Normalize color formats: convert both rgb(r,g,b) and hex to rgb for consistency
    def normalize_color_to_rgb(match):
        full_match = match.group(0)
        if full_match.startswith('rgb'):
            # Parse rgb(r,g,b) format
            import re
            rgb_match = re.search(r'rgb\((\d+),(\d+),(\d+)\)', full_match)
            if rgb_match:
                r, g, b = map(int, rgb_match.groups())
                return f"rgb({r},{g},{b})"
        elif full_match.startswith('#'):
            # Convert hex to rgb format for consistency
            hex_color = full_match[1:]
            if len(hex_color) == 3:
                hex_color = ''.join([c*2 for c in hex_color])  # Expand shorthand
            if len(hex_color) == 6:
                try:
                    r = int(hex_color[0:2], 16)
                    g = int(hex_color[2:4], 16)
                    b = int(hex_color[4:6], 16)
                    return f"rgb({r},{g},{b})"
                except ValueError:
                    pass
        return full_match
    
    # Find and replace rgb/hex colors with a consistent format (rgb)
    content = re.sub(r'rgb\(\d+,\d+,\d+\)', normalize_color_to_rgb, content)
    content = re.sub(r'#[0-9a-fA-F]{3,6}', normalize_color_to_rgb, content)
    
    # Extract path coordinates and normalize them to canonical representation
    def extract_and_sort_path_coords(path_d):
        # This function converts path data to a canonical form by extracting coordinates
        import re
        
        # Extract all coordinate pairs from the path data
        coords = re.findall(r'[-+]?\d*\.?\d+', path_d)
        
        # Convert to float and group into coordinate pairs
        coord_pairs = []
        for i in range(0, len(coords), 2):
            if i + 1 < len(coords):
                try:
                    x, y = float(coords[i]), float(coords[i + 1])
                    coord_pairs.append((x, y))
                except ValueError:
                    continue
        
        # Sort coordinate pairs to create canonical representation
        coord_pairs.sort()
        
        # Create a canonical string representation
        canonical_coords = ' '.join([f"{x:.3f},{y:.3f}" for x, y in coord_pairs])
        return canonical_coords
    
    # Apply coordinate normalization to path data
    def normalize_path_coords(match):
        path_d = match.group(1)
        canonical_coords = extract_and_sort_path_coords(path_d)
        return f'd="{canonical_coords}"'
    
    content = re.sub(r'd="([^"]*)"', normalize_path_coords, content)
    
    # Normalize whitespace
    content = re.sub(r'\s+', ' ', content)
    
    # Normalize numbers (round to 3 decimal places to handle floating point differences)
    def round_numbers(match):
        try:
            num = float(match.group())
            return f"{num:.3f}"
        except ValueError:
            return match.group()  # Return original if conversion fails
    
    content = re.sub(r'\d+\.?\d*', round_numbers, content)
    
    return content.strip()


def extract_groups_from_svg(svg_path: str) -> Dict[str, List[Element]]:
    """
    Extract all <g> groups from an SVG file.
    Returns a dictionary with 'find' and 'replace' groups separated by ID pattern.
    """
    tree = ET.parse(svg_path)
    root = tree.getroot()
    
    # Find all groups with IDs starting with 'find_' or 'replace_'
    find_groups = {}
    replace_groups = {}
    
    for g in root.iter('{http://www.w3.org/2000/svg}g'):
        group_id = g.get('id')
        if group_id:
            if group_id.startswith('find_'):
                find_groups[group_id] = g
            elif group_id.startswith('replace_'):
                replace_groups[group_id] = g
    
    return {'find': find_groups, 'replace': replace_groups}


def get_child_groups(element: Element) -> List[Element]:
    """
    Get all direct child <g> elements of an element.
    """
    child_groups = []
    for child in element:
        if child.tag.endswith('g'):
            child_groups.append(child)
    return child_groups


def normalize_coordinates_in_content(content: str) -> str:
    """
    Normalize coordinates in SVG content to handle relative vs absolute paths.
    This function converts path data to a standardized format that ignores absolute positions.
    """
    import re
    
    # Function to normalize path data by converting to relative coordinates from a base point
    def normalize_path_data(path_d):
        # This is a simplified approach - we'll just normalize the numbers but keep relative vs absolute distinction
        # Extract all numbers from the path
        numbers = re.findall(r'[-+]?\d*\.?\d+', path_d)
        if not numbers:
            return path_d
        
        # Convert to floats, handling potential conversion errors
        coords = []
        for n in numbers:
            if n:
                try:
                    coords.append(float(n))
                except ValueError:
                    # Skip values that can't be converted to float
                    continue
        
        # If the path has at least 2 coordinates, we can normalize
        if len(coords) >= 2:
            # For now, just normalize by rounding to 3 decimals which is already done
            # More sophisticated normalization would involve converting absolute to relative or vice versa
            pass
        
        return path_d
    
    # Find path elements and normalize their 'd' attributes
    def normalize_path(match):
        attr_part = match.group(0)
        # Find the d attribute
        d_match = re.search(r'd="([^"]*)"', attr_part)
        if d_match:
            original_d = d_match.group(1)
            normalized_d = normalize_path_data(original_d)
            return attr_part.replace(f'd="{original_d}"', f'd="{normalized_d}"')
        return attr_part
    
    # Apply to path elements
    content = re.sub(r'<[^>]*d="[^"]*"[^>]*>', normalize_path, content)
    
    # Also normalize points attributes in polygons/polylines
    def normalize_points_attr(match):
        attr_part = match.group(0)
        points_match = re.search(r'points="([^"]*)"', attr_part)
        if points_match:
            points_str = points_match.group(1)
            # Parse points and normalize by making them relative to the minimum coordinate
            points_list = points_str.split()
            coords = []
            for point in points_list:
                if ',' in point:
                    x, y = point.split(',')
                    try:
                        coords.append((float(x.strip()), float(y.strip())))
                    except ValueError:
                        # Skip invalid coordinates
                        continue
            
            if coords:
                # Find min x and y to normalize
                min_x = min(c[0] for c in coords)
                min_y = min(c[1] for c in coords)
                
                # Create normalized points string
                normalized_points = ' '.join(f"{x-min_x:.3f},{y-min_y:.3f}" for x, y in coords)
                return attr_part.replace(f'points="{points_str}"', f'points="{normalized_points}"')
        
        return attr_part
    
    content = re.sub(r'<[^>]*points="[^"]*"[^>]*>', normalize_points_attr, content)
    
    # Normalize whitespace
    content = re.sub(r'\s+', ' ', content)
    
    # Normalize numbers (round to 3 decimal places to handle floating point differences)
    def round_numbers(match):
        try:
            num = float(match.group())
            return f"{num:.3f}"
        except ValueError:
            return match.group()  # Return original if conversion fails

    content = re.sub(r'\d+\.?\d*', round_numbers, content)
    
    return content.strip()


def normalize_element_content(element: Element) -> str:
    """
    Normalize an SVG element content by removing IDs and normalizing coordinates.
    """
    # Create a deep copy to avoid modifying the original
    elem_copy = copy.deepcopy(element)
    
    # Remove attributes that shouldn't affect visual matching
    for attr in ['id', 'class', 'style']:
        if attr in elem_copy.attrib:
            del elem_copy.attrib[attr]
    
    # Convert to string
    content = ET.tostring(elem_copy, encoding='unicode')
    
    # Normalize coordinates in the content
    content = normalize_coordinates_in_content(content)
    
    return content.strip()


def normalize_group_content(group: Element) -> List[str]:
    """
    Normalize the content of a group by normalizing each child element.
    """
    child_contents = []
    for child in group:
        child_contents.append(normalize_element_content(child))
    return sorted(child_contents)  # Sort to make order-independent


def match_groups(input_groups: List[Element], find_groups: Dict[str, Element]) -> List[Tuple[List[Element], str]]:
    """
    Match input groups to find groups based on visual content.
    This function compares path elements found in the subgroups of find_ groups
    to path elements in input.svg. Once path elements that look the same are found,
    it looks for groups around the path element group that match the structure 
    of the subgroups of the find_ group.
    Returns a list of tuples: (matched_input_groups, find_group_id)
    """
    matches = []

    for find_id, find_group in find_groups.items():
        print(f"Looking for match for find group {find_id}")
        
        # Extract path elements from find group subgroups
        find_path_elements = []
        find_subgroups = []
        for subgroup in find_group:  # Direct children of the find group
            find_path_elements.extend(extract_path_elements(subgroup))
            find_subgroups.append(subgroup)
        
        print(f"Find group {find_id} has {len(find_path_elements)} path elements in {len(find_subgroups)} subgroups")
        
        if not find_path_elements:
            print(f"No path elements found in find group {find_id}, skipping...")
            continue

        # Normalize all path elements in the find group for comparison
        normalized_find_paths = [normalize_path_content(path_elem) for path_elem in find_path_elements]
        normalized_find_paths_set = set(normalized_find_paths)
        
        print(f"Normalized find path elements: {len(normalized_find_paths)} elements, {len(normalized_find_paths_set)} unique")

        # Look for input groups that contain path elements matching the find group
        matching_input_groups = []
        
        # Try to match sequences of consecutive groups that match the pattern
        for i in range(len(input_groups)):
            # Check if we have enough remaining groups to match the find pattern
            if i + len(find_subgroups) > len(input_groups):
                continue
                
            # Get a sequence of input groups to compare
            candidate_groups = input_groups[i:i+len(find_subgroups)]
            
            # Extract path elements from these candidate groups
            candidate_path_elements = []
            for group in candidate_groups:
                candidate_path_elements.extend(extract_path_elements(group))
            
            if not candidate_path_elements:
                continue
                
            # Normalize path elements in these candidate groups
            normalized_candidate_paths = [normalize_path_content(path_elem) for path_elem in candidate_path_elements]
            normalized_candidate_paths_set = set(normalized_candidate_paths)
            
            # Check if the candidate groups have the same path elements as the find group
            if normalized_find_paths_set == normalized_candidate_paths_set:
                print(f"Found matching sequence of {len(candidate_groups)} groups with same path elements as {find_id}")
                matching_input_groups.append(candidate_groups)
            elif normalized_find_paths_set.issubset(normalized_candidate_paths_set):
                # If the candidate groups contain all the path elements from the find group (and maybe more)
                print(f"Found matching sequence containing find group path elements: {find_id}")
                matching_input_groups.append(candidate_groups)
            else:
                # Additional check: see if the shapes are similar by comparing structure more loosely
                # This is especially important when dealing with relative vs absolute coordinates
                find_shape_signature = create_shape_signature(normalized_find_paths)
                candidate_shape_signature = create_shape_signature(normalized_candidate_paths)
                
                if find_shape_signature == candidate_shape_signature:
                    print(f"Found matching sequence based on shape signature for {find_id}")
                    matching_input_groups.append(candidate_groups)

        print(f"Found {len(matching_input_groups)} candidate group sequences for {find_id}")
        
        # Add the matching groups to results
        for input_group_sequence in matching_input_groups:
            group_ids = [g.get('id', 'no_id') for g in input_group_sequence]
            print(f"Adding match: group sequence {group_ids} matches {find_id}")
            matches.append((input_group_sequence, find_id))

    return matches


def create_shape_signature(path_contents):
    """
    Create a signature of shapes by counting types and attributes.
    This helps match paths that have different coordinate representations but same structure.
    """
    signature = {}
    for content in path_contents:
        # Count different element types
        if 'path' in content:
            key = 'path'
        elif 'polygon' in content:
            key = 'polygon'
        elif 'polyline' in content:
            key = 'polyline'
        else:
            key = 'other'
        
        # Count by element type and stroke/fill attributes
        stroke_match = re.search(r'stroke="([^"]*)"', content)
        fill_match = re.search(r'fill="([^"]*)"', content)
        
        stroke = stroke_match.group(1) if stroke_match else 'none'
        fill = fill_match.group(1) if fill_match else 'none'
        
        # Normalize colors for comparison
        stroke = normalize_color(stroke)
        fill = normalize_color(fill)
        
        attr_key = f"{key}_{stroke}_{fill}"
        signature[attr_key] = signature.get(attr_key, 0) + 1
    
    return signature