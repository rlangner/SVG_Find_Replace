#!/usr/bin/env python3
"""
SVG Group Matcher and Replacer

This script searches through an input.svg file, parses out every <g> group,
compares them to known .svg <g> groups found as subgroups in lookup.svg,
and replaces matching groups with corresponding replacement groups.
"""

import xml.etree.ElementTree as ET
from xml.etree.ElementTree import Element
import re
import copy
from typing import Dict, List, Tuple, Optional
import sys


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
            x = float(segments[i])
            i += 1
            y = float(segments[i])
            if cmd.islower():
                x += current_x
                y += current_y
            result.extend(['M', str(x), str(y)])
            current_x, current_y = x, y
            start_x, start_y = x, y
            i += 1
        elif cmd in 'LlTt':
            i += 1
            x = float(segments[i])
            i += 1
            y = float(segments[i])
            if cmd.islower():
                x += current_x
                y += current_y
            result.extend(['L', str(x), str(y)])
            current_x, current_y = x, y
            i += 1
        elif cmd in 'Hh':
            i += 1
            x = float(segments[i])
            if cmd.islower():
                x += current_x
            y = current_y
            result.extend(['L', str(x), str(y)])
            current_x = x
            i += 1
        elif cmd in 'Vv':
            i += 1
            y = float(segments[i])
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
    content = re.sub(r'clipId\\d+\\.\\d*-\\d+', 'clipId', content)
    content = re.sub(r'clipId\\d+\\.\\d*', 'clipId', content)
    
    # Normalize color formats: convert rgb(r,g,b) to hex or hex to rgb for consistency
    def normalize_color(match):
        full_match = match.group(0)
        if full_match.startswith('rgb'):
            # Parse rgb(r,g,b) format
            import re
            rgb_match = re.search(r'rgb\\((\\d+),(\\d+),(\\d+)\\)', full_match)
            if rgb_match:
                r, g, b = map(int, rgb_match.groups())
                return f"#{r:02x}{g:02x}{b:02x}".upper()
        return full_match
    
    # Find and replace rgb colors with hex colors
    content = re.sub(r'rgb\\(\\d+,\\d+,\\d+\\)', normalize_color, content)
    
    # Extract path coordinates and normalize them to canonical representation
    def extract_and_sort_path_coords(path_d):
        # This function converts path data to a canonical form by extracting coordinates
        import re
        
        # Extract all coordinate pairs from the path data
        coords = re.findall(r'[-+]?\\d*\\.?\\d+', path_d)
        
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
    content = re.sub(r'\\s+', ' ', content)
    
    # Normalize numbers (round to 3 decimal places to handle floating point differences)
    def round_numbers(match):
        num = float(match.group())
        return f"{num:.3f}"
    
    content = re.sub(r'\\d+\\.?\\d*', round_numbers, content)
    
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
    
    # Normalize color formats: convert rgb(r,g,b) to hex or hex to rgb for consistency
    def normalize_color(match):
        full_match = match.group(0)
        if full_match.startswith('rgb'):
            # Parse rgb(r,g,b) format
            import re
            rgb_match = re.search(r'rgb\((\d+),(\d+),(\d+)\)', full_match)
            if rgb_match:
                r, g, b = map(int, rgb_match.groups())
                return f"#{r:02x}{g:02x}{b:02x}".upper()
        return full_match
    
    # Find and replace rgb colors with hex colors
    content = re.sub(r'rgb\(\d+,\d+,\d+\)', normalize_color, content)
    
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
        num = float(match.group())
        return f"{num:.3f}"
    
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
        
        # Convert to floats
        coords = [float(n) for n in numbers if n]
        
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
                    coords.append((float(x.strip()), float(y.strip())))
            
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
        num = float(match.group())
        return f"{num:.3f}"
    
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

        print(f"Found {len(matching_input_groups)} candidate group sequences for {find_id}")
        
        # Add the matching groups to results
        for input_group_sequence in matching_input_groups:
            group_ids = [g.get('id', 'no_id') for g in input_group_sequence]
            print(f"Adding match: group sequence {group_ids} matches {find_id}")
            matches.append((input_group_sequence, find_id))

    return matches

def get_group_transform(group: Element) -> str:
    """
    Extract the transform attribute from a group or calculate it based on position.
    """
    transform = group.get('transform')
    if transform:
        return transform
    
    # If no transform, try to extract position from child elements
    # This is a simplified approach - in real usage, you might need more sophisticated logic
    return ''


def replace_groups_in_svg(input_svg_path: str, lookup_svg_path: str, output_svg_path: str):
    """
    Main function to replace matching groups in input SVG with replacement groups from lookup SVG.
    """
    print("Loading input SVG...")
    input_tree = ET.parse(input_svg_path)
    input_root = input_tree.getroot()
    
    print("Loading lookup SVG...")
    lookup_groups = extract_groups_from_svg(lookup_svg_path)
    find_groups = lookup_groups['find']
    replace_groups = lookup_groups['replace']
    
    # Find all <g> elements in the input SVG (excluding those with IDs that are find/replace)
    all_input_groups = []
    for g in input_root.iter('{http://www.w3.org/2000/svg}g'):
        group_id = g.get('id')
        if not group_id or (not group_id.startswith('find_') and not group_id.startswith('replace_')):
            all_input_groups.append(g)
    
    print(f"Found {len(all_input_groups)} groups in input SVG")
    print(f"Found {len(find_groups)} find groups in lookup SVG")
    print(f"Found {len(replace_groups)} replace groups in lookup SVG")
    
    # Find matches between input groups and find groups
    matches = match_groups(all_input_groups, find_groups)
    
    print(f"Found {len(matches)} matches")
    
    # Process each match
    for matched_input_groups, find_id in matches:
        # Get the corresponding replace group
        replace_id = find_id.replace('find_', 'replace_')
        if replace_id not in replace_groups:
            print(f"Warning: No replacement group found for {find_id} (expected {replace_id})")
            continue
        
        replace_group = replace_groups[replace_id]
        print(f"Replacing groups matching {find_id} with {replace_id}")
        
        # Determine the position/transform for the replacement
        # Use the first matched group's transform as a reference
        if matched_input_groups:
            first_group = matched_input_groups[0]
            original_transform = get_group_transform(first_group)
            
            # Apply the same transform to the replacement group
            replacement = copy.deepcopy(replace_group)
            if original_transform and 'transform' not in replacement.attrib:
                replacement.set('transform', original_transform)
            
            # Remove the matched groups from the input SVG
            for group in matched_input_groups:
                parent = None
                for p in input_root.iter():
                    if group in list(p):
                        parent = p
                        break
                if parent is not None:
                    parent.remove(group)
            
            # Add the replacement group to the input SVG
            input_root.append(replacement)
    
    # Write the output SVG
    print(f"Writing output to {output_svg_path}")
    input_tree.write(output_svg_path, encoding='unicode', xml_declaration=True)


def main():
    """Main function to run the script."""
    if len(sys.argv) != 4:
        print("Usage: python svg_replacer.py <input.svg> <lookup.svg> <output.svg>")
        sys.exit(1)
    
    input_svg = sys.argv[1]
    lookup_svg = sys.argv[2]
    output_svg = sys.argv[3]
    
    replace_groups_in_svg(input_svg, lookup_svg, output_svg)
    print("Processing complete!")


if __name__ == "__main__":
    main()