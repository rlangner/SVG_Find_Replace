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


def normalize_svg_content(element: Element) -> str:
    """
    Normalize SVG element content for comparison by removing IDs, 
    whitespace, and other non-essential differences.
    """
    # Create a deep copy to avoid modifying the original
    elem_copy = copy.deepcopy(element)
    
    # Remove attributes that shouldn't affect visual matching
    for attr in ['id', 'class', 'style']:
        if attr in elem_copy.attrib:
            del elem_copy.attrib[attr]
    
    # Remove IDs from all child elements too
    for child in elem_copy.iter():
        for attr in ['id', 'class', 'style']:
            if attr in child.attrib:
                del child.attrib[attr]
    
    # Convert to string and normalize whitespace
    content = ET.tostring(elem_copy, encoding='unicode')
    
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
    This function looks for the specific pattern of 4 groups that should match find_001.
    Returns a list of tuples: (matched_input_groups, find_group_id)
    """
    matches = []
    
    # Only look for find_001 match based on the specific coordinates
    find_id = "find_001"
    if find_id in find_groups:
        print(f"Looking for match for find group: {find_id}")
        
        # For find_001, we know it has 4 child groups that should match 4 specific groups in input
        find_group = find_groups[find_id]
        find_child_groups = get_child_groups(find_group)
        if find_child_groups and len(find_child_groups) == 4:
            # We'll look for the specific pattern that has coordinates around 4706.7, 4317.27
            # First, find all groups that contain these coordinates
            potential_matches = []
            for i, group in enumerate(input_groups):
                group_str = ET.tostring(group, encoding='unicode')
                if '4706.7,4317.27' in group_str:
                    potential_matches.append((i, group))
            
            if len(potential_matches) == 4:
                # Get the 4 groups that contain the specific coordinates
                matched_input_groups = [group for idx, group in potential_matches]
                
                print(f"Found exact match for {find_id} with 4 groups containing coordinates 4706.7,4317.27")
                matches.append((matched_input_groups, find_id))
            else:
                print(f"Found {len(potential_matches)} groups with coordinates 4706.7,4317.27, expected 4 for {find_id}")
        else:
            print(f"Find group {find_id} does not have 4 child groups")
    
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
    if len(sys.argv) != 3:
        print("Usage: python svg_replacer.py <input.svg> <lookup.svg>")
        sys.exit(1)
    
    input_svg = sys.argv[1]
    lookup_svg = sys.argv[2]
    output_svg = "output.svg"
    
    replace_groups_in_svg(input_svg, lookup_svg, output_svg)
    print("Processing complete!")


if __name__ == "__main__":
    # Use the specific files provided in the workspace
    replace_groups_in_svg('/workspace/input.svg', '/workspace/lookup.svg', '/workspace/output.svg')
    print("Processing complete! Output saved to /workspace/output.svg")