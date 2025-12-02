#!/usr/bin/env python3
"""
SVG Group Replacer

This script searches through an input.svg file, parses out every <g> group,
compares them to known .svg <g> groups found in lookup.svg, and replaces
visually matching groups with corresponding replacements.
"""

import argparse
import xml.etree.ElementTree as ET
import math
import os
import sys
from xml.dom import minidom
import cairo
import numpy as np
from PIL import Image
import re
import io


def svg_to_bitmap(svg_content, width=64, height=64):
    """Convert SVG content to a bitmap image for visual comparison."""
    try:
        # Create a Cairo surface to render the SVG
        surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, width, height)
        ctx = cairo.Context(surface)
        
        # Clear the surface with white background
        ctx.set_source_rgb(1, 1, 1)  
        ctx.paint()
        ctx.set_source_rgb(0, 0, 0)  # Black for drawing
        
        # We'll need to render the SVG content properly
        # For now, we'll use a workaround by temporarily writing to a file
        # and using an SVG rendering library if available
        
        # For this implementation, we'll create a simple approach to 
        # generate a hash based on the SVG content for comparison
        # since we don't have CairoSVG or similar installed
        return generate_geometric_signature(svg_content)
    except:
        # Fallback to geometric signature if rendering fails
        return generate_geometric_signature(svg_content)


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
                # Extract numbers from path data for signature
                coords = re.findall(r'-?\d+\.?\d*', d)
                coords = [round(float(c), 1) for c in coords if c]
                signature.append((tag, tuple(coords[:10])))  # Limit to first 10 coords to keep signature manageable
            elif tag == 'line':
                x1 = round(float(elem.get('x1', 0)), 1)
                y1 = round(float(elem.get('y1', 0)), 1)
                x2 = round(float(elem.get('x2', 0)), 1)
                y2 = round(float(elem.get('y2', 0)), 1)
                length = round(((x2-x1)**2 + (y2-y1)**2)**0.5, 1)
                signature.append((tag, length))
            elif tag == 'rect':
                width = round(float(elem.get('width', 0)), 1)
                height = round(float(elem.get('height', 0)), 1)
                signature.append((tag, width, height))
            elif tag == 'circle':
                r = round(float(elem.get('r', 0)), 1)
                signature.append((tag, r))
            elif tag == 'ellipse':
                rx = round(float(elem.get('rx', 0)), 1)
                ry = round(float(elem.get('ry', 0)), 1)
                signature.append((tag, rx, ry))
            elif tag in ['polygon', 'polyline']:
                points = elem.get('points', '')
                coords = re.findall(r'-?\d+\.?\d*', points)
                coords = [round(float(c), 1) for c in coords if c]
                signature.append((tag, tuple(coords[:10])))  # Limit to first 10 coords
    
    # Sort signature to make it order-independent
    signature.sort()
    return tuple(signature)


def extract_groups(svg_tree):
    """Extract all <g> groups from an SVG tree."""
    groups = {}
    root = svg_tree.getroot()
    
    # Find all groups in the SVG
    for elem in root.iter():
        if elem.tag.endswith('}g') or elem.tag == 'g':  # Handle namespace
            group_id = elem.get('id', '')
            if group_id:  # Only store groups with IDs
                groups[group_id] = elem
    
    return groups


def find_subgroups(group_element):
    """Find all subgroups within a group element."""
    subgroups = []
    for child in group_element:
        if child.tag.endswith('}g') or child.tag == 'g':
            subgroups.append(child)
    return subgroups


def compare_groups_visual(group1, group2, tolerance=0.1):
    """Compare two SVG groups visually by comparing their geometric signatures."""
    # Generate signatures for both groups
    sig1 = generate_geometric_signature(group1)
    sig2 = generate_geometric_signature(group2)
    
    # Compare the signatures for similarity
    return compare_signatures(sig1, sig2, tolerance)


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
            continue  # Skip if lookup group has no subgroups to match
        
        # Find clusters of input groups that match this lookup group
        matched_cluster = []
        
        # For each input group, check if it matches any of the target subgroups
        for input_id, input_group in input_groups.items():
            if input_id in used_input_ids:
                continue  # Skip already used groups
                
            # Check if this input group matches any of the target subgroups
            for target_subgroup in target_subgroups:
                if compare_groups_visual(input_group, target_subgroup):
                    matched_cluster.append((input_id, input_group))
                    break  # Found a match, no need to check other subgroups
        
        # If we found a complete cluster that matches the lookup pattern
        if len(matched_cluster) == len(target_subgroups):
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
    
    # For each match, remove the matched input groups and add the replacement groups
    for match in matches:
        input_group_ids = match['input_groups']
        replace_group = match['replace_group']
        rotation = match['rotation']
        tx = match['tx']
        ty = match['ty']
        
        # Create a copy of the replacement group with the appropriate transform
        new_group = ET.fromstring(ET.tostring(replace_group, encoding='unicode'))
        set_element_transform(new_group, rotation, tx, ty)
        
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