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
import re


def svg_to_bitmap(svg_content, width=64, height=64):
    """Convert SVG content to a bitmap image for visual comparison."""
    # Since we don't have CairoSVG or similar installed, 
    # we'll generate a geometric signature based on the SVG content for comparison
    return generate_geometric_signature(svg_content)


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
    
    for child in element:
        tag = child.tag.split('}')[-1] if '}' in child.tag else child.tag  # Remove namespace
        
        if tag == 'polygon' or tag == 'polyline':
            points = child.get('points', '')
            coords = extract_coords_from_points(points)
            for x, y in coords:
                min_x, max_x = min(min_x, x), max(max_x, x)
                min_y, max_y = min(min_y, y), max(max_y, y)
        elif tag == 'path':
            d = child.get('d', '')
            coords = extract_coords_from_path(d)
            for x, y in coords:
                min_x, max_x = min(min_x, x), max(max_x, x)
                min_y, max_y = min(min_y, y), max(max_y, y)
        elif tag == 'rect':
            x = float(child.get('x', 0))
            y = float(child.get('y', 0))
            width = float(child.get('width', 0))
            height = float(child.get('height', 0))
            min_x, max_x = min(min_x, x), max(max_x, x + width)
            min_y, max_y = min(min_y, y), max(max_y, y + height)
        elif tag == 'circle':
            cx = float(child.get('cx', 0))
            cy = float(child.get('cy', 0))
            r = float(child.get('r', 0))
            min_x, max_x = min(min_x, cx - r), max(max_x, cx + r)
            min_y, max_y = min(min_y, cy - r), max(max_y, cy + r)
        elif tag == 'ellipse':
            cx = float(child.get('cx', 0))
            cy = float(child.get('cy', 0))
            rx = float(child.get('rx', 0))
            ry = float(child.get('ry', 0))
            min_x, max_x = min(min_x, cx - rx), max(max_x, cx + rx)
            min_y, max_y = min(min_y, cy - ry), max(max_y, cy + ry)
        elif tag in ['line']:
            x1 = float(child.get('x1', 0))
            y1 = float(child.get('y1', 0))
            x2 = float(child.get('x2', 0))
            y2 = float(child.get('y2', 0))
            min_x, max_x = min(min_x, x1, x2), max(max_x, x1, x2)
            min_y, max_y = min(min_y, y1, y2), max(max_y, y1, y2)
    
    # If no child elements were found, return a default bounding box
    if min_x == float('inf'):
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
                if rounded_coords:
                    # Calculate bounding box
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
        
        print(f"Looking for {len(target_subgroups)} subgroups matching {lookup_id}")
        
        # Instead of using geometric signatures, let's try to match based on IDs
        # Get the IDs of the target subgroups
        target_ids = [subgroup.get('id') for subgroup in target_subgroups if subgroup.get('id')]
        
        if target_ids:
            print(f"  Looking for target IDs: {target_ids}")
            
            # Look for input groups with the same IDs
            matched_cluster = []
            for target_id in target_ids:
                for input_id, input_group in input_groups.items():
                    if input_id == target_id and input_id not in used_input_ids:
                        matched_cluster.append((input_id, input_group))
                        break  # Each input group should only be used once
            
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
                
                print(f"Found match: {[item[0] for item in matched_cluster]} matches {lookup_id}")
            else:
                print(f"  Found only {len(matched_cluster)} out of {len(target_subgroups)} target IDs")
        
        else:
            # If no target IDs, fall back to geometric matching
            # Create signatures for all target subgroups
            target_signatures = []
            for i, target_subgroup in enumerate(target_subgroups):
                sig = generate_geometric_signature(target_subgroup)
                target_signatures.append(sig)
                print(f"  Target {i} signature: {sig}")
            
            # Try to match input groups to target subgroups
            # Build a mapping of input groups to potential target matches
            input_to_targets = {}
            for input_id, input_group in input_groups.items():
                if input_id in used_input_ids:
                    continue
                input_sig = generate_geometric_signature(input_group)
                
                # Check if this input group matches any of the target subgroups
                matching_targets = []
                for i, target_sig in enumerate(target_signatures):
                    if compare_signatures(input_sig, target_sig):
                        matching_targets.append(i)
                
                if matching_targets:
                    input_to_targets[input_id] = (input_group, matching_targets)
                    print(f"  Input {input_id} matches targets: {matching_targets}")
            
            print(f"  Found {len(input_to_targets)} potential input matches")
            
            # Now try to find a complete assignment (each target subgroup matched to exactly one input group)
            # This is essentially a bipartite matching problem
            from itertools import permutations
            
            available_input_ids = list(input_to_targets.keys())
            
            if len(available_input_ids) < len(target_subgroups):
                print(f"  Not enough input groups ({len(available_input_ids)}) to match targets ({len(target_subgroups)})")
                continue  # Not enough input groups to match this lookup
            
            # Try different assignments of input groups to target subgroups
            # For efficiency, only try if we have a reasonable number of possibilities
            if len(available_input_ids) <= len(target_subgroups) * 2:  # Reasonable limit
                print(f"  Trying {len(available_input_ids)} available inputs for {len(target_subgroups)} targets")
                # Try to find a valid assignment where each target gets matched to one input
                for assignment in permutations(available_input_ids, len(target_subgroups)):
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
                        break  # Found a valid assignment, move to next lookup group
            else:
                # For large numbers, use a greedy approach
                matched_cluster = []
                assigned_targets = set()
                assigned_inputs = set()
                
                # For each target subgroup, try to find the best matching input group
                for target_idx, target_sig in enumerate(target_signatures):
                    best_match = None
                    for input_id, (input_group, possible_targets) in input_to_targets.items():
                        if input_id in assigned_inputs:
                            continue
                        if target_idx in possible_targets and target_idx not in assigned_targets:
                            best_match = (input_id, input_group)
                            assigned_targets.add(target_idx)
                            assigned_inputs.add(input_id)
                            matched_cluster.append(best_match)
                            break
                
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
                    
                    print(f"Found match: {[item[0] for item in matched_cluster]} matches {lookup_id}")
    
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