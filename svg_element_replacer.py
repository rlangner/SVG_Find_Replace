#!/usr/bin/env python3
"""
SVG Element Replacer

This module replaces matched SVG elements with replacement elements from a lookup file.
"""

import xml.etree.ElementTree as ET
from xml.etree.ElementTree import Element
import copy
import math
import uuid
import sys

# Import required functions from other modules
from svg_element_matcher import extract_groups_from_svg, calculate_group_bbox
from svg_pattern_matcher import find_pattern_matches
from svg_bounding_box import parse_transform


def get_transform_rotation_angle(transform_str: str) -> float:
    """
    Extract rotation angle from a transform string.
    Returns the rotation angle in degrees, or 0 if no rotation is found.
    """
    if not transform_str:
        return 0.0
    
    # Look for matrix transformations that might include rotation
    import re
    matrix_match = re.search(r'matrix\(([^)]+)\)', transform_str)
    if matrix_match:
        values = [float(x.strip()) for x in matrix_match.group(1).split(',') if x.strip()]
        if len(values) == 6:
            a, b, c, d, e, f = values
            # Extract rotation angle using atan2(b, a)
            rotation_radians = math.atan2(b, a)
            rotation_degrees = math.degrees(rotation_radians)
            return rotation_degrees
    
    return 0.0


def replace_groups_in_svg(input_svg_path: str, lookup_svg_path: str, output_svg_path: str):
    """
    Main function to replace matching groups in input SVG with replacement groups from lookup SVG.
    
    Args:
        input_svg_path: Path to the input SVG file
        lookup_svg_path: Path to the lookup SVG file containing find/replace patterns
        output_svg_path: Path to write the output SVG file
    """
    print("Loading input SVG...")
    input_tree = ET.parse(input_svg_path)
    input_root = input_tree.getroot()

    print("Loading lookup SVG...")
    lookup_groups = extract_groups_from_svg(lookup_svg_path)
    find_groups = lookup_groups['find']
    replace_groups = lookup_groups['replace']

    # Find all <g> elements in the input SVG (excluding find/replace groups)
    all_input_groups = []
    for g in input_root.iter('{http://www.w3.org/2000/svg}g'):
        group_id = g.get('id')
        if not group_id or (not group_id.startswith('find_') and not group_id.startswith('replace_')):
            all_input_groups.append(g)

    print(f"Found {len(all_input_groups)} groups in input SVG")
    print(f"Found {len(find_groups)} find groups in lookup SVG")
    print(f"Found {len(replace_groups)} replace groups in lookup SVG")

    # Find pattern matches between input groups and find groups
    pattern_matches = find_pattern_matches(all_input_groups, find_groups)
    print(f"Found {len(pattern_matches)} pattern matches")

    # Convert pattern matches to the format: [(parent_group, matched_elements, find_id), ...]
    matches = []
    for input_group, matched_elements, find_id in pattern_matches:
        matches.append(([input_group, matched_elements], find_id))
    
    # Process each match, avoiding overlapping matches
    used_groups = set()
    occurrence_counts = {}

    for matched_input_groups, find_id in matches:
        parent_group = matched_input_groups[0]
        matched_elements = matched_input_groups[1]

        # Check if any matched elements have already been used
        if any(elem in used_groups for elem in matched_elements):
            print(f"Skipping match for {find_id} - some elements already used by another match")
            continue

        # Get the corresponding replace group
        replace_id = find_id.replace('find_', 'replace_')
        if replace_id not in replace_groups:
            print(f"Warning: No replacement group found for {find_id} (expected {replace_id})")
            continue

        replace_group = replace_groups[replace_id]
        print(f"Replacing elements matching {find_id} with {replace_id}")

        # Mark matched elements as used
        for elem in matched_elements:
            used_groups.add(elem)

        # Create a deep copy of the replacement group
        replacement = copy.deepcopy(replace_group)

        # Generate a unique ID for this replacement instance
        occurrence_counts[find_id] = occurrence_counts.get(find_id, 0) + 1
        replacement.set('id', f"{replace_id}_{occurrence_counts[find_id]}")

        # Calculate the center of matched elements in parent's local coordinate system
        temp_group = ET.Element('g')
        temp_id = f"temp_matched_{uuid.uuid4().hex[:8]}"
        temp_group.set('id', temp_id)

        for elem in matched_elements:
            temp_group.append(copy.deepcopy(elem))

        parent_group.append(temp_group)

        try:
            bbox = calculate_group_bbox(parent_group, temp_id)
            if bbox:
                target_center_x = (bbox['min_x'] + bbox['max_x']) / 2
                target_center_y = (bbox['min_y'] + bbox['max_y']) / 2
            else:
                # Fallback: simple average
                target_center_x = 0
                target_center_y = 0
                for elem in matched_elements:
                    if elem.tag.endswith('rect'):
                        x = float(elem.get('x', 0))
                        y = float(elem.get('y', 0))
                        w = float(elem.get('width', 0))
                        h = float(elem.get('height', 0))
                        target_center_x += x + w/2
                        target_center_y += y + h/2
                    elif elem.tag.endswith('circle'):
                        target_center_x += float(elem.get('cx', 0))
                        target_center_y += float(elem.get('cy', 0))
                target_center_x /= len(matched_elements)
                target_center_y /= len(matched_elements)
        finally:
            parent_group.remove(temp_group)

        print(f"DEBUG: Matched elements center position (local): ({target_center_x:.2f}, {target_center_y:.2f})")

        # Get parent group's transform to understand the coordinate system
        parent_transform = parent_group.get('transform', '')
        parent_matrix = parse_transform(parent_transform) if parent_transform else [1, 0, 0, 1, 0, 0]
        parent_a, parent_b, parent_c, parent_d, parent_e, parent_f = parent_matrix

        # Extract parent's scale
        parent_scale_x = math.sqrt(parent_a * parent_a + parent_b * parent_b)
        parent_scale_y = math.sqrt(parent_c * parent_c + parent_d * parent_d)
        parent_scale = (parent_scale_x + parent_scale_y) / 2

        print(f"DEBUG: Parent transform: {parent_transform}")
        print(f"DEBUG: Parent scale: {parent_scale:.6f}")

        # Load lookup SVG to get original transforms
        lookup_tree = ET.parse(lookup_svg_path)
        lookup_root = lookup_tree.getroot()

        # Find original replacement group in lookup SVG
        original_lookup_group = None
        for g in lookup_root.iter('{http://www.w3.org/2000/svg}g'):
            if g.get('id') == replace_id:
                original_lookup_group = g
                break

        # Find original find group in lookup SVG
        original_find_group = None
        for g in lookup_root.iter('{http://www.w3.org/2000/svg}g'):
            if g.get('id') == find_id:
                original_find_group = g
                break

        # Get original transforms
        original_transform = replace_group.get('transform', '')
        original_find_transform = original_find_group.get('transform', '') if original_find_group else ''

        # Calculate center of replacement group's children in local coordinates
        if original_lookup_group is not None:
            temp_group2 = ET.Element('g')
            temp_id2 = f"temp_replace_children_{uuid.uuid4().hex[:8]}"
            temp_group2.set('id', temp_id2)

            for child in original_lookup_group:
                temp_group2.append(copy.deepcopy(child))

            lookup_root.append(temp_group2)

            try:
                bbox2 = calculate_group_bbox(lookup_root, temp_id2)
                if bbox2:
                    original_replacement_center_x = (bbox2['min_x'] + bbox2['max_x']) / 2
                    original_replacement_center_y = (bbox2['min_y'] + bbox2['max_y']) / 2
                else:
                    original_replacement_center_x = 0
                    original_replacement_center_y = 0
            finally:
                lookup_root.remove(temp_group2)
        else:
            original_replacement_center_x = 0
            original_replacement_center_y = 0

        print(f"DEBUG: Replacement children center (local coords): ({original_replacement_center_x:.2f}, {original_replacement_center_y:.2f})")

        # Calculate compensation scale to maintain original size
        compensation_scale = 1.0 / parent_scale if parent_scale > 0 else 1.0
        print(f"DEBUG: Compensation scale: {compensation_scale:.6f}")

        # Target position in parent's local coordinates
        local_x = target_center_x
        local_y = target_center_y
        print(f"DEBUG: Target position in parent's local coordinates: ({local_x:.2f}, {local_y:.2f})")

        # Calculate final transform
        if original_transform:
            original_matrix = parse_transform(original_transform)
            orig_a, orig_b, orig_c, orig_d, orig_tx, orig_ty = original_matrix

            # Get rotations from find and replace patterns
            find_rotation = get_transform_rotation_angle(original_find_transform)
            replace_rotation = get_transform_rotation_angle(original_transform)

            # Calculate net rotation (replace rotation minus find rotation)
            net_rotation = replace_rotation - find_rotation

            # Calculate scale magnitude from original transform
            orig_scale_x = math.sqrt(orig_a * orig_a + orig_b * orig_b)
            orig_scale_y = math.sqrt(orig_c * orig_c + orig_d * orig_d)
            orig_scale = (orig_scale_x + orig_scale_y) / 2

            # Create rotation matrix with scaled magnitude
            net_rotation_rad = math.radians(net_rotation)
            cos_angle = math.cos(net_rotation_rad)
            sin_angle = math.sin(net_rotation_rad)

            total_scale = orig_scale * compensation_scale
            final_a = total_scale * cos_angle
            final_b = total_scale * sin_angle
            final_c = total_scale * (-sin_angle)
            final_d = total_scale * cos_angle

            # Handle negative scales (flips)
            orig_det = orig_a * orig_d - orig_b * orig_c
            if orig_det < 0:
                final_c = -final_c
                final_d = -final_d

            # Calculate translation to position center correctly
            final_tx = local_x - final_a * original_replacement_center_x - final_c * original_replacement_center_y
            final_ty = local_y - final_b * original_replacement_center_x - final_d * original_replacement_center_y

            final_transform = f"matrix({final_a},{final_b},{final_c},{final_d},{final_tx},{final_ty})"
        else:
            # No original transform - just apply compensation scale
            final_tx = local_x - compensation_scale * original_replacement_center_x
            final_ty = local_y - compensation_scale * original_replacement_center_y
            final_transform = f"matrix({compensation_scale},0,0,{compensation_scale},{final_tx},{final_ty})"

        print(f"DEBUG: Final transform: {final_transform}")

        # Apply final transform to replacement group
        replacement.set('transform', final_transform)

        # Remove matched elements from parent group
        for elem in matched_elements:
            if elem in parent_group:
                parent_group.remove(elem)

        # Add replacement to parent group
        parent_group.append(replacement)
        print(f"DEBUG: Replaced {len(matched_elements)} elements in group '{parent_group.get('id', 'no_id')}' with '{replace_id}'")

    # Write output SVG
    print(f"Writing output to {output_svg_path}")
    input_tree.write(output_svg_path, encoding='unicode', xml_declaration=True)


def main():
    """Main function to run the script."""
    if len(sys.argv) != 4:
        print("Usage: python svg_element_replacer.py <input.svg> <lookup.svg> <output.svg>")
        sys.exit(1)

    input_svg = sys.argv[1]
    lookup_svg = sys.argv[2]
    output_svg = sys.argv[3]

    replace_groups_in_svg(input_svg, lookup_svg, output_svg)
    print("Processing complete!")


if __name__ == "__main__":
    main()


