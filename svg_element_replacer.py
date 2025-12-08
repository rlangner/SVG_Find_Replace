#!/usr/bin/env python3
"""
SVG Element Replacer

This module contains functions for replacing matched SVG elements with replacement elements.
"""

import xml.etree.ElementTree as ET
from xml.etree.ElementTree import Element
import re
import copy
from typing import Dict, List, Tuple, Optional
import sys

# Import the element matching functions
from svg_element_matcher import (
    extract_groups_from_svg,
    match_groups,
    get_child_groups,
    create_shape_signature,
    calculate_group_bbox
)


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


def calculate_group_bounding_box(groups: List[Element]) -> Tuple[float, float, float, float]:
    """
    Calculate the bounding box (min_x, min_y, max_x, max_y) of a sequence of groups by analyzing their path elements.
    """
    all_coords = []
    
    for group in groups:
        # Extract coordinates from path elements in this group
        for path_elem in group.iter():
            if path_elem.tag.endswith('path') and path_elem.get('d'):
                d_attr = path_elem.get('d')
                # Extract coordinates from path data
                coords = re.findall(r'[-+]?\d*\.?\d+', d_attr)
                # Process coordinates in pairs (x, y)
                for i in range(0, len(coords), 2):
                    if i + 1 < len(coords):
                        try:
                            x = float(coords[i])
                            y = float(coords[i + 1])
                            all_coords.append((x, y))
                        except ValueError:
                            continue
            elif path_elem.tag.endswith('polygon') or path_elem.tag.endswith('polyline'):
                points = path_elem.get('points')
                if points:
                    # Parse coordinates from points attribute
                    point_pairs = points.split()
                    for point_pair in point_pairs:
                        if ',' in point_pair:
                            x, y = point_pair.split(',')
                            try:
                                x = float(x.strip())
                                y = float(y.strip())
                                all_coords.append((x, y))
                            except ValueError:
                                continue
    
    if not all_coords:
        return 0.0, 0.0, 0.0, 0.0
    
    # Calculate bounding box
    min_x = min(coord[0] for coord in all_coords)
    min_y = min(coord[1] for coord in all_coords)
    max_x = max(coord[0] for coord in all_coords)
    max_y = max(coord[1] for coord in all_coords)
    
    return min_x, min_y, max_x, max_y


def calculate_group_center_improved(groups: List[Element], svg_root) -> Tuple[float, float]:
    """
    Calculate the center position of a sequence of groups using the improved bounding box calculation.
    This function creates a temporary group containing the elements to pass to calculate_group_bbox.
    """
    import xml.etree.ElementTree as ET
    import uuid
    
    # Create a temporary SVG structure to pass to calculate_group_bbox
    # We'll create a temporary group with a unique ID
    temp_group = ET.Element('g')
    temp_id = f"temp_group_{uuid.uuid4().hex[:8]}"
    temp_group.set('id', temp_id)
    
    # Add all the elements from the groups to the temporary group
    for group in groups:
        # Deep copy each element to avoid modifying the original
        temp_group.append(copy.deepcopy(group))
    
    # Add the temporary group to the SVG root temporarily
    svg_root.append(temp_group)
    
    try:
        # Calculate the bounding box of the temporary group
        bbox = calculate_group_bbox(svg_root, temp_id)
        
        if bbox:
            # Calculate the center of the bounding box
            center_x = (bbox['min_x'] + bbox['max_x']) / 2
            center_y = (bbox['min_y'] + bbox['max_y']) / 2
            
            return center_x, center_y
        else:
            # Fallback to the old method if bounding box calculation fails
            min_x, min_y, max_x, max_y = calculate_group_bounding_box(groups)
            center_x = (min_x + max_x) / 2
            center_y = (min_y + max_y) / 2
            return center_x, center_y
    finally:
        # Remove the temporary group from the root
        svg_root.remove(temp_group)


def calculate_group_center(groups: List[Element]) -> Tuple[float, float]:
    """
    Calculate the center position of a sequence of groups by analyzing their path elements.
    This is the old implementation for fallback use.
    """
    min_x, min_y, max_x, max_y = calculate_group_bounding_box(groups)

    # Calculate the center of the bounding box
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2

    return center_x, center_y


def find_parent(element: Element, root: Element) -> Optional[Element]:
    """
    Find the parent of an element in the SVG tree.
    """
    for parent in root.iter():
        if element in list(parent):
            return parent
    return None


def get_transform_rotation_angle(transform_str):
    """
    Extract rotation angle from transform string if it contains a rotation.
    Returns the rotation angle in degrees, or 0 if no rotation is found.
    """
    if not transform_str:
        return 0.0
    
    # Look for rotate() in the transform
    rotate_match = re.search(r'rotate\(([^)]+)\)', transform_str)
    if rotate_match:
        rotation_params = rotate_match.group(1).split(',')
        if rotation_params:
            try:
                angle = float(rotation_params[0].strip())
                return angle
            except ValueError:
                pass
    
    # Look for matrix transformations that might include rotation
    matrix_match = re.search(r'matrix\(([^)]+)\)', transform_str)
    if matrix_match:
        values = [float(x.strip()) for x in matrix_match.group(1).split(',') if x.strip()]
        if len(values) == 6:
            a, b, c, d, e, f = values
            # For a rotation matrix, the angle can be extracted using atan2
            # The rotation angle in radians is atan2(b, a) or atan2(-c, d)
            import math
            # Use atan2(b, a) to get the rotation angle in radians
            rotation_radians = math.atan2(b, a)
            rotation_degrees = math.degrees(rotation_radians)
            return rotation_degrees
    
    return 0.0


def calculate_element_orientation(element: Element) -> float:
    """
    Calculate the orientation/rotation of an element by analyzing its path elements.
    This function looks at both the transform attribute and the geometric orientation of the paths.
    """
    # First check for explicit transform
    transform = element.get('transform', '')
    explicit_rotation = get_transform_rotation_angle(transform)
    
    # Analyze geometric orientation from path elements
    geometric_rotation = get_geometric_orientation(element)
    
    # For now, return the explicit rotation if available, otherwise the geometric rotation
    if explicit_rotation != 0:
        return explicit_rotation
    else:
        return geometric_rotation


def get_geometric_orientation(element: Element) -> float:
    """
    Calculate the geometric orientation of an element by analyzing its path elements.
    This function determines the orientation based on the principal axis of the shape.
    """
    import math
    
    # Collect all coordinates from all path/polygon/polyline elements
    all_coords = []
    
    for path_elem in element.iter():
        if path_elem.tag.endswith('path') and path_elem.get('d'):
            d_attr = path_elem.get('d')
            # Extract all coordinates from path data
            coords = re.findall(r'[-+]?\d*\.?\d+', d_attr)
            # Process coordinates in pairs (x, y)
            for i in range(0, len(coords), 2):
                if i + 1 < len(coords):
                    try:
                        x = float(coords[i])
                        y = float(coords[i+1])
                        all_coords.append((x, y))
                    except ValueError:
                        continue
        elif path_elem.tag.endswith('polygon') or path_elem.tag.endswith('polyline'):
            points = path_elem.get('points')
            if points:
                # Parse all coordinates from points
                point_pairs = points.split()
                for point_pair in point_pairs:
                    if ',' in point_pair:
                        x, y = point_pair.split(',')
                        try:
                            x = float(x.strip())
                            y = float(y.strip())
                            all_coords.append((x, y))
                        except ValueError:
                            continue
    
    if len(all_coords) < 2:
        return 0.0  # Not enough points to determine orientation
    
    # Calculate centroid
    cx = sum(coord[0] for coord in all_coords) / len(all_coords)
    cy = sum(coord[1] for coord in all_coords) / len(all_coords)
    
    # Calculate covariance matrix components for PCA
    sum_xx = sum((coord[0] - cx)**2 for coord in all_coords)
    sum_yy = sum((coord[1] - cy)**2 for coord in all_coords)
    sum_xy = sum((coord[0] - cx) * (coord[1] - cy) for coord in all_coords)
    
    # Calculate the angle of the principal axis
    if sum_xx == sum_yy == 0:
        return 0.0  # All points are the same
    
    # Calculate the angle using atan2
    theta = 0.5 * math.atan2(2 * sum_xy, sum_xx - sum_yy)
    angle_degrees = math.degrees(theta)
    
    # Normalize to be between -90 and 90 degrees
    while angle_degrees > 90:
        angle_degrees -= 180
    while angle_degrees <= -90:
        angle_degrees += 180
    
    return angle_degrees


def apply_rotation_to_matrix(matrix_values, rotation_angle, center_x, center_y):
    """
    Apply a rotation transformation to an existing transformation matrix.
    Returns the new matrix values as a list.
    """
    import math
    
    if len(matrix_values) != 6:
        return matrix_values  # Return original if not a 2D transformation matrix
    
    a, b, c, d, e, f = matrix_values
    
    # Convert rotation angle to radians
    angle_rad = math.radians(rotation_angle)
    cos_angle = math.cos(angle_rad)
    sin_angle = math.sin(angle_rad)
    
    # Create rotation matrix around the center point
    # First, translate to center, apply rotation, then translate back
    # Rotation matrix: [cos -sin 0; sin cos 0; 0 0 1]
    # Translation matrix: [1 0 tx; 0 1 ty; 0 0 1]
    # Combined: [cos -sin -cx*cos+cy*sin+cx; sin cos -cx*sin-cy*cos+cy; 0 0 1]
    
    # The transformation is: translate(-center) -> rotate -> translate(center) -> original_matrix
    # First, create a rotation matrix around center
    rot_a = cos_angle
    rot_b = sin_angle
    rot_c = -sin_angle
    rot_d = cos_angle
    rot_e = center_x * (1 - cos_angle) + center_y * sin_angle
    rot_f = -center_x * sin_angle + center_y * (1 - cos_angle)
    
    # Then multiply the rotation matrix by the original matrix
    # [rot_a rot_c rot_e]   [a c e]
    # [rot_b rot_d rot_f] * [b d f]
    # [0     0     1  ]   [0 0 1]
    
    new_a = rot_a * a + rot_c * b
    new_b = rot_b * a + rot_d * b
    new_c = rot_a * c + rot_c * d
    new_d = rot_b * c + rot_d * d
    new_e = rot_a * e + rot_c * f + rot_e
    new_f = rot_b * e + rot_d * f + rot_f
    
    return [new_a, new_b, new_c, new_d, new_e, new_f]


def calculate_group_rotation(groups: List[Element], lookup_root: Element, find_id: str) -> float:
    """
    Calculate the rotation of a matched group sequence by comparing it with the original lookup group.
    Returns the rotation difference that should be applied to the replacement.
    """
    # Get the original find group from the lookup SVG
    original_find_group = None
    for g in lookup_root.iter('{http://www.w3.org/2000/svg}g'):
        if g.get('id') == find_id:
            original_find_group = g
            break
    
    if original_find_group is None:
        # If original find group not found, return 0
        return 0.0
    
    # Calculate rotation of the original find group
    original_rotation = calculate_element_orientation(original_find_group)
    
    # Calculate rotation of the matched input groups
    # For simplicity, we'll use the first group's transform as representative
    matched_rotation = calculate_element_orientation(groups[0])
    
    # Calculate the difference in rotation
    rotation_difference = matched_rotation - original_rotation
    
    # Normalize the angle to be within -180 to 180 degrees
    import math
    while rotation_difference > 180:
        rotation_difference -= 360
    while rotation_difference <= -180:
        rotation_difference += 360
    
    return rotation_difference


def calculate_original_transform(groups: List[Element], input_root: Element) -> str:
    """
    Calculate the transform needed to position the replacement group at the same location
    as the matched groups in the input SVG.
    """
    # Find the first group in the sequence to get its transform
    first_group = groups[0]
    
    # First, check if there are any transforms in the hierarchy
    transforms = []
    current = first_group
    while current is not None and current != input_root:
        transform_attr = current.get('transform')
        if transform_attr:
            transforms.append(transform_attr)
        parent = find_parent(current, input_root)
        current = parent
    
    if transforms:
        # If there are explicit transforms in the hierarchy, use those
        # Apply transforms in the correct order (from parent to child)
        transforms.reverse()
        return ' '.join(transforms)
    else:
        # If no explicit transforms found in the hierarchy, 
        # calculate the position by examining the coordinates of path elements
        # and find the center or a representative coordinate of the group
        
        # Collect all coordinates from all path/polygon/polyline elements in the group sequence
        all_coords = []
        
        for group in groups:
            for path_elem in group.iter():
                if path_elem.tag.endswith('path') and path_elem.get('d'):
                    d_attr = path_elem.get('d')
                    # Extract all coordinates from path data
                    coords = re.findall(r'[-+]?\d*\.?\d+', d_attr)
                    # Process coordinates in pairs (x, y)
                    for i in range(0, len(coords), 2):
                        if i + 1 < len(coords):
                            try:
                                x = float(coords[i])
                                y = float(coords[i+1])
                                all_coords.append((x, y))
                            except ValueError:
                                continue
                elif path_elem.tag.endswith('polygon') or path_elem.tag.endswith('polyline'):
                    points = path_elem.get('points')
                    if points:
                        # Parse all coordinates from points
                        point_pairs = points.split()
                        for point_pair in point_pairs:
                            if ',' in point_pair:
                                x, y = point_pair.split(',')
                                try:
                                    x = float(x.strip())
                                    y = float(y.strip())
                                    all_coords.append((x, y))
                                except ValueError:
                                    continue
        
        if all_coords:
            # Calculate the center of the matched groups for positioning
            matched_center_x = sum(coord[0] for coord in all_coords) / len(all_coords)
            matched_center_y = sum(coord[1] for coord in all_coords) / len(all_coords)
            
            # Find the minimum x and y to get the top-left position for comparison
            min_x = min(coord[0] for coord in all_coords)
            min_y = min(coord[1] for coord in all_coords)
            
            print(f"DEBUG: Positioning - Min coordinates: ({min_x}, {min_y}), Matched group center: ({matched_center_x}, {matched_center_y}), Total coords: {len(all_coords)}")
            
            return f"translate({matched_center_x},{matched_center_y})"
        else:
            # If no coordinates found in path elements, return empty string
            print("DEBUG: No coordinates found for positioning")
            return ''


def get_element_position_info(element: Element) -> Tuple[Optional[str], Optional[str]]:
    """
    Extract position information from an element by looking at its attributes and content.
    Returns (transform, coordinates_info) tuple.
    """
    # Get the transform attribute
    transform = element.get('transform')
    
    # Try to extract coordinate information from path/polygon/polyline elements
    coords_info = None
    
    # Look for path elements with d attribute
    for path_elem in element.iter():
        if path_elem.tag.endswith('path') and path_elem.get('d'):
            d_attr = path_elem.get('d')
            # Extract first coordinate from path data
            import re
            coords = re.findall(r'[-+]?\d*\.?\d+', d_attr)
            if coords and len(coords) >= 2:
                try:
                    x = float(coords[0])
                    y = float(coords[1])
                    coords_info = f"first_coord:({x},{y})"
                    break
                except ValueError:
                    continue
        elif path_elem.tag.endswith('polygon') or path_elem.tag.endswith('polyline'):
            points = path_elem.get('points')
            if points:
                # Parse first coordinate from points
                point_pairs = points.split()
                if point_pairs:
                    first_point = point_pairs[0]
                    if ',' in first_point:
                        x, y = first_point.split(',')
                        try:
                            x = float(x.strip())
                            y = float(y.strip())
                            coords_info = f"first_point:({x},{y})"
                            break
                        except ValueError:
                            continue
    
    return transform, coords_info


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
    
    # Sort matches to prioritize longer sequences over shorter ones to avoid overlapping matches
    # This will help ensure larger patterns get priority over smaller ones that might be subsets
    def match_priority(match_tuple):
        matched_input_groups, find_id = match_tuple
        # Priority based on the number of groups in the match (longer sequences first)
        # and then by the find_id for consistent ordering
        return (-len(matched_input_groups), find_id)
    
    matches.sort(key=match_priority)
    
    # Process each match, but avoid overlapping matches
    # Track which input groups have already been matched and removed
    used_groups = set()
    occurrence_counts = {}
    
    for matched_input_groups, find_id in matches:
        # Check if any of the matched groups have already been used
        already_used = any(group in used_groups for group in matched_input_groups)
        if already_used:
            print(f"Skipping match for {find_id} - some groups already used by another match")
            continue
        
        # Get the corresponding replace group
        replace_id = find_id.replace('find_', 'replace_')
        if replace_id not in replace_groups:
            print(f"Warning: No replacement group found for {find_id} (expected {replace_id})")
            continue
        
        replace_group = replace_groups[replace_id]
        print(f"Replacing groups matching {find_id} with {replace_id}")
        
        if matched_input_groups:
            # Mark these groups as used before processing
            for group in matched_input_groups:
                used_groups.add(group)
            
            # Create a deep copy of the replacement group
            replacement = copy.deepcopy(replace_group)
            
            # Generate a unique ID for this replacement instance
            occurrence_counts[find_id] = occurrence_counts.get(find_id, 0) + 1
            replacement.set('id', f"{replace_id}_{occurrence_counts[find_id]}")
            
            # Get the original transform of the replacement group
            original_transform = replace_group.get('transform', '')
            
            # Calculate the target position (center of matched input groups)
            target_center_x, target_center_y = calculate_group_center_improved(matched_input_groups, input_root)
            
            # Calculate the rotation difference between the matched input groups and the lookup find group
            lookup_tree_rotation = ET.parse(lookup_svg_path)
            lookup_root_rotation = lookup_tree_rotation.getroot()
            rotation_difference = calculate_group_rotation(matched_input_groups, lookup_root_rotation, find_id)
            
            # Calculate the original position of the replacement group in the lookup SVG
            # We need to determine where the replacement group would be positioned if placed without any transformation
            lookup_tree = ET.parse(lookup_svg_path)
            lookup_root = lookup_tree.getroot()
            
            # Find the original replacement group in the lookup SVG to get its original position
            original_lookup_group = None
            for g in lookup_root.iter('{http://www.w3.org/2000/svg}g'):
                if g.get('id') == replace_id:
                    original_lookup_group = g
                    break
            
            if original_lookup_group is not None:
                # Calculate the original position of the replacement group in the lookup SVG
                original_replacement_center_x, original_replacement_center_y = calculate_group_center_improved([original_lookup_group], lookup_root)
            else:
                # Fallback: calculate from the copied replacement group
                original_replacement_center_x, original_replacement_center_y = calculate_group_center_improved([replace_group], lookup_root)
            
            # For the positioning, we want to find where the original replacement element is located in the lookup SVG
            # The calculate_group_center_improved function returns the center of the element considering its transforms
            # So original_replacement_center_x, original_replacement_center_y is the actual position of the element in the lookup SVG
            original_center_x = original_replacement_center_x
            original_center_y = original_replacement_center_y
            
            # Calculate the translation needed to move the replacement to the target position
            translation_x = target_center_x - original_center_x
            translation_y = target_center_y - original_center_y
            
            # Create the final transform
            if original_transform:
                # Combine the original transform with the positioning transform and rotation
                # For matrix transforms, we need to apply the translation after the original matrix
                if original_transform.strip().startswith('matrix'):
                    # Decompose the matrix and apply translation
                    matrix_match = re.match(r'matrix\(([^)]+)\)', original_transform.strip())
                    if matrix_match:
                        # Safely convert matrix values to floats
                        values = []
                        for x in matrix_match.group(1).split(','):
                            try:
                                values.append(float(x.strip()))
                            except ValueError:
                                # Skip invalid values
                                continue
                        if len(values) == 6:
                            a, b, c, d, e, f = values
                            # Apply rotation to the matrix if there's a rotation difference
                            if rotation_difference != 0:
                                # Apply rotation to the existing matrix
                                new_values = apply_rotation_to_matrix([a, b, c, d, e, f], rotation_difference, original_center_x, original_center_y)
                                # Apply the translation to the rotated matrix components
                                final_e = new_values[4] + translation_x
                                final_f = new_values[5] + translation_y
                                final_transform = f"matrix({new_values[0]},{new_values[1]},{new_values[2]},{new_values[3]},{final_e},{final_f})"
                            else:
                                # Apply the translation to the existing translation components
                                final_e = e + translation_x
                                final_f = f + translation_y
                                final_transform = f"matrix({a},{b},{c},{d},{final_e},{final_f})"
                        else:
                            # If matrix format is wrong, combine with translate
                            if rotation_difference != 0:
                                final_transform = f"{original_transform} rotate({rotation_difference},{original_center_x},{original_center_y}) translate({translation_x},{translation_y})"
                            else:
                                final_transform = f"{original_transform} translate({translation_x},{translation_y})"
                    else:
                        # If matrix format is wrong, combine with translate
                        if rotation_difference != 0:
                            final_transform = f"{original_transform} rotate({rotation_difference},{original_center_x},{original_center_y}) translate({translation_x},{translation_y})"
                        else:
                            final_transform = f"{original_transform} translate({translation_x},{translation_y})"
                else:
                    # For other transforms, append rotation and translation
                    if rotation_difference != 0:
                        # Calculate the center point around which to rotate (use the original replacement center)
                        final_transform = f"{original_transform} rotate({rotation_difference},{original_center_x},{original_center_y}) translate({translation_x},{translation_y})"
                    else:
                        final_transform = f"{original_transform} translate({translation_x},{translation_y})"
            else:
                # No original transform, apply rotation and then translation
                if rotation_difference != 0:
                    final_transform = f"rotate({rotation_difference},{original_center_x},{original_center_y}) translate({translation_x},{translation_y})"
                else:
                    final_transform = f"translate({translation_x},{translation_y})"
            
            # Apply the final transform to the replacement group
            replacement.set('transform', final_transform)
            
            # Find the parent of the first matched group to replace the entire sequence in the same location
            parent = None
            for p in input_root.iter():
                if matched_input_groups[0] in list(p):
                    parent = p
                    break
            
            if parent is not None:
                # Find the index of the first matched group in its parent
                index = list(parent).index(matched_input_groups[0])
                
                # Remove all matched groups in the sequence (in reverse order to maintain indices)
                for matched_group in reversed(matched_input_groups[1:]):
                    if matched_group in parent:  # Check if still in parent before removing
                        parent.remove(matched_group)
                
                # Replace the first matched group with the replacement
                parent[index] = replacement
            else:
                # If no parent found, append to root (fallback)
                input_root.append(replacement)
    
    # Write the output SVG
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