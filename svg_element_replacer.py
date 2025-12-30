#!/usr/bin/env python3
"""
SVG Element Replacer

This module contains functions for replacing matched SVG elements with replacement elements.
"""

import xml.etree.ElementTree as ET
from xml.etree.ElementTree import Element
import re
import copy
import math
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

# Import the pattern matching functions
from svg_pattern_matcher import find_pattern_matches, get_elements_combined_position

# Import the bounding box functions for matrix operations
from svg_bounding_box import parse_transform, multiply_matrices


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


def get_full_transform_chain_to_root(element: Element, svg_root: Element):
    """
    Get the full transform chain from an element up to the root to calculate the complete transformation matrix.
    """
    from svg_bounding_box import parse_transform, multiply_matrices

    # Start with identity matrix
    full_matrix = [1, 0, 0, 1, 0, 0]

    # Traverse up the tree from the element to the root
    current = element
    while current is not None and current != svg_root:
        # Get transform of current element
        element_transform = current.get('transform')
        if element_transform:
            element_matrix = parse_transform(element_transform)
            # Pre-multiply the element's transform to the accumulated matrix
            # (parent transforms should be applied first)
            full_matrix = multiply_matrices(element_matrix, full_matrix)

        # Move to parent
        parent = None
        for p in svg_root.iter():
            if current in list(p):
                parent = p
                break

        current = parent

    return full_matrix


def calculate_group_center_improved(groups, svg_root) -> Tuple[float, float]:
    """
    Calculate the center position of a sequence of groups using the improved bounding box calculation.
    This function now properly accounts for the original transform context of the groups.
    """
    import xml.etree.ElementTree as ET
    import uuid
    from svg_bounding_box import calculate_group_bbox

    # Check if this is a subset match [parent_group, subset_elements] or a full group match [group1, group2, ...]
    if len(groups) == 2 and isinstance(groups[1], (list, tuple)):
        # This is a subset match: [parent_group, subset_elements]
        # Calculate center based on the subset elements
        subset_elements = groups[1]  # groups[0] is the parent group, groups[1] is the subset

        # Create a temporary group with the subset elements to calculate the center
        temp_group = ET.Element('g')
        temp_id = f"temp_group_{uuid.uuid4().hex[:8]}"
        temp_group.set('id', temp_id)

        # Add all the subset elements to the temporary group
        for element in subset_elements:
            # Deep copy each element to avoid modifying the original
            temp_group.append(copy.deepcopy(element))

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
                min_x, min_y, max_x, max_y = calculate_group_bounding_box(subset_elements)
                center_x = (min_x + max_x) / 2
                center_y = (min_y + max_y) / 2
                return center_x, center_y
        finally:
            # Remove the temporary group from the root
            svg_root.remove(temp_group)
    else:
        # This is a full group match: [group1, group2, ...]
        # If we have just one group and we want to get its position in the original context,
        # we should find its original position in the SVG hierarchy instead of creating a temporary group
        if len(groups) == 1:
            # For a single group, find its original position in the SVG by ID if it exists
            original_group = groups[0]
            original_id = original_group.get('id')

            # Look for the original group in the SVG by its ID
            found_original = None
            if original_id:
                for elem in svg_root.iter():
                    if elem.get('id') == original_id:
                        found_original = elem
                        break

            if found_original:
                # Calculate bounding box of the original group in its original context
                bbox = calculate_group_bbox(svg_root, original_id)
                if bbox:
                    center_x = (bbox['min_x'] + bbox['max_x']) / 2
                    center_y = (bbox['min_y'] + bbox['max_y']) / 2
                    return center_x, center_y

        # For multiple groups or if the single group approach doesn't work,
        # create a temporary group but with a better approach to maintain context
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
    
    # Calculate rotation of the original find group from its transform attribute
    original_transform = original_find_group.get('transform', '')
    original_rotation_from_transform = get_transform_rotation_angle(original_transform)
    
    # Calculate rotation of the matched input groups from their transform attribute
    # For the first group in the sequence
    matched_transform = groups[0].get('transform', '')
    matched_rotation_from_transform = get_transform_rotation_angle(matched_transform)
    
    # Calculate the difference from explicit transforms
    rotation_difference_from_transforms = matched_rotation_from_transform - original_rotation_from_transform
    
    # Only return rotation difference if there's a significant explicit transform difference
    # Round to nearest 5 degrees to avoid tiny floating point differences
    rotation_difference = round(rotation_difference_from_transforms / 5) * 5
    
    # Only return a meaningful rotation if the explicit transform difference is significantly different from 0
    # (at least 5 degrees to account for the rounding)
    if abs(rotation_difference) >= 5.0:
        return rotation_difference
    
    # If no significant explicit transform rotation difference found, return 0
    return 0.0


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

    # Find pattern matches between input groups and find groups
    pattern_matches = find_pattern_matches(all_input_groups, find_groups)

    print(f"Found {len(pattern_matches)} pattern matches")

    # Convert pattern matches to the format expected by the rest of the function
    # Each pattern match is (input_group, matched_elements, find_group_id)
    # We need to convert this to the format: (matched_input_groups, find_id)
    matches = []
    for input_group, matched_elements, find_id in pattern_matches:
        # For pattern matching, we have [input_group, matched_elements] structure
        matches.append(([input_group, matched_elements], find_id))
    
    # Process each match, but avoid overlapping matches
    # Track which input groups have already been matched and removed
    used_groups = set()
    occurrence_counts = {}

    for matched_input_groups, find_id in matches:
        # For pattern matching, we have [parent_group, matched_elements] structure
        parent_group = matched_input_groups[0]
        matched_elements = matched_input_groups[1]

        # Check if any of the matched elements have already been used
        already_used = any(elem in used_groups for elem in matched_elements)
        if already_used:
            print(f"Skipping match for {find_id} - some elements already used by another match")
            continue

        # Get the corresponding replace group
        replace_id = find_id.replace('find_', 'replace_')
        if replace_id not in replace_groups:
            print(f"Warning: No replacement group found for {find_id} (expected {replace_id})")
            continue

        replace_group = replace_groups[replace_id]
        print(f"Replacing elements matching {find_id} with {replace_id}")

        # Mark the matched elements as used before processing
        for elem in matched_elements:
            used_groups.add(elem)

        # Create a deep copy of the replacement group
        replacement = copy.deepcopy(replace_group)

        # Generate a unique ID for this replacement instance
        occurrence_counts[find_id] = occurrence_counts.get(find_id, 0) + 1
        replacement.set('id', f"{replace_id}_{occurrence_counts[find_id]}")

        # Calculate the target position (center of matched input elements)
        # We need to calculate this in the parent's LOCAL coordinate system
        # The matched elements are children of parent_group, so we calculate their center
        # relative to the parent, not the global SVG root

        # Calculate the center of the matched elements in the parent's local coordinate system
        from svg_bounding_box import calculate_group_bbox

        # Create a temporary group to hold the matched elements for bbox calculation
        import uuid
        temp_group = ET.Element('g')
        temp_id = f"temp_matched_{uuid.uuid4().hex[:8]}"
        temp_group.set('id', temp_id)

        # Add copies of matched elements to temp group
        for elem in matched_elements:
            temp_group.append(copy.deepcopy(elem))

        # Add temp group to parent (not root) to calculate in parent's coordinate system
        parent_group.append(temp_group)

        try:
            # Calculate bbox in parent's coordinate system
            # We need to pass the parent as the root for this calculation
            bbox = calculate_group_bbox(parent_group, temp_id)

            if bbox:
                target_center_x = (bbox['min_x'] + bbox['max_x']) / 2
                target_center_y = (bbox['min_y'] + bbox['max_y']) / 2
            else:
                # Fallback: calculate simple average of element positions
                target_center_x = 0
                target_center_y = 0
                for elem in matched_elements:
                    # Try to extract position from element
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
            # Remove temp group
            parent_group.remove(temp_group)

        print(f"DEBUG: Matched elements center position (local): ({target_center_x:.2f}, {target_center_y:.2f})")

        # Get the parent group's transform to understand the coordinate system
        parent_transform = parent_group.get('transform', '')
        parent_matrix = parse_transform(parent_transform) if parent_transform else [1, 0, 0, 1, 0, 0]
        parent_a, parent_b, parent_c, parent_d, parent_e, parent_f = parent_matrix

        # Extract the parent's scale (assuming uniform scaling for simplicity)
        parent_scale_x = math.sqrt(parent_a * parent_a + parent_b * parent_b)
        parent_scale_y = math.sqrt(parent_c * parent_c + parent_d * parent_d)
        parent_scale = (parent_scale_x + parent_scale_y) / 2

        print(f"DEBUG: Parent transform: {parent_transform}")
        print(f"DEBUG: Parent scale: {parent_scale:.6f}")

        # Calculate the original position of the replacement group in the lookup SVG
        lookup_tree = ET.parse(lookup_svg_path)
        lookup_root = lookup_tree.getroot()

        # Find the original replacement group in the lookup SVG to get its original position
        original_lookup_group = None
        for g in lookup_root.iter('{http://www.w3.org/2000/svg}g'):
            if g.get('id') == replace_id:
                original_lookup_group = g
                break

        # Find the original find group in the lookup SVG to get its transform
        original_find_group = None
        for g in lookup_root.iter('{http://www.w3.org/2000/svg}g'):
            if g.get('id') == find_id:
                original_find_group = g
                break

        # Get the original transform of the replacement group to preserve its internal structure
        original_transform = replace_group.get('transform', '')

        # Get the original transform of the find group
        original_find_transform = original_find_group.get('transform', '') if original_find_group is not None else ''

        # Calculate the center of the replacement group's CHILDREN in the group's local coordinate system
        # This is the center before the replacement group's transform is applied
        if original_lookup_group is not None:
            # Create a temporary group with the children (without the parent's transform)
            temp_group2 = ET.Element('g')
            temp_id2 = f"temp_replace_children_{uuid.uuid4().hex[:8]}"
            temp_group2.set('id', temp_id2)

            # Copy all children of the replacement group
            for child in original_lookup_group:
                temp_group2.append(copy.deepcopy(child))

            # Add to lookup root temporarily (without any transform)
            lookup_root.append(temp_group2)

            try:
                # Calculate bbox of the children
                bbox2 = calculate_group_bbox(lookup_root, temp_id2)
                if bbox2:
                    # The center of the children in their local coordinate system
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

        # Calculate the scale factor needed to compensate for the parent's scale
        # The replacement should maintain its original size from lookup.svg
        # So if the parent has scale 0.01, we need to apply scale 100 to compensate (100 * 0.01 = 1.0)
        if parent_scale > 0:
            compensation_scale = 1.0 / parent_scale
        else:
            compensation_scale = 1.0

        print(f"DEBUG: Compensation scale: {compensation_scale:.6f}")

        # The target_center_x and target_center_y are already in the parent's local coordinate system
        # No need to convert - we calculated them relative to the parent
        local_x = target_center_x
        local_y = target_center_y

        print(f"DEBUG: Target position in parent's local coordinates: ({local_x:.2f}, {local_y:.2f})")

        # Now calculate the position offset for the replacement
        # The replacement group's transform should position it at local_x, local_y
        # and scale it by compensation_scale to maintain its original size

        # Parse the original transform from the replacement group in lookup.svg
        if original_transform:
            original_matrix = parse_transform(original_transform)
            orig_a, orig_b, orig_c, orig_d, orig_tx, orig_ty = original_matrix

            # Get the rotation from the find pattern's transform
            find_rotation = get_transform_rotation_angle(original_find_transform)

            # Get the rotation from the replace pattern's transform
            replace_rotation = get_transform_rotation_angle(original_transform)

            # Calculate the net rotation (replace rotation minus find rotation)
            # This accounts for the fact that the find pattern's rotation is part of how
            # the pattern is defined, not part of what we want to preserve in the output
            net_rotation = replace_rotation - find_rotation

            # Calculate the scale magnitude from the original transform
            orig_scale_x = math.sqrt(orig_a * orig_a + orig_b * orig_b)
            orig_scale_y = math.sqrt(orig_c * orig_c + orig_d * orig_d)
            orig_scale = (orig_scale_x + orig_scale_y) / 2

            # Create a new transform with the net rotation and scaled magnitude
            # Convert net rotation to radians
            net_rotation_rad = math.radians(net_rotation)
            cos_angle = math.cos(net_rotation_rad)
            sin_angle = math.sin(net_rotation_rad)

            # Create the rotation matrix with the scaled magnitude
            # For a rotation matrix: [cos -sin; sin cos] scaled by the magnitude
            total_scale = orig_scale * compensation_scale
            final_a = total_scale * cos_angle
            final_b = total_scale * sin_angle
            final_c = total_scale * (-sin_angle)
            final_d = total_scale * cos_angle

            # Handle negative scales (flips)
            # Check if the original transform had a flip (negative determinant)
            orig_det = orig_a * orig_d - orig_b * orig_c
            if orig_det < 0:
                # Flip the y-axis
                final_c = -final_c
                final_d = -final_d

            # Calculate the translation to position the center correctly
            # The center of the replacement content (in the group's local coords) is at
            # (original_replacement_center_x, original_replacement_center_y)
            # After applying the transform matrix [final_a, final_b, final_c, final_d, final_tx, final_ty]
            # to this center point, we get:
            # x' = final_a * cx + final_c * cy + final_tx
            # y' = final_b * cx + final_d * cy + final_ty
            # We want (x', y') = (local_x, local_y), so:
            # local_x = final_a * cx + final_c * cy + final_tx
            # local_y = final_b * cx + final_d * cy + final_ty
            # Solving for final_tx and final_ty:
            # final_tx = local_x - final_a * cx - final_c * cy
            # final_ty = local_y - final_b * cx - final_d * cy

            final_tx = local_x - final_a * original_replacement_center_x - final_c * original_replacement_center_y
            final_ty = local_y - final_b * original_replacement_center_x - final_d * original_replacement_center_y

            final_transform = f"matrix({final_a},{final_b},{final_c},{final_d},{final_tx},{final_ty})"
        else:
            # If there's no original transform, just apply the compensation scale
            final_tx = local_x - compensation_scale * original_replacement_center_x
            final_ty = local_y - compensation_scale * original_replacement_center_y

            final_transform = f"matrix({compensation_scale},0,0,{compensation_scale},{final_tx},{final_ty})"

        print(f"DEBUG: Final transform: {final_transform}")

        # Apply the final transform to the replacement group
        replacement.set('transform', final_transform)

        # Remove the matched elements from the parent group
        for elem in matched_elements:
            if elem in parent_group:
                parent_group.remove(elem)

        # Add the replacement to the same parent group
        parent_group.append(replacement)
        print(f"DEBUG: Replaced {len(matched_elements)} elements in group '{parent_group.get('id', 'no_id')}' with '{replace_id}'")

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