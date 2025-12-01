import xml.etree.ElementTree as ET
import math
import os
import sys
from xml.dom import minidom


def svg_to_bitmap(svg_content, width=64, height=64):
    """Convert SVG content to a bitmap image."""
    # For now, we'll return a placeholder since we don't have Cairo
    # This function is used for visual comparison, so we'll implement string-based comparison instead
    return svg_content





def compare_images(svg1_content, svg2_content, threshold=0.9):
    """Compare two SVG elements and return similarity score."""
    # Convert elements to strings for comparison
    str1 = ET.tostring(svg1_content, encoding='unicode') if isinstance(svg1_content, ET.Element) else str(svg1_content)
    str2 = ET.tostring(svg2_content, encoding='unicode') if isinstance(svg2_content, ET.Element) else str(svg2_content)
    
    # Normalize the strings by removing transform attributes temporarily for comparison
    import re
    
    # Remove transform attributes for comparison since we want to match the shapes themselves
    norm1 = re.sub(r'\\s*transform\\s*=\\s*[\"\\\'][^\"\\\']*?[\"\\\']', '', str1)
    norm2 = re.sub(r'\\s*transform\\s*=\\s*[\"\\\'][^\"\\\']*?[\"\\\']', '', str2)
    
    # Remove id attributes for comparison to focus on structure
    norm1 = re.sub(r'\\s*id\\s*=\\s*[\"\\\'][^\"\\\']*?[\"\\\']', '', norm1)
    norm2 = re.sub(r'\\s*id\\s*=\\s*[\"\\\'][^\"\\\']*?[\"\\\']', '', norm2)
    
    # Remove extra whitespace
    norm1 = ' '.join(norm1.split())
    norm2 = ' '.join(norm2.split())
    
    if norm1 == norm2:
        return 1.0
    
    # For a more sophisticated comparison, we can check if the elements have the same structure
    try:
        elem1 = ET.fromstring(norm1)
        elem2 = ET.fromstring(norm2)
        
        # Compare the structure by comparing tags, attributes (excluding id/transform), and children
        similarity = compare_elements_structure(elem1, elem2)
        return similarity
    except:
        # If parsing fails, fall back to string comparison
        min_len = min(len(norm1), len(norm2))
        if min_len == 0:
            return 0.0 if len(norm1) != len(norm2) else 1.0
        
        # Calculate similarity based on common characters
        common_len = sum(1 for a, b in zip(norm1, norm2) if a == b)
        return common_len / max(len(norm1), len(norm2))


def get_geometric_signature(elem):
    """Create a geometric signature of an SVG element based on its visual properties."""
    import re
    
    # If it's a group element, combine signatures of children
    if elem.tag.endswith('}g') or elem.tag == 'g':
        signatures = []
        for child in elem:
            child_sig = get_geometric_signature(child)
            if child_sig:
                signatures.append(child_sig)
        # Sort signatures to make order-independent
        signatures.sort()
        return tuple(signatures)
    
    # For geometric elements (path, line, rect, circle, etc.)
    tag = elem.tag.split('}')[-1] if '}' in elem.tag else elem.tag
    
    if tag in ['path', 'line', 'rect', 'circle', 'ellipse', 'polygon', 'polyline']:
        # Extract coordinates and geometric properties
        if tag == 'path':
            # For path elements, extract d attribute and normalize coordinates
            d = elem.get('d', '')
            # Extract numbers from path data
            coords = re.findall(r'-?\d+\.?\d*', d)
            # Convert to float and round to reduce sensitivity to small differences
            coords = [round(float(c), 1) for c in coords if c]
            return (tag, tuple(coords))
        elif tag == 'line':
            # Extract x1, y1, x2, y2
            x1 = round(float(elem.get('x1', 0)), 1)
            y1 = round(float(elem.get('y1', 0)), 1)
            x2 = round(float(elem.get('x2', 0)), 1)
            y2 = round(float(elem.get('y2', 0)), 1)
            # Create a signature based on length and angle
            length = round(((x2-x1)**2 + (y2-y1)**2)**0.5, 1)
            angle = round(math.atan2(y2-y1, x2-x1), 2) if x2 != x1 or y2 != y1 else 0
            return (tag, length, angle)
        elif tag == 'rect':
            # Extract x, y, width, height
            x = round(float(elem.get('x', 0)), 1)
            y = round(float(elem.get('y', 0)), 1)
            width = round(float(elem.get('width', 0)), 1)
            height = round(float(elem.get('height', 0)), 1)
            return (tag, width, height)
        elif tag == 'circle':
            # Extract r (radius)
            r = round(float(elem.get('r', 0)), 1)
            return (tag, r)
        elif tag == 'ellipse':
            # Extract rx, ry
            rx = round(float(elem.get('rx', 0)), 1)
            ry = round(float(elem.get('ry', 0)), 1)
            return (tag, rx, ry)
        elif tag in ['polygon', 'polyline']:
            # Extract points
            points = elem.get('points', '')
            # Parse points string
            coords = re.findall(r'-?\d+\.?\d*', points)
            coords = [round(float(c), 1) for c in coords if c]
            # Group into (x,y) pairs and create signature based on distances between points
            if len(coords) >= 2 and len(coords) % 2 == 0:  # Ensure even number of coordinates
                # Calculate distances between consecutive points
                distances = []
                for i in range(0, len(coords)-2, 2):
                    x1, y1 = coords[i], coords[i+1]
                    if i+2 < len(coords):
                        x2, y2 = coords[i+2], coords[i+3]
                        dist = round(((x2-x1)**2 + (y2-y1)**2)**0.5, 1)
                        distances.append(dist)
                distances.sort()  # Make order-independent
                return (tag, tuple(distances))
            elif len(coords) >= 2:  # Odd number of coords, just return as is
                return (tag, tuple(coords))
    
    # For other elements, return a simple signature
    return (tag,)


def compare_elements_structure(elem1, elem2):
    """Compare two XML elements for structural similarity."""
    # Generate geometric signatures for both elements
    sig1 = get_geometric_signature(elem1)
    sig2 = get_geometric_signature(elem2)
    
    # Debug: print signatures if they're not empty
    if sig1 and sig2:
        similarity = 1.0 if sig1 == sig2 else 0.0
        if similarity < 1.0:
            # If exact match fails, try to calculate similarity based on geometric features
            similarity = calculate_geometric_similarity(sig1, sig2)
        return similarity
    elif not sig1 and not sig2:
        # Both have no geometric signature, compare tags
        return 1.0 if elem1.tag == elem2.tag else 0.0
    else:
        # One has signature, the other doesn't
        return 0.0

def calculate_geometric_similarity(sig1, sig2):
    """Calculate similarity between two geometric signatures."""
    # If both are tuples of the same length, compare element by element
    if isinstance(sig1, tuple) and isinstance(sig2, tuple):
        if len(sig1) == len(sig2):
            total_similarity = 0
            for s1, s2 in zip(sig1, sig2):
                if s1 == s2:
                    total_similarity += 1
                elif isinstance(s1, tuple) and isinstance(s2, tuple):
                    # Recursive comparison for nested tuples
                    total_similarity += calculate_geometric_similarity(s1, s2)
                else:
                    # Try to compare as geometric features
                    total_similarity += compare_geometric_features(s1, s2)
            return total_similarity / len(sig1) if sig1 else 1.0
        else:
            # Different number of elements, but if they share some common elements...
            common_elements = set(sig1) & set(sig2)
            max_len = max(len(sig1), len(sig2))
            return len(common_elements) / max_len if max_len > 0 else 0.0
    else:
        # Compare as geometric features
        return compare_geometric_features(sig1, sig2)

def compare_geometric_features(f1, f2):
    """Compare two geometric features for similarity."""
    if f1 == f2:
        return 1.0
    elif isinstance(f1, (int, float)) and isinstance(f2, (int, float)):
        # For numeric values, calculate similarity based on relative difference
        if f1 == 0 and f2 == 0:
            return 1.0
        elif f1 == 0 or f2 == 0:
            return 0.0
        else:
            # Calculate similarity based on relative difference
            diff = abs(f1 - f2) / max(abs(f1), abs(f2))
            return max(0, 1 - diff)  # Convert difference to similarity
    elif isinstance(f1, tuple) and isinstance(f2, tuple):
        # For tuples, check if they represent similar geometric patterns
        if len(f1) > 0 and len(f2) > 0:
            # If both tuples start with the same tag, compare the rest
            if len(f1) > 0 and len(f2) > 0 and f1[0] == f2[0]:
                # Same type of element, compare geometric properties
                if len(f1) == len(f2):
                    # Compare each geometric property
                    total_sim = 0
                    for i in range(1, len(f1)):
                        if i < len(f2):
                            if isinstance(f1[i], (int, float)) and isinstance(f2[i], (int, float)):
                                # Calculate similarity for numeric properties
                                if f1[i] == 0 and f2[i] == 0:
                                    total_sim += 1.0
                                elif f1[i] == 0 or f2[i] == 0:
                                    # If one is 0 and the other isn't, use a tolerance
                                    if abs(f1[i] - f2[i]) < 1:  # 1 unit tolerance
                                        total_sim += 0.5
                                    else:
                                        total_sim += 0
                                else:
                                    # Calculate similarity based on relative difference
                                    diff = abs(f1[i] - f2[i]) / max(abs(f1[i]), abs(f2[i]))
                                    total_sim += max(0, 1 - diff)
                            else:
                                total_sim += 1.0 if f1[i] == f2[i] else 0.0
                    return total_sim / (len(f1) - 1) if len(f1) > 1 else 1.0
                else:
                    # Different number of properties, find best match
                    min_len = min(len(f1), len(f2))
                    total_sim = 0
                    for i in range(1, min_len):
                        if isinstance(f1[i], (int, float)) and isinstance(f2[i], (int, float)):
                            if f1[i] == 0 and f2[i] == 0:
                                total_sim += 1.0
                            elif f1[i] == 0 or f2[i] == 0:
                                if abs(f1[i] - f2[i]) < 1:
                                    total_sim += 0.5
                                else:
                                    total_sim += 0
                            else:
                                diff = abs(f1[i] - f2[i]) / max(abs(f1[i]), abs(f2[i]))
                                total_sim += max(0, 1 - diff)
                        else:
                            total_sim += 1.0 if f1[i] == f2[i] else 0.0
                    return total_sim / (min_len - 1) if min_len > 1 else 1.0
            else:
                return 0.0  # Different element types
        else:
            return 0.0
    else:
        return 0.0


def rotate_image(svg_content, angle):
    """Rotate an SVG content by the given angle in degrees."""
    # For now, we'll just return the content since we're not doing actual image rotation
    # The angle parameter is used for testing different rotations in the matching algorithm
    return svg_content


def get_element_transform(element):
    """Extract transform information from an SVG element."""
    transform = element.attrib.get('transform', '')
    # Parse transform to extract rotation, translation, etc.
    # This is a simplified version
    import re
    
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
        element.attrib['transform'] = ' '.join(transforms)
    elif 'transform' in element.attrib:
        del element.attrib['transform']


def find_matching_groups(input_groups, lookup_groups, replace_groups):
    """Find matching groups between input and lookup SVGs."""
    matches = []
    used_input_indices = set()  # Track which input groups have been used to avoid duplicate matches
    
    for lookup_idx, (lookup_id, lookup_group) in enumerate(lookup_groups):
        # Extract all subgroups from the lookup group
        lookup_subgroups = []
        for elem in lookup_group:
            if (elem.tag.endswith('}g') or elem.tag == 'g') and not elem.get('id', '').startswith(('find_', 'replace_')):
                lookup_subgroups.append(elem)
        
        if not lookup_subgroups:
            continue  # Skip if lookup group has no subgroups to match
        
        # Create a signature for the lookup group based on its subgroups
        lookup_signature = create_group_signature(lookup_subgroups)
        
        # Find clusters of input groups that match this lookup group
        matched_clusters = find_matching_clusters(input_groups, used_input_indices, lookup_signature, lookup_subgroups)
        
        for cluster in matched_clusters:
            # Find the corresponding replace group
            find_num = lookup_id.split('_')[-1]  # Extract the 3-digit number
            replace_group = None
            replace_id = f"replace_{find_num}"
            for rep_id, rep_group in replace_groups:
                if rep_id == replace_id:
                    replace_group = rep_group
                    break
            
            if replace_group is not None and cluster:
                # Use the transform of the first matched group as reference
                first_input_idx = cluster[0]
                first_input_group = input_groups[first_input_idx]
                input_rotation, input_tx, input_ty = get_element_transform(first_input_group)
                
                matches.append({
                    'input_indices': cluster,  # Multiple input indices
                    'lookup_idx': lookup_idx,
                    'replace_group': replace_group,
                    'rotation': input_rotation,
                    'tx': input_tx,
                    'ty': input_ty
                })
                
                # Mark these input groups as used
                for idx in cluster:
                    used_input_indices.add(idx)
    
    return matches

def create_group_signature(groups):
    """Create a signature for a group based on its subgroups."""
    signatures = []
    for group in groups:
        sig = get_geometric_signature(group)
        if sig:
            signatures.append(sig)
    # Sort to make order-independent
    signatures.sort()
    return tuple(signatures)

def find_matching_clusters(input_groups, used_input_indices, lookup_signature, lookup_subgroups):
    """Find clusters of input groups that match the lookup group structure."""
    clusters = []
    
    # Try different combinations of unused input groups
    unused_indices = [i for i in range(len(input_groups)) if i not in used_input_indices]
    
    # For now, let's try matching individual groups first, then expand to clusters
    for i in unused_indices:
        if i in used_input_indices:
            continue
            
        input_group = input_groups[i]
        input_sig = get_geometric_signature(input_group)
        
        # Check if this single input group matches any of the lookup subgroups
        for j, lookup_subgroup in enumerate(lookup_subgroups):
            lookup_sub_sig = get_geometric_signature(lookup_subgroup)
            
            if input_sig and lookup_sub_sig and compare_signatures(input_sig, lookup_sub_sig):
                # Found a potential match, now look for other groups that might be part of the same cluster
                cluster = [i]
                
                # Try to find other groups that are close in position to form a cluster
                input_rotation, input_tx, input_ty = get_element_transform(input_group)
                
                # Look for nearby groups that might be part of the same pattern
                for k in unused_indices:
                    if k == i or k in used_input_indices or k in cluster:
                        continue
                    
                    other_group = input_groups[k]
                    other_rotation, other_tx, other_ty = get_element_transform(other_group)
                    
                    # Check if this group is close in position (within a tolerance)
                    if abs(other_tx - input_tx) < 50 and abs(other_ty - input_ty) < 50:
                        # Check if this group matches one of the remaining lookup subgroups
                        other_sig = get_geometric_signature(other_group)
                        if other_sig:
                            remaining_lookup_sigs = [get_geometric_signature(ls) for ls in lookup_subgroups if ls != lookup_subgroup]
                            for remaining_sig in remaining_lookup_sigs:
                                if remaining_sig and compare_signatures(other_sig, remaining_sig):
                                    cluster.append(k)
                                    break
                
                # Check if this cluster matches the full lookup signature
                cluster_groups = [input_groups[idx] for idx in cluster]
                cluster_signature = create_group_signature(cluster_groups)
                
                if compare_signatures(cluster_signature, lookup_signature):
                    clusters.append(cluster)
                    break  # Move to next unused index
    
    return clusters

def compare_signatures(sig1, sig2):
    """Compare two signatures for similarity."""
    if sig1 == sig2:
        return True
    
    # For more complex comparison, use the geometric similarity function
    return calculate_geometric_similarity(sig1, sig2) > 0.8


def remove_groups_at_same_position(input_root, input_groups, matched_indices):
    """Remove groups that are at the same position as matched groups."""
    # For each matched group, remove all groups at the same position
    positions = []
    for match_idx in matched_indices:
        group = input_groups[match_idx]
        _, tx, ty = get_element_transform(group)
        positions.append((tx, ty))
    
    # Remove groups at these positions
    groups_to_remove = []
    for i, group in enumerate(input_groups):
        if i not in matched_indices:  # Don't remove the matched groups themselves yet
            _, tx, ty = get_element_transform(group)
            for pos_tx, pos_ty in positions:
                if abs(tx - pos_tx) < 1 and abs(ty - pos_ty) < 1:  # Position tolerance
                    groups_to_remove.append(i)
                    break
    
    # Remove from the tree (in reverse order to maintain indices)
    for idx in sorted(groups_to_remove, reverse=True):
        input_root.remove(input_groups[idx])


def extract_input_groups(svg_root):
    """Extract all groups from the input SVG that could potentially match lookup patterns."""
    input_groups = []
    
    # Find all groups that are not lookup patterns
    for elem in svg_root.iter():
        if elem.tag.endswith('}g') or elem.tag == 'g':
            # Skip if this is a lookup pattern group
            elem_id = elem.get('id', '')
            if elem_id.startswith(('find_', 'replace_')):
                continue
            # Add this group to our list
            input_groups.append(elem)
    
    return input_groups


def main(input_svg_path, lookup_svg_path, output_svg_path):
    """Main function to process SVG files."""
    # Parse input SVG
    input_tree = ET.parse(input_svg_path)
    input_root = input_tree.getroot()
    
    # Extract all groups from the input SVG
    input_groups = extract_input_groups(input_root)
    
    # Parse lookup SVG
    lookup_tree = ET.parse(lookup_svg_path)
    lookup_root = lookup_tree.getroot()
    
    # Find all find_ and replace_ groups in lookup SVG
    lookup_groups = []
    replace_groups = []
    
    for elem in lookup_root.iter():
        elem_id = elem.get('id', '')
        if elem_id.startswith('find_'):
            lookup_groups.append((elem_id, elem))
        elif elem_id.startswith('replace_'):
            replace_groups.append((elem_id, elem))
    
    print(f"Found {len(input_groups)} input groups, {len(lookup_groups)} lookup groups, {len(replace_groups)} replace groups")
    
    # Find matches
    matches = find_matching_groups(input_groups, lookup_groups, replace_groups)
    
    print(f"Found {len(matches)} matches")
    
    # For each match, remove the matched input groups and add the replacement groups
    for match in matches:
        input_indices = match['input_indices']  # This is now a list of indices
        replace_group = match['replace_group']
        rotation = match['rotation']
        tx = match['tx']
        ty = match['ty']
        
        # Create a copy of the replacement group with the appropriate transform
        new_group = ET.fromstring(ET.tostring(replace_group))
        set_element_transform(new_group, rotation, tx, ty)
        
        # Add the new group to the root
        input_root.append(new_group)
        
        # Add the new group to the root
        input_root.append(new_group)
    
    # Remove matched input groups from the SVG
    for match in reversed(matches):  # Reverse to maintain indices when removing
        input_idx = match['input_idx']
        original_group = input_groups[input_idx]
        # Find and remove the original group from its parent
        for parent in input_root.iter():
            for child in list(parent):
                if child == original_group:
                    parent.remove(child)
                    break
    
    # Write the output SVG
    # Pretty print the XML
    rough_string = ET.tostring(input_root, encoding='unicode')
    
    # Fix namespace issue
    import re
    # Remove namespace prefixes but avoid creating duplicate attributes
    rough_string = re.sub(r'ns\d+:([a-zA-Z]+)', r'\1', rough_string)
    # Remove xmlns:ns0="..." and similar attributes
    rough_string = re.sub(r' xmlns:ns\d+="[^"]*"', '', rough_string)
    # Add proper xmlns attribute if not present
    if ' xmlns=' not in rough_string:
        rough_string = rough_string.replace('<svg', '<svg xmlns="http://www.w3.org/2000/svg"', 1)
    
    # Function to remove duplicate attributes from tags
    import re
    def remove_duplicate_attrs(match):
        full_match = match.group(0)
        tag_name = match.group(1)
        attrs_part = match.group(2) if match.group(2) else ""
        
        # Check if the tag is self-closing (ends with />)
        is_self_closing = full_match.endswith('/>')
        
        # Find all attributes with regex
        attr_pattern = r"([a-zA-Z_:][a-zA-Z0-9_:.-]*)\s*=\s*(?:\"([^\"]*)\"|'([^']*)')"
        found_attrs = []
        seen_attr_names = set()
        
        for attr_match in re.finditer(attr_pattern, attrs_part):
            attr_name = attr_match.group(1)
            attr_value = attr_match.group(2) or attr_match.group(3)  # Get value from double or single quotes
            if attr_name not in seen_attr_names:
                if attr_match.group(2):  # If it was in double quotes
                    found_attrs.append(f'{attr_name}="{attr_value}"')
                else:  # If it was in single quotes
                    found_attrs.append(f"{attr_name}='{attr_value}'")
                seen_attr_names.add(attr_name)
        
        # Return the tag with proper closing
        attrs_str = " ".join(found_attrs)
        if is_self_closing:
            return f'<{tag_name} {attrs_str}/>'
        else:
            return f'<{tag_name} {attrs_str}>'
    
    # Apply to all tags (including self-closing ones) - use a pattern that handles both <tag> and <tag/>
    rough_string = re.sub(r'<(\w+)([^>]*)/?>', remove_duplicate_attrs, rough_string)
    
    # Simple approach to output the SVG without complex processing that causes issues
    try:
        # Pretty print using minidom but without complex regex processing that causes malformed XML
        reparsed = minidom.parseString(rough_string.encode('utf-8'))
        pretty_xml = reparsed.toprettyxml(indent="  ")
        
        # Remove extra blank lines and fix potential issues
        lines = [line for line in pretty_xml.split('\n') if line.strip()]
        pretty_xml = '\n'.join(lines)
        
        # Ensure proper SVG closing tag
        if not pretty_xml.strip().endswith('</svg>'):
            if '</svg>' not in pretty_xml:
                # Add the closing tag if it's missing
                pretty_xml = pretty_xml.rstrip() + '\n</svg>'
    except Exception as e:
        print(f"Error processing XML: {e}")
        # If there's an error, just clean the rough string
        lines = [line for line in rough_string.split('\n') if line.strip()]
        pretty_xml = '\n'.join(lines)
        # Ensure proper closing tag
        if not pretty_xml.strip().endswith('</svg>'):
            if '</svg>' not in pretty_xml:
                pretty_xml = pretty_xml.rstrip() + '\n</svg>'
    
    with open(output_svg_path, 'w') as f:
        f.write(pretty_xml)
    
    print(f"Output written to {output_svg_path}")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python svg_group_replacer.py <input.svg> <lookup.svg> <output.svg>")
        sys.exit(1)
    
    input_svg = sys.argv[1]
    lookup_svg = sys.argv[2]
    output_svg = sys.argv[3]
    
    main(input_svg, lookup_svg, output_svg)