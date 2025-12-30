#!/usr/bin/env python3
"""
SVG Pattern Matcher

This module implements pattern matching for SVG elements based on element type, order, and style.
"""

import xml.etree.ElementTree as ET
from xml.etree.ElementTree import Element
import re
from typing import Dict, List, Tuple


def normalize_color(color: str) -> str:
    """
    Normalize color values to handle different formats (rgb, hex, named colors).
    
    Args:
        color: Color string in various formats (rgb(), #hex, or named)
    
    Returns:
        Normalized color string in rgb(r,g,b) format
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

    return color.lower()


def get_element_detailed_signature(element: Element) -> Dict:
    """
    Create a detailed signature for an element based on its tag, style, and attributes.
    
    Args:
        element: SVG element to create signature for
    
    Returns:
        Dictionary with element characteristics (tag, fill, stroke, stroke_width, id_base)
    """
    # Extract fill and stroke from both attributes and style
    fill = element.get('fill', 'none')
    stroke = element.get('stroke', 'none')
    stroke_width = element.get('stroke-width', '1')

    # Parse the style attribute if present (overrides attributes)
    style = element.get('style', '')
    if style:
        fill_match = re.search(r'fill:\s*([^;]+)', style)
        if fill_match:
            fill = fill_match.group(1).strip()

        stroke_match = re.search(r'stroke:\s*([^;]+)', style)
        if stroke_match:
            stroke = stroke_match.group(1).strip()

        stroke_width_match = re.search(r'stroke-width:\s*([^;]+)', style)
        if stroke_width_match:
            stroke_width = stroke_width_match.group(1).strip()

    # Extract base ID by removing:
    # 1. Trailing digits (e.g., "HATCH1" -> "HATCH")
    # 2. Hex suffixes like "---0xABCD" or " - 0xABCD" (e.g., "HATCH---0x3D44" -> "HATCH")
    element_id = element.get('id', '')
    id_base = re.sub(r'(\s*-+\s*0x[0-9A-Fa-f]+|\d+)$', '', element_id)

    signature = {
        'tag': element.tag.split('}')[-1] if '}' in element.tag else element.tag,
        'id_base': id_base,
        'fill': normalize_color(fill),
        'stroke': normalize_color(stroke),
        'stroke_width': stroke_width,
    }
    return signature


def match_all_patterns(find_group: Element, input_group: Element) -> List[List[Element]]:
    """
    Find ALL occurrences of the pattern in find_group within input_group.
    
    Args:
        find_group: The group from lookup.svg with the pattern to find
        input_group: A group from input.svg to check for matching pattern
    
    Returns:
        List of lists of matching elements from input_group (one list per match found)
    """
    find_signatures = [get_element_detailed_signature(child) for child in find_group]
    input_signatures = [get_element_detailed_signature(child) for child in input_group]

    if len(find_signatures) > len(input_signatures):
        return []  # Cannot match if find group has more elements

    all_matches = []
    used_indices = set()  # Track which indices have been used to avoid overlapping matches

    # Look for all contiguous sequences in input_signatures that match find_signatures
    for i in range(len(input_signatures) - len(find_signatures) + 1):
        # Skip if this starting position overlaps with a previous match
        if i in used_indices:
            continue

        match_found = True
        for j, find_sig in enumerate(find_signatures):
            # Skip if any element in this potential match has been used
            if (i + j) in used_indices:
                match_found = False
                break

            input_sig = input_signatures[i + j]

            # Check if tags match
            if find_sig['tag'] != input_sig['tag']:
                match_found = False
                break

            # Check if fills match (require exact match when both are specified)
            if (find_sig['fill'] != 'none' and input_sig['fill'] != 'none' and
                find_sig['fill'] != input_sig['fill']):
                match_found = False
                break

            # Check if strokes match (require exact match when both are specified)
            if (find_sig['stroke'] != 'none' and input_sig['stroke'] != 'none' and
                find_sig['stroke'] != input_sig['stroke']):
                match_found = False
                break

            # Note: We intentionally do NOT check stroke-width to allow matching
            # patterns across different SVG files that may have different stroke widths
            # but the same visual structure (colors, element types, order)

        if match_found:
            # Validate ID bases match
            find_id_bases = [sig['id_base'] for sig in find_signatures]
            input_id_bases = [sig['id_base'] for sig in input_signatures[i:i+len(find_signatures)]]

            id_match = True
            for find_base, input_base in zip(find_id_bases, input_id_bases):
                if find_base and input_base and not input_base.startswith(find_base):
                    id_match = False
                    break

            if id_match:
                # Validate that matched elements have expected characteristics
                valid_match = True
                for j, find_sig in enumerate(find_signatures):
                    input_sig = input_signatures[i + j]

                    # If find element has specific styling, input element should match
                    if find_sig['stroke'] != 'none' and input_sig['stroke'] != find_sig['stroke']:
                        valid_match = False
                        break

                    if find_sig['fill'] != 'none' and input_sig['fill'] != find_sig['fill']:
                        valid_match = False
                        break

                if valid_match:
                    # Extract the matched elements
                    matched_elements = list(input_group)[i:i+len(find_signatures)]
                    all_matches.append(matched_elements)

                    # Mark these indices as used
                    for j in range(len(find_signatures)):
                        used_indices.add(i + j)

    return all_matches


def find_pattern_matches(input_groups: List[Element], find_groups: Dict[str, Element]) -> List[Tuple[Element, List[Element], str]]:
    """
    Find all pattern matches in input groups.

    Args:
        input_groups: List of groups from input SVG to search
        find_groups: Dictionary of find patterns from lookup SVG (id -> group)

    Returns:
        List of tuples: (input_group, matched_elements, find_group_id)
        This function finds ALL instances of each pattern, including duplicates.
    """
    matches = []

    for find_id, find_group in find_groups.items():
        print(f"Looking for pattern match for {find_id}")

        for input_group in input_groups:
            # Find ALL matches in this group, not just the first one
            all_matches_in_group = match_all_patterns(find_group, input_group)
            for matched_elements in all_matches_in_group:
                print(f"Found pattern match in group '{input_group.get('id', 'no_id')}' for {find_id}")
                matches.append((input_group, matched_elements, find_id))

    return matches


