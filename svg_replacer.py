#!/usr/bin/env python3
"""
SVG Group Matcher and Replacer

This script searches through an input.svg file, parses out every <g> group,
compares them to known .svg <g> groups found as subgroups in lookup.svg,
and replaces matching groups with corresponding replacement groups.
"""

import sys

# Import the replacement function
from svg_element_replacer import replace_groups_in_svg


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