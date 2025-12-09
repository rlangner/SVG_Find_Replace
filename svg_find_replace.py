import xml.etree.ElementTree as ET
import re
from copy import deepcopy

def load_svg_content(svg_path):
    """Load SVG content from file."""
    with open(svg_path, 'r', encoding='utf-8') as f:
        return f.read()

def parse_svg_groups(svg_content):
    """Parse SVG and return the root element."""
    # Parse the SVG content
    root = ET.fromstring(svg_content)
    return root

def find_replace_003_element():
    """Extract the replace_003 element from lookup.svg using XML parsing."""
    tree = ET.parse('/workspace/lookup.svg')
    root = tree.getroot()
    
    # Define namespace map if needed
    namespaces = {}
    for event, elem in ET.iterparse('/workspace/lookup.svg', events=('start-ns',)):
        if event == 'start-ns':
            prefix, uri = elem
            namespaces[prefix] = uri
    
    # Find the element with id="replace_003"
    replace_element = None
    for elem in root.iter():
        if elem.get('id') == 'replace_003':
            replace_element = elem
            break
    
    if replace_element is None:
        raise ValueError("Could not find replace_003 element in lookup.svg")
    
    return replace_element

def find_and_replace_groups(svg_content):
    """Find groups with rotate(180,4349.6,4734.26) transform and replace them with replace_003 content."""
    # Parse the input SVG
    tree = ET.parse('/workspace/input.svg')
    root = tree.getroot()
    
    # Find the replace_003 element from lookup.svg
    replace_element = find_replace_003_element()
    
    # Find all elements with the specific transform
    elements_to_replace = []
    for elem in root.iter():
        transform = elem.get('transform')
        if transform == 'rotate(180,4349.6,4734.26)':
            elements_to_replace.append(elem)
    
    print(f"Found {len(elements_to_replace)} elements to replace")
    
    # For each matching element, replace it with the replace_003 content
    # but preserve the original transform
    for elem in elements_to_replace:
        # Create a copy of the replace element
        new_elem = deepcopy(replace_element)
        
        # Set the same transform as the original element
        new_elem.set('transform', elem.get('transform'))
        
        # Replace the element in the parent
        parent = None
        for p in root.iter():
            if elem in p:
                parent = p
                break
        
        if parent is not None:
            # Find the index of the element to replace
            index = list(parent).index(elem)
            # Replace the element
            parent[index] = new_elem
    
    # Write the modified SVG to a string
    ET.register_namespace('', 'http://www.w3.org/2000/svg')
    ET.register_namespace('sodipodi', 'http://sodipodi.sourceforge.net/DTD/sodipodi-0.dtd')
    ET.register_namespace('inkscape', 'http://www.inkscape.org/namespaces/inkscape')
    
    return ET.tostring(root, encoding='unicode')

def main():
    # Load the input SVG
    svg_content = load_svg_content('/workspace/input.svg')
    
    # Perform the find and replace
    modified_content = find_and_replace_groups(svg_content)
    
    # Save the result
    with open('/workspace/output.svg', 'w', encoding='utf-8') as f:
        f.write(modified_content)
    
    print("SVG find and replace operation completed. Output saved to output.svg")

if __name__ == "__main__":
    main()