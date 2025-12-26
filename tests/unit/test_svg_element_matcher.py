import unittest
import xml.etree.ElementTree as ET
from svg_element_matcher import (
    normalize_color,
    normalize_transform,
    normalize_path_data,
    extract_path_elements,
    normalize_path_content,
    normalize_svg_content,
    extract_groups_from_svg,
    get_child_groups,
    normalize_coordinates_in_content,
    normalize_element_content,
    normalize_group_content,
    create_detailed_shape_signature,
    groups_match_structure,
    create_shape_signature
)


class TestSvgElementMatcher(unittest.TestCase):
    
    def test_normalize_color_rgb(self):
        """Test normalizing RGB color format."""
        result = normalize_color("rgb(255, 0, 128)")
        self.assertEqual(result, "rgb(255,0,128)")
        
        result = normalize_color("rgb(100, 50, 200)")
        self.assertEqual(result, "rgb(100,50,200)")
    
    def test_normalize_color_hex(self):
        """Test normalizing hex color format."""
        result = normalize_color("#FF0080")
        self.assertEqual(result, "rgb(255,0,128)")
        
        result = normalize_color("#f00")  # shorthand
        self.assertEqual(result, "rgb(255,0,0)")
        
        result = normalize_color("#a1b2c3")
        self.assertEqual(result, "rgb(161,178,195)")
    
    def test_normalize_color_invalid(self):
        """Test normalizing invalid color format."""
        result = normalize_color("#xyz")  # invalid hex
        self.assertEqual(result, "#xyz")
        
        result = normalize_color("invalid_color")
        self.assertEqual(result, "invalid_color")
    
    def test_normalize_color_none(self):
        """Test normalizing None or empty color."""
        result = normalize_color(None)
        self.assertIsNone(result)
        
        result = normalize_color("")
        self.assertEqual(result, "")
    
    def test_normalize_transform_translate(self):
        """Test normalizing translate transform."""
        result = normalize_transform("translate(10, 20)")
        self.assertEqual(result, "translate(10.0000,20.0000)")
        
        result = normalize_transform("translate(5.5, -3.2)")
        self.assertEqual(result, "translate(5.5000,-3.2000)")
    
    def test_normalize_transform_scale(self):
        """Test normalizing scale transform."""
        result = normalize_transform("scale(2, 3)")
        self.assertEqual(result, "scale(2.0000,3.0000)")
        
        result = normalize_transform("scale(1.5)")
        self.assertEqual(result, "scale(1.5000)")
    
    def test_normalize_transform_rotate(self):
        """Test normalizing rotate transform."""
        result = normalize_transform("rotate(45)")
        self.assertEqual(result, "rotate(45.0000)")
        
        result = normalize_transform("rotate(30, 10, 20)")
        self.assertEqual(result, "rotate(30.0000,10.0000,20.0000)")
    
    def test_normalize_transform_multiple(self):
        """Test normalizing multiple transforms."""
        result = normalize_transform("translate(10, 20) scale(2) rotate(45)")
        expected = "translate(10.0000,20.0000) scale(2.0000) rotate(45.0000)"
        self.assertEqual(result, expected)
    
    def test_normalize_transform_none(self):
        """Test normalizing None transform."""
        result = normalize_transform(None)
        self.assertIsNone(result)
        
        result = normalize_transform("")
        self.assertEqual(result, "")
    
    def test_normalize_path_data_m_line(self):
        """Test normalizing path data with M and L commands."""
        path_data = "M 10 20 L 30 40"
        result = normalize_path_data(path_data)
        # Should convert to absolute coordinates
        self.assertIn("M", result)
        self.assertIn("L", result)
        # The coordinates should be normalized to 1 decimal place (as per actual function)
        self.assertIn("10.0", result)
        self.assertIn("20.0", result)
        self.assertIn("30.0", result)
        self.assertIn("40.0", result)
    
    def test_normalize_path_data_empty(self):
        """Test normalizing empty path data."""
        result = normalize_path_data("")
        self.assertEqual(result, "")
        
        result = normalize_path_data(None)
        self.assertEqual(result, "")
    
    def test_extract_path_elements(self):
        """Test extracting path elements from an SVG element."""
        # Create a test element with path and polygon children
        root = ET.Element('g')
        path_elem = ET.SubElement(root, 'path')
        path_elem.set('d', 'M10,10 L20,20')
        polygon_elem = ET.SubElement(root, 'polygon')
        polygon_elem.set('points', '0,0 10,0 10,10 0,10')
        circle_elem = ET.SubElement(root, 'circle')  # This should not be extracted
        
        elements = extract_path_elements(root)
        self.assertEqual(len(elements), 2)  # path and polygon, but not circle
        self.assertEqual(elements[0].tag, 'path')
        self.assertEqual(elements[1].tag, 'polygon')
    
    def test_extract_path_elements_nested(self):
        """Test extracting path elements from nested elements."""
        root = ET.Element('g')
        sub_group = ET.SubElement(root, 'g')
        path_elem = ET.SubElement(sub_group, 'path')
        path_elem.set('d', 'M10,10 L20,20')
        
        elements = extract_path_elements(root)
        self.assertEqual(len(elements), 1)
        self.assertEqual(elements[0].tag, 'path')
    
    def test_get_child_groups(self):
        """Test getting child groups from an element."""
        root = ET.Element('g')
        group1 = ET.SubElement(root, 'g')
        group1.set('id', 'group1')
        group2 = ET.SubElement(root, 'g')
        group2.set('id', 'group2')
        path_elem = ET.SubElement(root, 'path')  # This should not be returned
        
        groups = get_child_groups(root)
        self.assertEqual(len(groups), 2)
        self.assertEqual(groups[0].get('id'), 'group1')
        self.assertEqual(groups[1].get('id'), 'group2')
    
    def test_create_detailed_shape_signature(self):
        """Test creating a detailed shape signature for a group."""
        # Create a test group with various elements
        group = ET.Element('g')
        path_elem = ET.SubElement(group, 'path')
        path_elem.set('d', 'M10,10 L20,20')
        path_elem.set('fill', '#FF0000')
        path_elem.set('stroke', '#0000FF')
        
        signature = create_detailed_shape_signature(group)
        
        # Check that the signature has the expected structure
        self.assertIn('element_count', signature)
        self.assertIn('element_types', signature)
        self.assertIn('fill_colors', signature)
        self.assertIn('stroke_colors', signature)
        self.assertIn('has_path_elements', signature)
        
        # Check values
        self.assertEqual(signature['element_count'], 1)
        self.assertIn('path', signature['element_types'])
        self.assertIn('rgb(255,0,0)', signature['fill_colors'])  # normalized color
        self.assertIn('rgb(0,0,255)', signature['stroke_colors'])  # normalized color
        self.assertTrue(signature['has_path_elements'])
    
    def test_groups_match_structure_same(self):
        """Test that identical groups match structure."""
        group1 = ET.Element('g')
        path1 = ET.SubElement(group1, 'path')
        path1.set('d', 'M10,10 L20,20')
        path1.set('fill', '#FF0000')
        
        group2 = ET.Element('g')
        path2 = ET.SubElement(group2, 'path')
        path2.set('d', 'M30,30 L40,40')
        path2.set('fill', '#FF0000')
        
        result = groups_match_structure(group1, group2)
        self.assertTrue(result)
    
    def test_groups_match_structure_different(self):
        """Test that different groups don't match structure."""
        group1 = ET.Element('g')
        path1 = ET.SubElement(group1, 'path')
        path1.set('d', 'M10,10 L20,20')
        path1.set('fill', '#FF0000')
        
        group2 = ET.Element('g')
        circle2 = ET.SubElement(group2, 'circle')
        circle2.set('cx', '5')
        circle2.set('cy', '5')
        circle2.set('r', '10')
        circle2.set('fill', '#FF0000')
        
        result = groups_match_structure(group1, group2)
        # Should be False because element types are different (path vs circle)
        self.assertFalse(result)
    
    def test_create_shape_signature(self):
        """Test creating a shape signature."""
        # Create sample path content strings
        path_contents = [
            '<path stroke="red" fill="blue"/>',
            '<polygon stroke="red" fill="green"/>'
        ]
        
        signature = create_shape_signature(path_contents)
        
        # Check that signature contains expected keys
        self.assertIsInstance(signature, dict)
        # The signature should count different element types with their attributes
        found_keys = [k for k in signature.keys() if 'path' in k or 'polygon' in k]
        self.assertGreater(len(found_keys), 0)


if __name__ == '__main__':
    unittest.main()