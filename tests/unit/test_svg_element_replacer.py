import unittest
import xml.etree.ElementTree as ET
from svg_element_replacer import (
    get_group_transform,
    calculate_group_bounding_box,
    calculate_group_center,
    calculate_group_center_improved,
    find_parent,
    get_transform_rotation_angle,
    calculate_element_orientation,
    get_geometric_orientation,
    apply_rotation_to_matrix,
    calculate_group_rotation,
    calculate_original_transform,
    get_element_position_info,
    calculate_group_size
)


class TestSvgElementReplacer(unittest.TestCase):
    
    def test_get_group_transform(self):
        """Test getting transform from a group."""
        group = ET.Element('g')
        group.set('transform', 'translate(10, 20)')
        
        result = get_group_transform(group)
        self.assertEqual(result, 'translate(10, 20)')
    
    def test_get_group_transform_empty(self):
        """Test getting transform from a group with no transform."""
        group = ET.Element('g')
        
        result = get_group_transform(group)
        self.assertEqual(result, '')
    
    def test_calculate_group_bounding_box(self):
        """Test calculating bounding box for a group of elements."""
        # Create a group with path elements
        path1 = ET.Element('path')
        path1.set('d', 'M 10 10 L 20 20')
        
        path2 = ET.Element('path')
        path2.set('d', 'M 30 30 L 40 40')
        
        groups = [path1, path2]
        
        min_x, min_y, max_x, max_y = calculate_group_bounding_box(groups)
        
        # Based on the coordinates in the paths
        self.assertLessEqual(min_x, 10)  # At least 10
        self.assertLessEqual(min_y, 10)  # At least 10
        self.assertGreaterEqual(max_x, 40)  # At least 40
        self.assertGreaterEqual(max_y, 40)  # At least 40
    
    def test_calculate_group_bounding_box_empty(self):
        """Test calculating bounding box for an empty group."""
        groups = []
        
        min_x, min_y, max_x, max_y = calculate_group_bounding_box(groups)
        
        # Should return zeros for empty groups
        self.assertEqual(min_x, 0.0)
        self.assertEqual(min_y, 0.0)
        self.assertEqual(max_x, 0.0)
        self.assertEqual(max_y, 0.0)
    
    def test_calculate_group_center(self):
        """Test calculating center of a group."""
        # Create a group with path elements
        path1 = ET.Element('path')
        path1.set('d', 'M 10 10 L 20 20')
        
        path2 = ET.Element('path')
        path2.set('d', 'M 30 30 L 40 40')
        
        groups = [path1, path2]
        
        center_x, center_y = calculate_group_center(groups)
        
        # The center should be somewhere in the middle of the coordinates
        self.assertGreaterEqual(center_x, 10)
        self.assertLessEqual(center_x, 40)
        self.assertGreaterEqual(center_y, 10)
        self.assertLessEqual(center_y, 40)
    
    def test_find_parent(self):
        """Test finding parent of an element."""
        root = ET.Element('svg')
        group = ET.SubElement(root, 'g')
        child = ET.SubElement(group, 'path')
        
        parent = find_parent(child, root)
        self.assertIsNotNone(parent)
        self.assertEqual(parent, group)
    
    def test_find_parent_root(self):
        """Test finding parent when element is root."""
        root = ET.Element('svg')
        
        parent = find_parent(root, root)
        self.assertIsNone(parent)
    
    def test_get_transform_rotation_angle_simple(self):
        """Test extracting rotation angle from a simple rotation transform."""
        transform = "rotate(45)"
        angle = get_transform_rotation_angle(transform)
        self.assertEqual(angle, 45.0)
    
    def test_get_transform_rotation_angle_with_center(self):
        """Test extracting rotation angle from a rotation transform with center."""
        transform = "rotate(30, 10, 20)"
        angle = get_transform_rotation_angle(transform)
        self.assertEqual(angle, 30.0)
    
    def test_get_transform_rotation_angle_none(self):
        """Test extracting rotation angle from a non-rotation transform."""
        transform = "translate(10, 20)"
        angle = get_transform_rotation_angle(transform)
        self.assertEqual(angle, 0.0)
    
    def test_get_transform_rotation_angle_empty(self):
        """Test extracting rotation angle from an empty transform."""
        transform = ""
        angle = get_transform_rotation_angle(transform)
        self.assertEqual(angle, 0.0)
        
        angle = get_transform_rotation_angle(None)
        self.assertEqual(angle, 0.0)
    
    def test_apply_rotation_to_matrix(self):
        """Test applying rotation to a transformation matrix."""
        # Start with an identity matrix
        matrix_values = [1, 0, 0, 1, 0, 0]
        rotation_angle = 90  # 90 degrees
        center_x, center_y = 0, 0  # rotate around origin
        
        result = apply_rotation_to_matrix(matrix_values, rotation_angle, center_x, center_y)
        
        # After 90-degree rotation around origin, the matrix should be approximately:
        # [0, 1, -1, 0, 0, 0] (cos(90)=0, sin(90)=1)
        import math
        expected_cos = math.cos(math.radians(90))
        expected_sin = math.sin(math.radians(90))
        
        # Check that the result is close to expected values
        self.assertAlmostEqual(result[0], expected_cos, places=7)  # a = cos
        self.assertAlmostEqual(result[1], expected_sin, places=7)  # b = sin
        self.assertAlmostEqual(result[2], -expected_sin, places=7)  # c = -sin
        self.assertAlmostEqual(result[3], expected_cos, places=7)  # d = cos
        # e and f should remain 0 since rotation is around origin
        self.assertAlmostEqual(result[4], 0, places=7)
        self.assertAlmostEqual(result[5], 0, places=7)
    
    def test_apply_rotation_to_matrix_invalid(self):
        """Test applying rotation to an invalid matrix."""
        # Matrix with wrong number of values
        matrix_values = [1, 0, 0, 1, 0]  # Only 5 values instead of 6
        rotation_angle = 45
        center_x, center_y = 0, 0
        
        result = apply_rotation_to_matrix(matrix_values, rotation_angle, center_x, center_y)
        # Should return the original matrix unchanged
        self.assertEqual(result, matrix_values)


if __name__ == '__main__':
    unittest.main()