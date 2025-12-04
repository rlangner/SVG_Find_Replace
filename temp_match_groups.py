def match_groups(input_groups: List[Element], find_groups: Dict[str, Element]) -> List[Tuple[List[Element], str]]:
    """
    Match input groups to find groups based on visual content.
    This function compares path elements found in the subgroups of find_ groups
    to path elements in input.svg. Once path elements that look the same are found,
    it looks for groups around the path element group that match the structure 
    of the subgroups of the find_ group.
    Returns a list of tuples: (matched_input_groups, find_group_id)
    """
    matches = []

    for find_id, find_group in find_groups.items():
        print(f"Looking for match for find group {find_id}")
        
        # Extract path elements from find group subgroups
        find_path_elements = []
        for subgroup in find_group:  # Direct children of the find group
            find_path_elements.extend(extract_path_elements(subgroup))
        
        print(f"Find group {find_id} has {len(find_path_elements)} path elements in its subgroups")
        
        if not find_path_elements:
            print(f"No path elements found in find group {find_id}, skipping...")
            continue

        # Normalize all path elements in the find group for comparison
        normalized_find_paths = [normalize_path_content(path_elem) for path_elem in find_path_elements]
        normalized_find_paths_set = set(normalized_find_paths)
        
        print(f"Normalized find path elements: {len(normalized_find_paths)} unique elements")

        # Look for input groups that contain path elements matching the find group
        matching_input_groups = []
        
        for input_group in input_groups:
            # Extract path elements from this input group
            input_path_elements = extract_path_elements(input_group)
            if not input_path_elements:
                continue
                
            # Normalize path elements in this input group
            normalized_input_paths = [normalize_path_content(path_elem) for path_elem in input_path_elements]
            normalized_input_paths_set = set(normalized_input_paths)
            
            # Check if the input group has the same path elements as the find group
            if normalized_find_paths_set == normalized_input_paths_set:
                print(f"Found matching input group with same path elements as {find_id}")
                matching_input_groups.append(input_group)
            elif len(normalized_find_paths_set.intersection(normalized_input_paths_set)) == len(normalized_find_paths_set):
                # If the input group contains all the path elements from the find group (and maybe more)
                print(f"Found matching input group containing find group path elements: {find_id}")
                matching_input_groups.append(input_group)

        print(f"Found {len(matching_input_groups)} candidate input groups for {find_id}")
        
        # Add the matching groups to results
        for input_group in matching_input_groups:
            print(f"Adding match: input group with id {input_group.get('id', 'no_id')} matches {find_id}")
            matches.append(([input_group], find_id))

    return matches