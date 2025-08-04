from pathlib import Path
from typing import List, Tuple


def longest_common_substring(s1: str, s2: str) -> str:
    """
    Finds the longest common substring between two strings.
    
    Args:
        s1: First string
        s2: Second string
        
    Returns:
        The longest common substring between s1 and s2
    """
    m = [[0] * (1 + len(s2)) for _ in range(1 + len(s1))]
    longest, x_longest = 0, 0
    for x in range(1, 1 + len(s1)):
        for y in range(1, 1 + len(s2)):
            if s1[x - 1] == s2[y - 1]:
                m[x][y] = m[x - 1][y - 1] + 1
                if m[x][y] > longest:
                    longest = m[x][y]
                    x_longest = x
            else:
                m[x][y] = 0
    return s1[x_longest - longest: x_longest]


def match_files_by_lcs(files1: List[Path], files2: List[Path]) -> List[Tuple[Path, Path]]:
    """
    Match files from two lists using longest common substring algorithm.
    
    Args:
        files1: First list of file paths
        files2: Second list of file paths
        
    Returns:
        List of tuples containing matched file pairs (file1, file2)
    """
    # Determine the smaller list to be the "root" for matching
    root_files = files1
    other_files = files2
    # Swap if files2 are fewer, to iterate over the smaller list
    if len(files2) < len(files1):
        root_files, other_files = other_files, root_files

    potential_matches = []
    for root_file in root_files:
        best_match = None
        max_lcs = -1
        for other_file in other_files:
            lcs = longest_common_substring(root_file.stem, other_file.stem)
            if len(lcs) > max_lcs:
                max_lcs = len(lcs)
                best_match = other_file
        
        if best_match:
            # The order of the pair in the tuple depends on which list was the root
            if len(files2) < len(files1):
                 # root is files2, so (file2, file1) -> store as (file1, file2)
                potential_matches.append((max_lcs, best_match, root_file))
            else:
                # root is files1, so (file1, file2)
                potential_matches.append((max_lcs, root_file, best_match))

    # Sort potential matches by LCS score, descending
    potential_matches.sort(key=lambda x: x[0], reverse=True)

    matched_pairs = []
    used_files1 = set()
    used_files2 = set()

    for _, file1, file2 in potential_matches:
        if file1 not in used_files1 and file2 not in used_files2:
            matched_pairs.append((file1, file2))
            used_files1.add(file1)
            used_files2.add(file2)
    
    return matched_pairs 