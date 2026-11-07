from infx.workflows.hardware_literals import literal_hits, literal_pattern


def test_counts_whole_keys_in_code_strings_only():
    source = '''
"""Runs on b200."""
LANES = {"gb200-nv": 1, "gh200x": 2, "b2000": 3}
GLOB = f"*{name}-h100_*"
HELP = "e.g. h200, mi355x"
# b300
'''
    pattern = literal_pattern({"b200", "gb200", "h100", "h200", "mi355x", "b300"})

    assert literal_hits(source, pattern) == [
        (3, "gb200-nv"),
        (4, "-h100_*"),
        (5, "e.g. h200, mi355x"),
        (5, "e.g. h200, mi355x"),
    ]
