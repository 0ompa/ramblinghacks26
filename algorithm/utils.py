import math

def distance(p1, p2):
    """Euclidean distance between two (x, y) points."""
    if p1 is None or p2 is None:
        return 0.0
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def get_box_center(box):
    """Return center (x, y) of a bounding box (x1, y1, x2, y2)."""
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2, (y1 + y2) / 2)