import math

def calculate_angle(a, b, c):
    """
    Calculates angle at point b given three points a, b, c.
    Points are in (x, y) format.
    """
    
    ba = (a[0] - b[0], a[1] - b[1])
    bc = (c[0] - b[0], c[1] - b[1])


    dot = ba[0]*bc[0] + ba[1]*bc[1]
    mag_ba = math.hypot(ba[0], ba[1])
    mag_bc = math.hypot(bc[0], bc[1])

    if mag_ba * mag_bc == 0:
        return 0

    
    angle = math.acos(dot / (mag_ba * mag_bc))
    return math.degrees(angle)
