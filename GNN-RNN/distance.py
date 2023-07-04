from math import radians,sin,cos,asin,sqrt,fabs

EARTH_RADIUS = 6371  # Earth's mean radius


def hav(theta):
    s = sin(theta / 2)
    return s * s


def get_distance_hav(lng0, lat0, lng1, lat1):
    # haversine formula
    # -->radian
    lat0 = radians(lat0)
    lat1 = radians(lat1)
    lng0 = radians(lng0)
    lng1 = radians(lng1)

    dlng = fabs(lng0 - lng1)
    dlat = fabs(lat0 - lat1)
    h = hav(dlat) + cos(lat0) * cos(lat1) * hav(dlng)
    distance = 2 * EARTH_RADIUS * asin(sqrt(h))

    return distance
