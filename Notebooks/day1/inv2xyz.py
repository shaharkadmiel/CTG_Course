from obspy import read_inventory
from obspy.signal.util import util_geo_km
import numpy as np


def get_center(inv):
    """
    Calculate the center coordinates of an inventory object as the mean
    of all channel coordinates.
    """
    net = inv[0]
    lon, lat, elev = np.array(
        [(cha.longitude, cha.latitude, cha.elevation) for sta in net
         for cha in sta]
    ).T
    return lon.mean(), lat.mean(), elev.mean()


def get_offsets(inv):
    """
    Calculate the offset of each channel in an inventory object relative
    to the center and attach an ``x`` and ``y`` attributes to the
    channel.

    Parameters
    ----------
    inv : :class:`~obspy.core.inventory.inventory.Inventory` or str
        An :class:`~obspy.core.inventory.inventory.Inventory` object or
        a path (relative or absolute) to a StationXML file with the
        array elements.

    Returns
    -------
    x, y, x : list
        Offsets relative to array center coordinate.
    """
    if isinstance(inv, str):
        inv = read_inventory(inv)

    inv[0].stations.sort(key=lambda x: x.code)

    center_lon, center_lat, center_elev = get_center(inv)
    net = inv[0]
    x = []
    y = []
    z = []
    for sta in net:
        for cha in sta:
            x_, y_ = np.array(
                util_geo_km(center_lon, center_lat,
                            cha.longitude, cha.latitude)) * 1e3
        z_ = cha.elevation - center_elev

        x.append(x_)
        y.append(y_)
        z.append(z_)
    return x, y, z
