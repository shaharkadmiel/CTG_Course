# -*- coding: utf-8 -*-
"""
Python module for backprojection of travel-time - back-azimuth of
infrasound detections to trace coupling points at the ground-atmosphere
interface.

.. module:: backproject

:author:
    Shahar Shani-Kadmiel (S.Shani-Kadmiel@tudelft.nl)

:copyright:
    Shahar Shani-Kadmiel

:license:
    This code is distributed under the terms of the
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
"""

import numpy as np
from warnings import warn


def backproject(source, station, T_d, BAZ_d, extent, dh=0.05,
                c_s=3., c_i=0.3, smooth=3):
    """
    Backproject travel-time - back-azimuth of infrasound detections to
    trace coupling points at the ground-atmosphere interface.

    Parameters
    ----------
    source : tuple
        (lon, lat, depth [km]) of the source.

    station : tuple
        (lon, lat, elevation [km]) of the station.

    T_d : array-like
        Travel-time of detections.

    BAZ_d : array-like
        Back-Azimuth of detections.

    extent : tuple
        (lon_min, lon_max, lat_min, latmax) of the grid search domain.

    dh : float
        Spacing of the grid search, degrees.

    c_s : float
        Seismic propagation velocity.

    c_i : float
        Infrasonic propagation velocity.

    smooth : int
        Number of neighboring grid points to average over.

    Returns
    -------
    lons, lats : 2x 2d :class:`~numpy.ndarray`
        Coordinates of the coupling locations.
    """
    w, e, s, n = extent
    lons_, lats_ = np.meshgrid(np.arange(w, e, dh),
                               np.arange(n, s, -dh))

    # Seismic tt
    source_lon, source_lat = source[:2]
    try:
        depth = source[2]
    except IndexError:
        depth = 0

    t_s = (
        np.sqrt(
            depth**2 + ll2dist(source_lon, source_lat,
                               lons_, lats_)**2
        ) / c_s
    )

    # Infrasonic tt
    station_lon, station_lat = station[:2]
    t_i = (ll2dist(station_lon, station_lat, lons_, lats_) / c_i)

    # Total tt
    t = t_s + t_i

    # Back azimuth
    baz = ll2az(station_lon, station_lat, lons_, lats_)

    coupling_locations = np.empty((T_d.size, 2))
    coupling_locations.fill(-999)
    for i, (baz_d, t_d) in enumerate(zip(BAZ_d, T_d)):
        # check if time of detection places it outside the grid search and
        # issue a warning
        if (t_d > t).all():
            warn(
                ('Grid search domain not large enough: Detection at {} '
                 'seconds is outside the domain. Skipping...').format(t_d)
            )
            continue

        # rotate baz by baz_d so that the discontinuity is opposite baz_d
        baz_ = _rotate_baz_grid(baz, baz_d)

        M_BAZ = np.abs(baz_ - baz_d)
        M_T = np.abs(t - t_d)
        M = M_BAZ * M_T

        loc = M.argmin()
        coupling_locations[i, :] = lons_.flat[loc], lats_.flat[loc]
    return np.ma.masked_equal(coupling_locations.T, -999)


def _rotate_baz_grid(baz, baz_d):
    """
    Rotate `baz` grid by `baz_d` so that the discontinuity is opposite
    `baz_d`.

    Back azimuth detections with values close to due-north create a
    a large discontinuity around the 360 to 0 boundary, which introduces
    artifacts. This helper function makes sure the discontinuity is
    opposite (180 degrees away from) the detection and is no longer
    discontinuous.

    A copy of `baz` is returned.
    """
    baz_ = baz.copy()
    if baz_d > 180:
        baz_[baz_ < baz_d - 180] += 360
    else:
        baz_[baz_ > baz_d + 180] -= 360
    return baz_


########################## distance utility functions ########################

deg2rad = np.pi / 180


def ll2az(lon1, lat1, lon2, lat2):
    """
    Long./Lat. to Azimuth. Calculate the element-wise azimuth from
    P1(``lon1``, ``lat1``) to P2(``lon2``, ``lat2``).

    Parameters
    ----------
    lat1 : float
        Latitude of P1 in degrees (positive for northern,
        negative for southern hemisphere)

    lon1 : float
        Longitude of P1 in degrees (positive for eastern,
        negative for western hemisphere)

    lat2 : float
        Latitude of P2 in degrees (positive for northern,
        negative for southern hemisphere)

    lon2 : float
        Longitude of P2 in degrees (positive for eastern,
        negative for western hemisphere)

    Returns
    -------
    float or :class:`~numpy.ndarray`
        Azimuth in degrees relative to north.
    """

    # Convert to radians.
    lat1 = lat1 * deg2rad
    lat2 = lat2 * deg2rad
    lon1 = lon1 * deg2rad
    lon2 = lon2 * deg2rad

    lon_diff = lon2 - lon1

    az = np.arctan2(np.sin(lon_diff),
                    np.cos(lat1) * np.tan(lat2) -
                    np.sin(lat1) * np.cos(lon_diff))
    return (az / deg2rad) % 360


def ll2dist(lon1, lat1, lon2, lat2, a=6370.997, formula='haversine'):
    """
    Convenience function to calculate the great circle distance between
    two points on a spherical Earth.

    Parameters
    ----------
    lat1 : float
        Latitude of P1 in degrees (positive for northern,
        negative for southern hemisphere)

    lon1 : float
        Longitude of P1 in degrees (positive for eastern,
        negative for western hemisphere)

    lat2 : float
        Latitude of P2 in degrees (positive for northern,
        negative for southern hemisphere)

    lon2 : float
        Longitude of P2 in degrees (positive for eastern,
        negative for western hemisphere)

    a : float
        Radius of Earth in km. Set to :math:`\pi/180` to get distance in
        degrees instead of km.

    formula : {'haversine', 'vincenty'}
        The haversine formula (default) is faster and more accurate than
        the Vincenty formula for small distances. However, for nearly
        antipodal distances, the Vincenty formula is more accurate.

    Returns
    -------
    distance : float or :class:`~numpy.ndarray`
        Great circle distance in km on a sphere.
    """

    # Convert to radians.
    lat1 = lat1 * deg2rad
    lat2 = lat2 * deg2rad
    lon1 = lon1 * deg2rad
    lon2 = lon2 * deg2rad

    lon_diff = lon2 - lon1

    # Vincenty formula
    if formula == 'vincenty':
        dist = (
            a * np.arctan2(
                np.sqrt(
                    (np.cos(lat2) * np.sin(lon_diff))**2 +
                    (np.cos(lat1) * np.sin(lat2) - np.sin(lat1) *
                     np.cos(lat2) * np.cos(lon_diff))**2
                ),
                np.sin(lat1) * np.sin(lat2) + np.cos(lat1) * np.cos(lat2) *
                np.cos(lon_diff)
            )
        )

    # haversine formula is faster and just as accurate as the Vincenty formula
    else:
        lat_diff = lat2 - lat1
        dist = (
            a * 2 * np.arcsin(
                np.sqrt(
                    np.sin(lat_diff / 2)**2 +
                    np.cos(lat1) * np.cos(lat2) *
                    np.sin(lon_diff / 2)**2
                )
            )
        )
    return dist
