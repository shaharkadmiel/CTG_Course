# -*- coding: utf-8 -*-
"""
Python module for retrieving and handling atmospheric parameters.

.. module:: atmos

:author:
    Shahar Shani-Kadmiel (S.Shani-Kadmiel@tudelft.nl)

:copyright:
    Shahar Shani-Kadmiel

:license:
    This code is distributed under the terms of the
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
"""
from __future__ import absolute_import, print_function, division

import matplotlib.pyplot as plt
import numpy as np

import msise00
from pyhwm2014 import HWM14  # U = zonal, V = meridional


deg2rad = np.pi / 180


def get_c_adiabatic(T):
    """
    Compute the adiabatic sound speed as a function of temperature T in
    degrees K(elvin).
    """
    return np.sqrt(402.8 * T)


def get_c_eff(c_T, W_zon, W_mer, az):
    """
    Compute the effective speed of sound accounting for the temperature
    and horizontal wind vector in a given direction.

    If using ``pyhwm2014`` U = zonal, V = meridional
    """
    az_rad = az * deg2rad
    return (c_T +
            W_mer * np.cos(az_rad) +
            W_zon * np.sin(az_rad))


def get_c_eff_avg(c_eff, alt_vector, altmin=40, altmax=60):
    """
    Calculate the average speed of sound in the given altitude range.
    """

    return c_eff[..., (alt_vector >= altmin) *
                 (alt_vector <= altmax)].mean(axis=-1)


def get_atmos(time, lat, lon, altitude=(0, 140, 1), apindex=35):
    """
    Retrieve atmospheric conditions (temperature and wind) from MSISE-00
    and HWM2014 models.
    """
    alt_vector = np.arange(*altitude)
    Tn = msise00.run(time, altkm=alt_vector,
                     glat=lat, glon=lon).Tn.data.ravel()
    wind = HWM14(altlim=[alt_vector[0], alt_vector[-1]], altstp=altitude[2],
                 glat=lat, glon=lon, ap=[-1, apindex],
                 ut=time.hour, day=time.julday, year=time.year, verbose=0)
    W_zonal, W_meridional = np.array([wind.Uwind, wind.Vwind])

    return alt_vector, Tn, W_zonal, W_meridional


def plot_atmos_params(alt_vector, Tn, W_zonal, W_meridional):
    """
    Plot atmospheric parameters.

    Example
    -------
    >>> origintime = UTCDateTime('20180123093141')
    >>> station_lat, station_lon = (64.866929, -147.85810087500002)
    >>> source_lat, source_lon, source_depth = (56.046, -149.073, 25.0)
    >>> distance = ll2dist(station_lon, station_lat, source_lon, source_lat)
    >>> true_az = ll2az(source_lon, source_lat, station_lon, station_lat)
    >>> true_baz = ll2az(station_lon, station_lat, source_lon, source_lat)

    >>> alt_vector, Tn, W_zonal, W_meridional = get_atmos(
            origintime, station_lat, station_lon, apindex=35)

    >>> plot_atmos_params(alt_vector, Tn, W_zonal, W_meridional)
    """

    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(ncols=5, sharey=True,
                                                  figsize=(8, 3))
    ax1.plot(Tn, alt_vector, 'k')
    ax1.set_ylabel('Altitude, km')
    ax1.set_xlabel('T, K')
    ax1.set_title('Temperature')
    ax1.set_ylim(alt_vector[0], alt_vector[-1])

    c_T = get_c_adiabatic(Tn)
    ax2.plot(c_T, alt_vector, 'k')
    ax2.set_xlabel('$\mathrm{C_T}$, m/s')
    ax2.set_title('Adiabatic\nSound Speed')
    ax2.axvline(c_T[0], color='k', linestyle='dashed', zorder=0)

    ax3.plot(W_meridional, alt_vector, label='N')
    ax3.plot(W_zonal, alt_vector, label='E')
    ax3.set_xlabel('$\mathrm{C_W}$, m/s')
    ax3.set_title('Zonal & Meridional\nWinds')
    ax3.legend(loc=0, handlelength=1, frameon=False)

    c_eff_meridional = get_c_eff(c_T, W_zonal, W_meridional, 0)
    c_eff_zonal = get_c_eff(c_T, W_zonal, W_meridional, 90)

    altmin = 35
    altmax = 65
    ax4.axhspan(altmin, altmax, color='0.8', zorder=0, lw=0)
    c_mer_avg = get_c_eff_avg(c_eff_meridional, alt_vector, altmin, altmax)
    c_zon_avg = get_c_eff_avg(c_eff_zonal, alt_vector, altmin, altmax)

    ax4.plot(c_eff_meridional, alt_vector, label='N {:.0f}'.format(c_mer_avg))
    # ax4.fill_betweenx(alt_vector, c_eff_meridional, c_eff_meridional[0],
    #                   where=(c_eff_meridional > c_eff_meridional[0]),
    #                   alpha=0.3)
    ax4.plot(c_eff_zonal, alt_vector, label='E {:.0f}'.format(c_zon_avg))
    # ax4.fill_betweenx(alt_vector, c_eff_zonal, c_eff_zonal[0],
    #                   where=(c_eff_zonal > c_eff_zonal[0]),
    #                   alpha=0.3)

    # ax4.plot(c_T + W_meridional, alt_vector, 'kx', markevery=10, label='N')
    # ax4.plot(c_T + W_zonal, alt_vector, 'ko', markevery=10, mfc='none',
    # label='E')

    ax4.set_xlabel('$\mathrm{C_{eff}}$, m/s')
    ax4.legend(loc=0, handlelength=1, frameon=False)
    ax4.set_title('Effective\nSound Speed')
    ax4.axvline(c_eff_zonal[0], color='k', linestyle='dashed', zorder=0)

    c_eff_meridional_ratio = c_eff_meridional / c_eff_meridional[0]
    c_eff_zonal_ratio = c_eff_zonal / c_eff_zonal[0]

    ax5.plot(c_eff_meridional_ratio >= 1, alt_vector, label='N')
    ax5.plot(c_eff_zonal_ratio >= 1, alt_vector, label='E')
    ax5.set_xlabel('$\mathrm{C_{eff}}$/$\mathrm{C_0} >= 1$')
    ax5.set_title('$\mathrm{C_{eff}}$ ratio >= 1')

    return fig, (ax1, ax2, ax3, ax4, ax5)
