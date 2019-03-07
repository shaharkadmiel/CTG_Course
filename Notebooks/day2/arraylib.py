from itertools import combinations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patheffects import withStroke

from copy import deepcopy

from obspy import Inventory, read_inventory
from obspy.core.inventory import Station
from obspy.core.util import AttribDict
from obspy.geodetics import gps2dist_azimuth
from obspy.signal.util import util_geo_km


class Array(Inventory):
    """
    The Array class inherits the functionality and hierarchy of the
    :class:`~obspy.core.inventory.Inventory` class.

    This object can be conveniently initiated by selecting the desired
    stations and channels from an existing
    :class:`~obspy.core.inventory.Inventory` object.

    Example
    -------
    >>> inv = read_inventory()
    >>> array = Array(inv.select(network='GR', station='*',
    channel='*Z'))
    >>> print array
    Inventory created at 2017-03-17T11:17:20.149564Z
    Created by: ObsPy 1.0.2
                https://www.obspy.org
    Sending institution: Erdbebendienst Bayern
    Contains:
        Networks (1):
            GR
        Stations (2):
            GR.FUR (Fuerstenfeldbruck, Bavaria, GR-Net)
            GR.WET (Wettzell, Bavaria, GR-Net)
        Channels (7):
            GR.FUR..BHZ, GR.FUR..HHZ, GR.FUR..LHZ, GR.FUR..VHZ,
            GR.WET..BHZ, GR.WET..HHZ, GR.WET..LHZ
    ...

    """

    def __init__(self, inventory, name=None):
        """
        Parameters
        ----------
        inventory : :class:`~obspy.core.inventory.Inventory` or str
            An :class:`~obspy.core.inventory.Inventory` object
            containing the array elements (stations -> channels). The
            `inventory` object should contain one `network`. Optionally,
            pass a relative or absolute path to a `StationXML` file.

        name : str
            Name of the array. If not given, a name will be guessed from
            the network and station codes.
        """
        if isinstance(inventory, str):
            inventory = read_inventory(inventory)

        inventory[0].stations.sort(key=lambda x: x.code)

        super(Array, self).__init__(inventory.networks, inventory.source)

        self.network = self.networks[0].code

        self._stations_channels2elements()
        self.name = name or common_string(self.get_contents()['channels'])

        self._compute_offsets()  # sets x and y attribute to each element

    @property
    def nelements(self):
        """Number of array elements."""
        return len(self.get_contents()['channels'])

    def _stations_channels2elements(self):
        """
        Extract channel metadata and populate the `elements` attribute
        accounting for different metadata structures.

        Some structures (usually from IMS) already contain a separate
        `Station` object for each element. Others regard the array as
        one station with channels as elements.
        """
        if len(self[0].stations) == 1:
            station = self[0].stations[0]
            stations = []
            site = deepcopy(station.site)
            site_name = site.name
            for cha in station:
                site.name = 'Site {}, {}'.format(cha.location_code, site_name)
                station_ = Station(
                    station.code + cha.location_code,
                    cha.latitude, cha.longitude, cha.elevation,
                    channels=[deepcopy(cha)], site=deepcopy(site)
                )
                station_._original_code = station.code
                stations += [station_]
            self[0].stations = stations
            self.elements = self[0].stations
        else:
            self.elements = self[0].stations
            for ele in self.elements:
                ele._original_code = ele.code

    @property
    def aperture(self):
        """The array aperture is the largest inter-station distance."""
        aperture = 0
        for ele1, ele2 in combinations(self.elements, 2):
            dist = gps2dist_azimuth(ele1.latitude, ele1.longitude,
                                    ele2.latitude, ele2.longitude)[0]
            aperture = max(aperture, dist)
        return aperture

    @property
    def center(self):
        """Longitude, latitude, and elevation of the center of the array."""
        lon = np.mean([ele.longitude for ele in self.elements])
        lat = np.mean([ele.latitude for ele in self.elements])
        elev = np.mean([ele.elevation for ele in self.elements])
        return lon, lat, elev

    def _compute_offsets(self):
        """x, y offsets from array center in meters."""
        c_lon, c_lat, c_elev = self.center
        for i, ele in enumerate(self.elements):
            ele.x, ele.y = np.array(
                util_geo_km(c_lon, c_lat, ele.longitude, ele.latitude)) * 1e3
            ele.z = ele.elevation - c_elev

    @property
    def offsets(self):
        """
        Returns a dictionary with element code and x, y offsets from
        array center.
        """
        offsets = {}
        for ele in self.elements:
            offsets[ele.code] = {'x': ele.x, 'y': ele.y}
        return AttribDict(offsets)

    def get_coordinates(self, coordsys='xyz'):
        """
        Coordinates of array elements.

        Parameters
        ----------
        coordsys : {'xyz', 'lonlat'}
            x, y, z in meters (default), or longitude, latitude in
            degrees, elevation in meters.
        """
        coords = []
        if coordsys is 'lonlat':
            for ele in self.elements:
                coords += [
                    [ele.longitude, ele.latitude, ele.elevation]
                ]
        elif coordsys is 'xyz':
            for ele in self.elements:
                coords += [
                    [ele.x, ele.y, ele.elevation]
                ]
        else:
            raise ValueError(
                "{} not recognized. Please choose 'xyz', or 'lonlat'.".format(
                    coordsys
                )
            )

        return np.array(coords)

    def plot_array_geometry(self, ax=None, title=None, plot_center=True,
                            **kwargs):
        """
        Plot array geometry in Cartesian coordinates.

        Parameters
        ----------
        ax : None or :class:`~matplotlib.axes.Axes`
            Specify a :class:`~matplotlib.axes.Axes` instance to plot to.
            By default, plotting is done to the current active axes.

        title : str
            Override the default title which is the array name.

        plot_center : bool
            Set to ``False`` if undesired.

        .. rubric:: **Other useful keyword arguments are:**

        Keyword Arguments
        -----------------
        c : color, sequence, or sequence of color, default: ‘k’
            See :func:`~matplotlib.pyplot.scatter` for more details.

        marker : :class:`~matplotlib.markers.MarkerStyle'
            See :class:`~matplotlib.markers.MarkerStyle' for more
            information on the different styles of markers scatter
            supports.

        s : float
            size in points^2.

        Note
        ----
        *For other keyword arguments* see:
        :func:`~matplotlib.pyplot.scatter`.
        """
        if not ax:
            ax = plt.gca()
        fig = ax.figure

        if 'c' not in kwargs:
            kwargs['c'] = 'k'
        if 'marker' not in kwargs:
            kwargs['marker'] = '^'
        if 's' not in kwargs:
            kwargs['s'] = 50

        if plot_center:
            ax.scatter(0, 0, marker='+', c='k', lw=2, zorder=100)

        x_max, y_max = 0, 0
        string = common_string([ele.code for ele in self.elements])
        for name, values in self.offsets.items():
            x_max = max(x_max, abs(values.x))
            y_max = max(y_max, abs(values.y))

            ax.scatter(values.x, values.y, **kwargs)
            try:
                label = name.split(string)[-1]
            except ValueError:
                label = name

            ax.annotate(label,
                        xy=(values.x, values.y),
                        xycoords='data',
                        xytext=(0, np.sqrt(kwargs['s'])),
                        textcoords='offset points',
                        arrowprops=None,
                        ha='center', va='center')

        ax.set_aspect(1)
        ax.set_xlabel('x, m')
        ax.set_ylabel('y, m')
        title = title or self.name
        ax.set_title(title)

        r_max = max(x_max, y_max) * 1.1
        ax.axis((-r_max, r_max, -r_max, r_max))
        return fig, ax

    def plot_array_response(self, c_app=280, c_steps=50,
                            f_min=0.2, f_max=0.2, f_steps=1,
                            px_0=0, py_0=0, plot=True, title=None,
                            circles=(340, 3000), ax=None, ax_top=None,
                            cb_ax=None, cmap='inferno_r', **kwargs):
        """
        Calculate and plot the array transfer response functions.

        Parameters
        ----------
        c_app : float
            The extent of the slowness grid will span (-1/c_app, 1/c_app) in
            the x-direction and (-1/c_app, 1/c_app) in the y-direction.

        c_steps : int
            The number of slowness vectors in each direction.

        f_min, f_max : float
            The minimum and maximum frequency range to evaluate.

        f_steps : int
            The number of frequencies in the frequency range to evaluate.

        px_0, py_0 : float
            Offset the incoming direction. ``px_0=0, py_0=0`` means
            vertical incident planar wave.

        plot : bool
            Set to False to suppress plotting and only return the
            array response.

        title : str
            Override the default title which is the array name.

        circles : array-like
            Sequence of velocities to plot circles of in slowness space.

        ax : None or :class:`~matplotlib.axes.Axes`
            Specify a :class:`~matplotlib.axes.Axes` instance to plot
            the response to. By default, plotting is done to the current
            active axes.

        ax_top : None or :class:`~matplotlib.axes.Axes`
            If specified, the response at py=0 will be plotted to the
            specified axes.

        cb_ax : None or :class:`~matplotlib.axes.Axes`
            By default, the colorbar is drawn into a new axes created on
            the fly. Alternatively, specify a dedicated axes for the
            colorbar.

        Note
        ----
        *For other keyword arguments* see:
        :func:`~matplotlib.pyplot.imshow`.

        Returns
        -------
        response : :class:`~numpy.ndarray`
            A 2d array with the array response.
        """
        x, y, z = self.get_coordinates().T
        resp = get_array_response(x, y, c_app, c_steps,
                                  f_min, f_max, f_steps,
                                  px_0, py_0)

        if plot:
            if not ax:
                ax = plt.gca()

            title = title or 'Array response: {}-{} Hz, {} steps'.format(
                f_min, f_max, f_steps
            )

            s_bound = 1 / (c_app * 1.2)
            circles = circles

            extent = [-s_bound, s_bound] * 2
            im = ax.imshow(resp, cmap, vmin=0, vmax=1,
                           interpolation='bilinear', extent=extent)
            plt.colorbar(im, ax=ax, cax=cb_ax,
                         label='Normalized array response')

            if ax_top:
                ax_top.plot(np.linspace(-s_bound, s_bound, resp.shape[1]),
                            resp[resp.shape[1] // 2], 'k', lw=0.5)
                ax_top.set_ylim(0, 1)
                ax_top.set_title(title)
            else:
                ax.set_title(title)

            for i, circle in enumerate(circles):
                ax.add_patch(
                    plt.Circle(
                        (0, 0), radius=1 / circle, color='none',
                        ec='k', lw=0.5,
                        path_effects=[withStroke(foreground='w', linewidth=2)]
                    )
                )
            ax.axis(extent)

            ax.set_xlabel('px, s/m')
            ax.set_ylabel('py, s/m')
        return resp


def get_array_response(x, y, c_app=280, c_steps=50,
                       f_min=0.2, f_max=0.2, f_steps=1,
                       px_0=0, py_0=0):
    """
    Calculate array response on a square slowness grid for
    an arbitrary array of N elements.

    Parameters
    ----------
    x, y : array-like
        Offset coordinates of array elements relative to array center.

    c_app : float
        The extent of the slowness grid will span (-1/c_app, 1/c_app) in
        the x-direction and (-1/c_app, 1/c_app) in the y-direction.

    c_steps : int
        The number of slowness vectors in each direction.

    f_min, f_max : float
        The minimum and maximum frequency range to evaluate.

    f_steps : int
        The number of frequencies in the frequency range to evaluate.

    px_0, py_0 : float
        Offset the incoming direction. ``px_0=0, py_0=0`` means
        vertical incident planar wave.

    Returns
    -------
    response : :class:`~numpy.ndarray`
        A 2d array with the array response.
    """

    x = np.array(x)
    y = np.array(y)

    # construct the px, py square grid
    s_max = 1 / c_app
    px = np.linspace(-s_max, s_max, c_steps)
    py = np.linspace(-s_max, s_max, c_steps)
    px, py = np.meshgrid(px, py)

    # construct the omega vector and the complex i
    i = 1j
    omega = 2 * np.pi * np.linspace(f_min, f_max, f_steps)

    # construct the P(px, py) * r(x, y) product
    p_r_product = (
        (px[..., np.newaxis] - px_0) * x +
        (py[..., np.newaxis] - py_0) * y
    )

    # put it all together
    complex_part = -i * omega * p_r_product[..., np.newaxis]

    # calculate the response in one step!
    # note: shape of array is now (npx, npy, N, nf). Sum over N! then
    # normalize by N, take abs and square, then sum over f
    resp = np.sum(
        np.abs(
            np.sum(
                np.exp(complex_part), 2
            )  # / x.size  ## normalization here is not necessary
        )**2, 2
    )
    resp /= resp.max()
    return resp[::-1]


def common_string(strings):
    """
    Find the longest string that is a prefix of all the strings.
    """
    if not strings:
        return ''
    prefix = strings[0]
    for s in strings:
        if len(s) < len(prefix):
            prefix = prefix[:len(s)]
        if not prefix:
            return ''
        for i in range(len(prefix)):
            if prefix[i] != s[i]:
                prefix = prefix[:i]
                break
    return prefix
