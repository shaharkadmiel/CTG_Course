import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.patheffects import withStroke

from itertools import combinations
from inv2xyz import get_offsets
from array_response import get_array_response

import ipywidgets as widgets

out = widgets.Output()


class Array():
    """
    An interactive Array class for the purpose of array design and
    response experimentation.
    """
    def __init__(self, array=None):
        """
        Parameters
        ----------
        array : :class:`~obspy.core.inventory.inventory.Inventory` or str
            An :class:`~obspy.core.inventory.inventory.Inventory` object
            or a path (relative or absolute) to a StationXML file with
            the array elements.
        """

        global out

        self.array = array

        # set the plot
        fwidth = 7
        fheight = 4.5
        fig = plt.figure(figsize=(fwidth, fheight))
        wspace = 0.12
        hspace = 0.05
        left = 0.1
        right = 0.15
        bottom = 0.15
        ncols = 2
        width = (1 - left - right - wspace) / ncols
        aspect = fwidth / fheight
        height = width * aspect
        ax1 = fig.add_axes((left, bottom,
                            width, height))
        ax2 = fig.add_axes((left + width + wspace, bottom,
                            width, height))
        ax02 = fig.add_axes((left + width + wspace, bottom + height + hspace,
                             width, 0.2), sharex=ax2)
        plt.setp(ax02.get_xticklabels(), visible=False)

#         out.clear_output(wait=True)
        # array configuration
        ax1.set_xlabel('x, m')
        ax1.set_ylabel('y, m')

        ax1.scatter(0, 0, s=20, c='r', marker='+', zorder=2)
        if array is None:
            self.r_max = 1500
            self.points = ax1.scatter(
                self.r_max, self.r_max, s=7, c='k', zorder=2
            )
            self.x = []
            self.y = []
        else:
            self.x, self.y, z = get_offsets(array)
            self.r_max = np.max(np.abs(self.x + self.y)) * 1.5
            self.points = ax1.scatter(self.x, self.y, s=7, c='k', zorder=2)

        self.cid = self.points.figure.canvas.mpl_connect(
            'button_press_event', self)

        ax1.axis([-self.r_max, self.r_max] * 2)
        ax1.xaxis.set_major_locator(
            MultipleLocator(round(self.r_max / 2, -1))
        )
        ax1.xaxis.set_minor_locator(
            MultipleLocator(self.r_max / 20)
        )
        ax1.yaxis.set_major_locator(
            MultipleLocator(round(self.r_max / 2, -1))
        )
        ax1.yaxis.set_minor_locator(
            MultipleLocator(self.r_max / 20)
        )

        ax1.grid(True, which='both', c='0.9', lw=0.5, zorder=0)

        self.ax1_title = ('Array configuration\n'
                          'N={} elements, Aperture={:.0f} m')
        ax1.set_title(self.ax1_title.format(self.nelements, self.aperture))

        # array response
        self.label = 'Normalized array response'

        ax2.set_xlabel('px, s/m')
        ax2.set_ylabel('py, s/m')

        cbx = fig.add_axes((left + 2 * width + wspace + 0.015, bottom,
                            0.015, height))
        cb = None

        ax02.set_title('Array response')
        ax02.set_ylim(0, 1)

        self.fig, self.ax1, self.ax2, self.ax02, self.cbx, self.cb = (
            fig, ax1, ax2, ax02, cbx, cb
        )

        self.plot_array_response()

        print('Click in the left frame to add array elements...')

    @out.capture(clear_output=True, wait=True)
    def __call__(self, event):
        if event.inaxes != self.points.axes:
            print('Clicked outside the axes')
            return

        print('Adding element x: {:.2f}, y: {:.2f}'.format(
            event.xdata, event.ydata))

        # update array
        self.x.append(event.xdata)
        self.y.append(event.ydata)

        self.points.set_offsets(list(zip(self.x, self.y)))
        self.points.figure.canvas.draw()

        self.ax1.set_title(
            self.ax1_title.format(self.nelements, self.aperture)
        )

        # update response
        self.plot_array_response(
            self.c_app, self.c_steps,
            self.f_min, self.f_max, self.f_steps,
            self.log, self.cmap
        )

    @property
    def aperture(self):
        aperture = 0
        for (x1, y1), (x2, y2) in combinations(zip(self.x, self.y), 2):
            dist = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            aperture = max(aperture, dist)
        return aperture

    @property
    def nelements(self):
        return len(self.x)

    @out.capture(clear_output=True, wait=True)
    def plot_array_response(self, c_app=280, c_steps=50,
                            f_min=0.2, f_max=0.2, f_steps=1,
                            log=False, cmap='inferno_r', circles=(340, 3000)):
        """
        Get the array transfer response functions.

        Parameters
        ----------
        c_app, c_step : float
            Minimum velocity and resolution (in meters). The velocity at
            which the wavefront propagates through the array.

        f_min, f_max, f_step : float
            Frequency parameters in Hz. Array response can either be
            modeled using a center frequency of a narrow-band signal or
            a range of frequencies for a broadband signal not accounting
            for instrument response or frequency contents of signal.

        cmap : str or :class:`~matplotlib.cm.ScalarMappable`
            Colormap to use for plotting.

        circles : array-like
            Sequence of velocities to plot circles of in slowness space.
        """
        self.c_app = c_app
        self.c_steps = c_steps
        self.f_min = f_min
        self.f_max = f_max
        self.f_steps = f_steps
        self.log = log
        self.cmap = cmap

        self.s_bound = 1 / (self.c_app * 1.2)
        self.circles = circles

        extent = [-self.s_bound, self.s_bound] * 2

        try:
            self.im
            resp = get_array_response(
                self.x, self.y, self.c_app * 1.2, self.c_steps,
                self.f_min, self.f_max, self.f_steps
            )
        except AttributeError:  # initialize
            resp = get_array_response(
                self.x, self.y, self.c_app * 1.2, self.c_steps,
                self.f_min, self.f_max, self.f_steps
            )
            # resp = np.zeros((2, 2))
            self.im = self.ax2.imshow(
                resp, self.cmap, vmin=0, vmax=1,
                interpolation='bilinear', extent=extent
            )
            self.cb = plt.colorbar(self.im, cax=self.cbx, label=self.label,
                                   fraction=0.1)
            self.l, = self.ax02.plot(
                np.linspace(-self.s_bound, self.s_bound, resp.shape[1]),
                resp[resp.shape[1] // 2], 'k', lw=0.5
            )
            self.ax2.axis(extent)

            for i, circle in enumerate(self.circles):
                vars(self)['circle' + str(i)] = plt.Circle(
                    (0, 0), radius=1 / circle, color='none', ec='k', lw=0.5,
                    path_effects=[withStroke(foreground='w', linewidth=2)])
                self.ax2.add_patch(vars(self)['circle' + str(i)])

        if len(self.x) < 2:  # no array response for less than 2 elements
            return

        # resp = get_array_response(
        #     self.x, self.y, self.c_app * 1.2, self.c_steps,
        #     self.f_min, self.f_max, self.f_steps
        # )

        if log:
            resp = 20 * np.log10(resp)
            self.label = 'Log of normalized array response, dB'
        else:
            self.label = 'Normalized array response'

        self.resp = resp

        vmin, vmax = resp.min(), resp.max()
        vmin = 0 if self.log is False else vmin

        self.im.set_array(resp)
        self.im.set_clim(vmin, vmax)
        self.im.set_cmap(cmap)
        self.im.set_extent(extent)
        self.ax2.axis(extent)
        self.im.figure.canvas.draw()
        self.cb.update_bruteforce(self.im)
        self.cb.set_label(self.label)
        self.l.set_data(
            (np.linspace(-self.s_bound, self.s_bound, self.resp.shape[1]),
             self.resp[self.resp.shape[1] // 2])
        )
        pad = (vmax - vmin) * 0.1
        self.ax02.set_ylim(vmin - pad, vmax + pad)

    @out.capture(clear_output=True, wait=True)
    def __update_widget__(self, c_app, c_steps, fminmax, f_steps,
                          log, cmap):
        f_min, f_max = fminmax
        self.plot_array_response(
            c_app, c_steps, f_min, f_max, f_steps, log, cmap
        )

    @out.capture(clear_output=True, wait=True)
    def __reset__(self, event):
        print('Resetting...')
        self.points.remove()
        self.points = self.ax1.scatter(
            self.ax1.get_xlim()[1], self.ax1.get_ylim()[1],
            s=7, c='k', zorder=2
        )
        self.x = []
        self.y = []
        self.cid = self.points.figure.canvas.mpl_connect(
            'button_press_event', self)
        self.resp[...] = 0
        self.im.set_array(self.resp)
        self.l.set_data(
            (np.linspace(-self.s_bound, self.s_bound, self.resp.shape[1]),
             self.resp[self.resp.shape[1] // 2])
        )
        out.clear_output(wait=True)

    @out.capture(clear_output=True, wait=True)
    def __center__(self, event):
        print('Centering...')
        x = np.array(self.x)
        self.x = list(x - x.mean())
        y = np.array(self.y)
        self.y = list(y - y.mean())

        self.r_max = np.max(np.abs(self.x + self.y)) * 1.5

        self.points.set_offsets(list(zip(self.x, self.y)))
        self.points.figure.canvas.draw()

        self.ax1.axis([-self.r_max, self.r_max] * 2)
        self.ax1.xaxis.set_major_locator(
            MultipleLocator(round(self.r_max / 2, -1))
        )
        self.ax1.xaxis.set_minor_locator(
            MultipleLocator(self.r_max / 20)
        )
        self.ax1.yaxis.set_major_locator(
            MultipleLocator(round(self.r_max / 2, -1))
        )
        self.ax1.yaxis.set_minor_locator(
            MultipleLocator(self.r_max / 20)
        )
        out.clear_output(wait=True)

