# -*- coding: utf-8 -*-
"""
Grid generation routines for pysabeam.

.. module:: grid

:author:
    Shahar Shani-Kadmiel (S.Shani-Kadmiel@tudelft.nl)

:copyright:
    Shahar Shani-Kadmiel

:license:
    This code is distributed under the terms of the
    GNU General Public License, Version 3
    (https://www.gnu.org/licenses/gpl-3.0.en.html)
"""
from __future__ import absolute_import, print_function, division

import matplotlib.pyplot as plt
import numpy as np

from copy import deepcopy

deg2rad = np.pi / 180


def pxpy2theta_app_vel(px, py):
    """
    Transform `px`, `py` coordinates in slowness space to theta and
    apparent velocity.
    """
    theta = (np.arctan2(px, py) / deg2rad) % 360
    app_vel = 1. / np.sqrt(px**2 + py**2)
    return theta, app_vel


def theta_app_vel2pxpy(theta, app_vel):
    """
    Transform `theta`, `app_vel` to px, py coordinates in slowness
    space.
    """
    slowness = 1 / app_vel
    theta *= deg2rad
    return slowness * np.sin(theta), slowness * np.cos(theta)


SCATTER_ARGS = dict(c='k', s=2, marker='.', zorder=0, label='Slowness vector')


class Grid():
    """
    Slowness grid class.
    """
    def __init__(self, app_vel_params=(280, 6000, 10, 450, 3),
                 theta_params=(0, 360, 2), grid_shape='doughnut',
                 verbose=True):
        """
        Compose a grid of all slowness vectors in px, py space.

        The grid can be 'doughnut' or 'square', evenly spaced or log-spaced,
        or any combination by adding several grids together.

        Parameters
        ----------
        app_vel_params : tuple or None
            The apparent velocity vector can be constructed as an evenly
            spaced array or a two-part lin-log array (evenly spaced upto
            a `linear_max`, and evenly log-spaced beyond that).

            - if 2 element tuple: (min, steps)
                `steps` px, py, vectors evenly spaced in slowness space
                in the range -1/`min`, 1/`min`. Note: this results in a
                square slowness grid regardless of `grid_shape` and
                `theta_params` are ignored.

            - if 3 element tuple: (min, max, step)
                evenly space app_vel vector.

            - if 5 element tuple (default):
              (min, max, step, linear_max, step_factor)

                - linear part evenly spaced at `step` between
                    `min` and `linear_max`.

                - logarithmic part evenly log-spaced between `linear_max`
                    and `max`. The first step in the log part is
                    `step` * `step_factor` and gradually increases after
                    that.

            If `None`, an empty :class:`~.Grid` object is returned.

        theta_params : tuple or None
            (theta_min, theta_max, theta_step) This should be self
            explanatory.

            If `None`, an empty :class:`~.Grid` object ir returned.

        grid_shape : {'doughnut', 'square'}
            The shape of the grid. By default, a ``'doughnut'`` grid of
            apparent velocity and theta is generated. A gridsearch over
            a doughnut grid is more efficient than over a ``'square'``
            grid as many points which are in the corners, outside the
            minimum apparent velocity value are not computed.
        """
        # return an empty Grid object
        if app_vel_params is None or theta_params is None:
            return

        self.app_vel_params = app_vel_params
        self.theta_params = theta_params
        self.unique = None

        self.grid_shape = grid_shape
        print_string = '{} grid\n{}-----\n'

        string_template = '{:>40}{:>8}{:>12}{:>12}\n'
        print_string += string_template.format('min', 'max', 'step', 'points')
        print_string += '{:>30} :\n'.format('Apparent velocity params, m/s')

        # Construct the Apparent velocity vector:
        # lin-log velocity vector
        try:
            (app_vel_min, app_vel_max, app_vel_step, linear_max,
             app_vel_step_factor) = [float(item) for item in app_vel_params]

            linear_part = np.arange(app_vel_min, linear_max, app_vel_step)
            linear_part_size = linear_part.size

            print_string += (string_template.format(
                app_vel_min, linear_max, app_vel_step, linear_part_size))

            # figure out the number of points in the log part
            delta = app_vel_max - linear_max
            alpha = app_vel_step / linear_max
            logarithmic_part_size = int(
                np.log((delta / linear_max)) /
                (alpha * app_vel_step_factor)
            )

            logarithmic_part = np.logspace(
                np.log(linear_max),
                np.log(app_vel_max),
                logarithmic_part_size,
                base=np.e)

            log_part_step = np.gradient(logarithmic_part)

            app_vel_vector = np.hstack((linear_part, logarithmic_part))
            print_string += (string_template.format(
                linear_max, app_vel_max, '{:.0f}->{:.0f}'.format(
                    log_part_step.min(), log_part_step.max()
                ), logarithmic_part_size))

            self.app_vel_min = app_vel_min
            self.app_vel_max = app_vel_max
            self.linear_max = linear_max

        # linear velocity vector
        except ValueError:
            try:
                (app_vel_min, app_vel_max,
                 app_vel_step) = [float(item) for item in app_vel_params]

                app_vel_vector = np.arange(
                    app_vel_min, app_vel_max + app_vel_step, app_vel_step)

                print_string += (string_template.format(
                    app_vel_min, app_vel_max, app_vel_step,
                    app_vel_vector.size))

                self.app_vel_min = app_vel_min
                self.app_vel_max = app_vel_max
                self.linear_max = None
            # square grid
            except ValueError:
                c_min, s_steps = app_vel_params
                s_min = 1 / c_min
                self.grid_shape = 'square'
                self.slownesses = np.linspace(-s_min, s_min, s_steps)
                app_vel_vector = 1 / self.slownesses
                theta_params = -1

                print_string += (string_template.format(
                    np.abs(app_vel_vector).min().round(),
                    np.abs(app_vel_vector).max().round(),
                    '---', s_steps))

                self.app_vel_min = c_min
                self.app_vel_max = None
                self.linear_max = None

        self.app_vel_vector = app_vel_vector

        # Construct the theta vector:
        try:
            theta_min, theta_max, theta_step = theta_params
            theta_vector = np.unique(
                np.arange(theta_min, theta_max + theta_step, theta_step) % 360
            ) * deg2rad

            self.theta_vector = theta_vector

            print_string += '{:>30} :\n'.format('Theta params, degrees')
            print_string += (
                string_template.format(
                    theta_min, theta_max, theta_step, theta_vector.size))
        except TypeError:
            self.theta_vector = None

        self.print_string = print_string

        if self.grid_shape == 'doughnut':
            self._doughnut_grid()
        else:
            self._square_grid()
            # raise NotImplementedError("At the moment only 'doughnut' grid"
            #                           "shape is implemented")

        if verbose:
            print(self, flush=True)

    def _doughnut_grid(self):
        """
        A private helper function to generate a slowness grid from doughnut
        theta, apperent velocity vectors.
        """
        # init the slowness grid
        self.slownesses = 1 / self.app_vel_vector
        # ss, tt = np.meshgrid(slownesses, self.theta_vector)

        px = self.slownesses * np.sin(self.theta_vector[..., np.newaxis])
        py = self.slownesses * np.cos(self.theta_vector[..., np.newaxis])

        self.px = px.ravel()
        self.py = py.ravel()

    def _square_grid(self):
        """
        A private helper function to generate a square, evenly spaced
        slowness grid.
        """
        # init the slowness grid
        px, py = np.meshgrid(self.slownesses, self.slownesses)

        self.px = px.ravel()
        self.py = py.ravel()

    def __str__(self):
        out = self.print_string + '{}\nSlowness vectors : {}\n'.format(
            '-' * 72, self.px.size
        )
        return out.format(
            self.grid_shape.title(), '-' * len(self.grid_shape)
        )

    def __add__(self, other):
        """
        Merge two grids together discarding of duplicate slowness
        vectors.
        """
        grid = Grid(None)
        grid.grid_shape = 'Composite grid'
        grid.app_vel_params = (self.app_vel_params, other.app_vel_params)
        grid.theta_params = (self.theta_params, other.theta_params)

        grid.app_vel_vector = (self.app_vel_vector, other.app_vel_vector)
        grid.theta_vector = (self.theta_vector, other.theta_vector)
        grid.px = np.hstack((self.px, other.px))
        grid.py = np.hstack((self.py, other.py))

        grid.app_vel_min = self.app_vel_min
        grid.app_vel_max = self.app_vel_max
        grid.linear_max = self.linear_max
        grid.print_string = self.print_string + other.print_string
        return grid

    def __iadd__(self, other):
        """
        Inplace merge two grids together discarding of duplicate slowness
        vectors.
        """
        self.grid_shape = 'Composite grid'
        self.app_vel_params = (self.app_vel_params, other.app_vel_params)
        self.theta_params = (self.theta_params, other.theta_params)

        self.app_vel_vector = (self.app_vel_vector, other.app_vel_vector)
        self.theta_vector = (self.theta_vector, other.theta_vector)
        self.px = np.hstack((self.px, other.px))
        self.py = np.hstack((self.py, other.py))
        self.print_string += other.print_string
        return self

    def plot_pxpy(self, indices=None, ax=None, title=None, **kwargs):
        """
        Plot the `px, py` slowness grid.

        Parameters
        ----------
        indices : array-like
            Indicies of slowness vectors to plot.

        ax : :class:`~matplotlib.axes.Axes`
            Plot grid into a given :class:`~matplotlib.axes.Axes`
            object. Otherwize will plot into the active axes or make a
            new one.

        title : str
            Set the title of the axes. If ``None``, a default title will
            be set.

        Other parameters
        ----------------
        **kwargs : `~matplotlib.pyplot.scatter` and
            `~matplotlib.collections.Collection` properties
        """
        try:
            plt.sca(ax)
        except ValueError:
            ax = plt.gca()

        SCATTER_ARGS_ = SCATTER_ARGS.copy()
        SCATTER_ARGS_.update(**kwargs)
        if indices is not None:
            plt.scatter(self.px[indices], self.py[indices], **SCATTER_ARGS_)
        else:
            plt.scatter(self.px, self.py, **SCATTER_ARGS_)

        circles = [self.app_vel_min, self.linear_max, self.app_vel_max]
        for circle in circles:
            try:
                ax.add_patch(
                    plt.Circle(
                        (0, 0), radius=1 / circle, color='none',
                        ec='r', lw=0.5, zorder=1
                    )
                )
            except TypeError:
                pass

        width = max(self.px.max(), self.py.max())
        extent = np.array((-width, width,
                           -width, width)) * 1.02

        plt.axis(extent)
        plt.xlabel('px, s/m')
        plt.ylabel('py, s/m')
        title = title or 'px,py slowness grid'
        plt.title(title)
        ax = plt.gca()
        ax.set_aspect(1)

    def get_sample_shifts(self, x, y, samp_rate, unique=True, dtype=np.int32):
        """
        Calculate the set of sample shifts associated with each slowness
        for each array element.

        A $n$-element array with coordinates $(x_i, y_i), i=1... n$
        per element, and sampling rate $sps$ are considered. For each
        slowness vector $s_j(p_{xj}, p_{yj})$, a set of $n$ sample
        shifts are calculated as:

        $SS_{i,j} = sps \cdot (x_i \cdot p_{xj} + y_i \cdot p_{yj})$.

        Parameters
        ----------
        x, y : array-like
            x and y coordinates of the array (`n`-elements).

        samp_rate : float
            Sampling rate of the data in samples per second (Hz).

        unique : bool
            If `True` (default), slowness vectors that result in
            identical sample shifts are grouped and once the resultant
            vector is calculated the other, non-unique slowness vectors,
            are discarded. This saves a lot of processing time as these
            are not evaluated over and over again to generate the exact
            same result.

            ..note:: This adds a bit of overhead when generating the
                     grid but will save a lot of time if many time bins
                     are to be evaluated.
        """
        self.unique = unique
        sample_shifts = (-samp_rate *
                         (x * self.px[..., np.newaxis] +
                          y * self.py[..., np.newaxis]) + 0.5).astype(dtype)

        sample_shifts_unique, i_unique = np.unique(
            sample_shifts, return_index=True, axis=0
        )
        if unique:
            print('Keeping {} unique sample shifts out of {}...'.format(
                i_unique.size, sample_shifts.shape[0]), flush=True)
            for i in i_unique:
                group = (
                    sample_shifts == sample_shifts[i]
                ).all(axis=1).nonzero()[0]

                self.px[group] = self.px[group].mean()
                self.py[group] = self.py[group].mean()

            self.px = self.px[i_unique]
            self.py = self.py[i_unique]
            self.sample_shifts = sample_shifts_unique
            return sample_shifts_unique
        else:
            print(('** Only {}% of sample shifts are unique,\n'
                   '** setting `unique=True` will speed up beamforming...'
                   '').format(100 * i_unique.size // sample_shifts.shape[0]),
                  flush=True)
            self.sample_shifts = sample_shifts
            return sample_shifts

    def copy(self):
        return deepcopy(self)
