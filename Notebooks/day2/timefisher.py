"""
Python implementation of Time-Domain Fisher Detector based on
"Multiple Signal Correlators", by Melton & Bailey, Geophysics 1957

.. module:: timefisher

:author:
    Shahar Shani-Kadmiel (s.shanikadmiel@tudelft.nl)
    Pieter Smets (p.s.m.smets@tudelft.nl)

:copyright:
    Shahar Shani-Kadmiel

:license:
    This code is distributed under the terms of the
    GNU General Public License, Version 3
    (https://www.gnu.org/licenses/gpl-3.0.en.html)
"""

import numpy as np
from scipy.signal import spectrogram as _spectrogram
from scipy.ndimage import uniform_filter
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable

from obspy import Trace
from time import time
from grid import Grid, pxpy2theta_app_vel

from numba import njit, prange


def stream2data_array(stream, dtype=None):
    """
    Function to convert traces in an obspy stream object to a data array.

    Parameters
    ----------
    stream : :class:`~obspy.core.stream.Stream`
        A :class:`~obspy.core.stream.Stream` object containing the
        waveforms feed to the beamforming algorithm. Any and all signal
        processing should be applied in the preprocessing phase. Data in
        `stream` will be processed as is.

        ..note:: Each :class:`~obspy.core.trace.Trace` in
                 :class:`~obspy.core.stream.Stream` must have an ``x``
                 and a ``y`` attributes corresponding to the offset of
                 each element from the array origin (center or a
                 specific element).

    dtype : :class:`~numpy.dtype`
        The data type of the returned array. If ``None``, the same data
        type as the data in `stream` will be used.

    Returns
    -------
    coordinates : :class:`~numpy.ndarray`
        An `n`-traces by `2` array with x, y coordinates of each element
        relative to array origin.

    signals : :class:`~numpy.ndarray`
        An `n`-traces by `npts` array with waveform data from each
        element.
    """
    starttime = stream[0].stats.starttime
    endtime = stream[0].stats.endtime
    stream.trim(starttime, endtime, pad=True, fill_value=0)
    count = stream.count()
    dtype = dtype or stream[0].data.dtype
    signals = np.empty((count, stream[0].stats.npts), dtype)
    coordinates = np.empty((count, 2))
    print('Using the following traces:')
    for coordinate, signal, tr in zip(coordinates, signals, stream):
        coordinate[:] = tr.x, tr.y
        signal[:] = tr.data
        print(tr.id, end=', ')
    print('\n')
    return coordinates, signals


def fratio2snr(fratio, N):
    return np.sqrt((fratio - 1) / N)


@njit('Tuple((f4, f8[:]))(f8[:, :], i4, i4)', fastmath=True)
def compute_fratio(win, N, T):
    """
    Calculate the Fisher ratio based on Melton and Bailey (1957).

    Parameters
    ----------
    win : array-like
        2D array with `N`-traces (rows) of length `T`-samples (cols).

    N : int
        Number of traces (array elements).

    T : int
        Number of samples in each trace.

    Returns
    -------
    fratio : float
        The Fisher ratio value.

    sum_stack : :class:`~numpy.ndarray`
        1D array of the sum stack of the traces.
    """
    sum_stack = np.sum(win, axis=0)
    fratio = (
        (np.sum(sum_stack**2) - np.sum(win)**2 / T) /
        (np.sum(win**2) - np.sum(sum_stack**2) / N)
    )
    return fratio, sum_stack


@njit('Tuple((f4, f8[:]))(f8[:, :], i4, i4, i4, i4[:])', fastmath=True)
def compute_fratio_numba(signals, N, binsize, start_sample, shifts):
    """
    Calculate the Fisher ratio based on Melton and Bailey (1957).

    Parameters
    ----------
    signals : array-like
        2D array with `N`-traces (rows).

    N : int
        Number of traces (array elements).

    binsize : int
        Number of samples in bin.

    start_sample : int
        Index of the first sample of the bin.

    shifts : array-like
        1D array of size `N` with sample shifts of each trace.

    Returns
    -------
    fratio : float
        The Fisher ratio value.

    sum_stack : :class:`~numpy.ndarray`
        1D array of the sum stack of the traces.
    """
    sum_stack = np.zeros(binsize, np.float64)
    term3 = 0
    for e in range(N):
        start_sample_ = start_sample + shifts[e]
        trace = signals[
            e, start_sample_: start_sample_ + binsize
        ]
        sum_stack += trace
        term3 += np.sum(trace * trace)

    term1 = np.sum(sum_stack * sum_stack)
    term2 = np.sum(sum_stack)**2 / binsize
    term4 = term1 / N
    fratio = ((term1 - term2) / (term3 - term4))

    return fratio, sum_stack


@njit(('i4, i4, i4[:, :], f8[:, :], i4,'
       'f8[:], f4[:], i4[:], f4[:, :]'),
      parallel=True, fastmath=True, nogil=True)
def _do_loops_numba(
    step, nskipbins, all_shifts, signals, binsize,  # inputs
    bestbeam, fratio_max, fratio_maxloc, fgrid      # outputs
):
    """
    Numba optimized version of the python code.

    This is more or less along the same lines as the loops in the python
    version but the routine that computes the Fisher ratio is slightly
    different and the loops have been optimized to run on multiple
    threads.
    """
    nslowvecs, nelements = all_shifts.shape
    nelements = np.int32(nelements)
    nslowvecs = np.int32(nslowvecs)
    nbins_ = np.int32(fratio_max.size)
    npts = np.int32(bestbeam.size)

    for bin_ in prange(nbins_):
        start_sample = step * (nskipbins + bin_)
        for sv in range(nslowvecs):

            fratio, sum_stack = compute_fratio_numba(
                signals, nelements, binsize, start_sample, all_shifts[sv]
            )

            if fratio > fratio_max[bin_]:
                fratio_max[bin_] = fratio
                fratio_maxloc[bin_] = sv
                bestbeam[start_sample: start_sample + binsize] = sum_stack

            # store the entire grid
            fgrid[bin_, sv] = fratio


def _do_loops_python(
    step, nskipbins, all_shifts, signals, binsize,  # inputs
    bestbeam, fratio_max, fratio_maxloc, fgrid  # outputs
):
    """
    The python implementation of the beamforming loops.

    Consists of looping over all windows (timebins), and over all slowness
    vectors to evaluate the maximum Fisher ratio.
    """
    nslowvecs, nelements = all_shifts.shape
    nelements = np.int32(nelements)
    nslowvecs = np.int32(nslowvecs)
    nbins_ = np.int32(fratio_max.size)
    npts = np.int32(bestbeam.size)

    timebin_data = np.zeros((nelements, binsize), np.float64)
    for bin_ in range(nbins_):
        start_sample = step * (nskipbins + bin_)
        print(f'Processing window {bin_ + 1} out of {nbins_} windows...',
              end='\r')
        for sv in range(nslowvecs):
            for e in range(nelements):
                start_sample_ = start_sample + all_shifts[sv, e]
                timebin_data[e, :] = signals[
                    e, start_sample_: start_sample_ + binsize
                ]

            fratio, sum_stack = compute_fratio(
                timebin_data, nelements, binsize
            )

            if fratio > fratio_max[bin_]:
                fratio_max[bin_] = fratio
                fratio_maxloc[bin_] = sv
                bestbeam[start_sample: start_sample + binsize] = sum_stack

            # store the entire grid
            fgrid[bin_, sv] = fratio


def beamform(stream, grid, wlen, overlap=0.5, unique=True, version='python'):
    """
    Time-Domain Fisher Detector.

    Parameters
    ----------
    stream : :class:`~obspy.core.stream.Stream`
        A :class:`~obspy.core.stream.Stream` object containing the data
        to beamform. Any and all signal processing should be applied in
        the preprocessing phase. Data in `stream` will be processed as
        is. All traces in `stream` should have the same number of
        samples and span over the same time period.

        ..note:: Each :class:`~obspy.core.trace.Trace` in
                 :class:`~obspy.core.stream.Stream` must have an ``x``
                 and a ``y`` attributes corresponding to the offset of
                 each element from the array origin (center or a
                 specific element).

    grid : :class:`Grid`
        The slowness grid object over which the grid search is
        performed.

    wlen : float
        Window length in seconds.

    overlap : float
        Fraction of overlap (0 < overlap < 1). Default is 0.5 (50%).
        Larger overlap yields longer processing time but better time
        resolution.

        Example overlaps:

                - ``overlap=0.9`` - 90% overlap
                - ``overlap=0.5`` - 50% overlap
                - ``overlap=0.1`` - 10% overlap

    unique : bool
        Passed to the :meth:`~Grid.get_sample_shifts` method.
        If `True` (default), slowness vectors that result in identical
        sample shifts are grouped and once the resultant vector is
        calculated the other, non-unique slowness vectors, are
        discarded. This saves a lot of processing time as these are not
        evaluated over and over again to generate the exact same result.

        ..note:: This adds a bit of overhead when generating the grid
                 but will save a lot of time if many windows are to be
                 evaluated.

    Returns
    -------
    bestbeam : :class:`~obspy.core.trace.Trace`
        A :class:`~obspy.core.trace.Trace` object containing the
        bestbeam.

    times : :class:`~numpy.ndarray`
        Time since start time of `stream`. Represents the middle of each
        time window.

    fratio_max : :class:`~numpy.ndarray`
        The maximum F-ratio in each time window.

    baz : :class:`~numpy.ndarray`
        The bearing from which the wavefront arrives, aka: back-azimuth.

    app_vel : :class:`~numpy.ndarray`
        The horizontal velocity at which the wavefront propagated over
        the array.

    fgrid : :class:`~numpy.ndarray`, optional
        The F-ratio over the entire slowness grid. Only provided if
        `full_grid` is `True`.
    """

    _t0 = time()

    ############################ Setting up ##################################
    stats = stream[0].stats.copy()
    starttime = stats.starttime
    endtime = stats.endtime
    samp_rate = stats.sampling_rate
    npts = np.int32(stats.npts)
    binsize = np.int32(wlen * samp_rate)
    overlap = np.int32(binsize * overlap)
    step = binsize - overlap

    # Construct data array with all signals
    coordinates, signals = stream2data_array(stream, np.float64)
    x, y = coordinates.T

    # Get all sample shifts from the grid object
    print(grid)
    try:
        all_shifts = grid.sample_shifts
    except AttributeError:
        all_shifts = grid.get_sample_shifts(x, y, samp_rate, unique)

    nslowvecs = grid.px.size

    # Figure out how many bins to skip to avoid out of bound sample
    # shift index
    nskipsamples = max(np.absolute(all_shifts).max(), binsize)
    nskipbins = np.int32(nskipsamples / step + 0.99)
    nbins = np.int32((npts - binsize) / step)
    nbins_ = nbins - 2 * nskipbins
    nelements = np.int32(stream.count())

    str_width = len(str(nbins)) + 3
    string_template = '{:>30} : '

    print(
        '\nTime domain Fisher beamforming'
        '\n------------------------------\n',
        (string_template + '{:>{width}}\n').format(
            'Number of elements', nelements, width=str_width - 3),
        (string_template + '{:>{width}.2f}\n').format(
            'Sampling rate, Hz', samp_rate, width=str_width),
        (string_template + '{:>{width}.2f} [{:.0f} samples]\n').format(
            'Sliding window lengh, s', wlen, binsize, width=str_width),
        (string_template +
         '{:>{width}.2f} [{:.0f} samples, {:.0f}%]\n').format(
            'Overlap, s', overlap / samp_rate, overlap,
            100 * overlap / binsize, width=str_width),
        (string_template + '{:>{width}}\n').format(
            'Total number of windows', nbins, width=str_width - 3),
        (string_template +
         '{:>{width}}    [Skipping {} windows left and right]\n').format(
            'Number of processing windows', nbins_, nskipbins,
            width=str_width - 3),
        (string_template + '{}\n').format('Start time', starttime),
        (string_template + '{}\n').format('End time', endtime),
        flush=True)

    # Construct the time vector with each element corresponding to the
    # middle of the timebin
    times = np.arange(
        nskipbins * step + 0.5 * binsize,
        (nskipbins + nbins_) * step + 0.5 * binsize,
        step
    ) / samp_rate

    # Beamform the data
    print(f'Beamforming {nslowvecs} slowness vectors in {nbins_} timebins...',
          flush=True)

    f_dof = (
        1 * binsize * (nelements - 1) /
        (1 * nelements * (binsize - 1))
    )

    ########################### Beamforming ##################################
    # Construct the result arrays to be filled by the beamforming loops
    fratio_max = np.zeros(nbins_, np.float32)
    bestbeam = np.zeros(npts, np.float64)
    fratio_maxloc = np.zeros(nbins_, np.int32)
    fgrid = np.zeros((nbins_, nslowvecs), np.float32)

    if version is 'python':
        _do_loops_python(
            # inputs
            step, nskipbins, all_shifts, signals, binsize,

            # outputs
            bestbeam, fratio_max, fratio_maxloc, fgrid
        )

    elif version is 'numba':
        _do_loops_numba(
            # inputs
            step, nskipbins, all_shifts, signals, binsize,

            # outputs
            bestbeam, fratio_max, fratio_maxloc, fgrid
        )


    ######################### Post processing ################################
    fgrid *= f_dof
    fratio_max *= f_dof

    # Convert slowness vector to back azimuth and apparent velocity
    theta, app_vel = pxpy2theta_app_vel(grid.px[fratio_maxloc],
                                        grid.py[fratio_maxloc])

    # Populate an ObsPy Trace object with the bestbeam data
    stats.mseed, stats.processing, stats.response = [], [], []
    stats.starttime += nskipbins * step / samp_rate
    stats.station = 'BBEAM'
    bestbeam = Trace(bestbeam / nelements, stats)

    ############################### Done #####################################
    print('\nDone! [Processing time: {:.2f} seconds]'.format(
        time() - _t0), flush=True
    )

    return bestbeam, times, fratio_max, theta, app_vel, fgrid


################################ Plotting routines ###########################
def custom_scatter(x, y, ax=None, **kwargs):
    """
    A wrapper to pyplot's scatter function to plot symbol edges behind
    symbols.
    """
    if not ax:
        ax = plt.gca()
    try:
        ec = kwargs['edgecolor']
        kwargs.pop('edgecolor')
    except KeyError:
        ec = 'k'

    try:
        c = kwargs['c']
        kwargs.pop('c')
    except KeyError:
        c = 'k'

    ax.scatter(x, y, c='none', edgecolor=ec, zorder=1, **kwargs)
    sp = ax.scatter(x, y, c=c, edgecolor='none', zorder=2, **kwargs)
    return sp


celerities = (6, 3, 2, 1, 0.7, 0.5, 0.3, 0.25, 0.2)


def plot_celerities(celerities, distance, ax=None):
    """
    Plot the celerity axes on top of the active axes or ``ax``.

    Parameters
    ----------
    celerities : array-like
        Sequence of celerity values to plot.

    distance : float
        Source-receiver distance in km.

    ax : :class:`~matplotlib.pyplot.axes`
        Axes to plot to. If ``None``, plotting is done to the active axes.
    """
    ticks = (distance / np.array(celerities))
    ax = ax or plt.gca()
    divider = make_axes_locatable(ax)

    ax_ = divider.append_axes('top', 0.1, -0.1)
    ax_.get_shared_x_axes().join(ax_, ax)
    ax_.set_xticks(ticks)
    ax_.set_xticklabels(celerities)
    ax_.set_xticks([], minor=True)
    ax_.set_xlabel('Celerity, km/s')
    ax_.xaxis.set_label_position('top')
    ax_.set_xlim(ax.get_xlim())

    ax_.patch.set_visible(False)
    ax_.xaxis.tick_top()
    ax_.set_yticks([])
    ax_.spines['bottom'].set_visible(False)
    ax_.spines['left'].set_visible(False)
    ax_.spines['right'].set_visible(False)
    ax_.set_xlim(ax.get_xlim())

    ax.xaxis.tick_bottom()
    ax.spines['top'].set_visible(False)

    plt.sca(ax)


def plot_utc_time(utc_time, ax=None, fmt='%H:%M'):
    """
    Plot an UTC time axes on top of the active axes or ``ax``.

    Parameters
    ----------
    utc_time : UTCDateTime
        UTC datetime of the startsample.

    ax : :class:`~matplotlib.pyplot.axes`
        Axes to plot to. If ``None``, plotting is done to the active axes.

    fmt : str
        Specify the tick datetime format.
    """
    divider = make_axes_locatable(ax)
    ax_ = divider.append_axes('bottom', 0.1, 0.4)
    ax_.get_shared_x_axes().join(ax_, ax)
    ticks = ax.get_xticks()
    ax_.set_xticks(ticks)
    utc_times = [utc_time + t for t in ticks]
    ticklabels = [t.strftime(fmt) for t in utc_times]
    ax_.set_xticklabels(ticklabels)
    ax_.set_xticks([], minor=True)
    ax_.set_xlabel('UTC')
    ax_.xaxis.set_label_position('bottom')
    ax_.set_xlim(ax.get_xlim())

    ax_.patch.set_visible(False)
    ax_.xaxis.tick_bottom()
    ax_.set_yticks([])
    ax_.spines['top'].set_visible(False)
    ax_.spines['left'].set_visible(False)
    ax_.spines['right'].set_visible(False)
    ax_.set_xlim(ax.get_xlim())

    plt.sca(ax)


def _nearest_pow_2(x):
    """
    Find power of two nearest to x

    >>> _nearest_pow_2(3)
    2
    >>> _nearest_pow_2(15)
    16
    """
    a = np.power(2, np.ceil(np.log2(x)))
    b = np.power(2, np.floor(np.log2(x)))
    if abs(a - x) < abs(b - x):
        return int(a)
    else:
        return int(b)


def spectrogram(trace, t_offset=0, wtype='hann', wlen=None, overlap=0.9,
                ax=None, smooth=5, utctime=False, verbose=True, **kwargs):
    """
    Convenience wrapper to :func:`~scipy.signal.spectrogram` combined
    with a plotting routine.

    Parameters
    ----------
    trace : :class:`~obspy.core.trace.Trace`
        An ObsPy Trace object.

    t_offset : float
        Time in seconds to offset the spectrogram in.

    wtype : str or tuple or array_like
        Desired window to use. If window is a string or tuple, it is
        passed to :func:`~scipy.signal.get_window` to generate the
        window values, which are DFT-even by default. See
        :func:`~scipy.signal.get_window` for a list of windows and
        required parameters. If window is array_like it will be used
        directly as the window and its length must be `wlen`. Defaults
        to a symmetric ``hann`` window.

    wlen : int
        Length of each segment in sample points. Defaults to
        sampling rate * 100. `wlen` is rounded to the nearest
        power of 2 whether specified or not.

    overlap: float
        Fraction of overlap between sliding windows. Defaults to 0.9
        which is 90%.

    ax : :class:`~matplotlib.pyplot.axes`
        Axes to plot to. If ``None``, plotting is done to the active axes.

    smooth : int
        Passed as size to :func:`~scipy.ndimage.filters.uniform_filter`.
        Set to ``False`` to forgo smoothing.

    utctime : bool
        Set the x-axis to utc time.

    Returns
    -------
    image : `~matplotlib.image.AxesImage`

    Other parameters
    ----------------
    **kwargs : `~matplotlib.pyplot.imshow` and
        `~matplotlib.artist.Artist` properties.
    """
    data = trace.data
    samp_rate = trace.stats.sampling_rate

    wlen = wlen or samp_rate * 100.
    wlen = _nearest_pow_2(wlen)
    overlap = int(overlap * wlen)

    print(data.size, samp_rate, wlen, overlap)
    if verbose:
        print(('Computing spectrogram with {} samples long windows and '
               '{} points overlap.\n'
               'Total number of windows is {}...').format(
              wlen, overlap, int((data.size - wlen) / (wlen - overlap))))

    freq, time, spect = _spectrogram(data, samp_rate, wtype, wlen, overlap,
                                     scaling='spectrum', mode='magnitude')

    if utctime:
        time = time/86400. + trace.times('matplotlib')[0]
    else:
        time += t_offset

    # calculate half bin width
    halfbin_time = (time[1] - time[0]) / 2.0
    halfbin_freq = (freq[1] - freq[0]) / 2.0

    # this method is much much faster!
    spect = np.flipud(spect)
    if smooth:
        spect = uniform_filter(spect, smooth)

    extent = (time[0] - halfbin_time, time[-1] + halfbin_time,
              freq[0] - halfbin_freq, freq[-1] + halfbin_freq)

    try:
        plt.sca(ax)
    except ValueError:
        pass

    try:
        aspect = kwargs.pop('aspect')
    except KeyError:
        aspect = 'auto'

    im = plt.imshow(spect, extent=extent, aspect=aspect, **kwargs)
    return im


def plot_results(bestbeam, times, fratio_max, baz, app_vel, snr,
                 stream=None, origintime=None, distance=None,
                 utctime=False, vmin=None, vmax=None):
    """
    Quick and dirty plotting function to preview beamforming results.

    Parameters
    ----------
    bestbeam : :class:`~obspy.core.trace.Trace`
        A :class:`~obspy.core.trace.Trace` object containing the
        bestbeam.

    times : :class:`~numpy.ndarray`
        Time since start time of `stream`. Represents the middle of each
        time window.

    fratio_max : :class:`~numpy.ndarray`
        The maximum F-ratio in each time window.

    baz : :class:`~numpy.ndarray`
        The bearing from which the wavefront arrives, aka: back-azimuth.

    app_vel : :class:`~numpy.ndarray`
        The horizontal velocity at which the wavefront propagated over
        the array.

    snr : :class:`~numpy.ndarray`
        The signal-to-noise ratio.

    stream : :class:`~obspy.core.stream.Stream`
        A :class:`~obspy.core.stream.Stream` object containing the data
        used for beamforming. Traces will be plotted in the background
        behind the bestbeam.

    origintime : :class:`~obspy.core.utcdatetime.UTCDateTime`
        Origin time of the source.

    distance : float
        Source-receiver distance in km.

    utctime : bool
        Plot time axis in UTC.

    Note
    ----
    If both `origintime` and `distance` are not ``None``, a celerity
    axes will be plotted above the top axes.
    """
    origintime = origintime or 0
    t_offset = bestbeam.stats.starttime - origintime

    if utctime:
        t_offset = 0
        origintime = None
        bestbeam_times = bestbeam.times('matplotlib')
        detection_times = times / 86400. + bestbeam_times[0]
    else:
        bestbeam_times = bestbeam.times('relative', origintime)
        detection_times = times + t_offset

    fig, ax = plt.subplots(5, figsize=(7, 8), sharex=True)

    if origintime and distance:
        plot_celerities(celerities, distance, ax[0])

    im = spectrogram(bestbeam, t_offset, ax=ax[0], cmap='Greys',
                     utctime=utctime)
    data_ = im.get_array()
    im.set_clim(0.1 * data_.std(), 0.85 * data_.max())
    cb1 = plt.colorbar(im, ax=ax[0], extend='both', pad=0.02, aspect=10,
                       label='Power')
    ax[0].set_yscale('log')
    ax[0].yaxis.set_major_locator(LogLocator(numticks=4))
    ax[0].yaxis.set_minor_locator(LogLocator(subs='all', numticks=10))
    ax[0].yaxis.set_minor_formatter(NullFormatter())
    ax[0].set_ylabel('Frequency, Hz')

    try:
        for tr in stream:
            if utctime:
                ax[1].plot(tr.times('matplotlib'), tr.data, lw=0.5, alpha=0.3)
            else:
                ax[1].plot(tr.times('relative', origintime), tr.data,
                           lw=0.5, alpha=0.3)
    except TypeError:
        pass

    ax[1].plot(bestbeam_times, bestbeam.data, 'k', lw=0.5)

    ax[1].set_ylabel('Amplitude')
    cb_fake = plt.colorbar(im, ax=ax[1:-1], extend='both', pad=0.02,
                           aspect=10)
    cb_fake.remove()

    vmax = vmax or 0.85 * snr.max()
    vmin = vmin or 0.6

    cmap = plt.get_cmap('inferno_r')
    cmap.set_under('w')

    sort_indices = snr.argsort()
    custom_scatter(
        detection_times[sort_indices], baz[sort_indices], ax[2],
        s=15, c=snr[sort_indices], cmap=cmap,
        vmin=vmin, vmax=vmax, edgecolor='k', lw=0.5)
    ax[2].set_ylim(0, 360)
    ax[2].set_ylabel(u'Back azimuth, °')

    custom_scatter(
        detection_times[sort_indices], app_vel[sort_indices], ax[3],
        s=15, c=snr[sort_indices], cmap=cmap,
        vmin=vmin, vmax=vmax, edgecolor='k', lw=0.5)
    ax[3].set_ylabel('App. vel., m/s')

    sp = custom_scatter(
        detection_times[sort_indices], fratio_max[sort_indices], ax[4],
        s=15, c=snr[sort_indices], cmap=cmap,
        vmin=vmin, vmax=vmax, edgecolor='k', lw=0.5)
    ax[4].set_ylabel('F-ratio')

    if utctime:
        for axi in ax:
            axi.xaxis_date()

        ax[4].set_xlabel('Time (UTC)')
        # fig.autofmt_xdate()
    elif origintime:
        ax[4].set_xlabel('Time since origin {}'.format(
            origintime.strftime('%FT%T')
        ))
    else:
        ax[4].set_xlabel('Time since {}'.format(
            bestbeam.stats.starttime.strftime('%FT%T')
        ))

    cb2 = plt.colorbar(sp, ax=ax[-1], extend='both', pad=0.02, aspect=10,
                       label='SNR')

    for axi in ax:
        axi.grid(True, axis='x', lw=0.5, color='0.7')
        axi.yaxis.set_label_coords(-0.1, 0.5)

    ax[4].set_xlim(bestbeam_times[0], bestbeam_times[-1])

    ax[0].set_title('Beamforming results', y=1.4)

    return fig, ax, cb1, cb2
