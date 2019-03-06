import numpy as np


def get_array_response(x, y, c_app=280, c_steps=50,
                       f_min=0.2, f_max=0.2, f_steps=1,
                       px_0=0, py_0=0):
    """
    Calculating array response on a square slowness grid for
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
