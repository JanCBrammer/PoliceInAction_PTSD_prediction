# -*- coding: utf-8 -*-
"""
@author: Jan C. Brammer <jan.c.brammer@gmail.com>
"""

import numpy as np
from scipy.signal import decimate
from scipy.interpolate import interp1d


def _prime_factors(n):
    """
    https://rosettacode.org/wiki/Prime_decomposition#Python:_Using_floating_point
    Returns list of prime factors of n (ascending order).
    """
    step = lambda x: 1 + (x << 2) - ((x >> 1) << 1)
    maxq = int(np.floor(np.sqrt(n)))
    d = 1
    q = 2 if n % 2 == 0 else 3
    while q <= maxq and n % q != 0:
        q = step(d)
        d += 1
    return [q] + _prime_factors(n // q) if q <= maxq else [n]


def decimate_signal(signal, decimation_factor):
    """
    Decimate signal according to https://dspguru.com/dsp/faqs/multirate/decimation/.
    Perform decimation in multiple stages for decimation facors larger than 13:
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.decimate.html.

    """
    if decimation_factor > 13:
        decimation_factors = _prime_factors(decimation_factor)

        if len(decimation_factors) == 1:
            print("Cannot decimate with factors larger than 13 that are prime"
                  " numbers.")
            return

        # Sort decendingly, since decimation should start with largest factor.
        decimation_factors.reverse()

    else:
        decimation_factors = [decimation_factor]

    decimated = signal
    for i in decimation_factors:

        decimated = decimate(decimated, i, ftype="iir", zero_phase=True)

    return decimated


def invert_signal(signal):

    signal_mean = np.mean(signal)
    signal_inverted = (signal_mean - signal) + signal_mean

    return signal_inverted


def interpolate_signal(peaks, signal, nsamples):

    samples = np.linspace(0, peaks[-1], nsamples)

    f = interp1d(peaks, signal, kind='slinear', bounds_error=False,
                 fill_value=([signal[0]], [signal[-1]]))

    signal_interpolated = f(samples)

    return signal_interpolated


def consecutive_samples(signal, condition):
    """
    Parameters
    ----------
    signal : array
        The signal in which to evaluate condition.
    condition : lambda function
        The condition to evaluate.
    Returns
    -------
    begs : array
        Indices of signal at which the consecutive chunks of samples start that
        fulfill the condition.
    ends : array
        Indices of signal at which the consecutive chunks of samples end that
        fulfill the condition.
    n : array
        Number of samples in each consecutive chunk that fulfills the
        condition.

    """
    idcs_bool = condition(signal)    # Boolean array
    # Find the indices of changes.
    change = np.diff(idcs_bool)
    idcs, = change.nonzero()

    idcs += 1    # Get indices following the change.

    if idcs_bool[0]:
        # If the first sample fulfills the condition, prepend a zero.
        idcs = np.r_[0, idcs]

    if idcs_bool[-1]:
        # If the last sample fulfills the condition, append an (out of bounds)
        # index corresponding to the length of signal in order to correctly
        # compute the duration of chunks that fulfill the condition.
        idcs = np.r_[idcs, idcs_bool.size]

    begs = idcs[0::2]
    ends = idcs[1::2]
    n = ends - begs

    if idcs_bool[-1]:
        # If the last sample fulfills the condition, set the end of the last
        # chunk that fulfills the condition to the final index of the original
        # array in order to prevent off-by-one errors.
        # Note that if the last chunk that fulfills the condition consists of
        # a single sample, that last chunk will be empty when indexed with
        # beg[-1]:end[-1].
        ends[-1] = idcs_bool.size - 1

    return begs, ends, n


def cop_radius(ap_displ, ml_displ):
    """Compute the radial center-of-pressure displacement.

    Parameters
    ----------
    ap_displ : 1D array
        The anterior-posterior displacement.
    ml_displ : 1D array
        The medio-lateral displacement.
    Returns
    -------
    avg_radius : 1D array
        The radial center-of-pressure displacement of each sample pair in
        ap_displ and ml_displ.
    """
    # Compute radius as Pythagorean hypotenuse (equivalent to vector norm).
    radius = np.sqrt(ap_displ**2 + ml_displ**2)
    
    return radius
