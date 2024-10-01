#!/usr/bin/env python

# Utilize a wavelet basis to model a BBH signal merger.
# Based on the work of Finch and Moore (https://arxiv.org/abs/2108.09344,
# https://arxiv.org/abs/2205.07809)
# Author: Alex Correia

import numpy as np
import math as m
from pycbc.types import TimeSeries

pi = np.pi

def parse_params(**kwargs):
    """Generate dictionaries for each wavelet's parameters.
    Checks if the minimum required parameters are provided.
    """
    # number of wavelets
    try:
        w = kwargs['wavelets']
    except KeyError:
        raise ValueError('Must provide number of wavelets to generate')

    amps = {}
    freqs = {}
    taus = {}
    phis = {}
    etas = {}

    for i in len(w):
        s = str(i)
        # amplitude
        try:
            amps[i] = kwargs['amp' + s]
        except KeyError:
            raise ValueError(f'missing amp{i}')

	# frequencies
        try:
            freqs[i] = kwargs['freq' + s]
        except KeyError:
            raise ValueError(f'missing freq{i}')

	# damping times
        try:
            taus[i] = kwargs['tau' + s]
        except KeyError:
            raise ValueError(f'missing tau{i}')

	# phases
        try:
            phis[i] = kwargs['phi' + s]
        except KeyError:
            raise ValueError(f'missing phi{i}')

	# ref times
        try:
            etas[i] = kwargs['eta' + s]
        except KeyError:
            raise ValueError(f'missing eta{i}')

        return w, amps, freqs, taus, phis, etas

def get_td_wavelet(f, tau, amp, phi, eta, end_time):
    r"""Generate a single wavelet in the time domain.
        This uses the Morlet-Gabot formula as listed in arXiv:2108.09344:

	.. math::

	   h(t) &:= h_{+} + ih_{\cross} \\
		&:= \sum_{w=1}^W A_w \exp \Big[-2\pi i\nu_w (t-\eta_w) \\
		& - \big( \frac{t-\eta_w}{\tau_w} \big)^2 - i\phi_w \Big], t_i < t < t_f

    Parameters
    ----------
    f : float
        The wavelet frequency.

    tau : float
        The wavelet width.

    amp : float
        The wavelet amplitude.

    phi : float
        The wavelet phase.

    eta : float
        The central time in seconds of the wavelet. This time corresponds to:

	.. math::

	   h_w(t = \eta_w) = A_w\exp (i \phi_w)

    end_time : float
	    The end time in seconds of the wavelet. For example, if using for a merger model,
	    this corresponds to the coalescence time of the signal.

    Returns
    -------
    (array, array)
        The time domain plus and cross polarizations of the wavelet.
    """
    # generate a time series for the wavelet
    start_time = end_time - 2*eta
    l = int((end_time - start_time)/dt + 1)
    t = np.linspace(start_time, end_time, l)

    # evaluate the wavelet
    offset = t - eta
    nondim_offset = offset/tau
    wf = amp*np.exp(-2*pi*1j*f*offset - nondim_offset*nondim_offset + 1j*phi)

    # retrieve the plus and cross polarizations
    hp = wf.real
    hc = -wf.imag
    return hp, hc

def wavelet_sum_base(input_params):
    """Base function for returning a superposition of wavelets.

    Parameters
    ----------
    input_params : dict
	Dictionary of parameters for generating wavelets. See
	get_td_wavelet for list of params.
    """
    # parse parameters
    w, amps, freqs, taus, phis, etas = parse_params(**input_params)
    tc = params['tc']
    dt = params['delta_t']

    # allocate hp, hc vectors using the maximum wavelet length
    max_eta = max(etas.items())
    max_length = int(2 * (tc - max_eta)/delta_t + 1)
    hp_out = TimeSeries(zeros(max_length, dtype=float64), delta_t=delta_t)
    hc_out = TimeSeries(zeros(max_length, dtype=float64), delta_t=delta_t)

    # generate wavelets and add to out vectors
    for i in range(w):
        wavelet_start_idx = ceil(int((tc - 2*etas[i])/dt))
        hp, hc = get_td_wavelet(freqs[i], taus[i], amps[i], phis[i], etas[i], tc)
        hp_out[wavelet_start_idx:] += hp
        hc_out[wavelet_start_idx:] += hc

    return hp_out, hc_out

### Approximants ###################################################################

def get_td_wavelet_basis(**kwargs):
    """Generate the time domain wavelet basis for a signal.
    """
    return wavelet_sum_base(kwargs)