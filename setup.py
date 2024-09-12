#!/usr/bin.env python
"""
setup.py file to allow PyCBC to read in wavelet basis code as a plugin
Author: Alex Correia
"""

from setuptools import Extension, setup, Command
from setuptools import find_packages

VERSION = '0.0.dev0'

setup(
    name = 'pycbc_wavelet_plugin',
    version = VERSION,
    author = 'Alex Correia',
    author_email = 'alcorrei@syr.edu',
    url = 'http://www.pycbc.org',
    download_url = f'https://github.com/acorreia61201/wavelet_merger_model/v{VERSION}',
    keywords = ['pycbc', 'gravitational waves'],
    install_requires = ['pycbc'],
    py_modules = ['wavelet'],
    entry_points = {'pycbc.waveform.td':'wavelet = wavelet:get_td_wavelet_basis',
		    'pycbc.waveform.fd':'wavelet = wavelet:get_fd_wavelet_basis'},
)

