# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 15:55:24 2021

@author: wicki
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from fitutils_P3 import FitObject, fit_parameter

filename = 'Diamantane\di_spectra_3.csv'
fitcol = '30'

# Possible data scales
possible_scales = ['ppm', 'kHz']

# If scale is given in multiple units prefer this one. None to select only scale
prefer_scale = 'ppm'

data = np.genfromtxt(filename, delimiter=',', names=True)
Tnames = list(data.dtype.names)

scaleset = set()

for scale in possible_scales:
    if scale in Tnames:
        Tnames.remove(scale)
        scaleset.add(scale)

nscale = len(scaleset)
if nscale == 0:
    sys.exit("None of the expected frequency scales found")
if prefer_scale is None:
    if nscale == 1:
        sys.exit("Multiple scales found. Need to select prefer_scale")
    use_scale = next(scaleset)
else:
    if prefer_scale not in scaleset:
        sys.exit(f"Preferred scale ({prefer_scale}) not set")
    use_scale = prefer_scale


def rawgauss(x, scale, X, sigma, c=0.0):
    zvalues = x - X
    return scale * np.exp(-(zvalues**2)/(2*sigma**2)) + c


# Functions to be fitted
# These have the "signature" of xvalues, dictionary of parameters, optional additional (fixed) arguments
def fitgauss1(x, pardict, args):
    X = pardict['shift'].value
    sigma = pardict['sigma'].value
    scale = pardict['scale'].value
    c = pardict['c'].value if 'c' in pardict else 0.0
    return rawgauss(x, scale, X, sigma, c)


parscale = fit_parameter('scale', 400000.)
parX = fit_parameter('shift', 0., displayname=f'shift / {use_scale}')
parsigma = fit_parameter('sigma', 1., displayname=f'sigma / {use_scale}')
parc = fit_parameter('c', 0., displayname='offset')

scale = data[prefer_scale]
fitobj = FitObject(fitgauss1, [parscale, parX, parsigma])
fitdata = fitobj.fit(scale, data[fitcol])

# Plot the results
plt.plot(scale, data[fitcol], 'k')
plt.plot(scale, fitdata, 'r')

ax = plt.gca()
ax.set_xlabel(f'Frequency / {use_scale}')
ax.invert_xaxis()
ax.get_yaxis().set_visible(False)
plt.show()