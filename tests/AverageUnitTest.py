# -*- coding: utf-8 -*-
"""
Test of averaging two site jump using analytical formula vs numerical
"""

import numpy as np
from math import sin, cos, sqrt
from soprano.nmr.tensor import NMRTensor
from soprano.data.nmr import nmr_gamma
from soprano.nmr.utils import _dip_constant
from soprano.calculate.powder import ZCW
from ase.quaternions import Quaternion
import matplotlib.pyplot as plt

# Ordering convention for dipolar tensors.
# Confirmed not to affect calculated results
dipole_tensor_convention = NMRTensor.ORDER_NQR  # HAEBERLEN

halfangle = 45  # degrees

ZCWorients_list = [21, 34, 55, 89, 144]

r = 1.75  # internuclear distance. Typical for CH2
gamma1H = nmr_gamma('H', iso=1)[0]  # nmr_gamma() returns a list
d = _dip_constant(r*1e-10, gamma1H, gamma1H)*1e-3
print("Unaveraged dipolar coupling constant: {} kHz".format(d))

def average_dipolar_tensor(halfangle, weighting=0.5, zorientation=None, verbose=False):
    """ Return average dipolar tensor given half angle """

    if (weighting < 0.0) or (weighting > 1.0):
        raise ValueError("weighting out of range (must be between 0 and 1)")

# Create internuclear unit vectors in yz plane, bisector along z
    r1 = np.array([0.0, -cos(halfangle), sin(halfangle)])
    r2 = np.array([0.0,  cos(halfangle), sin(halfangle)])
    if zorientation is not None:  # rotate vectors
        quat = Quaternion.from_euler_angles(zorientation[1], zorientation[0], 0.0)
        r1 = quat.rotate(r1)
        r2 = quat.rotate(r2)

# Create corresponding dipolar coupling tensors

    D1 = d * (3*r1[None, :]*r1[:, None]-np.eye(3))
    D2 = d * (3*r2[None, :]*r2[:, None]-np.eye(3))

    if verbose:
        print("Dipolar tensor 1:\n", D1)
        print("Dipolar tensor 2:\n", D2)

    avD = weighting * D1 + (1.0 - weighting) * D2
    return avD


avD = average_dipolar_tensor(halfangle, verbose=False)
avDtensor = NMRTensor(avD, order=dipole_tensor_convention)
dav, etaav = 0.5*avDtensor.eigenvalues[2], avDtensor.asymmetry
print("Averaged tensor: d = {} kHz, eta = {}".format(dav, etaav))

contrib_RMSD_powder = abs(dav)*sqrt(0.2 + (etaav ** 2)/15.0)

print("Analytical RMS D (powder) contribution: %g kHz" % contrib_RMSD_powder)


def dcontrib_single(angles):
    """ Return d_zz for average dipolar tensor evaluated at given orientation """

    avD = average_dipolar_tensor(halfangle, zorientation=angles)
    return 0.5*avD[2, 2]


avdlist = []
for ZCWorients in ZCWorients_list:
    angles, weights = ZCW('sphere').get_orient_angles(ZCWorients)
    print("Powder averaging over {} ZCW orientations".format(len(weights)))

    dcontribs = [dcontrib_single(o) for o in angles]
    d2contribs = [dip*dip for dip in dcontribs]
    avd = np.average(dcontribs, weights=weights)  # NB weights should be uniform for ZCW
    avd2 = np.average(d2contribs, weights=weights)
    if len(ZCWorients_list) == 1:
        print("Average D (should be close to zero): %g kHz" % avd)
        print("Numerically averaged RMS D: %g kHz" % sqrt(avd2))
    else:
        avdlist.append(sqrt(avd2))

if len(ZCWorients_list) > 1:
    ax = plt.gca()
    artist, = ax.plot(ZCWorients_list, avdlist,'bx')
    artist.set_label('Explicit powder average')
    plt.xlabel('Number of powder orientations (ZCW sampling)')
    plt.ylabel('RMSD / kHz')
    artist, = ax.plot([21, 144],[contrib_RMSD_powder]*2,'k--')
    artist.set_label('Analytical prediction')
    ax.legend(frameon=False)
