# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 12:16:18 2023

@author: PAUL
"""

import numpy as np
from math import sin, cos, sqrt
from soprano.nmr.tensor import NMRTensor
from soprano.calculate.powder import ZCW
from ase.quaternions import Quaternion

# Test of averaging dipolar tensor with asymmetry

ZCWorients = 5

dav = 1.0
etaav = 0.3
print("Starting (averaged) tensor: d = {} kHz, eta = {}".format(dav, etaav))

contrib_RMSD_powder = abs(dav)*sqrt(0.2 + (etaav ** 2)/15.0)
print("RMS D (powder) contrib (formula 1): {} kHz".format(contrib_RMSD_powder))

angles, weights = ZCW('sphere').get_orient_angles(ZCWorients)
N = len(weights)
print("Powder averaging over {} ZCW orientations".format(N))


def P2(x):
    return 0.5*(3.0*x*x-1)


dcontribs = np.zeros(N)
danal = np.zeros(N)
gamma2s = np.zeros(N)
zeta2s = np.zeros(N)
gamma4s = np.zeros(N)
zeta4s = np.zeros(N)
gamma2zeta2s = np.zeros(N)
for i, zorientation in enumerate(angles):
    theta, phi = zorientation
    cosg , coszeta = map(cos, zorientation)
    P2cosg, P2coszeta = map(P2, (cosg, coszeta))
    quat = Quaternion.from_euler_angles(phi, theta, 0.0)
    DCM = quat.rotation_matrix()
    print("DCM\n", DCM)
    evals = dav * np.array([-1-etaav, -1+etaav, 2.0])
#    RouterR = np.outer(DCM[:,2], DCM[:,2])
#    print("R outer R\n", RouterR)
#    SouterS = np.outer(DCM[:,1], DCM[:,1])
#    print("S outer S\n", SouterS)
#    TouterT = np.outer(DCM[:,0], DCM[:,0])
#    print("T outer T\n", TouterT)
#    newD = dav * (evals[0] *TouterT + evals[1] * SouterS + evals[2] * RouterR)
#    print("Reconstructed D\n", newD)
#   Confirmed that statement below is implementing Eq. (30)-(31)

    D = np.linalg.multi_dot([DCM, np.diag(evals), DCM.T])
    print("D\n", D)
    dOmega = 0.5*D[2, 2]  # d_zz at given orientation
    CSApred = dav * (P2cosg - (0.5*etaav)*(sin(theta)**2)*cos(2*phi))
    print(zorientation* (180.0/np.pi))

    print("gamma = cos(beta): {} \t zeta = cos(phi): {}".format(cosg, coszeta))
    gamma2s[i] = cosg**2
    zeta2s[i] = coszeta**2
    gamma4s[i] = cosg**4
    zeta4s[i] = coszeta**4
    gamma2zeta2s[i] = cosg**2 * coszeta**2
    print(P2cosg, P2coszeta)
    danal[i] = dOmegaanalytical = dav*((1+etaav/3)*P2cosg + (2*etaav/3)*P2coszeta)
    dcontribs[i] = dOmega
    print("dOmega: {} \t CSAform: {}".format(dOmega, CSApred))

d2contribs = [dip*dip for dip in dcontribs]
avd = np.average(dcontribs, weights=weights)  # NB weights should be uniform for ZCW
avd2 = np.average(d2contribs, weights=weights)
print("Average D (should be close to zero): {} kHz".format(avd))
print("RMS D: {} kHz".format(sqrt(avd2)))
print("<gamma^2>: %g" % np.average(gamma2s, weights=weights))
print("<zeta^2>: %g" % np.average(zeta2s, weights=weights))
print("<gamma^4>: %g" % np.average(gamma4s, weights=weights))
print("<zeta^4>: %g" % np.average(zeta4s, weights=weights))
print("<gamma^2 zeta^2>: %g" % np.average(gamma2zeta2s, weights=weights))
