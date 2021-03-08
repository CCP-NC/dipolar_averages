#!/usr/bin/env python

"""vanVleckCalculator.py

A script to compute and average the Van Vleck second moments in molecular
crystals, including the effects of rotational motion.

Made by Simone Sturniolo and Paul Hodgkinson for CCP-NC (2021)
"""

import re
import warnings
import argparse as ap
import numpy as np

from collections import defaultdict, Counter
from soprano.nmr.tensor import NMRTensor
from soprano.nmr.utils import _get_isotope_data, _dip_constant
from soprano.properties.transform import Rotate
from soprano.properties.linkage import Molecules, MoleculeCOM
from soprano.utils import minimum_supcell, supcell_gridgen

from ase import io
from ase.quaternions import Quaternion


def read_with_labels(fname):
    """ Loads a structure using ASE, ensuring that site labels are present.

    Parameters
    ----------
    fname : str
        filename

    Returns
    -------
    Atoms
        ASE structure object

    Raises
    ------
    KeyError
        If site label information could not be found.

    Notes
    -----
    Requires `spacegroup_kinds` array and `_atom_site_label` information
    to be present in loaded structure. This requires an up-to-date ASE.
    """

    # Load the structure with ASE
    # Note: this can produce some warnings, they're useless, we're silencing
    # them
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        struct = io.read(args.structure, store_tags=True)
    # (download from Gitlab, https://gitlab.com/ase/ase)
    try:
        kinds = struct.get_array('spacegroup_kinds')
        struct.new_array('site_labels',
                         np.array(struct.info['_atom_site_label'])[kinds])
    except KeyError as exc:
        raise KeyError("Didn't find site labels in input file %s" %
                       fname) from exc
    return struct


def molecule_crystallographic_types(s, mols):

    kinds = s.get_array('spacegroup_kinds')
    # Find which ones are equivalent
    mol_types = defaultdict(list)
    for i, m in enumerate(mols):
        m_signature = frozenset(Counter(kinds[m.indices]).items())
        mol_types[m_signature].append(i)

    return mol_types


def average_dipolar_tensor(p1, p2, gamma, intra=False):
    """Average dipolar tensor from a range of positions
    for two spins of given gamma. If intra=True, considers
    the two spins part of the same molecule and thus rotating together.

    Returns values in kHz.

    """

    p1 = np.array(p1)
    p2 = np.array(p2)

    if intra:
        r = p2-p1
    else:
        r = (p2[:, None, :]-p1[None, :, :]).reshape((-1, 3))

    rnorm = np.linalg.norm(r, axis=-1)
    r /= rnorm[:, None]
    d = _dip_constant(rnorm*1e-10, gamma, gamma)*1e-3

    D = d[:, None, None]*(3*r[:, None, :]*r[:, :, None]-np.eye(3)[None])
    D = np.average(D, axis=0)

    D = NMRTensor(D, order=NMRTensor.ORDER_HAEBERLEN)

    d = D.eigenvalues[2]
    eta = D.asymmetry
    z = D.eigenvectors[:, 2]
    y = D.eigenvectors[:, 1]

    return (d, z, eta, y)


def van_vleck_contribution(d, z, eta, y, I, axis=None):
    """Compute the Van Vleck second moment contribution of a dipolar tensor
    given the largest (absolute) eigenvalue d, its eigenvector z, the 
    asymmetry eta, and the eigenvector y of the smallest (absolute) eigenvalue.
    Needs also the magnitude of the spin, I.

    Also takes an axis for an applied field direction. If missing, a powder
    average is returned instead.
    """

    if axis is not None:
        ct = np.dot(axis, z)
        cp = np.dot(axis, y)

        Bjk = d*(1.5*(1+eta/3)*(3*ct**2-1)/2+eta*(3*cp**2-1)/2.0)
        return I*(I+1)*Bjk**2/3.0
    else:
        B2 = d**2*(9/4*(1+eta/3)**2+eta**2-1.5*(1+eta/3)*eta)
        return I*(I+1)*B2/15.0


class RotationAxis(object):

    def __init__(self, axis_str, force_com=False):

        axre = re.compile('([A-Za-z0-9]+)(,[A-Za-z0-9]+)*(:[0-9])*')
        m = axre.match(axis_str)

        if not m:
            raise ValueError('Invalid axis definition {0}'.format(axis_str))

        a1, a2, n = m.groups()
        a2 = a2[1:] if a2 else None
        n = int(n[1:]) if n else 0

        if (n < 0 or n > 6):
            raise ValueError('Invalid n = {0} for axis definition'.format(n))

        self.axis_str = axis_str
        self.n = n
        self.a1 = a1
        self.a2 = a2
        self.com = force_com

    def validate(self, rmol):
        # Check if this axis applies to the given RotatingMolecule
        labels = rmol.labels

        ax_indices = np.concatenate((np.where(labels == self.a1)[0],
                                     np.where(labels == self.a2)[0]))
        if ax_indices.shape != (2,):
            raise ValueError('Invalid axis; axis does not define two atoms'
                             ' in molecule')

        i1, i2 = ax_indices
        p1, p2 = rmol.positions[ax_indices]

        v = p2-p1
        v /= np.linalg.norm(v)  # Vector

        if self.com:
            r = (rmol.com-p1)
            if not(np.isclose(r @ v, np.linalg.norm(r))):
                raise ValueError('Axis does not pass through COM')

        # Quaternion
        if self.n > 0:
            q = Quaternion.from_axis_angle(v, 2*np.pi/self.n)
            R = q.rotation_matrix()
            n = self.n
        else:
            R = v[:, None]*v[None, :]
            n = 1

        return (R, p1, n)

    def __repr__(self):
        return self.axis_str


class RotatingMolecule(object):

    def __init__(self, s, mol, ijk=[0, 0, 0], axes=[]):

        self.cell = s.get_cell()
        self.s = mol.subset(s, use_cell_indices=True)
        self.s.set_positions(self.s.get_positions() +
                             self.cell.cartesian_positions(ijk))
        self.positions = self.s.get_positions()
        self.symbols = np.array(self.s.get_chemical_symbols())
        self.com = self.s.get_center_of_mass()

        # CIF labels
        self.labels = self.s.get_array('site_labels')

        # Now analyze rotation axes
        self.rotations = []
        for ax in axes:
            try:
                R, o, n = ax.validate(self)
            except ValueError as e:
                print('Skipping axis {0}: {1}'.format(ax, e))
            self.rotations.append((R, o, n))

        # Do they all commute?
        for i, (R1, o1, n1) in enumerate(self.rotations):
            for j, (R2, o2, n2) in enumerate(self.rotations[i+1:]):
                # Check that o2 is invariant under R1
                o2r = (R1 @ (o2-o1)) + o1
                if not np.isclose(np.linalg.norm(o2r-o2), 0):
                    raise ValueError('Molecule has multiple rotations with'
                                     ' non-intersecting axes')
                # Check that R1 and R2 commute
                comm = R1 @ R2 - R2 @ R1
                if not np.isclose(np.linalg.norm(comm), 0):
                    raise ValueError('Molecule has multiple non-commuting'
                                     ' rotations')

        rot_positions = []

        for p in self.positions:
            all_rotations = []
            for (R, o, n) in self.rotations:
                for i in range(n):
                    x = all_rotations[-1] if i > 0 else p
                    x = (R @ (x-o)) + o
                    all_rotations.append(x)
            rot_positions.append(all_rotations)

        if len(self.rotations) == 0:
            # Just regular positions...
            rot_positions = self.positions[:, None, :]

        self.rot_positions = np.array(rot_positions)

    def atoms_rot_positions(self, element=None):

        if element:
            indices = np.where(self.symbols == element)[0]
        else:
            indices = np.arange(len(self.s))
        return [(self.labels[i], self.rot_positions[i]) for i in indices]


class VanVleckCalculator(object):

    def __init__(self, rmols, element='H'):

        self.element = element
        self.el_gamma = _get_isotope_data([element], 'gamma')[0]
        self.el_I = _get_isotope_data([element], 'I')[0]


if __name__ == "__main__":

    parser = ap.ArgumentParser()
    parser.add_argument('structure',
                        help="A structure file containing site labels in an "
                        "ASE "
                        "readable format, typically .cif")
    parser.add_argument('--element', '-e', default='H', dest='element',
                        help="Element for which to compute the (homonuclear) "
                        "dipolar couplings")
    parser.add_argument('--central_label', '-c', default=None,
                        dest='central_label',
                        help="Crystallographic label of atom to use to define"
                        " a central molecule in case of more than one type")
    parser.add_argument('--axis', '-a', action='append', dest='axes',
                        default=[],
                        help="Specify an axis through "
                        "first_atom_label[,second_atom_label][:n]")
    parser.add_argument('--CoMaxis', action='append', dest='CoMaxes',
                        default=[],
                        help="Specify an axis through Centre of "
                        "Mass as first_atom_label[,second_atom_label][:n]")
    parser.add_argument('--nomerge', dest='nomerge', action="store_true",
                        help="Don't merge results from sites with same label")
    parser.add_argument('--radius', '-r', dest="radius", type=float,
                        default=10.0,
                        help="Radius over which to include molecules (in Å)")
    parser.add_argument('--euler_rotation', '-er', dest="euler_rotation",
                        nargs=3, type=float, default=[0., 0., 0.],
                        help="Overall rotation of crystal system expressed at"
                        " ZYZ Euler angles in degrees")
    parser.add_argument('--powder', '-p', dest="powder", action="store_true",
                        default=False,
                        help="Use powder averaging for all second moments")
    parser.add_argument('--verbose', '-v',
                        help="Increase verbosity", action='count')

    args = parser.parse_args()
    structure = read_with_labels(args.structure)

    # Orientation
    B_axis = np.array([0, 0, 1.0])
    if args.powder:
        B_axis = None
    else:
        euler = np.array(args.euler_rotation)*np.pi/180.0
        rotation_quat = Quaternion.from_euler_angles(*euler)
        axis, angle = rotation_quat.axis_angle()
        structure.rotate(180/np.pi*angle, v=axis, rotate_cell=True)

    # NMR data
    element = args.element
    el_I = _get_isotope_data([element], 'I')[0]
    el_gamma = _get_isotope_data([element], 'gamma')[0]

    # Parse axes
    axes = [RotationAxis(a) for a in args.axes]
    axes += [RotationAxis(a, True) for a in args.CoMaxes]

    mols = Molecules.get(structure)
    mol_coms = MoleculeCOM.get(structure)
    mol_types = molecule_crystallographic_types(structure, mols)
    R = args.radius

    # Types of molecule?
    Z = len(mols)
    Zp = len(mol_types)

    print('Structure analysed: Z = {}, Z\' = {}'.format(Z, Zp))

    # Which is the central molecule?
    mol0_i = None
    if Zp == 1:
        mol0_i = 0
    else:
        cl = args.central_label
        for i, m in enumerate(mols):
            if cl in m.get_array('site_labels'):
                mol0_i = i
                break

    if mol0_i is None:
        raise RuntimeError("Must specify a central label for systems with "
                           "Z' > 1")

    # Find the origin
    mol0_com = mol_coms[mol0_i]

    # Find the necessary molecules
    scell = minimum_supcell(R, structure.get_cell())
    fxyz, xyz = supcell_gridgen(structure.get_cell(), scell)

    mol_dists = np.linalg.norm(mol_coms[:, None, :]-mol0_com+xyz[None, :, :],
                               axis=-1)
    sphere_i = np.where((mol_dists <= R)*(mol_dists > 0))

    # Always start with the centre
    rmols = [RotatingMolecule(structure, mols[mol0_i], axes=axes)]
    for mol_i, cell_i in zip(*sphere_i):
        rmol = RotatingMolecule(structure, mols[mol_i], fxyz[cell_i], axes)
        rmols.append(rmol)

    print("Number of molecules for intermolecular interactions: "
          "{}".format(len(rmols[1:])))

    # Now go on to compute the actual couplings
    rmols_rotpos = [rmol.atoms_rot_positions(element) for rmol in rmols]

    rmol0_rotpos = rmols_rotpos[0]

    intra_moments = np.zeros(len(rmol0_rotpos))
    inter_moments = np.zeros(len(rmol0_rotpos))

    for i, (l1, p1) in enumerate(rmol0_rotpos):

        # Intramolecular couplings
        for j, (l2, p2) in enumerate(rmol0_rotpos[i+1:]):
            D = average_dipolar_tensor(p1, p2, el_gamma, True)
            vv2 = van_vleck_contribution(*D, el_I, B_axis)

            intra_moments[i] += vv2
            intra_moments[i+j+1] += vv2

        # For everything else we can't save time the same way
        for rmol_apos in rmols_rotpos[1:]:
            for l2, p2 in rmol_apos:
                D = average_dipolar_tensor(p1, p2, el_gamma)
                vv2 = van_vleck_contribution(*D, el_I, B_axis)
                inter_moments[i] += vv2

    intra_moments_dict = defaultdict(list)
    inter_moments_dict = defaultdict(list)
    for i, m in enumerate(intra_moments):
        l = rmol0_rotpos[i][0]
        intra_moments_dict[l].append(m)
        inter_moments_dict[l].append(inter_moments[i])

    def getstats(moments):
        mean = np.mean(moments)
        if np.isclose(mean, 0):
            return mean, 0
        frac = (max(moments)-min(moments))/mean
        return mean, frac

    print("Label\tIntra-drss/kHz\tInter-drss/kHz")

    if args.nomerge:
        for i, intram in enumerate(intra_moments):
            l = rmol0_rotpos[i][0]
            interm = inter_moments[i]
            print("{}\t{:.4f}\t{:.4f}".format(l, intram**0.5, interm**0.5))
    else:
        for l, intram in intra_moments_dict.items():
            intra_mean, intra_frac = getstats(intram)
            inter_mean, inter_frac = getstats(inter_moments_dict[l])

            m1, pc1, m2, pc2 = (intra_mean**0.5,
                                100.*intra_frac,
                                inter_mean**0.5,
                                100.*inter_frac)
            print("{}\t{:.4f} ({:.2f}%)\t{:.4f} ({:.2f}%)".format(l, m1, pc1,
                                                                  m2, pc2))

    total_dRSS_intra = np.mean(intra_moments)**0.5
    total_dRSS_inter = np.mean(inter_moments)**0.5
    total_dRSS = np.mean(intra_moments+inter_moments)**0.5

    print("Intramolecular contribution to mean d_RSS: {:g} kHz".format(
        total_dRSS_intra))
    print("Intermolecular contribution to mean d_RSS at "
          "{:g} Å: {:g} kHz".format(R, total_dRSS_inter))
    print("Overall d_RSS: {:g} kHz".format(total_dRSS))
