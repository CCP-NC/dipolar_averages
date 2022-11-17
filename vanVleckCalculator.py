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
from soprano.properties.linkage import Molecules, MoleculeCOM
from soprano.utils import minimum_supcell, supcell_gridgen

from ase import io, Atoms
from ase.quaternions import Quaternion

verbose = 0


#np.seterr(all='raise')

# Typical command line arguments:
# --radius 10 --axis C1,C53:2 -v ../TRIAMT01_geomopt-out.cif
# --radius 1 -v TEST --axis H1,C1:3

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
        struct = io.read(fname, store_tags=True)
    try:
        kinds = struct.get_array('spacegroup_kinds')
        struct.new_array('site_labels',
                         np.array(struct.info['_atom_site_label'])[kinds])
    except KeyError as exc:
        raise KeyError("Didn't find site labels in input file %s" %
                       fname) from exc
    return struct


def molecule_crystallographic_types(struct):
    """ Loads a structure using ASE, ensuring that site labels are present.

    Parameters
    ----------
    struct : ASE Atoms object
        Source structure

    Returns
    -------
    AtomSelection array
        Set of molecules found in structure
    dict of list of indices
        Molecules grouped by `chemical signature` as dict key. The signatures
        differentiate the atoms but are not readily parseable. The lists
        are indices into the molecules array

    Notes
    -----
    Requires `spacegroup_kinds` array to be present in loaded structure.
    This requires an up-to-date ASE.
    """

    mols = Molecules.get(struct)
    kinds = struct.get_array('spacegroup_kinds')
    # Find which ones are equivalent
    mol_types = defaultdict(list)
    for i, m in enumerate(mols):
        m_signature = frozenset(Counter(kinds[m.indices]).items())
        mol_types[m_signature].append(i)

    return mols, mol_types


def average_dipolar_tensor(p1, p2, gamma, intramolecular):
    """Average dipolar tensor from a range of positions for two spins

    Parameters
    ----------
    p1 : numpy 1D array of 3-vectors
        Positions of atom 1
    p2 : numpy 1D array of 3-vectors
        Positions of atom 2
    gamma : scalar
        Common magnetogyric ratio
    intramolecular : bool
        If `True` the two spins are treated as part of the same molecule
        and thus rotating together.

    Returns
    -------
    NMRTensor object
        Averaged dipolar coupling tensor (in kHz)

    Notes
    -----
    Limited to homonuclear couplings (i.e. between same isotope) due to
    common gamma. Uses internal Soprano function `_dip_constant` which should
    be public. Axes are ordered using Haeberlen convention, which is not
    typical for zero-trace tensors, but X vs. Y distinction is not significant
    for calculation.
    """

    if intramolecular:
        r = p2 - p1
    else:
        # Urgh. Presumably evaluates all combinations of internuclear vectors
        r = (p2[:, None, :]-p1[None, :, :]).reshape((-1, 3))

    rnorm = np.linalg.norm(r, axis=-1)
    # Evaluate dipolar coupling(s) (in kHz) and internuclear unit vectors
    r /= rnorm[:, None]
    d = _dip_constant(rnorm*1e-10, gamma, gamma)*1e-3

    # Construct dipolar tensors (Cartesian 3 x 3 matrices) and average
    D = d[:, None, None]*(3*r[:, None, :]*r[:, :, None]-np.eye(3)[None])
    D = np.average(D, axis=0)

    # Convert to tensor object for readout
    return NMRTensor(D, order=NMRTensor.ORDER_HAEBERLEN)


def van_vleck_contribution(D, I, axis=None):
    """Compute the Van Vleck second moment contribution of a (homonuclear)
    dipolar tensor

    Parameters
    ----------
    D : Soprano NMRTensor object
        Dipolar tensor
    I : float
        Common I of spins involved
    axis : 3-vector, optional
        Axis of applied field direction. If `None`, an analytical powder
        average value is returned.

    Returns
    -------
    float :
        Contribution to second moment (in kHz^2?)

    Notes
    -----
    The eigenvalues are assumed to be ordered with the largest component at
    index 2.
   """

    d = D.eigenvalues[2]
    eta = D.asymmetry

    if axis is not None:
        Z = D.eigenvectors[:, 2]
        Y = D.eigenvectors[:, 1]
        ct = np.dot(axis, Z)
        cp = np.dot(axis, Y)

        Bjk = d*(1.5*(1+eta/3)*(3*ct**2-1)/2+eta*(3*cp**2-1)/2.0)
        return I*(I+1)*Bjk**2/3.0
    else:
        B2 = d**2*(9/4*(1+eta/3)**2+eta**2-1.5*(1+eta/3)*eta)
        return I*(I+1)*B2/15.0

def D2_contribution(D, axis=None):
    """Compute the contribution of a (homonuclear) dipolar tensor to D^2

    Parameters
    ----------
    D : Soprano NMRTensor object
        Dipolar tensor
    axis : 3-vector, optional
        Axis of applied field direction. If `None`, an analytical powder
        average value is returned.

    Returns
    -------
    float :
        Contribution to second moment (in kHz^2?)

    Notes
    -----
    The eigenvalues are assumed to be ordered with the largest component at
    index 2.
   """

    d = D.eigenvalues[2]
    eta = D.asymmetry

    if axis is not None:
        raise RuntimeWarning("Not implemented")
    else:
        B2 = d**2*(9/4*(1+eta/3)**2+eta**2-1.5*(1+eta/3)*eta)
        return (4/9)*B2

class RotationAxis(object):
    """ Class defining an axis of rotation

    Attributes
    ----------
    axis_str : string
        User specification of axis <label>[,<label>][:<n>]
        where <label> is an atom site label. `n` specifies a :math:`C_n`
        axis.
    n : integer
        `n` for a :math:`C_n` specification, with 0 corresponding to free
        rotation
    a1 : string
        Site label for atom 1
    a2 : string
        Site label for atom 2. ``None`` if axis passes through a pair of atoms
        with label **a1**
    force_com : bool, optional
        If ``True``, translate axis so that it passes through centre-of-mass
        (default ``False``).This allows the axis to be defined using a pair
        of atoms that have the correct relative orientation, but which do
        not sit on the desired axis.
    off_com_tol : float, optional
        Tolerance for distance (in A) between centre of mass (CoM) of molecule
        and axis (default 0.1 A). Rotation about an axis away from the CoM
        is unphysical and would suggest that the axis is passing through
        the wrong atoms. ``None`` disables check.
   """

    def __init__(self, axis_str, force_com=False, off_com_tol=0.1):
        """
        Raises
        ------
        ValueError
            If `n` in the :math:`C_n` specification in `axis_str` is outside
            the range 2 to 6
        """

        axre = re.compile('([A-Za-z0-9]+)(?:,([A-Za-z0-9]+))?(?::([0-9]))?')
        m = axre.match(axis_str)

        if not m:
            raise ValueError('Invalid axis definition {}'.format(axis_str))

        a1, a2, n = m.groups()
        if n:
            n = int(n)
            if (n < 2 or n > 6):
                raise ValueError('Invalid n = {} for axis definition'.format(n))
        else:
            n = 0
        if a1 == a2:  # capture a1 = a2 special case (code will fail later if not)
            a2 = None

        self.axis_str = axis_str
        self.n = n
        self.a1 = a1
        self.a2 = a2
        self.force_com = force_com
        self.off_com_tol = off_com_tol

    def validate(self, rmol):
        """ Check if this axis is valid for the given molecule

        Parameters
        ----------
        rmol : RotatingMolecule
            Molecule for which to validate axis

        Returns
        -------
        3 x 3 numpy array:
            Rotation matrix for :math:`C_n` rotation about axis
        3 vector:
            Point on axis
        integer:
            Value of n

        Raises
        ------
        ValueError
            If the axis is not valid for **rmol**

        Notes
        -----
        The code will behave unexpectedly if either of the two labels are not
        present in the molecule, but other label returns 2 atoms. Rather
        than generate an error (expected), the axis will be based on
        the pair of atoms that are present.

        Return values for n = 0 currently unclear.
        """

        labels = rmol.labels

# Suspect code here (see Notes)
# Really should check that labels are positively present
        ax_indices = np.concatenate((np.where(labels == self.a1)[0],
                                     np.where(labels == self.a2)[0]))
        if ax_indices.shape != (2,):
            raise ValueError('Invalid axis: {} does not define two atoms'
                             ' in molecule'.format(self.axis_str))

        i1, i2 = ax_indices
        p1, p2 = rmol.positions[ax_indices]

        v = p2 - p1
        v /= np.linalg.norm(v)  # unit vector defining axis direction

        if self.force_com:
            # Force this to pass through the center of mass
            p1 = rmol.com
            p2 = p1 + v
        elif self.off_com_tol is not None:
            # Does the axis pass through the CoM?
            r = rmol.com - p1
            d = np.linalg.norm(r-(r@v)*v)
            if d > self.off_com_tol:
                raise ValueError('Axis does not pass through centre-of-mass '
                    '(within tolerance of {} A)'.format(self.off_com_tol))

        # Quaternion
        if self.n > 0:
            q = Quaternion.from_axis_angle(v, 2*np.pi/self.n)
            R = q.rotation_matrix()
            n = self.n
        else:
            R = v[:, None]*v[None, :]
            n = 1  # Why returning 1 and not 0?

        return (R, p1, n)

    def __repr__(self):
        return self.axis_str


class RotatingMolecule(object):
    """ Class defining a rotating molecule

    Attributes
    ----------
    cell : ASE cell object
        Unit cell definition from passed structure
    s : Atoms object
        Subset of the original structure containing atoms of molecule with
        co-ordinates adjusted for given supercell position.
    positions : array of 3-vector
        Positions of atoms of `s`
    com : 3-vector
        Centre of mass of molecule (recalculated for shifted position)
    symbols : array of string
        Element labels of atoms of `s`
    labels : array of string
        Site labels of atoms of `s`
    rot_positions : 3-dimensional array
        All atomic positions rotated through each combination of rotations
        First index is atom number, second is rotation index, last
        dimension is 3-vector
    selected_rotpos : list of tuples of (atom label, atom position)
        Resclicing of `rot_positions` selecting only `element`

    Notes
    -----
    Currently code creates a RotatingMolecule for every molecule within the
    selection radius, even if unaffected by motion (Z' > 1). Not clear whether
    this is the Right Thing.

    The code has two tests for non-commuting rotations. The first should always
    be passed for Z' = 1, since all the axes should pass through the CoM
    unless the tolerance is large (no allowance is made for a large
    off_com_tol). The second is expected to fail, e.g. for 2 C3 rotations.
    But it's not clear why this should generate an exception rather than a
    warning.

    All atomic positions in molecule are evaluated, even though only a subset
    corresponding to selected isotope are needed. Similarly symbols, labels
    arrays duplicate information. It might be cleaner to have an overall
    super object factoring out common information. Should be easier now that
    element is passed to initialiser.
    """

    def __init__(self, s, mol, axes, ijk=[0, 0, 0], element=None,
                 ignoreinvalid=False, checkcommuting=True):
        """
        Parameters
        ----------
        s : ASE Atoms object
            Initial structure object
        mol : AtomSelection
            Selector for molecule of interest
        axes : list of RotationAxis objects
            Axis definitions (empty list corresponds to no rotation)
        ijk : 3-vector, optional
            Cell offset in fractional co-ordinates. Default is no offset
        element : character
            Select atom type (optional). `None` corresponds to all atoms
        ignoreinvalid : bool
            Ignore axes that are invalid for this molecule (default `False`)
        checkcommuting : bool
            Apply commutation test to axes (see Notes). Default is `True`
        """

        self.cell = s.get_cell()
        self.s = mol.subset(s, use_cell_indices=True)
        self.s.set_positions(self.s.get_positions() +
                             self.cell.cartesian_positions(ijk))
        self.positions = self.s.get_positions()
        self.symbols = np.array(self.s.get_chemical_symbols())
        self.com = self.s.get_center_of_mass()  # Duplicates previous CoM calculation?

        # CIF labels
        self.labels = self.s.get_array('site_labels')

        # Now analyze how each axis affects the molecule
        self.rotations = []
        for ax in axes:
            try:
                self.rotations.append(ax.validate(self))
            except ValueError as e:
                if ignoreinvalid:
                    print('Skipping axis {0}: {1}'.format(ax, e))
                else:
                    raise

        # Do they all commute (see Notes)?
        if checkcommuting:
            for i, (R1, o1, n1) in enumerate(self.rotations):
                for j, (R2, o2, n2) in enumerate(self.rotations[i+1:]):
                    # Check that o2 is invariant under R1
                    o2r = (R1 @ (o2-o1)) + o1
                    if not np.isclose(np.linalg.norm(o2r-o2), 0.0):
                        raise ValueError('Molecule has rotations with'
                                         ' non-intersecting axes')
                    # Check that R1 and R2 commute
                    comm = R1 @ R2 - R2 @ R1
                    if not np.isclose(np.linalg.norm(comm), 0.0):
                        raise ValueError('Molecule has non-commuting rotations')
            if verbose:
                print("Axis commutation check passed")

        if len(self.rotations) == 0:
            # Just regular positions...
            rot_positions = self.positions[:, None, :]
        else:
            rot_positions = []
            for p in self.positions:
                all_rotations = [p]
                # successively apply rotations to initial point
                for rot_def in self.rotations:
                    all_rotations = [self._expand_rotation(x, *rot_def)
                                     for x in all_rotations]
                    all_rotations = np.concatenate(all_rotations) # Some kind of flattening?
                rot_positions.append(all_rotations)

        self.rot_positions = np.array(rot_positions)

        if element:
            indices = np.where(self.symbols == element)[0]
        else:
            indices = np.arange(len(self.s))
        self.selected_rotpos = [(self.labels[i], self.rot_positions[i]) for i in indices]


    @staticmethod
    def _expand_rotation(x, R, o, n):
        """ Internal function to generate set of positions corresponding
        to rotation about :math:`C_n`

        Parameters
        ----------
        x : 3-vector
            Starting position (Cartesian axes)
        R : 3 x 3 array
            Rotation matrix
        o : 3-vector
            Origin (point on rotation axis)
        n : integer
            `n` of :math:`C_n`, where `n` is guaranteed to be >1

        Returns
        -------
        array of 3-vectors:
            Set of `n` positions
        """
        all_rotations = [x]
        for _ in range(1, n):
            x = all_rotations[-1]
            x = (R @ (x-o)) + o
            all_rotations.append(x)
        return np.array(all_rotations)


if __name__ == "__main__":

    parser = ap.ArgumentParser()
    parser.add_argument('structure',
                        help="A structure file containing site labels in an "
                        "ASE readable format, typically .cif")
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
                        nargs=3, type=float,
                        help="Calculate for single orientation (rather than "
                        "powder) with an overall rotation of crystal"
                        "system expressed as ZYZ Euler angles in degrees")
#    parser.add_argument('--powder', '-p', dest="powder", action="store_true",
#                        default=False,
#                        help="Use powder averaging for all second moments")
    parser.add_argument('--verbose', '-v', default=0,
                        help="Increase verbosity", action='count')

    args = parser.parse_args()
    verbose = args.verbose

    testmode = args.structure.startswith("TEST")
    if testmode:
        cell_dimensions = [3.0, 4.0, 5.0]
        structure = Atoms(['H', 'H', 'C'],
                       positions=[(1.0, 1.0, 1.0), (2.0, 1.0, 1.0), (1.0, 2.0, 1.0)],
                       cell=cell_dimensions,  # orthorhombic cell
                       pbc=True)
        structure.new_array('site_labels', np.array(['H1', 'H2', 'C1']))
    else:
        structure = read_with_labels(args.structure)

    # Orientation
    if args.euler_rotation is None:
        B_axis = None
    else:
        B_axis = np.array([0, 0, 1.0])
        euler = np.array(args.euler_rotation)*np.pi/180.0
        rotation_quat = Quaternion.from_euler_angles(*euler)
        axis, angle = rotation_quat.axis_angle()
        structure.rotate(180/np.pi*angle, v=axis, rotate_cell=True)

    # NMR data
    # Not ideal - should be isotope rather than element based
    element = args.element
    el_I = _get_isotope_data([element], 'I')[0]
    el_gamma = _get_isotope_data([element], 'gamma')[0]

    # Parse axes. Note that application order shouldn't matter
    axes = [RotationAxis(a, False) for a in args.axes]
    axes += [RotationAxis(a, True) for a in args.CoMaxes]

    # Find molecules
    if testmode:
        mols = Molecules.get(structure)
        mol_types = {None: mols}
    else:
        mols, mol_types = molecule_crystallographic_types(structure)
    mol_coms = MoleculeCOM.get(structure)

    # Types of molecule?
    Z = len(mols)
    Zp = len(mol_types)

    print('Structure analysed: Z = {}, Z\' = {}'.format(Z, Zp))

    R = args.radius

    # Which is the central molecule?
    # This is rather confusing. If Zp > 1, then it would make sense to
    # determine which axis definitions work for which molecules.
    # The code probably does not work for axes operating on different molecules
    # The 'central label' seems unnecessary
    if Zp == 1:
        mol0_i = 0
    else:
        mol0_i = None
        cl = args.central_label
        for i, m in enumerate(mols):
            if cl in m.get_array('site_labels'):
                mol0_i = i
                break
        if mol0_i is None:
            raise RuntimeError("Must specify a central label for systems with "
                               "Z' > 1")

    # Find the centre of mass
    mol0_com = mol_coms[mol0_i]

    # Determine the shape of supercell required to include a given radius
    # Not related to molecules at this stage.
    scell = minimum_supcell(R, structure.get_cell())
    # xyz is supercell grid (offset from 0,0,0?) in Cartesian co-ordinates
    # fxy in fractional
    fxyz, xyz = supcell_gridgen(structure.get_cell(), scell)

    # Determine distances between CoM of each molecule combined
    # with each supercell offset and CoM of reference molecule
    mol_dists = np.linalg.norm(mol_coms[:, None, :]-mol0_com+xyz[None, :, :],
                               axis=-1)
    # Identify indices of molecules (within mol_dists) that lie within radius
    # excluding reference molecule (zero distance)
    # Note latter test assumes no rounding errors introduced
    # Should be obvious if this goes wrong
    sphere_i = np.where((mol_dists <= R)*(mol_dists > 0))

    # For Z' > 1, skip over axes that are not valid for molecule
    # Apparently no check that axes are valid for at least one molecule in
    # asymmetric unit, i.e. code is flawed / surprising for Z' > 1
    ignoreinvalid = (Zp > 1)

    # Always start with the centre
    rmols = [RotatingMolecule(structure, mols[mol0_i], axes, element=element, ignoreinvalid=ignoreinvalid)]
    # Now create RotatingMolecule objects for the `other' molecules
    for mol_i, cell_i in zip(*sphere_i):
        rmols.append(RotatingMolecule(structure, mols[mol_i], axes, fxyz[cell_i],
                        element=element, ignoreinvalid=ignoreinvalid))

    print("Number of molecules for intermolecular interactions: "
          "{}".format(len(rmols)-1))

    # Now go on to compute the actual couplings
    # rmols_rotpos = [rmol.atoms_rot_positions(element) for rmol in rmols]

    rmol0_rotpos = rmols[0].selected_rotpos

    natoms = len(rmol0_rotpos)
    intra_moments = np.zeros(natoms)
    inter_moments = np.zeros(natoms)

    for i, (_, pos1) in enumerate(rmol0_rotpos):

        # Intramolecular couplings
        for j, (_, pos2) in enumerate(rmol0_rotpos[i+1:]):
            D = average_dipolar_tensor(pos1, pos2, el_gamma, intramolecular=True)
            D2 = D2_contribution(D, B_axis)

            # Add contribution to intramolecular sum for both spins
            intra_moments[i] += D2
            intra_moments[i+j+1] += D2

        # For everything else we can't save time the same way
        for rmol2 in rmols[1:]:
            rmol2_rotpos = rmol2.selected_rotpos
            for _, pos2 in rmol2_rotpos:
                D = average_dipolar_tensor(pos1, pos2, el_gamma, intramolecular=False)
                inter_moments[i] += D2_contribution(D, B_axis)

    intra_moments_dict = defaultdict(list)
    inter_moments_dict = defaultdict(list)
    for i in range(natoms):
        lab = rmol0_rotpos[i][0]
        intra_moments_dict[lab].append(intra_moments[i])
        inter_moments_dict[lab].append(inter_moments[i])

    print("Label\tIntra-drss/kHz\tInter-drss/kHz")

    if args.nomerge:
        for i in range(natoms):
            lab = rmol0_rotpos[i][0]
            intram = intra_moments[i]
            interm = inter_moments[i]
            print("{}\t{:.2f}\t{:.2f}".format(lab, intram**0.5, interm**0.5))
    else:
        def getstats(vals):
            """ Return average of values that are expected to be same within
            rounding error (+range as fraction) """

            mean = np.mean(vals)
            if np.isclose(mean, 0):
                return mean, 0
            frac = (max(vals)-min(vals))/mean
            return mean, frac

        for lab, intram in intra_moments_dict.items():
            intra_mean, intra_frac = getstats(intram)
            inter_mean, inter_frac = getstats(inter_moments_dict[lab])

            print("{}\t{:.2f} ({:.2f}%)\t{:.2f} ({:.2f}%)".format(lab,
                                intra_mean**0.5, 100.*intra_frac,
                                inter_mean**0.5, 100.*inter_frac))

    mean_dSS_intra = np.mean(intra_moments)
    mean_dSS_inter = np.mean(inter_moments)
    total_dSS = mean_dSS_intra + mean_dSS_inter

    print("Intramolecular contribution to mean d_SS: {:.2f} kHz^2".format(
        mean_dSS_intra))
    print("Intermolecular contribution to mean d_SS at "
          "{:g} Å: {:.2f} kHz^2".format(R, mean_dSS_inter))
    print("Overall mean d_SS: {:.2f} kHz^2    Mean d_RSS {:.2f} kHz".format(total_dSS, total_dSS**0.5))
    M2 = (3.0/20)*total_dSS*el_I*(el_I+1)
    print("Second moment: {:.2f} kHz^2".format(M2))
