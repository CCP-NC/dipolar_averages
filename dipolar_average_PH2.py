#!/usr/bin/env python

from collections import defaultdict, Counter
from types import SimpleNamespace
import argparse
import warnings
import sys
from math import sqrt, pi, cos, sin, atan2
import numpy as np
from numpy.linalg import norm
from ase import io, Atoms, Atom
from ase.quaternions import Quaternion
from soprano.selection import AtomSelection
from soprano.properties.linkage import Molecules
from soprano.properties.transform import Rotate
#from soprano.properties.nmr import DipolarCoupling
from soprano.utils import minimum_supcell
from soprano.nmr.utils import _dip_constant, _get_isotope_data
from soprano.nmr import NMRTensor

# Typical command line arguments:
# TRIAMT01_geomopt-out.cif 15 --logfile dsstest15A.txt --phiaverage 42 --thetaaverage 40

default_el = 'H'
off_axis_tol = 0.1  # warning if centre of mass is more than this distance (in A) from rotation axis
centre_molecule = 0
skip_intramolecular = False
testmode = True # test behaviour with model system
verbose = 0  # Default verbose level
probe_indices = None
#{(0, 2)}
dump_mol = False


def prettyprintmatrix33(a, lead='\t\t'):
    for k in range(3):
        print('{}{:.3f} {:.3f} {:.3f}'.format(lead, *a[k]))


deg_to_rad = pi/180.0
rad_to_deg = 180.0/pi


def molecule_generator(mol_list, radius, centre_molecule=None, use_symmetry=True):
    """ Return indices of molecules within a given radius of an origin molecule.

    Parameters
    ----------
    mol_list : list-type of RotatedMolecule
        List of molecules within unit cell.
    radius : float
        Cut-off radius (in angstrom) for molecules to be included.
    centre_molecule : int
        Index of molecule to be used as centre. Not currently optional
        since `use_symmetry` option is invalid without it.
    use_symmetry : bool
        Only return molecules that are unique by translation symmetry.
        Only cell indices >= 0 will be returned, with the weighting factor
        indicated how many symmetry-equivalent molecules are involved.
        Default: True.

    Yields
    ------
    int
        Molecule index (into original `mol_list` parameter).
    list of int:
        Cell indices (nx, ny, nz).
    int
        Number of symmetry-equivalent molecules (1 if `use_symmetry` is `False`).

    Notes
    -----
    May silently fail if origin molecule is not within centre unit cell.
    """

    try:
        lattice = mol_list[0].molecule.get_cell()
    except (TypeError, IndexError):
        raise TypeError("molecule_generator: first argument must be a non-empty list-type object")

    if centre_molecule is None:
        raise ValueError("molecule_generator: origin is not specified")
    else:
        try:
            ignore_molecule = mol_list[centre_molecule]
        except IndexError:
            raise IndexError("{} is not a valid molecule index (maximum is {})".format(centre_molecule, len(mol_list)))
        origin = ignore_molecule.get_center_of_mass()
        yield centre_molecule, [0, 0, 0], 1  # Return centre molecule first (so can be easily skipped)

    # minimum_supcell is returns a range that is always >= the range needed
    # Will do the Right Thing for non-periodic dimensions (range is 0)
    scell = minimum_supcell(radius, lattice)
    for nx in range(-scell[0], scell[0]+1):
        for ny in range(-scell[1], scell[1]+1):
            for nz in range(-scell[2], scell[2]+1):
                cell_indices = [nx, ny, nz]
                cell_offset_cart = lattice.cartesian_positions(cell_indices)
                total_offset_cart = cell_offset_cart - origin
                for mol_no, mol in enumerate(mol_list):
                    if (mol is ignore_molecule) and (nx == 0) and (ny == 0) and (nz == 0):
                        continue  # skip centre molecule (already returned)
                    r = norm(mol.get_center_of_mass() + total_offset_cart)  # Note won't be very efficient of c_of_mass is slow (Atoms?)
                    if r > radius:
                        continue
                    if (mol_no != centre_molecule) or not use_symmetry:
                        yield mol_no, cell_indices, 1
                        continue
                    if (nx < 0) or (ny < 0) or (nz < 0):
                        continue
                    nonzerocount = int(nx != 0) + int(ny != 0) + int(nz != 0)
                    weight = 1 << nonzerocount
                    yield mol_no, cell_indices, weight


#thirdeye3 = (1.0/3)*eye3
#def dipolarmatrix_from_versor(D, dircos):
#    return (1.5*D)*(dircos[:,None]*dircos[None,:]-thirdeye3)

# def averageDipTens(D, Daxis, Raxis):
#     """ Average dipolar tensor specified as magnitude of coupling +
#     direction cosine vector, on the basis of continous rotation about given axis vector"""

#     # Make sure vectors are normalised
#     Daxis /= np.linalg.norm(Daxis)  # Not really necessary but no harm
#     Raxis /= np.linalg.norm(Raxis)
#     # Split orientation vector for dipolar coupling into components parallel and orthogonal to rotation axis
#     Dpar = np.dot(Daxis, Raxis)*Raxis
#     Dortho = Daxis - Dpar

#     # Get their norms squared
#     Dparnorm2 = np.sum(Dpar**2)
#     Dorthonorm2 = np.sum(Dortho**2)
#     cos2matrix = Raxis[:,None]*Raxis[None,:]

#     # And compute the averaged dipolar tensor
#     return 0.5*D*((3*Dparnorm2-1)*cos2matrix+
#                   (1.5*Dorthonorm2-1)*(eye3-cos2matrix))


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
# Note: this can produce some warnings, they're useless, we're silencing them
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        struct = io.read(args.structure, store_tags=True)

# (download from Gitlab, https://gitlab.com/ase/ase)
    try:
        kinds = struct.get_array('spacegroup_kinds')
        struct.new_array('site_labels', np.array(struct.info['_atom_site_label'])[kinds])
    except KeyError as exc:
        raise KeyError("Didn't find site labels in input file %s" % fname) from exc

    return struct

axis_list = []
logfile = None

if testmode:
#Create dummy molecule and set of arguments
    args = SimpleNamespace(element=default_el, axes=['A1,A2', 'A1,A3'],
                           verbose=2, nomerge=True, CoMaxes=None, radius=0.0,
                           euler_rotation=None, phiaverage=None,
                           thetaaverage=None, usesymmetry=False,
                           noorientation=True, logfile=None)
    cell_dimensions = [3.0, 4.0, 5.0]
    struct = Atoms([default_el]*3,
                   positions=[(1.0, 1.0, 1.0), (2.0, 1.0, 1.0), (1.0, 2.0, 1.0)],
                   cell=cell_dimensions,  # orthorhombic cell
                   pbc=True
                   )
    struct.new_array('site_labels', np.array(['A1', 'A2', 'A3']))
# 1st test case - rotations along interatom vector
    first_orient = (1.0, 0., 0.)
    second_orient = (0., 1.0, 0.)
# 2nd test case - rotations along perpendicular vector (will scale intramolecular couplings by -0.5)
#    first_orient = (0.0, 1.0, 0.)
#    second_orient = (1.0, 0.0, 0.)
# 3rd test case - rotations about axis at magic angle (will remove intramolecular couplings)
#    MA = 54.7*pi/180.0
#    first_orient = (cos(MA), sin(MA), 0.)
#    second_orient = (0.0, cos(MA), sin(MA))
#    explicit_axes = [((0., 0., 0.), first_orient), ((1.5, 2.0, 2.0), second_orient)]
else:
    parser = argparse.ArgumentParser()
    parser.add_argument('--element', default=default_el, dest='element',
                        help="Element for which to compute the (homonuclear) dipolar couplings")
    parser.add_argument('structure',
                        help="A structure file containing site labels in an ASE "
                        "readable format, typically .cif")
    parser.add_argument('radius', type=float,
                        help="radius (in A) over which to calculator intermolecular contributions")
    parser.add_argument('--axis', help="Specify an axis through first_atom_label[,second_atom_label][:n]",
                        action='append', dest='axes')
    parser.add_argument('--CoMaxis', help="Specify an axis through Centre of Mass as first_atom_label[,second_atom_label][:n]",
                        action='append', dest='CoMaxes')
    parser.add_argument('--nomerge', dest='nomerge', action="store_true",
                        help="Don't merge results from sites with same label")
    parser.add_argument('--usesymmetry', dest='usesymmetry', action="store_true",
                        help="Skip over symmetry-equivalent unit cells")
    parser.add_argument('--noorientation', dest='noorientation',
                        action="store_true", help="Don't include orientation dependence of dipolar coupling Hamiltonian")
    parser.add_argument('--radius', dest="radius", type=float,
                        help="Radius over which to include molecules (in A)")
    parser.add_argument('--euler_rotation', dest="euler_rotation", nargs=3, type=float,
                        help="Overall rotation of crystal system expressed at ZYZ Euler angles in degrees")
    parser.add_argument('--thetaaverage', dest='thetaaverage', type=int,
                        help="Number of steps for theta polar angle averaging")
    parser.add_argument('--phiaverage', dest='phiaverage', type=int, nargs='?', const=3,
                        help="Number of steps for phi polar angle averaging (default 3)")
    parser.add_argument('--logfile', dest='logfile', metavar='FILE',
                        help="Output file for dRSS data as a function of powder angle")
    parser.add_argument('--verbose', '-v', help="Increase verbosity", action='count')

    args = parser.parse_args()
    struct = read_with_labels(args.structure)

include_orientation = not args.noorientation

if include_orientation:
    if (args.phiaverage is None) and (args.thetaaverage is None):
        print("Warning: no powder averaging defined without --noorientation flag", file=sys.stderr)
else:
    if args.phiaverage is not None:
        print("Warning: --phiaverage makes no sense with --noorientation", file=sys.stderr)
    if args.thetaaverage is not None:
        print("Warning: --thetaaverage makes no sense with --noorientation", file=sys.stderr)
    if args.logfile:
        print("Warning: --logfile makes no sense with --noorientation", file=sys.stderr)
        args.logfile = None

if args.logfile is not None:
    try:
        logfile = open(args.logfile, 'w')
        logfile.write("#Theta/deg\tPhi/deg\tdRSS_intra/kHz\tdRSS_inter/kHz\n")
    except IOError:
        raise IOError("Failed to open {} for writing".format(args.logfile))


def add_axes(axisspec, forceCoM):
    for axis in axisspec:
        precolon, _, postcolon = axis.partition(':')
        n = 0
        ok = False
        if postcolon:
            try:
                n = int(postcolon)
                if (n > 0) and (n < 7):   # silently allow n=1 for testing
                    ok = True
            except ValueError:
                pass
            if not ok:
                sys.exit("{} could not be parsed as a n for C_n axis (integer from 2-6)".format(postcolon))
        axiscomps = precolon.split(',')
        if len(axiscomps) > 2:
            sys.exit("Axis specification ({}) should contain 1 or 2 site comma-separated site labels".format(axis))
        cur_iscone = (n == 0)
        if not hasattr(add_axes, 'iscone'):
            add_axes.iscone = cur_iscone
        elif add_axes.iscone != cur_iscone:
            sys.exit("Can't mix jump-type axes (C_n) with cone model.")
        axis_list.append((axiscomps, n, forceCoM))


if args.axes:
    add_axes(args.axes, forceCoM=False)
if args.CoMaxes:
    add_axes(args.CoMaxes, forceCoM=True)
if len(axis_list) > 2:
    print("Warning: More than 2 ({}) axes - this doesn't really make sense and will be very slow!".format(len(axis_list)), file=sys.stderr)
if axis_list and not include_orientation and not add_axes.iscone:
    print("Warning: --noorientation should not generally be used if system is dynamic", file=sys.stderr)

if args.verbose:
    verbose = args.verbose


def point_to_line_distance(a, origin, direction):
    """ Return perpendicular distance of point to line defined
    by origin and unit vector (must be normalised).
    """
    Oa = a - origin
    Oa_par = np.dot(Oa, direction) * direction
    Oa -= Oa_par
    return norm(Oa)


class AxisObject():
    """ Object representing rotation about single axis. """
    def __init__(self, molecule, axis_spec, CoM, labels=None, axislabel=None):
        axisstr = '' if axislabel is None else ' '+axislabel
        if labels is None:
            labels = molecule.get_array('site_labels')
        axis_labels, self.Cn, forceCoM = axis_spec
        self._parse_axis_from_labels(molecule, labels, axis_labels)
        if forceCoM:
            self.axis_origin = CoM
        else:
            perpdist = point_to_line_distance(CoM, self.axis_origin, self.axis_unit_vector)
            if verbose:
                print("Distance of centre of mass from axis{}: {:g} A".format(axisstr, perpdist))
            if perpdist > off_axis_tol:
                print("RotatingMolecule: distance of centre of mass from axis{} ({:g} A) exceeds {:g} A".format(axisstr, perpdist, off_axis_tol), file=sys.stderr)
        if self.Cn > 0:
            rotation_quaternion = Quaternion.from_axis_angle(self.axis_unit_vector, 2.0*pi/self.Cn)
            self._rotate_Cn = Rotate(center=self.axis_origin, quaternion=rotation_quaternion)
        else:
            self._rotate_Cn = None
        if verbose:
            print("Axis{} origin: {}".format(axisstr, self.axis_origin))
            print("Axis{} unit vector: {}".format(axisstr, self.axis_unit_vector))

    def _parse_axis_from_labels(self, molecule, labels, axis_labels):
        axis_atoms = []
        for spec in axis_labels:
            axis_atoms += np.where(labels == spec)[0].tolist()
        if len(axis_atoms) != 2:
            raise ValueError("Axis specification ({}) should identify two atoms (found {} atoms with matching label(s))".format(",".join(axis_labels), len(axis_atoms)))
        self.axis_origin = molecule.positions[axis_atoms[0]].flatten()
        axis_vector = molecule.positions[axis_atoms[1]].flatten() - self.axis_origin
        self.axis_unit_vector = axis_vector / norm(axis_vector)

    def rotate_Cn(self, molecule):
        """ Return copy of molecule rotated one step about Cn axis. """

        if self.Cn is None:
            raise ValueError("rotate_Cn: Cn axis not set")
        return self._rotate_Cn(molecule)

    def rotated_molecule_Cn(self, molecule):
        """ Generator yielding new molecules rotated successively by one about Cn axis. """
        if self.Cn is None:
            raise ValueError("rotate_Cn: Cn axis not set")
        yield molecule
        rotated_mol = molecule
        for step in range(self.Cn - 1):
            rotated_mol = self.rotate_Cn(rotated_mol)
            yield rotated_mol

    def _rawproject(self, pos):
        return self.axis_origin + (np.dot(self.axis_unit_vector, pos-self.axis_origin))*self.axis_unit_vector

    def project(self, initialpos):
        return map(self._rawproject, initialpos)



class RotatingMolecule():
    """ Object to represent an (optionally) rotating molecule.

    Note that currently *contains* an Atoms object, but could possibly
    be derived *from* an Atoms object."""

    def __init__(self, struct, molselector, axis_list=None, element=None):
        self.molecule_all = molselector.subset(struct, use_cell_indices=True)   # This is the point at which atomic positions are unwrapped
        labels_all = self.molecule_all.get_array('site_labels')
        self.center_of_mass = self.molecule_all.get_center_of_mass()
        if verbose > 1:
            print("Creating new RotatingMolecule centred at {} A".format(self.center_of_mass))

        if element is None:
            self.molecule = self.molecule_all
        else:
            selection = AtomSelection.from_element(self.molecule_all, element)
            self.molecule = selection.subset(self.molecule_all)

        self.labels = self.molecule.get_array('site_labels')
        self.isrotating = bool(axis_list)
        self.isconemodel = False
        self.rotationsteps = None
        if self.isrotating:
            self.axes = [AxisObject(self.molecule_all, axis, self.center_of_mass, labels=labels_all, axislabel=str(naxis+1)) for naxis, axis in enumerate(axis_list)]
            if self.axes[0].Cn > 0:
                self.averaged_positions = None  # Flag that averaged positions are meaningless for Cn rotation
                self.rotationsteps = 1
                for axis in self.axes:
                    self.rotationsteps *= axis.Cn
            else:
                self.averaged_positions = self.molecule.get_positions()
                for axis in self.axes:
                    self.averaged_positions = axis.project(self.averaged_positions)
                self.isconemodel = True
        else:
            self.averaged_positions = self.molecule.get_positions()  # Hmmm. Do we need this?

    # @staticmethod
    # def _check_3vector(a, name):
    #     """ Verify that input argument looks like a 3 vector """
    #     try:
    #         if len(a) == 3:
    #             return a
    #     except TypeError:
    #         pass
    #     raise TypeError("RotatingMolecule: input argument {} is not a 3-vector".format(name))

    def get_center_of_mass(self):
        return self.center_of_mass

    def _rotated_molecule_axis(self, n, molecule):
        for rotated_mol in self.axes[n].rotated_molecule_Cn(molecule):
            if n == 0:
                yield rotated_mol
            else:
                yield from self._rotated_molecule_axis(n-1, rotated_mol)

    def rotated_molecule_Cn(self):
        """ Return positions of atoms with successive rotations about Cn axis. """
        if self.rotationsteps is None:
            raise ValueError("rotated_positions_Cn: Cn axes have not been specified")
        yield from self._rotated_molecule_axis(len(self.axes)-1, self.molecule)

    def write(self, outname, selected_only=False, add_COM=False):
        source = self.molecule if selected_only else self.molecule_all
        if add_COM:
            newatoms = source.copy()
            newatoms.append(Atom('N', position=self.center_of_mass))
            io.write(outname, newatoms)
        else:
            io.write(outname, source)


gamma = _get_isotope_data(args.element, 'gamma')[0]
gammas = [gamma, gamma]

eye3 = np.eye(3)
thirdeye3 = (1.0/3)*eye3
def dipolarmatrix_from_vector(r):
    """ Return tensor corresponding to dipolar coupling between two nuclei
    separated by given vector.

    Note: can be deleted as overlaps with functionality of make_dipolar. """

    norm_r = norm(r)
    d = _dip_constant(norm_r*1e-10, gamma, gamma)
    if verbose > 2:
        print("r = {} A   d= {} Hz".format(norm(r), d))
    r /= norm_r
    return (d*1.5)*(np.outer(r, r)-thirdeye3)
#    return (1.5*D)*(dircos[:,None]*dircos[None,:]-thirdeye3)


# def orientation_scaling(tens):
#     theta = tens.euler_angles()[1]
#     return 0.5*(3*cos(theta)**2-1)

def orientation_scaling_from_ivec(r):
    x, y = r[0], r[1]
    rxy = sqrt(x*x+y*y)
    theta = atan2(rxy, r[2])
    return 0.5*(3*cos(theta)**2-1)


def get_cell_list(mol_list, radius):
    molgen = molecule_generator(mol_list, radius, centre_molecule=centre_molecule, use_symmetry=args.usesymmetry)
    next(molgen)  # skip centre_molecule - already processed
    molecule_list = list(molgen)
    print("Number of molecules for intermolecular interactions: {}".format(len(molecule_list)))
    return molecule_list

def calculate_intermolecular(mol_list, cell_list):
    idss_kHz2 = np.zeros((natoms, ))
    if not cell_list:
        return idss_kHz2
    lattice = mol_list[0].molecule.get_cell()
    for mol_no, cell_indices, cell_weight in cell_list:
        cell_coords = lattice.cartesian_positions(cell_indices)
        mol = mol_list[mol_no]
        if verbose > 1:
            print("Considering molecule {} centred at {}  Cell indices: {}".format(mol_no, mol.center_of_mass, cell_indices))
        if not mol.isrotating or mol.isconemodel:
            for i in range(natoms):
                for j in range(natoms):
                    if (probe_indices is not None) and ((i, j) not in probe_indices):
                        continue
                    remote_position = mol.averaged_positions[j] + cell_coords
                    cart_vector = remote_position - mol0.averaged_positions[i]
                    R_ij = norm(cart_vector)
                    d_ij_kHz = 1e-3*_dip_constant(R_ij*1e-10, gamma, gamma)
                    if include_orientation:
                        d_ij_kHz *= orientation_scaling_from_ivec(cart_vector)
                    if (verbose > 2) or (probe_indices is not None):
                        print("Internuclear vector: {}".format(cart_vector))
                        print("{}({}) and {}({}): Distance: {:g} A  Coupling: {:g} kHz  Weighting: {}".format(mol0.labels[i], centre_molecule, mol.labels[j], mol_no, R_ij, d_ij_kHz, cell_weight))
                    if R_ij < 0.5:
                        print("Origin atom position: {} A".format(mol0.averaged_positions[i]))
                        print("Remote atom position: {} A".format(remote_position))
                        raise ValueError("Something horrible has happened here! Use dump_mol option to investigate further")
                    idss_kHz2[i] += cell_weight*d_ij_kHz**2
        else:
            totalD = dict()
            rotcount = 0
            for rotatedmol_i in mol0.rotated_molecule_Cn():  # A little inefficient to repeat this for each mol - but outer loop ...
                for rotatedmol_j in mol.rotated_molecule_Cn():
                    rotcount += 1
                    for i, positioni in enumerate(rotatedmol_i.positions):
                        for j, positionj in enumerate(rotatedmol_j.positions):
                            r_ij = positionj + cell_coords - positioni
                            D_ij = dipolarmatrix_from_vector(r_ij)
                            curtens = totalD.get((i, j))
                            if curtens is None:
                                totalD[(i, j)] = D_ij
                            else:
                                curtens += D_ij
            totsteps = mol0.rotationsteps * mol.rotationsteps
            if totsteps != rotcount:
                sys.exit("This shouldn't happen!")
            scaling = 1e-3/totsteps
            for i in range(natoms):
                for j in range(natoms):
                    if (probe_indices is not None) and ((i, j) not in probe_indices):
                        continue
                    tens = NMRTensor(totalD[(i, j)])
                    d_ij_kHz = scaling * tens.reduced_anisotropy
                    if include_orientation:
                        axis = (tens.quaternion.axis_angle())[0]
                        d_ij_kHz *= orientation_scaling_from_ivec(axis)
                    if (verbose > 2) or (probe_indices is not None):
                        print("{}({}) and {}({}): Coupling: {:g} kHz  Weighting: {}".format(mol0.labels[i], centre_molecule, mol.labels[j], mol_no, d_ij_kHz, cell_weight))
                    idss_kHz2[i] += cell_weight*d_ij_kHz**2

    return idss_kHz2


def calculate_intramolecular(mol0):
    dss_kHz2 = np.zeros((natoms, ))
    if skip_intramolecular:
        return dss_kHz2

    if verbose:
        print("Determining intramolecular contributions to dss")
    labels = mol0.labels
    if mol0.isconemodel or not mol0.isrotating:
        curaxis = mol0.axes[0] if mol0.isrotating else None
        for i, j in ij_generator():
            if mol0.isrotating:
                avgDtens = NMRTensor.make_dipolar(mol0.molecule, i, j, rotation_axis=curaxis.axis_unit_vector, gammas=gammas)
                ivec = curaxis.axis_unit_vector
                avgD_kHz = 1e-3 * avgDtens.reduced_anisotropy
                for axiscount in range(1, len(mol0.axes)):  # For further cone rotations, multiply by P2(cothe theta) of angle between successive axes
                    curaxis_vector = mol0.axes[axiscount].axis_unit_vector
                    costheta = np.dot(ivec, curaxis_vector)
                    P2costheta = 0.5*(3*costheta*costheta-1)
                    if verbose > 2:
                        print("Scaling for cone axis {} by {}".format(axiscount+1, P2costheta))
                    avgD_kHz *= P2costheta
                    ivec = curaxis_vector
            else:
                avgDtens = NMRTensor.make_dipolar(mol0.molecule, i, j, gammas=gammas)
                ivec = mol0.molecule.positions[j] - mol0.molecule.positions[i]
                if (verbose > 2) or probe_indices is not None:
                    R_ij = mol0.molecule.get_distance(i, j)
                    prettyprintmatrix33(avgDtens.data)
                    print("Internuclear vector: {} A".format(ivec))
#                    D2 = dipolarmatrix_from_vector(ivec)
                    dcoupling2 = _dip_constant(norm(R_ij)*1e-10, *gammas)
#                    avgD_tens = D2
#                    prettyprintmatrix33(D2.data)
#                    avgD_kHz = 1e-3 * _dip_constant(R_ij*1e-10, gamma, gamma) * orientation_scaling(avgDtens)
                    print("R = {:g} A  d={:g} kHz".format(R_ij, dcoupling2))
#                    scale2 = orientation_scaling_from_ivec(effective_ivec)
#                    print("{:g}".format(scale2))
                avgD_kHz = 1e-3 * avgDtens.reduced_anisotropy

            if include_orientation:
                avgD_kHz *= orientation_scaling_from_ivec(ivec)

            dss_kHz2[i] += avgD_kHz**2
            dss_kHz2[j] += avgD_kHz**2
            if (verbose > 1) or probe_indices is not None:
                print('\tAdding coupling of {:g} kHz to dss of spins {} and {} (static intra)'.format(avgD_kHz, labels[i], labels[j]))
    else:
        totalD = dict()
        molcount = 0
        for rotated_mol in mol0.rotated_molecule_Cn():
            molcount += 1
            for i, j in ij_generator():
                ivec = rotated_mol.positions[j] - rotated_mol.positions[i]
                tens = dipolarmatrix_from_vector(ivec)
                curD = totalD.get((i, j))
                if curD is None:
                    totalD[(i, j)] = tens
                else:
                    curD += tens
                if (verbose > 2) or (probe_indices is not None):
                    print("Dipolar tensor:")
                    prettyprintmatrix33(tens)
                    fulltens = NMRTensor.make_dipolar(rotated_mol, i, j, gammas=gammas)
                    MDtens = fulltens.data
#                    R_ij = rotated_mol.get_distance(i, j)
#                    print("Internuclear vector beween spins {} and {}: {} A".format(labels[i], labels[j], ivec))
                    print("(make_dipolar tensor between spins {} and {} in step {}:".format(labels[i], labels[j], molcount))
                    prettyprintmatrix33(MDtens)
        if molcount != mol0.rotationsteps:
            raise RuntimeError("This shouldn't happen!")

        dscaling = 1.0
        scaling = 1e-3/mol0.rotationsteps
        for i, j in ij_generator():
            tens = NMRTensor(totalD[(i, j)])
            d_ij_kHz = scaling * tens.reduced_anisotropy
            if verbose > 2:
                print("Total dipolar tensor between spins {} and {}:".format(labels[i], labels[j]))
                prettyprintmatrix33(tens.data)
            if include_orientation:
                axis = (tens.quaternion.axis_angle())[0]
                dscaling = orientation_scaling_from_ivec(axis)
                d_ij_kHz *= dscaling
            dss_kHz2[i] += d_ij_kHz**2
            dss_kHz2[j] += d_ij_kHz**2
            if (verbose > 1) or (probe_indices is not None):
                print('\tAdding coupling of {:g} kHz to dss of spins {} and {} (dynamic intra, scaled by {:g})'.format(d_ij_kHz, labels[i], labels[j], dscaling))

    return dss_kHz2


def getfirstmolecule():
    if testmode:
        mols = Molecules.get(struct)
        mol_types = {None: mols}
        print("Cell dimensions: {} A".format(cell_dimensions))
    else:
        kinds = struct.get_array('spacegroup_kinds')
        # Now find the molecules (using VdW radii)
        mols = Molecules.get(struct)
        # Find which ones are equivalent
        mol_types = defaultdict(list)
        for m in mols:
            m_signature = frozenset(Counter(kinds[m.indices]).items())
            mol_types[m_signature].append(m)

    print("Structure analysed: Z = {}, Z' = {}".format(len(mols),
                                                       len(mol_types)))
    if len(mol_types) > 1:
        sys.exit("Currently only tested for structures containing a single molecule")

    return next(iter(mol_types.values()))  # 1st molecule type


if args.euler_rotation:
    base_rotation = args.euler_rotation
else:
    base_rotation = [0.0, 0.0, 0.0]


def powder_generator(ntheta, nphi, base_rotation):
    gammaang = base_rotation[2]
    if (nphi == 1) or (nphi is None):
        phiangles = [base_rotation[0]]
        baseweighting = 1.0
    else:
        phiangles = base_rotation[0] + np.linspace(0.0, 360.0, num=args.phiaverage, endpoint=False)
        baseweighting = 1.0/args.phiaverage

    if (ntheta == 1) or (ntheta is None):
        betaangles = [base_rotation[1]]
        betaweightings = [1.0]
    else:
        betaangles = np.linspace(90.0, 0.0, num=ntheta, endpoint=False)
        if nphi == 1:
            betaweightings = [1.0] * ntheta
        else:  # Note that sin(beta) factor is only appropriate for two-angle integration
            betaweightings = np.array([sin(beta*deg_to_rad) for beta in betaangles])
            betaweightings *= 1.0/sum(betaweightings)
    for beta, betaweighting in zip(betaangles, betaweightings):
        totweighting = baseweighting * betaweighting
        for phi in phiangles:
            yield (phi, beta, gammaang), totweighting


cell_list = None
struct_orig = None
rawmol_list = None
sumdSS_inter = 0.0
sumdSS_intra = 0.0

isaveraged = (args.thetaaverage is not None) and (args.thetaaverage > 1)
if (args.phiaverage is not None) and (args.phiaverage > 1):
    isaveraged = True
for euler_rotation, weighting in powder_generator(args.thetaaverage, args.phiaverage, base_rotation):
    theta = euler_rotation[1]
    phi = euler_rotation[0]
    if include_orientation or isaveraged:
        print("Theta: {:g}  Phi: {:g} degrees".format(theta, phi))
    if any(euler_rotation):
        if verbose > 0:
            print("Rotating crystal structure by Euler angles: {} degrees".format(euler_rotation))
        euler_rotation_radians = [x*deg_to_rad for x in euler_rotation]
        quat_overall_rotation = Quaternion.from_euler_angles(*euler_rotation_radians)
        overall_rotation_axis, overall_rotation_angle = quat_overall_rotation.axis_angle()
        if struct_orig is None:
            struct_orig = struct.copy()
        else:
            struct = struct_orig.copy()
        struct.rotate(overall_rotation_angle*180.0/pi, v=overall_rotation_axis, rotate_cell=True)

    if verbose > 1:
        print(struct.cell)

    if rawmol_list is None:
        rawmol_list = getfirstmolecule()

    if not axis_list:
        mol_list = [RotatingMolecule(struct, mol, element=args.element) for mol in rawmol_list]
    else:
        mol_list = [RotatingMolecule(struct, mol, axis_list=axis_list, element=args.element) for mol in rawmol_list]
    if dump_mol:
        for mol_no, mol in enumerate(mol_list):
            mol.write("Molecule{}.pdb".format(mol_no))

    mol0 = mol_list[centre_molecule]
    if verbose:
        print("Number of {} atoms in molecule: {}".format(args.element, len(mol0.molecule)))

    natoms = len(mol0.molecule)

    def ij_generator():
        if probe_indices is not None:
            yield from probe_indices
        else:
            for i in range(natoms):
                for j in range(i+1, natoms):
                    yield i, j

    dss_kHz2 = calculate_intramolecular(mol0)

# if len(radii) > 1:
#     drss = []
#     for radius in radii:
#         intermolecular_drss = calculate_intermolecular(mol_list, radius)
#         print("Intermolecular contribution to mean d_RSS at {:g} A: {:g} kHz".format(radius, intermolecular_drss))
#         drss.append(intramolecular_drss + intermolecular_drss)

#     plt.plot(radii, drss, 'x-')
#     plt.xlabel('Radius / A')
#     plt.ylabel('d_rss / kHz')
#     plt.savefig('drss_vs_radius_C2.pdf')
# #    plt.show()
# else:
    if cell_list is None:
        cell_list = get_cell_list(mol_list, args.radius)
        if (verbose > 0) and cell_list:
            print("Included molecules in intermolecular couplings:")
            for mol_no, cell_indices, weight in cell_list:
                print("Molecule {} with cell_indices {} with multiplicity {}".format(mol_no, cell_indices, weight))

    idss_kHz2 = calculate_intermolecular(mol_list, cell_list)

    if logfile is not None:
        meandRSS_intra = sqrt(np.mean(dss_kHz2))
        meandRSS_inter = sqrt(np.mean(idss_kHz2))
        logfile.write("{:g}\t{:g}\t{:g}\t{:g}\n".format(theta, phi, meandRSS_intra, meandRSS_inter))

    sumdSS_intra += weighting * dss_kHz2
    sumdSS_inter += weighting * idss_kHz2

if logfile is not None:
    logfile.close()

print("Label\tIntra-drss/kHz\tInter-drss/kHz")
if args.nomerge:
    try:
        labnum = [int(x[1:]) for x in mol0.labels]
        displayorder = np.argsort(labnum)
    except ValueError:
        print("Unexpected label form (expecting <element><number>). Labels will not be sorted.", file=sys.stderr)
        displayorder = range(len(mol0.labels))

    for i in displayorder:
        print("{}\t{:g}\t{:g}".format(mol0.labels[i], sqrt(sumdSS_intra[i]), sqrt(sumdSS_inter[i])))
else:
    intra_dss_by_label = defaultdict(list)
    inter_dss_by_label = defaultdict(list)
    for i in range(natoms):
        curlabel = mol0.labels[i]
        intra_dss_by_label[curlabel].append(sumdSS_intra[i])
        inter_dss_by_label[curlabel].append(sumdSS_inter[i])

    def getstats(curlist):
        mean = np.mean(curlist)
        if mean == 0.0:
            return 0.0, 0.0
        frac = (max(curlist) - min(curlist))/mean
        return mean, frac

    for label in sorted(intra_dss_by_label):
        intra_mean, intra_frac = getstats(intra_dss_by_label[label])
        inter_mean, inter_frac = getstats(inter_dss_by_label[label])
        print("{}\t{:g} ({:g}%)\t{:g} ({:g}%)".format(label, sqrt(intra_mean), 100.*intra_frac, sqrt(inter_mean), 100.*inter_frac))

if probe_indices is None:
    meandRSS_intra = sqrt(np.mean(sumdSS_intra))  # Note using un-merged lists to maintain weighting
    meandRSS_inter = sqrt(np.mean(sumdSS_inter))
    meandRSS = sqrt(meandRSS_intra**2 + meandRSS_inter**2)
    if not skip_intramolecular:
        print("Intramolecular contribution to mean d_RSS: {:g} kHz".format(meandRSS_intra))
    print("Intermolecular contribution to mean d_RSS at {:g} A: {:g} kHz".format(args.radius, meandRSS_inter))
    print("Overall d_RSS: {:g} kHz".format(meandRSS))
    if not include_orientation and not axis_list:
        print("Including sqrt(1/5) scaling for orientation dependence:")
        scale = sqrt(0.2)
        if not skip_intramolecular:
            print("Intramolecular contribution to mean d_RSS: {:g} kHz".format(scale*meandRSS_intra))
        print("Intermolecular contribution to mean d_RSS at {:g} A: {:g} kHz".format(args.radius, scale*meandRSS_inter))
        print("Overall d_RSS: {:g} kHz".format(scale*meandRSS))
