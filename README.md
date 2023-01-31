# dipolar_averages
Python script for calculating dipolar 2nd moments under molecular motion.

**vanVleckCalculator.py** is a Python script for calculating the "second moment" of NMR lineshapes
for organic molecules in the limiting conditions of no molecular motion or motion that is
fast compared to the linewidth.

The script uses the [Soprano](https://ccp-nc.github.io/soprano/intro.html) library, which in itself
is built on the [Atomic Simulation Environment (ASE)](https://wiki.fysik.dtu.dk/ase/). Installing
Soprano, e.g. with `pip install soprano` will automatically install ASE.

**AverageUnitTest.py** is a test program that confirms that the analytical formula for the orientationally
averaged root-mean-square dipolar coupling used in the main script gives the same result as an explicit 
numerical powder integration. It is supporting information only.

## Overview

Rotation axes are expressed in terms of site labels, so it is necessary to supply a starting
structure containing crystallographic site labels. In practice, this means using a CIF, since
`read_with_labels` searches for CIF-specific data arrays.

It is assumed that the structure is of a three-dimensional molecular crystal. The lattice parameters
and space group information are used to determine the atomic coordinates of molecules within a specified
radius of the "origin" molecule.

The principal "use case" is motion in plastic crystals, where there is one molecule type
which share the same dynamic process, with no significant correlation between the motion of different
molecules. In principle, the code supports systems in which there is more than one molecule in the unit
cell (Z' > 1), but this code has not been tested and issues can be expected.

Motions are currently limited to rotational diffusion about specified axes. Axis specifications are in
terms of site labels within an individual molecule, with three specifications supported:
- Axes passing through a pair of atoms e.g. --axis C1,C2. Such axes must pass
through the centre of mass (CoM) within a tolerance, currently 0.1 Å,
- Axes passing through the CoM, parallel to the interatomic vector between two atoms,
- Axes in a plane through the CoM and normal to a given interatomic vector (arbitrary orientation within plane).

Note that these axis specifications all resolve internally to specify a rotation in terms of a vector providing an origin and direction, i.e.
it would be straightforward to extend the code to specify axes in different ways.

Multiple axes (of any type) can be provided. The motions are assumed to be uncorrelated, and since the
motions are always fast, the order in which multiple axes are given is not important.

## Usage

`vanVleckCalculator.py --help` will give the command line arguments and usage. A few clarifying points:
- Since H positioning is critical for quantitative results, either use neutron-diffraction structures,
- or DFT-optimised structures, with H positions relaxed.
- The symmetry of a rotation must be specified explictly, e.g. --axis C1,C2:2 denotes a two-fold jump motion
about a vector between C1 and C2. No check is made against the symmetry of the molecule, so specifying a two-fold
jump along an axis with 3-fold symmetry will **not** generate a warning!
- If an axis involves atoms with the same label, it is sufficient to give the label once, e.g. --CoMaxis C3:3
denotes a 3-fold rotation about an axis passing through both C3 atoms in a molecule. For this to work, the molecule
must have exactly two atoms of type C3.
- The calculation time will increase rapidly (roughly as the cube) with the radius parameter.
Depending on the degree of convergence sought,
there is little value in exceeding a radius of 20 Å. Note that a radius of 0
can be used to effetively calculate second moments on isolated molecules.
- The static (non-dynamic) limit is calculated when no axes are given.

## Examples

The **Examples** directory contains DFT-optimised structures for diamantane (CSD refcode CONGRS) and triamantane (refcode TRIAMT01), hence

`vanVleckCalculator.py --radius 15 --axis C1:3 Examples/CONGRSrelaxed_geomopt-out.cif`

will output

>Structure analysed: Z = 4, Z' = 1<br>
>Number of molecules for intermolecular interactions: 54<br>
>Label	Intra-drss/kHz	Inter-drss/kHz<br>
>H1	3.32 	12.06<br>
>H9	13.56 	8.25<br>
>H33	13.58 	8.41<br>
>H57	11.17 	8.47<br>
>Intramolecular contribution to mean d_SS: 148.98 kHz^2<br>
>Intermolecular contribution to mean d_SS at 15 Å: 77.68 kHz^2<br>
>Overall mean d_SS: 226.67 kHz^2    Mean d_RSS 15.06 kHz<br>
>Second moment: 102.00 (Intra: 67.04  Inter: 34.96) kHz^2<br>

In other words, the structure contains four molecules of diamantane in the unit cell, one unique molecule. 54 molecules are within 15 Å of the reference molecule. The following table gives the intramolecular and intermolecular contributions to the root-sum-square dipolar coupling at the 4 crystallographically distinct H sites. The final lines give the mean sum-square and root-sum-square couplings, followed by the corresponding second moments (proportional to the mean sum-square-coupling).

## Contributing

The code is being made available through GitHub to encourage further development. You are welcome to contribute to development by raising issues or forking your own version to develop and potentially merge back (through a pull request).


