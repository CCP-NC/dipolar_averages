# dipolar_averages
Python scripts for calculating dipolar 2nd moments under molecular motion

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

The `--help` argument details the command line arguments and usage. A few clarifying points:
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




