# dipolar_averages

**`dipolar_averages/vanVleckCalculator.py`** is a Python script for calculating the "second moment" of NMR lineshapes
for organic molecules in the limiting conditions of no molecular motion or motion that is
fast compared to the linewidth.

The script uses the [Soprano](https://ccp-nc.github.io/soprano/intro.html) library, which in itself
is built on the [Atomic Simulation Environment (ASE)](https://wiki.fysik.dtu.dk/ase/). Installing
Soprano, e.g. with `pip install soprano` will automatically install ASE.

**`tests/AverageUnitTest.py`** is a test program that confirms that the analytical formula for the orientationally
averaged root-mean-square dipolar coupling used in the main script gives the same result as an explicit 
numerical powder integration. It is supporting information only.

## Installation

It's best to install the package in a virtual environment, e.g. using `virtualenv` or `conda`.
Instructions for installing and using `virtualenv` can be found [here](https://virtualenv.pypa.io/en/latest/installation.html).

Once you have a virtual environment set up, clone the repository with

`git clone https://github.com/CCP-NC/dipolar_averages.git`

After obtaining the code, install the dependencies with

`pip install .`

from the top-level directory (i.e. where this `README.md` file is). 


*Alternatively*, you can install the package directly from GitHub with

`pip install git+https://github.com/CCP-NC/dipolar_averages.git`


This will install the `dipolar_averages` package, which contains the
`vanVleckCalculator` script. It will add this script to the path, so it can be run from anywhere. You can check that it is installed correctly by running:

`vanVleckCalculator --help`

and you should see the usage information.



## Overview

Rotation axes are expressed in terms of site labels, so it is necessary to supply a starting
structure containing crystallographic site labels. In practice, this means using a CIF, since
`read_with_labels` searches for CIF-specific data arrays.

It is assumed that the structure is of a three-dimensional molecular crystal. The lattice parameters
and space group information are used to determine the atomic coordinates of molecules within a specified
radius of the "origin" molecule.

The initial "use case" was motion in plastic crystals, where there is one molecule type
which share the same dynamic process, with no significant correlation between the motion of different
molecules. The code has recently been extended to improve support for systems with more than one molecule 
in the unit cell (Z' > 1). 

Motions are currently limited to rotational diffusion of whole molecules about specified axes. Axis specifications are in
terms of site labels within an individual molecule, with four specifications supported:
- Axes passing through a pair of atoms e.g. --axis C1,C2. Such axes must pass
through the centre of mass (CoM) within a tolerance, currently 0.1 Å,
- Axes passing through the CoM, parallel to the interatomic vector between two atoms,
- Axes in a plane through the CoM and normal to a given interatomic vector (arbitrary orientation within plane),
- Axes bisecting a pair of atoms through the CoM.

Note that these axis specifications all resolve internally to specify a rotation in terms of a vector providing an origin and direction, i.e.
it would be straightforward to extend the code to specify axes in different ways.

Multiple axes (of any type) can be provided. The motions are assumed to be uncorrelated, and since the
motions are always fast, the order in which multiple axes are given is not important.

## Usage

`vanVleckCalculator --help` will give the command line arguments and usage. A few clarifying points:
- Since H positioning is critical for quantitative results, either use neutron-diffraction structures,
- or DFT-optimised structures, with H positions relaxed.
- The symmetry of a rotation must be specified explictly, e.g. `--axis C1,C2:2` denotes a two-fold jump motion
about a vector between C1 and C2. No check is made against the symmetry of the molecule, so specifying a two-fold
jump along an axis with 3-fold symmetry will **not** generate a warning!
- If an axis involves atoms with the same label, it is sufficient to give the label once, e.g. --CoMaxis C3:3
denotes a 3-fold rotation about an axis passing through both C3 atoms in a molecule. For this to work, the molecule
must have exactly two atoms of type C3.
- The `perpCoMaxis` specification should only be used for truely "pseudo" axes, since the orientation of the axis
is undefined in terms of individual atoms. Prefer the `bisectorCoMaxis` option if possible, for example, for C2 axes that 
are perpendicular to the principal symmetry axis.
- Uniquely the `bisectorCoMaxis` specification permits atom specifications that return more than two atoms. In this case,
a pair of atoms that are closest together are identified and the bisector found by averaging these atomic positions. A warning
is given if two equivalent short distances are not found; it is assumed that the set of atoms are related by symmetry, e.g.
a C3 axis, and so a given atom should be equidistant to two others in the "ring".
- The calculation time will increase rapidly (roughly as the cube) with the radius parameter.
Depending on the degree of convergence sought,
there is little value in exceeding a radius of 20 Å. Note that a radius of 0
can be used to effetively calculate second moments on isolated molecules.
- The static (non-dynamic) limit is calculated when no axes are given.

## Examples

The **Examples** directory contains DFT-optimised structures for diamantane (CSD refcode CONGRS) and triamantane (refcode TRIAMT01), hence

`vanVleckCalculator --radius 15 --axis C1:3 Examples/CONGRSrelaxed_geomopt-out.cif`

will output

>Structure analysed: Z = 4, Z' = 1<br>
>Number of molecules for intermolecular interactions: 54<br>
>Label	Intra-drss/kHz	Inter-drss/kHz  Total drss/kHz<br>
>H1	  3.32 	12.06   12.51<br>
>H9	 13.56 	 8.25   15.87<br>
>H33 13.58 	 8.41   15.97<br>
>H57 11.17 	 8.47   14.02<br>
>Intramolecular contribution to mean d_SS: 148.98 kHz^2<br>
>Intermolecular contribution to mean d_SS at 15 Å: 77.68 kHz^2<br>
>Overall mean d_SS: 226.67 kHz^2    Mean d_RSS 15.06 kHz<br>
>Second moment: 102.00 (Intra: 67.04  Inter: 34.96) kHz^2<br>

In other words, the structure contains four molecules of diamantane in the unit cell, one unique molecule. 54 molecules are within 15 Å of the reference molecule. The following table gives the intramolecular and intermolecular contributions to the root-sum-square dipolar coupling at the 4 crystallographically distinct H sites. The final lines give the mean sum-square and root-sum-square couplings, followed by the corresponding second moments (proportional to the mean sum-square-coupling).

## Test script

**test/AverageUnitTest.py** is a test script to validate the key result that the averaging of motionally averaged dipolar tensors can be carried out analytically.

## Contributing

The code is being made available through GitHub to encourage further development. You are welcome to contribute to development by raising issues or forking your own version to develop and potentially merge back (through a pull request).


