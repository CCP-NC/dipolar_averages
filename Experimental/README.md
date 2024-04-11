# Experimental

This folder contains Python scripts used to quantify second moments from experimental data.
The `Diamantane` folder contains processed versions of diamantane spectra stored as CSV files, with slightly different layouts.
The full original data associated with this work can be found in the [data archive](https://collections.durham.ac.uk/files/r1qz20ss572).

These scripts have been tidied up post publication to make them easier to use and more flexible. But they are definitely not polished analytical tools, and are essentially starting points for adaptation.

## By spectral integration

`M2_integrator.py` performs simple numerical integration of the spectra in order to estimate M2, with the example of the diamantane spectrum acquired at low temperature. 

Notes:
- The input data must be a simple text file, either white space delimited or comma separated. By default, this is determined by the file extension, i.e. a file ending `.csv` will be assumed to be comma separated, otherwise white space delimited.
- The horizontal scale is assumed to be in Hz. 
- By default, both wings of the spectrum are integrated to give a sense of the reproducibility of the values. The behaviour is controlled by the `mode` parameter of the `M2integrate` function.
- The spectrum baseline must be absolutely flat and lie at zero for the integration to be stable. Similarly the signal to noise ratio needs to be extremely high.
- The zero of the spectrum should be close to the midpoint of the frequency scale, but does not need to be exactly at the zero; a search is made for the maximum.  
- The `selector` determines the how far into the wing to integrate, set in terms of fraction of the simple integral (which always converges quickly and smoothly). Default is 99.9%

## By Gaussian lineshape fitting

`Spec-gaussian_fit_low_T.py` fits a spectrum to a Gaussian function. It requires functions from the `fit_utils_P3.py` file to function. 

Notes:
- The input file can contain multiple data columns and is assumed to be comma separated. The column to be fitted is selected by `fitcol`.
- The frequency scale can be given in different (even multiple) units, although only a frequency-based unit makes sense for M2 values.
- By default, 3 parameters are fitted, an overall scale, a Gaussian sigma parameter (M2 = sigma^2) and a shift for the peak origin (which should be close to zero). A baseline offset can be optionally included by adding `parc` to the list of parameters to be fitted.
- The initial value of sigma may need adjusted to be close to the correct value for the minimisation to function correctly.
