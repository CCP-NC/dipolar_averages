""" Helper functions for friendlier interface to `curve_fit`.

Helper module for friendlier interface to `curve_fit` with specific
functionality for fitting NMR relaxation data.

Key objects
------------
fit_parameter()
    Class containing information about a fitting parameter.
FitObject()
    Class to handle data fitting using the SciPy `curve_fit` function.

Version history
---------------
    2022-12-13  Add print_correlations and print_currentpars
    2022-07-28  Add fitR1rhofunc_info
    2022-02-18  Consolidation of different versions
    2020-05-06  New interface. Python 3.
    2020-05-13  Add fitR1funcQ.
    2020-05-16  Update interface, bug fixes. Add randomise, Tmin formulation.
    2020-05-22  Add formatuncertainty, bug fixes.
"""

from itertools import chain
from math import ceil, floor, log10, isnan
import sys
from collections import defaultdict
from collections.abc import Mapping, Iterable
import numpy as np
from scipy.optimize import curve_fit
from scipy.constants import R, pi

default_randomise = 0.2


# def finergrid(x, factor):
#     """ Create finer grid by adding `factor` linearly spaced intermediate points """

#     if factor < 1:
#         raise ValueError("finergrid: smoothing factor must be >1")
#     if len(x) < 2:
#         raise ValueError("finergrid: input list must have >1 member")

#     npoints = factor + 1
#     finerx = np.concatenate([np.linspace(x[n], x[n+1], npoints) for n in range(len(x)-1)])
#     return np.concatenate((finerx, [x[-1]]))


def sumrows(a):
    """ Sum rows of matrix - used with combine e.g. for R1 data """
    return np.sum(a, 0)


def inversesumrows(a):
    """ Add rows of matrix as reciprocals - used use combine e.g. for T1 data """

    total = 0.0
    for r in range(a.shape[0]):
        total += 1.0 / a[r, :]
    return 1.0 / total


def formatuncertainty(x, xe, precision=0):
    """ Pretty print value and uncertainty.

    Returns shortest string representation of `x +- xe` either as
        x.xx(ee)e+xx
    or as
        xxx.xx(ee)

    Parameters
    ----------
    x
        Nominal value
    xe
        Uncertainty
    precision
        Number of significant digits in uncertainty (0 for default behaviour
        of 2 if leading digit is one, else 1). Maximum value 4.

    Returns
    -------
    str
        Formatted value

    Raises
    ------
    ValueError
        If ``xe`` is not positive or precision is out of range.
    """

    if (xe <= 0.0) or isnan(xe):
        raise ValueError("formatuncertainty: uncertainty must be non-zero and positive")

    if (precision < 0) or (precision > 4):
        raise ValueError("formatuncertainty: precision must be between 0 and 4")

    # base 10 exponents
    #print ('xe below')
    #print (xe)
    x_exp = int(floor(log10(abs(x))))
    xe_exp = int(floor(log10(xe)))

    tmpprecision = precision if precision > 0 else 2
    un_exp = xe_exp-tmpprecision+1
    un_int = int(ceil(xe*10**(-un_exp)) + 0.1)
    if (str(un_int)[0] != '1') and (precision == 0):
        un_exp = xe_exp
        un_int = int(ceil(xe*10**(-un_exp)) + 0.1)

    # nominal value
    no_int = round(x*10**(-un_exp))

    # format - nom(unc)exp

    fieldw = x_exp - un_exp
    if fieldw > 0:
        fmt = '%%.%df' % fieldw
        #print (fmt,no_int,fieldw,un_int,x_exp)
        result1 = (fmt + '(%.0f)e%d') % (no_int*10**(-fieldw), un_int, x_exp)
    else:
        result1 = None


    # format - nom(unc)
    fieldw = max(0, -un_exp)
    fmt = '%%.%df' % fieldw
    result2 = (fmt + '(%.0f)') % (no_int*10**un_exp, un_int*10**max(0, un_exp))

    # return shortest representation
    if result1 is None:
        return result2
    return result2 if len(result2) <= len(result1) else result1


def J(nu, tauc):
    """ Return (unscaled) spectral density, *J*.

    Function evaluated is :math:`\\frac{2\\tau_c}{1 + (2\\pi\\tau_c)^2}`

    Parameters
    ----------
    nu : float-type
        Frequency (in Hz not angular units) at which to evaluate *J*
    tauc
        Correlation time at which to evaluate *J*. Note that function
        is intended to accept numpy-type 1D arrays.
    """

    omegatauc = (2.0 * pi * nu)*tauc
    omegatauc2 = omegatauc * omegatauc
    return (2.0 * tauc)/(1.0 + omegatauc2)


#def rate_from_T(T, parlogtau, parEa):
#    """ Calculate rate at given absolute *T* from current parameters. """
#
#    try:
#        Ea = parEa.value
#        log10tau0 = parlogtau.value
#    except AttributeError:
#        log10tau0 = parlogtau
#        Ea = parEa
#    tauc = (10.0 ** log10tau0) * np.exp(Ea*1000.0/(R*T))
#    return 1.0 / (2.0 * pi * tauc)

def tauc_from_T(T, pardict, args):
    """ Return tau_c value(s) at given temperature(s)

    Calculate correlation times at given temperatures using fitting
    parameters.

    Parameters
    ----------
    T - float-type OR numpy 1D array
        Temperature (in K). If not a scalar, must be object that numpy
        recognises as 1D vector.
    pardict - dict
        Dictionary of parameter values, as used by ``FitObject``. Activation
        energy is provided by ``'Ea'`` parameter (kJ mol-1), if supplied,
        otherwise ``'EaK'`` (activation energy expressed as temperature in K).
        Must also contain one of ``'Tmin'`` (temperature at which correlation
        time returned is :math:`1/2\\pi\\nu_0`, where :math:`\\nu_0` is first
        member of ``args``) OR ``'log10(tau0)'`` (:math:`\\log_{10}\\tau_0`) OR
        ``'log10(f0)'`` (:math:`\\tau_0` calculated from :math:`1/2\\pi f_0`).
    args - sequence
        Fixed arguments for relaxation functions (only used for ``Tmin``
        formulation).

    Returns
    -------
    float-type OR numpy 1D array
        Correlation time(s) in s

    Raises
    ------
    ValueError
        ``args`` is empty, but is needed for ``Tmin`` formulation.
    KeyError
        If unable to find needed parameters in ``pardict``, i.e. one of
        ``Ea`` OR ``EaK`` plus one of ``Tmin`` OR ``log10(tau0)`` OR
        ``log10(f0)``.
    """
    parEa = pardict.get('Ea')
    if parEa is None:
        Ea = (R/1000.0)*pardict['EaK'].value
    else:
        Ea = parEa.value
    if len(args) < 1:
        raise ValueError("tauc_from_T: expecting at least one Larmor frequency")
    nu0 = args[0]
    Tminpar = pardict.get('Tmin')
    if Tminpar:
        exparg = (Ea*1000.0/R)*(1.0/T - 1.0/Tminpar.value)
        return np.exp(exparg)/(2*pi*nu0)
    tau0par = pardict.get('log10(tau0)')
    tau0 = (10.0 ** tau0par.value) if tau0par else 1.0/(2.0*pi*(10.0 ** pardict['log10(f0)'].value))
    return tau0 * np.exp((Ea*1000.0)/(R*T))


def fitR1func_info(fixedparams):
    """ Return information string about current fitting using ``fitR1func``.

    Use of this function is strongly recommended to ensure that the data is
    being fitted to an appropriate model!

    Parameters
    ----------
    fixedparams
        Fixed parameters being passed to ``fitR1func``, i.e. the Larmor
        frequency(ies) in Hz. Relaxation model is either homonuclear dipolar
        relaxation (1 frequency) or heteronuclear dipolar (two frequencies).

    Returns
    -------
    str
        Information string

    Raises
    ------
    ValueError
        If `fixedparams` does not contain either 1 or 2 frequency values
    """

    if len(fixedparams) == 1:
        return "Relaxation model: homonuclear dipolar relaxation driven by isotropic motion at Larmor frequency of %g MHz" % (fixedparams[0]/1e6)
    if len(fixedparams) == 2:
        nu0, nu0I = fixedparams
        if nu0 >= nu0I:
            sys.stderr.write("WARNING: Expecting observe Larmor frequency to be < 'remote' Larmor frequency. fixedparams wrong way round?\n")
        return "Relaxation model: heteronuclear dipolar relaxation to spins of Larmor frequency %g MHz driven by isotropic motion at Larmor frequency of %g MHz" % (nu0I/1e6, nu0/1e6)
    raise ValueError("fitR1func expects either 1 or 2 Larmor frequencies in fixed parameters")


def fitR1func(tscale, pardict, args):
    """
    Calculate R1 as a function of given temperatures (in K)
    based on dipolar coupling between spin-1/2 nuclei.
    See p179-180 of Solid-State NMR: Basic Principles & Practice
    """
    log10A = pardict['log10(A)'].value
    tauc = tauc_from_T(tscale, pardict, args)
    nfixed = len(args)
    if nfixed > 0:
        nu0 = args[0]
        A = 10.0 ** log10A
        if nfixed > 1:
            nu0I = args[1]
        else:
            return A*J(nu0, tauc) + (4.0*A)*J(2.0*nu0, tauc)
        if nfixed == 2:
            return A*J(nu0I-nu0, tauc) + (3.0*A)*J(nu0, tauc) + (6.0*A)*J(nu0+nu0I, tauc)
    raise ValueError("fitR1func: expecting either 1 or 2 Larmor frequencies")


def fitT1func(tscale, pardict, args):
    return 1.0/fitR1func(tscale, pardict, args)


def fitR1funcQ_info(fixedparams):
    """ Return information string about current fitting using ``fitR1funcQ``.

    Use of this function is strongly recommended to ensure that the data is
    being fitted to an appropriate model!

    Parameters
    ----------
    fixedparams
        Fixed parameters being passed to ``fitR1funcQ``, i.e. the Larmor
        frequency (in Hz).

    Returns
    -------
    str
    Information string

    Raises
    ------
    ValueError
        If `fixedparams` does not contain 1 (Larmor frequency) value
    """

    if len(fixedparams) == 1:
        return "Relaxation model: quadrupolar for spin-1 nucleus driven by isotropic motion at Larmor frequency of %g MHz" % (fixedparams[0]/1e6)
    raise ValueError("fitR1funcQ expect 1 Larmor frequency in fixed parameters")


def fitR1funcQ(tscale, pardict, args):
    """
    Calculate R1 as a function of given temperatures (in K)
    based on quadrupolar relaxation mechanism
    See P129 of Dong (NMR of Liquid Crystals or P56 of Bakhmutov (Practical NMR Relaxation for Chemists)
    Note A is proportional to the C_Q squared, but care would be needed to establish the proportionality
    """
    if len(args) != 1:
        sys.exit("fitR1funcQ: expecting 1 Larmor frequency")
    tauc = tauc_from_T(tscale, pardict, args)
    A = 10.0 ** pardict['log10(A)'].value
    nu0 = args[0]
    return A*J(nu0, tauc) + (4.0*A)*J(2*nu0, tauc)


def fitT1funcQ(tscale, pardict, args):
    return 1.0/fitR1funcQ(tscale, pardict, args)


def fitR1rhofunc(tscale, pardict, args):
    """ Calculate R1rho for dipolar relaxation between spin-1/2 nuclei.

    :math:`R_{1\\rho} = A J(\\nu_1, \\tau_c)`
    where :math:`\\nu_1` is the RF nutation frequency (taken from ``args[0]``)
    and :math:`\\tau_c = A \\exp(E_a/RT)`.
    See p181 of Solid-State NMR: Basic Principles & Practice

    Parameters
    ----------
    tscale : sequence
        Temperatures (in K) at which to evaluate :math:`R_{1\\rho}`
    pardict : dict
        Dictionary of parameters: ``'log10(A)'`` (:math:`\\log_{10}A`) plus
        parameters for ``tauc_from_T``.
    args : sequence
        Fixed parameters: RF nutation frequency (Hz)

    Raises
    ------
    ValueError
        If `args` does not contain a single (frequency) value

    Notes
    -----
        Valid for limit where :math:`T_1` relaxation is much slower than
        :math:`T_{1\\rho}` relaxation.
    """
    A = 10.0 ** pardict['log10(A)'].value
    tauc = tauc_from_T(tscale, pardict, args)
    if len(args) == 1:
        return A*J(args[0], tauc)
    raise ValueError("fitR1rhofunc: expecting only 1 nutation frequency")


def fitT1rhofunc(tscale, pardict, args):
    return 1.0/fitR1rhofunc(tscale, pardict, args)


def fitR1rhofunc_info(params):
    if len(params) != 1:
        raise ValueError("fitR1rhofunc_info: invalid number of fixed parameters (1)")
    return("Relaxation model: T1rho relaxation for spin-1/2 nuclei driven by simple isotropic motion at a nutation frequency of %g kHz (ignoring contribution from T1)" % (params[0]/1e3))


class ConstError(TypeError):
    pass


class fit_parameter:
    """ Class containing information about a fitting parameter.

    Attributes
    ----------
    name : str
        Primary name of parameter. This will be used in the fitting functions
        to retrieve the parameter from the supplied ``dict``.
    value : float-type
        Current value.
    displayname : str, optional
        Full name, recommend to include units, e.g. ``Ea / kJ mol-1``.
    fitted : bool, optional
        ``True`` if parameter is fitted (default). Note that fixed parameters
        that cannot be meaningfully fitted should be passed separately.
    dataset : int, optional
        Data set number, counting from 1. If specified, this parameter is
        unique to a particular data set and will only be included its
        parameter ``dict``. If ``None`` (default), then the parameter will
        be shared across all data sets. Ignored if fitting single data set.
    error: float-type OR ``None``
        Standard error, if determined, else ``None``.

    Notes
    -----
    **value** attribute is updated during fitting. **error** attribute is
    initially ``None`` and is set once a fit has concluded.  **dataset**
    attribute cannot be subsequently changed (as this would leave an associated
    ``FitObject`` in an inconsistent state).
    """

    def __init__(self, name, value, displayname=None, fitted=True, dataset=None):
        self.name = name
        self.displayname = displayname if displayname is not None else name
        self.value = value
        self.fitted = fitted
        self.error = None
        self.dataset = dataset
        bad = True
        try:
            if (dataset is None) or (dataset > 0):
                bad = False
        except TypeError:
            pass
        if bad:
            raise ValueError('fit_parameter: dataset must be an integer >=1')

    def __setattr__(self, key, value):
        if (key == 'dataset') and key in self.__dict__:
            raise ConstError("Can't change value of fit_parameter.%s" % key)
        self.__dict__[key] = value

    def prettyvalue(self, simpleformat='g'):
        """ Return pretty string version of value + uncertainty """

        return self.value.__format__(simpleformat) if self.error is None else formatuncertainty(self.value, self.error)

    def __str__(self):
        name = self.displayname
        if self.dataset is not None:
            name += ",%i" % self.dataset
        base = "%s: %f" % (name, self.value)
        if self.error is None:
            if not self.fitted:
                base += " (fixed)"
        else:
            base += (" +/- %f" % self.error)
            try:
                base += " [" + formatuncertainty(self.value, self.error) + "]"
            except (ValueError, OverflowError):
                pass
        if name.startswith("log10("):
            base += " (%s = %g) " % (self.displayname[6:-1], 10.0**self.value)
        return base

    def shortname(self):
        return self.name if self.dataset is None else "%s,%i" % (self.name, self.dataset)



class RejectingDict(dict):
    """ Subclass of ``dict`` that raises exception if key already exists. """

    def __init__(self, inp=None):
        if isinstance(inp, dict):
            super(RejectingDict, self).__init__(inp)
        else:
            super(RejectingDict, self).__init__()
            if isinstance(inp, (Mapping, Iterable)):
                for k, v in inp:
                    self.__setitem__(k, v)

    def __setitem__(self, k, v):
        if self.get(k) is None:
            super(RejectingDict, self).__setitem__(k, v)
        else:
            raise KeyError("duplicate key '{0}' found".format(k))


def forcelist(a):
    """ Force a sequence to be a simple list """
    if isinstance(a, list):
        return a
    return list(a)



class FitObject:
    """ Class to handle data fitting using the SciPy curve_fit function."""

    def __init__(self, fitfunc, parlist, args=None, verbose=1, combine=None):
        """ Initialise FitObject.

        A ``FitObject`` is initialised with a single global set of functions
        and parameters. The points at which the function is evaluated
        are specified in individual calls.

        Parameters
        ----------
        fitfunc
            Fitting function expecting a dictionary of ``fit_parameter``
            and an optional list of fixed parameters. If fitting multiple
            sets of parameters, argument is a list of such functions,
            one per distinct value of ``dataset``.

        parlist
            List of ``fit_parameter`` objects defining the parameters that
            can potentially be varied. If fitting multiple data sets, the
            ``dataset`` attribute is used to associate a parameter with
            an individual data set; otherwise parameters are passed to
            all fitting functions. Note that the ``fit_parameter`` objects
            are updated during the fitting, i.e. ``FitObject`` does not make
            local copies.

        args
            Optional fixed arguments that can be passed to the fitting
            functions. If fitting multiple data sets, a non-``None`` `args`
            needs to be a list-type object of the same length as `fitfunc`.

        combine
            Function to combine data sets. It should take a 2D array of n
            columns and combine into a 1D array of n values.


        Raises
        ------
        ValueError
            ``dataset`` value in a parameter exceeds number of supplied
            data sets (determined from number of fitting functions) OR number
            of fitting functions doesn't match number of ``args`` sets.
        RuntimeError
            ``parlist`` contains two parameters with same name (and ``dataset``
            parameter in case of multi-set fitting).
        """

        self.fitfunc = fitfunc
        self.hasmultifuncs = hasattr(fitfunc, '__getitem__')
        if not self.hasmultifuncs and combine:
            raise ValueError("FitObject: combine function applied, but not multiple calculation functions")
        self.combine = combine
        if self.hasmultifuncs:
            self.ndatasets = len(fitfunc)
            if (args is not None) and ((len(args) != self.ndatasets) or not hasattr(args[0], '__getitem__')):
                raise ValueError("FitObject: args for multiple fits must match list of fit functions")
            if (verbose > 0) and (self.ndatasets == 1) and (combine is None):
                sys.stderr.write("Warning: multi-set data set fitting enabled (as fit functions supplied as list), but with only one data set\n")
            self.pardict = [RejectingDict() for _ in range(self.ndatasets)]
        else:
            self.ndatasets = None
            self.pardict = RejectingDict()
        self.parlist = parlist
        self.args = args
        self.verbose = verbose
        self.sigma = None  # effective flag for initialisation
        self.corm = None # Flag correlation matrix does not exist

        self.parorig = dict()
        for par in self.parlist:
            try:
                if self.hasmultifuncs:
                    if par.dataset is None:
                        for ldict in self.pardict:
                            ldict[par.name] = par
                    else:
                        if par.dataset > self.ndatasets:
                            raise ValueError("Parameter %s applies to data set %i, but only %i fitting functions supplied!" % (par.name, par.dataset, self.ndatasets))
                        self.pardict[par.dataset-1][par.name] = par
                else:
                    self.pardict[par.name] = par
                self.parorig[(par.name, par.dataset)] = par.value
            except KeyError as exc:
                raise RuntimeError("Duplicate parameter name (%s) encountered" % par.name) from exc

    def _reset_initial(self):
        """ Reset parameter values to those passed when object created """

        for k, v in self.parorig.items():
            if self.randompar != 0.0:
                v *= 1.0 + 2.0*self.randompar * (np.random.rand() - 0.5)
            if self.hasmultifuncs:
                name, dataset = k
                if dataset is None:
                    for curdict in self.pardict:
                        curdict[name].value = v
                else:
                    self.pardict[dataset-1][name].value = v
            else:
                self.pardict[k[0]].value = v

    def print_currentpars(self, file=sys.stdout):
        """ Output current contents of the fitting parameters. """

        for parval in self.parlist:
            print(str(parval), file=file)

    def _initialise(self, x, y, noiselevel, randomise, allownone=False):
        """ Internal function preparing for fitting.

        Creates list of actively fitted parameters in ``which``, and
        list of initial parameter values for ``curve_fit`` in ``p0``.
        Stores values of x, y (flattened into vector if neceessary).
        Determine individual :math:`\\sigma` values for data vector.
        """

        if self.hasmultifuncs and (self.combine is None):
            try:
                lens = set(map(len, [self.fitfunc, x, y, noiselevel]))
            except TypeError as exc:
                raise TypeError('For multiple fits, fitfuncs, xs, ys and noiselevels arguments must all be list-type objects') from exc
            if len(lens) > 1:
                raise ValueError('For multiple fits, fitfuncs, xs, ys and noiselevels arguments must have matching lengths')

        self.which = [par for par in self.parlist if par.fitted]
        if not self.which:
            raise ValueError("Can't initialise fitting as no free variables")
        self.p0 = [par.value for par in self.which]
        self.npars = len(self.p0)
        if self.hasmultifuncs and (self.combine is None):
            self.sigma = []
            for ly, lnoiselevel in zip(y, noiselevel):
                self.sigma += forcelist(self._makenoise(lnoiselevel, len(ly), allownone=False))
        else:
            self.sigma = self._makenoise(noiselevel, len(y), allownone=allownone)
        if randomise is False:
            self.randompar = 0.0
        elif randomise is True:
            self.randompar = default_randomise
        else:
            try:
                self.randompar = float(randomise)
                if self.randompar > 1.0:
                    raise ValueError(r"FitObject: randomise parameter cannot exceed 1.0 (100\%)")
            except ValueError as exc:
                raise ValueError("FitObject: cannot parse randomise parameter (%s) as floating point" % str(randomise)) from exc

    @staticmethod
    def _makenoise(noiselevel, leny, allownone):
        """ Internal function that creates :math:`\\sigma`` values for data set.

        Parameters
        ----------
        noiselevel
            noiselevel specified as :math:`\\sigma` or as list of values.
            In principle list can be any iterable object. In practice simple
            ``list`` object will be safer.
        leny
            length of data set
        allownone : bool
            If ``True`` allow ``None`` as ``noiselevel`` value.

        Returns
        -------
        list
            List of sigma values of length `leny`


        Raises
        ------
        ValueError:
            ``noiselevel`` is ``None`` in combination with ``allownone``
            being False.
        RuntimeError:
            Mismatch between length of ``noiselevel`` vector and ``leny``.
        """

        if not hasattr(noiselevel, '__iter__'):
            use_noiselevel = noiselevel
            if noiselevel is None:
                if not allownone:
                    raise ValueError("Noise level must be specified explicitly e.g. for multi-dataset fitting")
                use_noiselevel = 1.0
            return [use_noiselevel] * leny
        if len(noiselevel) != leny:
            raise RuntimeError("Number of entries in noise vector (%i) doesn't match number of data items (%i)" % (len(noiselevel), leny))
        return noiselevel


    def _docurvefit(self, curriedfunc, estimatenoise=False):
        """ Internal function that executes ``curve_fit``

        Parameters
        ----------
        curriedfunc : function
            Curried function that passes parameters from ``curve_fit`` to
            original function(s) expecting parameter dictionaries.
        estimatenoise : bool
            If ``True``, use std. dev. of residuals as noise level, otherwise
            use user-specified :math:`\sigma` values

        Raises
        ------
        RuntimeError
            If `sigma` attribute has not been set (via ``_initialise``)
        """

        if self.sigma is None:
            raise RuntimeError("docurvefit: called before initialisation")
        if self.randompar != 0.0:
            self._reset_initial()
        if self.verbose > 0:
            print("Initial parameters:")
            self.print_currentpars()

        if self.hasmultifuncs and (self.combine is None):
            self.popt, self.pcov = curve_fit(curriedfunc, self.rawx, self.rawy, self.p0, sigma=self.sigma, absolute_sigma=True)
        else:
            self.popt, self.pcov = curve_fit(curriedfunc, self.x, self.y, self.p0, sigma=self.sigma, absolute_sigma=True)
        self.rawfitdata = curriedfunc(self.x, *(self.popt))
        if self.hasmultifuncs and (self.combine is None):
            self.fitdata = []
            curp = 0
            for n in map(len, self.x):
                self.fitdata.append(self.rawfitdata[curp:curp+n])
                curp += n
        else:
            self.fitdata = self.rawfitdata
        errors = np.sqrt(np.diag(self.pcov))
        self.sigmay = np.std(self.rawfitdata - self.y, ddof=self.npars)  # y noise-level based on residuals

        if self.verbose or estimatenoise:
            print("Noise-level estimated from residuals of %i data points with %i DOF: %g" % (len(self.y), self.npars, self.sigmay))

        if estimatenoise:
            est_noise = self.sigmay  # Use population std dev.
            print("WARNING: basing errors on std. dev. of residuals. Dubious if systematic errors present!")
        else:
            est_noise = 1.0
    #        print("Factor = %g" % ((est_noise*est_noise)/(len(y)-npars)))
        for i, par in enumerate(self.which):
            par.value = self.popt[i]
            par.error = est_noise * errors[i]

        if self.verbose > 0:
            print("\nFitted parameters:")
            self.print_currentpars()
            print("")

        self.corm = np.zeros((self.npars, self.npars))

        for i in range(self.npars):
            for j in range(i+1, self.npars):
                coeff = self.pcov[i, j]/(errors[i]*errors[j])
                self.corm[j, i] = self.corm[i, j] = coeff

        if self.verbose > 0:
            self.print_correlations()


    def print_correlations(self, threshold=0.0, dest=sys.stdout):
        """ Output current contents of correlation matrix """

        if self.corm is None:
            raise RuntimeError("print_correlations: called before fitting")
        if (threshold < 0.0) or (threshold >=1.0):
            raise ValueError("print_correlations: threshold parameter cannot be <0 or >1")

        overthreshold = 0
        for i in range(self.npars):
            for j in range(i+1, self.npars):
                coeff = self.corm[i, j]
                if abs(coeff) >= threshold:
                    overthreshold += 1
                    print("Correlation coefficient between %s and %s: %g"
                          % (self.which[i].shortname(), self.which[j].shortname(), self.corm[i, j]), file=dest)
        return overthreshold

    def montecarlo(self, x, noiselevel, repeats=1000, randomise=False, verbose=0):
        """ Monte Carlo analysis of uncertainties.

        Parameters
        ----------
        x : sequence
            `x` points at which to evaluate function(s).
        noiselevel
            Initialisation parameters for :math:`\\sigma` values
        repeats : int
            Number of repeats (default 1000)
        randomise
            See ``fit`` (default is ``False``)
        verbose
            Verbosity level (default 0), which overrides current value
            in ``FitObject``

        Returns
        -------
        ``dict`` of fitting (i.e. variable) parameters (key) and list of
        values fitted (value)
        """

        y = self.evaluate(x)
        self._initialise(x, y, noiselevel, randomise, allownone=False)
        retdict = defaultdict(list)
        old_verbose = self.verbose
        try:
            self.verbose = verbose
            for _ in range(repeats):
                ynoisy = y.copy()
                for i, sigma in enumerate(self.sigma):
                    ynoisy[i] += np.random.normal(scale=sigma)
                self._reset_initial()
                self._fit_current(x, ynoisy)
                for par in self.which:
                    retdict[par].append(par.value)
        finally:
            self.verbose = old_verbose
        return retdict

    def evaluate(self, x):
        """ Evaluate function(s) at current parameter values.

        Parameters
        ----------
        x : sequence
            `x` points at which to evaluate function(s).

        Returns
        -------
        List of `y` values (single dataset)
        list of list of `y` values (multiple datasets)
        matrix of `y` values (combined functions)
        """

        if not self.hasmultifuncs:
            return self.fitfunc(x, self.pardict, args=self.args)
        if self.combine:
            totmat = np.empty((self.ndatasets, len(x)))
            for i in range(self.ndatasets):
                totmat[i, :] = self.fitfunc[i](x, self.pardict[i], args=self.args[i])
            return totmat
        retfunc = []
        if len(x) != self.ndatasets:
            raise ValueError("evaluate: number of x data sets (%i) doesn't match number of data sets (%i)" % (len(x), self.ndatasets))
        for i in range(self.ndatasets):
            retfunc.append(self.fitfunc[i](x[i], self.pardict[i], args=self.args[i]))
        return retfunc

    def _singlefit(self, x, estimatenoise):
        """ Core internal function for single-dataset fitting

        See ``fit`` for parameters.
        """

        def curriedfunc(_, *locargs):
            """ Converts dict-based function to function expected by ``curve_fit``

            Parameters
            ----------
            locargs : sequence
                Parameters passed by ``curve_fit`` to target function

            Returns
            -------
            function
                Curried function
            """
            for i, locarg in enumerate(locargs):
                self.which[i].value = locarg
            return self.fitfunc(x, self.pardict, args=self.args)

        self._docurvefit(curriedfunc, estimatenoise=estimatenoise)

    def _multifit(self, xs):
        """ Core internal function for multi-dataset fitting

        See ``fit`` for parameters
        """

        def curriedfunc(_x, *locargs):
            """ Converts dict-based function to function expected by ``curve_fit``

            Parameters
            ----------
            _x
                Ignored - enclosing ``xs`` used
            locargs : sequence
                Parameters passed by ``curve_fit`` to target function

            Returns
            -------
            function
                Curried function
            """
            for i, locarg in enumerate(locargs):
                self.which[i].value = locarg
            retfunc = []
            for datasetm1 in range(self.ndatasets):
                retfunc += self.fitfunc[datasetm1](xs[datasetm1], self.pardict[datasetm1], args=self.args[datasetm1]).tolist()
            return retfunc

        self._docurvefit(curriedfunc, estimatenoise=False)

    def _combinefit(self, x, estimatenoise):
        """ Core internal function for combined-dataset fitting

        See ``fit`` for parameters.
        """

        def curriedfunc(_, *locargs):
            """ Converts dict-based function to function expected by ``curve_fit``

            Parameters
            ----------
            locargs : sequence
                Parameters passed by ``curve_fit`` to target function

            Returns
            -------
            function
                Curried function
            """
            for i, locarg in enumerate(locargs):
                self.which[i].value = locarg
            return self.combine(self.evaluate(x))

        self._docurvefit(curriedfunc, estimatenoise=estimatenoise)

    def _fit_current(self, x, y, estimatenoise=False):
        """ Internal function for fitting when already initialised.

        See ``fit`` for parameters.
        """

        self.x = x
        self.y = y
        if self.hasmultifuncs:
            if self.combine:
                self._combinefit(x, estimatenoise)
            else:
                self.rawx = list(chain(*x))  # Not used directly - curriedfunc passes the correct subset of x, y to the fit functions
                self.rawy = list(chain(*y))
                self._multifit(x)
        else:
            self._singlefit(x, estimatenoise)

    def fit(self, x, y, noiselevel=None, randomise=False):
        """ Perform least-squares fitting.

        Parameters
        ----------
        x : sequence OR sequence of sequences
            `x` values at which function(s) will be evaluated
        y : sequence OR sequence of sequences
            Target `y` values
        noiselevel
            Initialising parameter(s) for :math:`\sigma` values. ``None`` means
            :math:`\sigma` will be determined from residuals (only valid for
            single data set fitting)
        randomise : bool OR float-type
            If ``False`` (default) initial fitting parameters specified at
            initialisation are used unchanged, otherwise the parameter
            specifies a fractional variation, *f*, for the initial parameter
            values, i.e. each initial parameter will randomly vary from -*f*/2
            to + *f*/2 either side of its value at initialisation. If ``True``,
            a default value of *f* is used (0.2).

        Returns
        -------
        list OR list of lists
            Fitted *y* points either as flat list (single data set) or
            list of lists (multiple data sets)
        """

        self._initialise(x, y, noiselevel, randomise, allownone=True)
        self._fit_current(x, y, noiselevel is None)
        return self.fitdata
