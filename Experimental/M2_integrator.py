import os
from enum import Enum
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import InterpolatedUnivariateSpline

plt.rcParams.update({'figure.max_open_warning': 0})

# Sam's program for calculating m2 integrals on all text files in a directory. throws up each graph one at a time.

####----Options
frange = 120.0     # range of spectrum (in kHz) to plot / integrate over
showplot = True    # True to show plot
verbose = True
####

default_selector = 0.999

testfile = 'Diamantane/di_m30.csv'
# True if data is given in kHz rather than Hz
iskHz = True
# None auto=detect based on file type
default_iscsv = None
#testfile = 'cooling_20.txt'
scandir = None   # Set to scan a directory (only looks for .txt)

selector = {'cooling_20.txt': 0.999}


States = Enum('States', ['NegLead', 'NegWing', 'PosWing', 'PosLead'])


# Courtesy Stackoverflow
def quadratic_spline_roots(spl):
    roots = []
    knots = spl.get_knots()
    for a, b in zip(knots[:-1], knots[1:]):
        u, v, w = spl(a), spl((a+b)/2), spl(b)
        t = np.roots([u+w-2*v, w-u, 2*v])
        t = t[np.isreal(t) & (np.abs(t) <= 1)]
        roots.extend(t*(b-a)/2 + (b+a)/2)
    return np.array(roots)


def M2integrate(file, mode='both', verbose=verbose):
    """ Perform M2 integration for a given data file """

    filename = os.path.basename(file)
    iscsv = default_iscsv
    if iscsv is None:
        iscsv = filename.endswith(".csv")

    to_kHz_scale_factor = 1.0 if iskHz else 1e-3

    curselector = selector.get(filename, default_selector)
    if (curselector >= 1.0) or (curselector <= 0.0):
        raise ValueError(f"Invalid convergence selector (must be >0 and <1): {curselector}")

    #Extract raw data (assumed data is in Hz)
    unsorteddata = np.genfromtxt(file, delimiter=',' if iscsv else None)
    # Sort on frequency column. This is probably unnecessary, but ensures order is well-defined and ascending
    rawdata = unsorteddata[unsorteddata[:, 0].argsort()]

    f = InterpolatedUnivariateSpline(rawdata[:,0], rawdata[:,1], k=3)
    cr_pts = quadratic_spline_roots(f.derivative())
    cr_vals = f(cr_pts)
    max_index = np.argmax(cr_vals)
    f_max = to_kHz_scale_factor*cr_pts[max_index]
    print("Interpolated data has maximum value at {:.1f} Hz".format(1e3*f_max))

    curstate = States.NegLead
    negstart = None
    posstart = None
    posend = None
    nrawpts = rawdata.shape[0]
    for i in range(nrawpts):
        f = rawdata[i, 0] * to_kHz_scale_factor
        if curstate == States.NegLead:
            if f > -frange + f_max:
                negstart = i
                curstate = States.NegWing
                continue
        elif curstate == States.NegWing:
            if f > f_max:
                posstart = i
                curstate = States.PosWing
                continue
        elif curstate == States.PosWing:
            if f > frange + f_max:
                posend = i
                curstate = States.PosLead
                break

    if curstate != States.PosLead:
        raise ValueError("Failed to find expected bandshape")

    if verbose:
        print(f"Negative wing: Points {negstart}-{posstart-1}")
        print(f"Positive wing: Points {posstart}-{posend}")

    freq_kHz = np.zeros((nrawpts,))
    # These are integration bin widths. If the scale is linear (almost certainly) these will be equal and could be ignored
    freqdelta_kHz = np.zeros((nrawpts,))

    freq_kHz[negstart-1:posend+1] = rawdata[negstart-1:posend+1, 0] * to_kHz_scale_factor
    for i in range(negstart, posend):
        freqdelta_kHz[i] = 0.5*(freq_kHz[i+1] - freq_kHz[i-1])
    f_0 = 0.5*(freq_kHz[posstart-1] + freq_kHz[posstart])
# In principle, we could interpolate data to account for the integration grid not be exactly centred around the peak max
    f_corr = f_max - f_0

    if verbose:
        print("Delta at midpoint: {:.2f} kHz".format(freqdelta_kHz[posstart]))
        print("Data / integration grid discrepancy: {:.0f} Hz".format(1e3*f_corr))
        print("f's either side of midpoint: {:.0f} and {:.0f} Hz".format(1e3*freq_kHz[posstart-1], 1e3*freq_kHz[posstart]))

    def doM2(ax2, ax3, wing, start, end):

        if wing == 'right':
            userange = range(start, end)
        else:
            userange = range(end-1, start-1, -1)

        npts = end - start
        cumul_int = 0.0
        cumul_rawM2 = 0.0
        cumul_ints = np.zeros((npts,))
        cumul_M2s = np.zeros((npts,))
        for outi, i in enumerate(userange):
            fdelta = freqdelta_kHz[i]
            cumul_int += rawdata[i, 1] * fdelta
            cumul_ints[outi] = cumul_int
            cumul_rawM2 += rawdata[i, 1] * (freq_kHz[i] - f_0)**2 * fdelta
            cumul_M2s[outi] = cumul_rawM2

    # Normalise
        norm_cumul_ints = cumul_ints / cumul_int
        cumul_M2s /= cumul_int

        final_M2 = None
        for i in range(npts):
            if norm_cumul_ints[i] > curselector:
                select_i = i
                final_M2 = cumul_M2s[i]
                if wing == 'left':
                    select_i = posstart - i - 1
                else:
                    select_i = i + posstart
                break

        if wing == 'left':
            cumul_ints = list(reversed(cumul_ints))
            cumul_M2s = list(reversed(cumul_M2s))
        ax2.plot(freq_kHz[start:end], cumul_ints, color='green')
        ax3.plot(freq_kHz[start:end], cumul_M2s, color='blue')
        ax3.plot([freq_kHz[select_i]], [final_M2], 'bs')
        qualifier = f"'{wing}' " if mode == 'both' else ''
        print(f"M2 {qualifier}for {file}: {final_M2:.0f} kHz^2")

        return cumul_int, final_M2

    fig, ax1 = plt.subplots()
    ax1.plot(freq_kHz[negstart:posend], rawdata[negstart:posend, 1], color='black', label=str(filename))
    ax1.set_ylabel('')
    ax1.set_yticks([])
    ax2 = ax1.twinx()
    ax3 = ax2.twinx()

    results = []
    if (mode == 'left') or (mode == 'both'):
        results.append(doM2(ax2, ax3, 'left', negstart, posstart))
    if (mode == 'right') or (mode == 'both'):
        results.append(doM2(ax2, ax3, 'right', posstart, posend))

    ax1.set_xlabel("Frequency / kHz", size='x-large')
    ax2.yaxis.set_label_position('left')
    ax2.set_ylabel("Integral", color='green', size='x-large')
    ax2.set_yticks([])

    ax3.set_ylabel("$M_2$ / kHz$^2$", color='blue', size='x-large')
    ax3.tick_params(axis='y', colors='blue')

    plt.savefig('{}.png'.format(filename))
    plt.legend([filename], loc="upper left")

    if showplot:
        plt.show()

    if mode == 'both':
        meanM2 = 0.5*(results[0][1] + results[1][1])
        print(f"M2 (mean) for {file}: {meanM2:.0f} kHz^2")
        print("Ratio of integrals: 1 : {:.2f}".format(results[1][0]/results[0][0]))
        return meanM2

    return results[0][1]

# with open('results.xy','w+') as output:


if scandir:
    for file in os.listdir(scandir):
        if file.endswith('.txt'):
            M2integrate(file)
else:
    M2 = M2integrate(testfile)
