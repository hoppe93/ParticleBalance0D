#!/usr/bin/env python3
#
# Calculation of fractions of D and Ne at given temperatures.
#

import matplotlib.pyplot as plt
import numpy as np
import sys

sys.path.append('/W/software/particlebalance')

import config
import ionrate
from DistributionFunction import DistributionFunction
from IonHandler import IonHandler

import continuous


Z = {'D': 1, 'Ne': 10, 'Ar': 18}
VOLUME = 4.632


def calcEquilibrium(species, fre=None, pn=None, Ne_frac=0.1):
    """
    Calculate the equilibrium distribution for the given species, as
    a function of measured neutral pressure.

    :param species: Name of ion species to calculate equilibrium for.
    :param fre:     RE distribution function to assume.
    :param pn:      Range of neutral pressures to consider.
    """
    ions = IonHandler()
    ions.addIon(species, Z=Z[species], n=Ne_frac * 7.2e18 / VOLUME)

    if species != 'D':
        ions.addIon('D', Z=1, n=5e18)

    if fre:
        ions.cacheImpactIonizationRate(fre)

    if pn is None:
        pn = np.logspace(-1.5, -.5, 40)

    s0, s1, s = np.zeros(pn.shape), np.zeros(pn.shape), np.zeros((pn.size, Z[species]+1))
    for i in range(pn.size):
        ne = continuous.density(pn[i])
        Te = continuous.temperature(pn[i])

        try:
            _, ni = ionrate.equilibriumAtNfree(ions, ne, Te, fre=fre)

            n = sum(ions[species].solution)
            s0[i] = ions[species].solution[0] / n
            s1[i] = ions[species].solution[1] / n
            s[i,:] = ions[species].solution / n
        except:
            s0[i] = np.nan
            s1[i] = np.nan
            s[i,:] = np.nan * np.ones(((Z[species]+1),))

    return s0, s1, s


def plotEquilibrium(species, fre=None, pn=None):
    if pn is None:
        pn = np.logspace(-1.5, -.5, 40)

    s0, s1, s = calcEquilibrium(species, fre=fre, pn=pn)

    fig, ax = plt.subplots(1, 1, figsize=(4.3, 2.5))

    ax.semilogx(pn, s0*100, '-', color='tab:blue', label=f'{species}-0')
    ax.semilogx(pn, s1*100, '-', color='tab:red', label=f'{species}-1')

    ax.set_xlim([min(pn), max(pn)])
    ax.set_ylim([0, 110])
    ax.set_xlabel(r'Neutral pressure $p^{\rm B}_{\rm n}$ (Pa)')
    ax.set_ylabel(r'Particle fraction (%)')
    ax.set_title(species)
    ax.legend()

    fig.tight_layout()

    if species == 'Ne':
        fig2, ax2 = plt.subplots(1, 1, figsize=(4.3, 2.5))

        Zv = np.sum([Z0*s[:,Z0] for Z0 in range(Z[species])], axis=0) / np.sum(s, axis=1)
        ax2.semilogx(pn, Zv)
        ax2.set_xlim([min(pn), max(pn)])
        ax2.set_ylim([0, Z[species]+1])
        ax2.set_xlabel(r'Neutral pressure $p^{\rm B}_{\rm n}$ (Pa)')
        ax2.set_ylabel(r'Average net charge')
        ax2.set_title(species)

        fig2.tight_layout()

    return fig, ax


def main():
    fre = DistributionFunction()
    fre.setStep(nre=1.6e16, pMin=1, pMax=20, pUpper=20, nP=4000)
    fre = None

    figD,  axD  = plotEquilibrium('D', fre=fre, pn=np.logspace(-2, 1, 40))
    figNe, axNe = plotEquilibrium('Ne', fre=fre, pn=np.logspace(-2, 1, 40))

    plt.show()

    return 0


if __name__ == '__main__':
    sys.exit(main())


