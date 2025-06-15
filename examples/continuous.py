#!/usr/bin/env python3

import argparse
import h5py
import matplotlib.pyplot as plt
import numpy as np
import sys

from scipy.constants import e, k

import config
import ionrate
from DistributionFunction import DistributionFunction
from IonHandler import IonHandler


SANS_FONT = False


if SANS_FONT:
    plt.rcParams.update({
        'text.usetex': False,
        'font.family': 'FreeSans',
        'font.size': 14,
        'axes.linewidth': 2,
        'legend.fancybox': False,
        'legend.edgecolor': 'k',
        'legend.labelspacing': 0,
        'legend.columnspacing': 0,
        'legend.handletextpad': 0.3,
        'legend.fontsize': 13
    })
else:
    plt.rcParams.update({
        'text.usetex': True,
        'font.family': 'serif',
        'font.size': 14,
        'axes.linewidth': 2,
        'legend.fancybox': False,
        'legend.edgecolor': 'k',
        'legend.labelspacing': 0,
        'legend.columnspacing': 0,
        'legend.handletextpad': 0,
        'legend.fontsize': 13
    })

VOLUME = 4.632
#NRE = 1.6e16
NRE = 2.73e16
PRE = 20


def temperature(pn):
    return (0.7478 + 0.3418/np.sqrt(pn))


def density(pn):
    x = np.log(pn)
    #n = 2.59481028 + 0.35098082*x + 5.62132258*x**2 - 2.66429666*x**3
    #n = 2.98961493 + 1.09276014*x + 0.61516866*x**2 - 0.18579151*x**3
    n = 2.99 + 1.09*x + 0.62*x**2 - 0.19*x**3
    return n*1e18


def simulate(Ne_frac=1, nre_frac=1):
    """
    Run the simulation.
    """
    nImp = Ne_frac * 7.2e18 / VOLUME
    nD = 2e19

    ions = IonHandler()
    ions.addIon('D', Z=1, n=nD)
    if nImp > 0:
        ions.addIon('Ne', Z=10, n=nImp)

    fre = DistributionFunction()
    #fre.setStep(nre=nre_frac*NRE, pMin=1, pMax=PRE, pUpper=PRE, nP=4000)
    fre.setDelta(nre=nre_frac*NRE, pStar=PRE)
    #fre = None

    if fre is not None:
        ions.cacheImpactIonizationRate(fre)
    
    pn = np.logspace(-2, 1, 40)
    Te = []
    nD = []
    ne = []
    nN = []
    I = []
    R = []
    Ire = []
    IN = []
    RN = []
    IreN = []
    ZNe = []
    for i in range(len(pn)):
        nfree = density(pn[i])
        _Te = temperature(pn[i])
        
        n0, ni = ionrate.equilibriumAtNfree(
            ions=ions, nfree=nfree, Te=_Te, fre=fre, maxiter=5000
        )

        Te.append(_Te)
        nD.append(n0)
        nN.append(ions.getNeutralDensity())
        ne.append(nfree)
        I.append(ions['D'].scd(Z0=0, n=nfree, T=_Te)*nfree*ni[0])
        R.append(ions['D'].acd(Z0=1, n=nfree, T=_Te)*nfree*ni[1])
        if fre is not None:
            Ire.append(ions['D'].evaluateImpactIonizationRate(Z0=0, fre=fre)*ni[0])
        else:
            Ire.append(np.nan)

        nNe = ions['Ne'].solution
        _ZNe = np.sum([Z0*nNe[Z0] for Z0 in range(nNe.size)]) / np.sum(nNe)
        ZNe.append(_ZNe)

        _IN = 0
        _RN = 0
        _IreN = 0
        for Z0 in range(10):
            _IN += ions['Ne'].scd(Z0=Z0, n=nfree, T=_Te)*nfree*ions['Ne'].solution[Z0]
            _RN += ions['Ne'].acd(Z0=Z0+1, n=nfree, T=_Te)*nfree*ions['Ne'].solution[Z0+1]

            if fre is not None:
                _IreN += ions['Ne'].evaluateImpactIonizationRate(Z0=Z0, fre=fre)*ions['Ne'].solution[Z0]
            else:
                _IreN = np.nan

        IN.append(_IN)
        RN.append(_RN)
        IreN.append(_IreN)

    return {
        'pn': pn,
        'Te': np.array(Te),
        'nD': np.array(nD),
        'ne': np.array(ne),
        'nN': np.array(nN),
        'I': np.array(I),
        'R': np.array(R),
        'Ire': np.array(Ire),
        'IN': np.array(IN),
        'RN': np.array(RN),
        'IreN': np.array(IreN),
        'ZNe': np.array(ZNe)
    }


def main():
    sdata = simulate(Ne_frac=.1)

    return 0


if __name__ == '__main__':
    sys.exit(main())


