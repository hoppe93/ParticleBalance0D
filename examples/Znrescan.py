#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import calcFracs
import sys

import config
import ionrate
from DistributionFunction import DistributionFunction
from IonHandler import IonHandler

import continuous

from scipy.constants import e


NRE = 1.6e16
PN = 0.5
VOLUME = 4.632


def simulate():
    """
    Run a simulation.
    """
    nre = np.logspace(-3, 1, 40)

    Zavg = []
    for i in range(nre.size):
        fre = DistributionFunction()
        fre.setStep(nre=nre[i]*NRE, pMin=1, pMax=20, pUpper=20, nP=4000)

        _, _, s = calcFracs.calcEquilibrium('Ne', fre=fre, pn=np.array([PN]))

        Zv = np.sum([Z0*s[:,Z0] for Z0 in range(s.shape[1])], axis=0) / np.sum(s, axis=1)
        Zavg.append(Zv)


    #mxZavg = max(Zavg)[0]
    mxZavg = 4
    T0 = calculate_equivalent_Te(mxZavg)
    print(f'Equivalent temperature for <Z0> = {mxZavg:.2f} is Te = {T0:.3f} eV.')

    plot_data(nre, Zavg, s)


def calculate_equivalent_Te(Zavg):
    """
    Calculate the temperature required for <Z0> of Ne to be 'Zavg'.
    """
    ions = IonHandler()
    ions.addIon('D', Z=1, n=5e18)
    ions.addIon('Ne', Z=10, n=0.1*7.2e18 / VOLUME)

    ne = continuous.density(PN)

    a = 5
    b = 20

    for i in range(20):
        T0 = (a+b)*0.5

        _, ni = ionrate.equilibriumAtNfree(ions, ne, T0, fre=None)

        Zv = sum([Z0*ni[2+Z0] for Z0 in range(11)]) / np.sum(ni[2:])
        
        if Zv > Zavg:
            b = T0
        else:
            a = T0

    return T0


#def plot_data(mid, llim, ulim, lblm, lbll, lblu, NeIonizRange=False):
def plot_data(nre, Zavg, n):
    """
    Plot the simulated data.
    """
    NeLims = (6e-2, 2e-1)

    fig1, ax1 = plt.subplots(1, 1, figsize=(4.3, 2.8))
    ax1.semilogx(nre*NRE, Zavg, lw=2)

    p = np.polyfit(np.log10(nre*NRE), Zavg, 1)
    print(f'Fit: {p[0][0]:.3f}*log10(nre) + {p[1][0]:.3f}')
    #ax1.semilogx([1e13, 2e16], [p[0]*13+p[1], p[0]*np.log10(2e16)+p[1]], 'r--', lw=2)
    
    ax1.set_xlabel(r'$n_{\rm re}$ (m$^{-3}$)')
    ax1.set_ylabel(r'$\left\langle Z_0 \right\rangle$')
    #ax1.set_xlim([1e-3, 1])
    ax1.set_xlim([min(nre)*NRE, max(nre)*NRE])
    ax1.set_ylim([0, 7])
    #ax1.set_xticks([1e-3, 1e-2, 1e-1, 1], [r'$0.1\%$', r'$1\%$', r'$10\%$', r'$100\%$'])
    ax1.set_xticks([1e14, 1e15, 1e16, 1e17])
    ax1.set_yticks(list(range(8)), [f'{i:d}' for i in range(8)])
    ax1.grid('on')

    fig1.tight_layout()

    fig2, ax2 = plt.subplots(1, 1, figsize=(4.3, 2.5))
    ax2.semilogy(list(range(n.shape[1])), n[0,:])

    ax2.set_xlabel(r'Charge state $Z_0$')
    ax2.set_ylabel(r'Ne density (m$^{-3}$)')
    ax2.set_xlim([0, 10])

    fig2.tight_layout()

    plt.show()


def main():
    simulate()

    return 0


if __name__ == '__main__':
    sys.exit(main())


