#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import continuous
import sys

from scipy.constants import e

import config
import ionrate
from DistributionFunction import DistributionFunction
from IonHandler import IonHandler


def simulate(Ne_frac=.1, pfac=120):
    """
    Run a simulation.
    """
    nImp = Ne_frac * 7.2e18 / continuous.VOLUME
    nD = 2e19

    ions = IonHandler()
    ions.addIon('D', Z=1, n=nD)
    if nImp > 0:
        ions.addIon('Ne', Z=10, n=nImp)

    fre = DistributionFunction()

    pn = np.logspace(-2, 1, 40)
    nre_facs = [0.25, 0.50, 1.0, 2.0, 4.0]
    nre_dicts = []

    for nre_fac in nre_facs:
        print(f'\n:: nre_fac  {(nre_fac-1)*100:+.0f}%')
        fre.setStep(
            nre=continuous.NRE*nre_fac, pMin=1,
            pMax=continuous.PRE, pUpper=continuous.PRE,
            nP=4000
        )
        ions.cacheImpactIonizationRate(fre)

        d = {
            'pn': [],
            'Te': [],
            'nD': [],
            'ne': [],
            'nN': [],
            'I': [],
            'R': [],
            'Ire': [],
        }

        for i in range(len(pn)):
            _Te = continuous.temperature(pn[i])

            n0, ni = ionrate.equilibriumAtPressure(
                ions=ions, p0=pn[i], pfac=pfac,
                Te=_Te, Tn=1*e, fre=fre
            )

            nfree = ions.getElectronDensity()
            d['pn'].append(pn[i])
            d['Te'].append(_Te)
            d['nD'].append(n0)
            d['nN'].append(ions.getNeutralDensity())
            d['ne'].append(nfree)
            d['I'].append(ions['D'].scd(Z0=0, n=nfree, T=_Te)*nfree*ni[0])
            d['R'].append(ions['D'].acd(Z0=1, n=nfree, T=_Te)*nfree*ni[1])
            d['Ire'].append(ions['D'].evaluateImpactIonizationRate(Z0=0, fre=fre)*ni[0])

            print(f'{i+1}... ', end='' if (i+1)%10!=0 else '\n', flush=True)

        d['nre'] = continuous.NRE * nre_fac
        # Turn into numpy arrays
        for k in d.keys():
            d[k] = np.array(d[k])

        nre_dicts.append(d)

    plot_data(continuous.NRE, nre_facs, nre_dicts)

    return nre_dicts


def plot_data(nre0, nre_facs, nre_dicts):
    """
    Plot data.
    """
    i0 = int(len(nre_facs) / 2)

    fig1, ax1 = plt.subplots(1, 1, figsize=(4.3, 2.8))

    d = nre_dicts[i0]
    ax1.semilogx(d['pn'], d['ne']/1e19, 'k', lw=2)

    pidx = np.argmin(np.abs(d['pn']-1))
    print(f'\n:: Density variation (at p = {d["pn"][pidx]:.2f} Pa):')

    # Mark variation around this value
    colors = ['tab:blue', 'tab:red', 'tab:green']
    for i in range(i0):
        d1 = nre_dicts[i]
        d2 = nre_dicts[-(i+1)]

        #lbl = f'$n_{{\\rm re}}\\pm{abs(nre_facs[i]-1)*100:.0f}\\%$'
        lbl = f'$n_{{\\rm re}}\\times \\frac{{1}}{{{1/nre_facs[i]:.0f}}}, {1/nre_facs[i]:.0f}$'

        #ax1.semilogx(d1['pn'], d1['ne'], '--', color=colors[i], label=lbl)
        #ax1.semilogx(d2['pn'], d2['ne'], '--', color=colors[i])
        ax1.fill_between(d1['pn'], d1['ne']/1e19, d2['ne']/1e19, color=colors[i], alpha=.4, label=lbl)

        pidx = np.argmin(np.abs(d1['pn']-1))
        maxvarP = (d1['ne']-d['ne'])/d['ne']*100
        maxvarP = maxvarP[np.argmax(np.abs(maxvarP))]
        maxvarN = (d2['ne']-d['ne'])/d['ne']*100
        maxvarN = maxvarN[np.argmax(np.abs(maxvarN))]

        nomvarP = (d1['ne'][pidx]-d['ne'][pidx]) / d['ne'][pidx] * 100
        nomvarN = (d2['ne'][pidx]-d['ne'][pidx]) / d['ne'][pidx] * 100

        print(f"  /{1/nre_facs[i]:.0f}: {nomvarP:+.1f}% (max {maxvarP:+.1f}%)")
        print(f"  x{nre_facs[-(i+1)]:.0f}: {nomvarN:+.1f}% (max {maxvarN:+.1f}%)")

    ax1.set_xlabel(r'Neutral pressure $p^{\rm B}_{\rm n}$ (Pa)')
    ax1.set_ylabel(r'$n_{\rm e}$ ($10^{19}\,$m$^{-3}$)')
    ax1.set_xlim([1e-2, 10])
    ax1.grid('on')
    ax1.legend(handletextpad=0.4)

    fig1.tight_layout()



    ###############################
    # IONIZATION RATES
    ###############################
    fig2, ax2 = plt.subplots(1, 1, figsize=(4.3, 2.5))

    d = nre_dicts[i0]
    ax2.loglog(d['pn'], d['I'], 'k--', lw=2, label=r'$I_D^{(0)}$')
    ax2.loglog(d['pn'], d['Ire'], 'k', lw=2, label=r'$I_{\rm re}^{(0)}$')

    for i in range(i0):
        d1 = nre_dicts[i]
        d2 = nre_dicts[-(i+1)]

        lbl = f'$\\pm{abs(nre_facs[i]-1)*100:.0f}\\%$'

        ax2.fill_between(d1['pn'], d1['Ire'], d2['Ire'], color=colors[i], alpha=.4, label=lbl)
        ax2.fill_between(d1['pn'], d1['I'], d2['I'], color=colors[i], alpha=.4)
        #ax2.loglog(d1['pn'], d1['Ire'], '--', color=colors[i], label=lbl)
        #ax2.loglog(d2['pn'], d2['Ire'], '--', color=colors[i])

    ax2.set_xlabel(r'Neutral pressure $p^{\rm B}_{\rm n}$ (Pa)')
    ax2.set_ylabel(r'Ioniz.\ rate (m$^{-3}$s$^{-1}$)')
    ax2.set_xlim([1e-2, 10])
    ax2.grid('on')
    ax2.legend(ncol=2, loc=(.03, .66))
    fig2.tight_layout()


    plt.show()


def main():
    simulate()
    return 0


if __name__ == '__main__':
    sys.exit(main())


