#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import continuous
import sys

from scipy.constants import e


def simulate():
    """
    Run a simulation.
    """
    sdataU = continuous.simulate(Ne_frac=0.1, nre_frac=1)
    sdataM = continuous.simulate(Ne_frac=0.1, nre_frac=.1)
    sdataL = continuous.simulate(Ne_frac=0.1, nre_frac=.01)
    sdataLL = continuous.simulate(Ne_frac=0.1, nre_frac=0)

    #plot_data(sdataM, sdataL, sdataU, lblm=r'$10\%$ Ne', lbll=r'$1\%$ Ne', lblu=r'$40\%$ Ne')
    data = [
        (sdataLL, r'$0\%\,n_{\rm re}$'),
        (sdataL, r'$1\%\,n_{\rm re}$'),
        (sdataM, r'$10\%\,n_{\rm re}$'),
        (sdataU, r'$100\%\,n_{\rm re}$')
    ]
    plot_data(data)


#def plot_data(mid, llim, ulim, lblm, lbll, lblu, NeIonizRange=False):
def plot_data(data, NeIonizRange=False):
    """
    Plot the simulated data.
    """
    NeLims = (6e-2, 2e-1)

    colors = ['tab:blue', 'k', 'tab:green', 'tab:red']
    styles = ['-', '--', ':', '-.']

    fig1, ax1 = plt.subplots(1, 1, figsize=(4.3, 2.5))
    for i in range(len(data)):
        ax1.loglog(data[i][0]['pn'], data[i][0]['nD'], color=colors[i], linestyle=styles[i], label=data[i][1])
    
    ax1.set_xlabel(r'Neutral pressure $p^{\rm B}_{\rm n}$ (Pa)')
    ax1.set_ylabel(r'$n_{\rm D/H}$ (m$^{-3}$)')
    ax1.set_xlim([1e-2, 10])
    ax1.grid('on')
    ax1.legend()

    fig1.tight_layout()

    ##########################
    # IONIZATION RATES
    ##########################
    fig2, ax2 = plt.subplots(1, 1, figsize=(4.3, 2.8))

    for i in range(len(data)):
        ax2.loglog(data[i][0]['pn'], data[i][0]['I'], color='tab:green', linestyle=styles[i], label=data[i][1])
        ax2.loglog(data[i][0]['pn'], data[i][0]['Ire'], color='tab:red', linestyle=styles[i])

    #ax2.loglog([], [], '-', color='tab:green', label='$I n_e n_i^{(0)}$')
    #ax2.loglog([], [], '-', color='tab:red', label=r'$\mathcal{I} n_i^{(0)}$')

    ax2.set_xlabel(r'Neutral pressure $p^{\rm B}_{\rm n}$ (Pa)')
    ax2.set_ylabel(r'Rate (m$^{-3}$s$^{-1}$)')
    ax2.set_xlim([1e-2, 10])
    ax2.set_ylim([2e15, 1e22])
    ax2.grid('on')
    #ax2.legend(ncols=2, loc=(.06, .74), framealpha=1, handletextpad=0.4)
    ax2.legend(ncols=2, framealpha=1, handletextpad=0.4)
    #ax2.set_title('D')

    fig2.tight_layout()

    #############################
    # NEUTRAL PRESSURE
    #############################
    fig4, ax4 = plt.subplots(1, 1, figsize=(4.3, 2.5))

    for i in range(len(data)):
        p = e*data[i][0]['nN']*data[i][0]['Te']
        ax4.loglog(data[i][0]['pn'], p, color=colors[i], linestyle=styles[i], label=data[i][1])

    ax4.loglog([1e-3, 10], [1e-3, 10], '--', color='tab:red')
    ax4.text(1e-1, 2e-1, r'$p_{\rm n}^{\rm B}=p_{\rm n}$', color='tab:red', rotation=28)

    if NeIonizRange:
        ax4.axvspan(*NeLims, color='r', alpha=.3)

    ax4.set_xlabel(r'Neutral pressure $p^{\rm B}_{\rm n}$ (Pa)')
    ax4.set_ylabel(r'$p_{\rm n}=T_e \sum n_i^{(0)}$ (Pa)')
    ax4.set_xlim([3e-4, 10])
    ax4.set_ylim([3e-4, 10])
    ax4.grid('on')
    ax4.legend()

    fig4.tight_layout()


    plt.show()


def main():
    simulate()

    return 0


if __name__ == '__main__':
    sys.exit(main())


