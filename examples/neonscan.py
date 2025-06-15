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
    sdataUU = continuous.simulate(Ne_frac=0.4)
    sdataU = continuous.simulate(Ne_frac=0.2)
    sdataM = continuous.simulate(Ne_frac=0.1)
    sdataL = continuous.simulate(Ne_frac=0.01)

    #plot_data(sdataM, sdataL, sdataU, lblm=r'$10\%$ Ne', lbll=r'$1\%$ Ne', lblu=r'$40\%$ Ne')
    data = [
        (sdataL, r'$1\%$ Ne'),
        (sdataM, r'$10\%$ Ne'),
        (sdataU, r'$20\%$ Ne'),
        (sdataUU, r'$40\%$ Ne')
    ]
    plot_data(data)


#def plot_data(mid, llim, ulim, lblm, lbll, lblu, NeIonizRange=False):
def plot_data(data, NeIonizRange=False):
    """
    Plot the simulated data.
    """
    NeLims = (6e-2, 2e-1)
    fig1, ax1 = plt.subplots(1, 1, figsize=(4.3, 2.5))

    colors = ['tab:blue', 'k', 'tab:green', 'tab:red']
    styles = ['--', '-', '--', '--']

    for i in range(len(data)):
        ax1.loglog(data[i][0]['pn'], data[i][0]['nD'], color=colors[i], linestyle=styles[i], label=data[i][1])
    
    if NeIonizRange:
        ax1.axvspan(*NeLims, color='r', alpha=.3)

    ax1.set_xlabel(r'Neutral pressure $p^{\rm B}_{\rm n}$ (Pa)')
    ax1.set_ylabel(r'$n_{\rm D/H}$ (m$^{-3}$)')
    ax1.set_xlim([1e-2, 10])
    ax1.grid('on')
    ax1.legend()

    fig1.tight_layout()


    #############################
    # NEUTRAL PRESSURE
    #############################
    fig4, ax4 = plt.subplots(1, 1, figsize=(4.3, 2.5))

    for i in range(len(data)):
        p = e*data[i][0]['nN']*data[i][0]['Te']
        ax4.loglog(data[i][0]['pn'], p, color=colors[i], linestyle=styles[i], label=data[i][1])

    if NeIonizRange:
        ax4.axvspan(*NeLims, color='r', alpha=.3)

    ax4.set_xlabel(r'Neutral pressure $p^{\rm B}_{\rm n}$ (Pa)')
    ax4.set_ylabel(r'$T_e \sum n_i^{(0)}$ (Pa)')
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


