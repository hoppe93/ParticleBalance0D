# Routines for solving the ion rate equation


import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import solve
import sys
import time
from scipy.constants import e

from IonHandler import IonHandler


def equilibrium(ions, Te, fre, reltol=1e-3, V_plasma=1, V_vessel=1):
    """
    Calculates the ion equilibrium for all species at the given temperature,
    with the given RE distribution function.
    """
    # Initial guess for electron density
    # (assume all species singly ionized on average)
    ne0 = ions.getTotalIonDensity()
    neprev = ne0

    # Initial step
    A, b = construct_matrix(ions, ne0, Te, fre=fre, V_plasma=V_plasma, V_vessel=V_vessel)
    n = solve(A, b)

    ions.setSolution(n)
    ne = ions.getElectronDensity()

    # Solve equilibrium using a bisection algorithm
    itr = 0
    a, b = 0, ions.getTotalElectronDensity()

    while abs(a/b-1) > reltol:
        itr += 1

        ne = 0.5*(a+b)

        _A, _b = construct_matrix(ions, ne, Te, fre=fre, V_plasma=V_plasma, V_vessel=V_vessel)
        n = solve(_A, _b)
        ions.setSolution(n)

        nenew = ions.getElectronDensity()
        if nenew > ne:
            a = ne
        else:
            b = ne

    ne = 0.5*(a+b)
    A, b = construct_matrix(ions, ne, Te, fre=fre, V_plasma=V_plasma, V_vessel=V_vessel)
    n = solve(A, b)
    ions.setSolution(n)

    if abs(ne-ions.getElectronDensity()) > 2*reltol*ne:
        raise Exception(f"Bisection algorithm failed to converge to the correct solution. Solution ne={ne:.6e} does not agree with electron density from quasi-neutrality: {ions.getElectronDensity():.6e}.")
    
    #print(f'Iterations {itr}')
    return ions.getSolution()


def construct_matrix(
    ions: IonHandler, ne, Te, fre=None, V_plasma=1, V_vessel=1
):
    """
    Construct the matrix for the ion rate equation.
    """
    N = ions.getNumberOfStates()
    A = np.zeros((N, N))
    b = np.zeros((N,))

    iVf = lambda j : (V_plasma / V_vessel) if j==0 else 1

    off = 0
    for ion in ions:
        I = lambda j : 0 if j==ion.Z else ion.scd(Z0=j, n=ne, T=Te)
        R = lambda j : 0 if j==0     else ion.acd(Z0=j, n=ne, T=Te)

        for j in range(ion.Z+1):
            if j > 0:
                A[off+j,off+j-1] = iVf(j)*I(j-1)*ne

            A[off+j,off+j] = -iVf(j)*(I(j) + R(j))*ne

            if j < ion.Z:
                A[off+j,off+j+1] = iVf(j)*R(j+1)*ne

            # Add fast-electron impact ionization
            if fre is not None:
                if j < ion.Z:
                    A[off+j,off+j] -= iVf(j)*ion.evaluateImpactIonizationRate(Z0=j, fre=fre)
                    if j > 0:
                        A[off+j,off+j-1] += iVf(j)*ion.evaluateImpactIonizationRate(Z0=j-1, fre=fre)

        A[off+ion.Z,off:(off+ion.Z+1)] = 1
        b[off+ion.Z] = ion.n

        off += ion.Z+1

    return A, b


def construct_nT_jacobian(
    ions: IonHandler, ne, Te, fre=None, V_plasma=1, V_vessel=1
):
    """
    Constructs the ne / Te part of a jacobian matrix. Returns a
    nIonStates-by-2 matrix with elements

      [ dF(iZ=0, Z0=0)/dn  dF(iZ=0, Z0=0)/dT
        dF(iZ=0, Z0=1)/dn  dF(iZ=0, Z0=1)/dT
        ...
        dF(iZ=iZmax, Z0=iZ0max)/dn  dF(iZ=iZmax, Z0=iZ0max)/dT ].
    """
    N = ions.getNumberOfStates()
    iVf = lambda j : (V_plasma / V_vessel) if j==0 else 1
    J = np.zeros((N, 2))

    for k in range(2):
        off = 0
        for ion in ions:
            dIdn = lambda j : 0 if j==ion.Z else ion.scd.deriv_ne(Z0=j, n=ne, T=Te)
            dRdn = lambda j : 0 if j==0     else ion.acd.deriv_ne(Z0=j, n=ne, T=Te)
            dIdT = lambda j : 0 if j==ion.Z else ion.scd.deriv_Te(Z0=j, n=ne, T=Te)
            dRdT = lambda j : 0 if j==0     else ion.acd.deriv_Te(Z0=j, n=ne, T=Te)

            I = [dIdn, dIdT]
            R = [dRdn, dRdT]

            for j in range(ion.Z+1):
                if j > 0:
                    J[off+j,k] += iVf(j)*I[k](j-1)*ne * ion.solution[j-1]

                J[off+j,k] -= iVf(j)*(I[k](j) + R[k](j))*ne * ion.solution[j]

                if j < ion.Z:
                    J[off+j,k] += iVf(j)*R[k](j+1)*ne * ion.solution[j+1]

            J[off+ion.Z,:] = 0
            off += ion.Z+1

    return J


def equilibriumAtPressure(
    ions, p0, pfac, Te, Tn, fre, n0=2e19, reltol=1e-3, species='D',
    returnany=False
):
    """
    Evaluate the equilibrium charge-state distribution at the specified
    neutral pressure.

    :param ions:      IonHandler object.
    :param p:         Target pressure.
    :param Te:        Electron temperature.
    :param Tn:        Neutral temperature (may be a list with one element per ion species; same order as ``ions``).
    :param fre:       Runaway electron distribution function.
    :param n0:        Initial guess for main species density.
    :param reltol:    Relative tolerance within which to determine main species density.
    :param species:   Name of species to vary density of in order to change neutral pressure.
    :param returnany: Even the neutral pressure cannot be satisfied, return the solution for the lowest achievable pressure.
    """
    ions[species].n = n0
    n = equilibrium(ions, Te=Te, fre=fre)

    def press():
        """
        Evaluates the neutral pressure.
        """
        if np.isscalar(Tn):
            pn = ions.getNeutralDensity() * Tn
        else:
            pn = 0
            for ion, T in zip(ions, Tn):
                pn += ion.getNeutralDensity() * T
        
        return pn*pfac

    nS0 = ions[species].getNeutralDensity()
    nS  = ions[species].n

    # Construct better guess
    pk = press()
    if np.isscalar(Tn):
        Tnavg = Tn
    else:
        Tnavg = sum(Tn) / len(Tn)

    dnS = nS/nS0 * (p0-pk) / Tnavg

    if dnS < 0 and abs(dnS/nS) > 1:
        dnS = -0.9*nS

    # Bisection algorithm
    ions[species].n = nS+dnS
    n = equilibrium(ions, Te=Te, fre=fre)
    pk2 = press()

    # Set upper limit
    if pk > p0:
        b = nS
    elif pk2 > p0:
        b = nS+dnS
    else:
        # Iteratively find upper limit
        itr = 0
        nSk = nS
        while pk2 < p0 and itr <= 10:
            nSk *= 2
            ions[species].n = nSk
            n = equilibrium(ions, Te=Te, fre=fre)
            pk2 = press()

            itr += 1

        if itr > 10:
            raise Exception("Unable to find an upper limit for the bisection algorithm. The desired neutral pressure cannot be reached.")

        b = nSk

    # Set lower limit
    if pk < p0 and pk2 < p0:
        a = max(pk, pk2)
    elif pk < p0:
        a = nS
    elif pk2 < p0:
        a = ions[species].n
    else:
        a = 1e10

    nS = 0.5*(a+b)

    # Do bisection
    while abs(a/b-1) > reltol:
        ions[species].n = nS
        n = equilibrium(ions, Te=Te, fre=fre)

        if press() > p0:
            b = nS
        else:
            a = nS

        nS = 0.5*(a+b)

    if abs(press()/p0-1) > max(1e-2, 10*reltol):
        if not returnany:
            raise Exception(f"Unable to reach the desired neutral pressure. press = {press():.4f}, p0 = {p0:.4f}")

    ions.setSolution(n)
    nfree = ions.getElectronDensity()

    return nfree, n


def equilibriumAtNfree(
    ions, nfree, Te, fre, reltol=1e-3, species='D', maxiter=50
):
    """
    Calculate ion equilibrium configuration that yields the specified
    free electron density.
    """
    # Initial guess for 'species' density
    # (assume all species are singly ionized on average)
    n0 = nfree - ions.getTotalIonDensity() + ions[species].n
    ions[species].setDensity(n0)

    # Initial step
    A, b = construct_matrix(ions, nfree, Te, fre=fre)
    ni = solve(A, b)

    ions.setSolution(ni)
    ne = ions.getElectronDensity()

    # Find equilibrium solution
    itr = 0
    while abs(ne/nfree-1) > reltol and itr < maxiter:
        itr += 1

        # Construct new solution
        n0 = ions[species].n
        n0e = ions[species].getElectronDensity()
        #n0 += (nfree - ne) * (n0e / n0)
        n0 += (nfree - ne) * (n0 / n0e)
        ions[species].setDensity(n0)

        # Solve
        A, b = construct_matrix(ions, nfree, Te, fre=fre)
        ni = solve(A, b)

        ions.setSolution(ni)
        ne = ions.getElectronDensity()

    if abs(ne/nfree-1) > reltol:
        raise Exception(f"Failed to converge to the correct solution after {itr} iterations.")
    
    return n0, ions.getSolution()


