# Class representing a runaway electron distribution function

import numpy as np
from DREAM.Formulas import getAvalancheDistribution, getRelativisticMaxwellJuttnerDistribution


class DistributionFunction:


    p = None
    xi = None
    f = None

    dp = None
    dxi = None


    def __init__(self):
        """
        Constructor.
        """
        pass


    def _generateGrid(self, pMin, pMax, nP, nXi):
        """
        Generate a uniform grid on which to evaluate the distribution function.
        """
        self.pMin = pMin
        self.pMax = pMax

        pp = np.linspace(pMin, pMax, nP+1)
        p = 0.5*(pp[:-1] + pp[1:])
        xip = np.linspace(-1, 1, nXi+1)
        xi = 0.5*(xip[:-1] + xip[1:])

        self.p = p
        self.xi = xi
        if nP > 1: self.dp = pp[1:] - pp[:-1]
        else: self.dp = np.ones(p.shape)

        if nXi > 1: self.dxi = xip[1:] - xip[:-1]
        else: self.dxi = np.ones(xi.shape)

        self.P, self.XI = np.meshgrid(self.p, self.xi)
        self.DP  = np.repeat(self.dp.reshape((1,self.p.size)), self.xi.size, axis=0)
        self.DXI = np.repeat(self.dxi.reshape((self.xi.size,1)), self.p.size, axis=1)

        return p, xi


    def moment(self, weight):
        """
        Evaluate a moment of the distribution function with the given weight.
        """
        fac = 1
        #fac = 0.6905
        #fac = 0.805
        I = fac*2*np.pi * np.sum(weight * self.f*self.P**2*self.DP*self.DXI)
        return I


    def setAvalanche(self, nre, pMin, pMax, nP=100, nXi=90, E=2, Z=9):
        """
        Set this distribution function according to an analytical
        avalanche distribution.
        """
        p, xi = self._generateGrid(pMin=pMin, pMax=pMax, nP=nP, nXi=nXi)
        f = getAvalancheDistribution(p, xi, E=E, Z=Z, nre=nre)

        self.f = f


    def setMaxwellian(self, nre, pMin, pMax, nP=100, nXi=10, T=10):
        """
        Set a Maxwellian distribution function with temperature T.
        """
        p, xi = self._generateGrid(pMin=pMin, pMax=pMax, nP=nP, nXi=nXi)
        self.f = getRelativisticMaxwellJuttnerDistribution(self.P, T, nre)


    def setStep(self, nre, pMin, pMax, nP=100, nXi=2, pUpper=20):
        """
        Set this distribution function to a step function in momentum
        space.

        :param pUpper: Maximum momentum of runaway electrons.
        """
        p, xi = self._generateGrid(pMin=pMin, pMax=pMax, nP=nP, nXi=nXi)

        idx = np.argmin(np.abs(p-pUpper))
        f = np.zeros((nXi, nP))
        f[:,0:(idx+1)] = 3*nre / (4*np.pi*(pUpper**3 - pMin**3))

        self.f = f


    def setDelta(self, nre, pStar):
        """
        Set the distribution function to a step function.
        """
        p, xi = self._generateGrid(pMin=0, pMax=2*pStar, nP=1, nXi=1)
        self.f = nre / pStar**2 * np.ones((1, 1))


    def saveToHDF5(self, f, saveto='fre'):
        """
        Save to an HDF5 file using the given HDF5 file handle.
        """
        g = f.create_group(saveto)

        g['p'] = self.p
        g['xi'] = self.xi
        g['dp'] = self.dp
        g['dxi'] = self.dxi
        g['f'] = self.f


