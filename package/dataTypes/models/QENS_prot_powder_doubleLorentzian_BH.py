import numpy as np

from collections import namedtuple
from scipy import optimize

from ..QENSType import DataTypeDecorator
from ...fit.fitQENS_models import protein_powder_2Lorentzians as model



class Model(DataTypeDecorator):
    """ This class stores data as resolution function related. It allows to perform a fit using a 
        pseudo-voigt profile as a model for instrument resolution. """

    def __init__(self, dataType):
        super().__init__(dataType)

        self.model      = model
        self.params     = None
        self.paramsNames = ['s0', 's1', 's2', 'g1', 'g2', 'msd', 'bkgd'] #_For plotting purpose

        self.BH_iter    = 100
        self.disp       = True



    def fit(self, p0=None, bounds=None):
        print("\nStarting basinhopping fitting for file: %s" % self.fileName, flush=True)
        print(50*"-", flush=True)

        if not p0: #_Using default initial values
            p0 = [0.6, 0.2, 0.2, 2, 15, 1, 0.001]

        if not bounds: #_Using default bounds
            bounds = [(0., 1), (0., 1), (0., 1), (0., 1000), (0., 1000), (0., 10), (0., 0.1)] 


        result = optimize.basinhopping( self.model, 
                                        p0,
                                        niter = self.BH_iter,
                                        niter_success = 0.5*self.BH_iter,
                                        disp=self.disp,
                                        minimizer_kwargs={ 'args':(self,), 'bounds':bounds } )



        #_Creating a list with the same parameters for each q-values (makes code for plotting easier)
        out = []
        for qIdx in self.data.qIdx:
            out.append(result)

        self.params = out    




    def qWiseFit(self, p0=None, bounds=None):
        print("\nStarting basinhopping fitting for file: %s\n" % self.fileName, flush=True)
        print(50*"-" + "\n", flush=True)

        if not p0: #_Using default initial values
            p0 = [0.6, 0.2, 0.2, 2, 15, 1, 0.001]

        if not bounds: #_Using default bounds
            bounds = [(0., 1), (0., 1), (0., 1), (0., 1000), (0., 1000), (0., 10), (0., 0.1)] 


        result = []
        for qIdx in self.data.qIdx:
            result.append(optimize.basinhopping( self.model, 
                                        p0,
                                        niter = self.BH_iter,
                                        niter_success = 0.5*self.BH_iter,
                                        disp=self.disp,
                                        minimizer_kwargs={ 'args':(self, qIdx), 'bounds':bounds } ))



        self.params = result

