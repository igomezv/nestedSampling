####Nested sampling based on Skilling (2009)
import numpy as np
import pandas as pd
from time import sleep

####################### Here begin the nested sampling#########################
class SkillingNS:
    def __init__(self, logLike, priorTransform, nDims, bounds, nlivepoints = 10,\
                 accuracy = 0.5, **kwargs):
        """
        Parameters
        -----------
        logLike : log-likelihood function
        priorTransform : priorTransform function that maps (0,1) -> (bound_inf, bound_sup)
        nDims : int # dimensions
        accuracy : float
        #dim is the dimension or the number of free parameters

        **kwargs:
              
        """
        self.logLike = logLike
        self.priorTransform = priorTransform
        self.nDims = nDims
        self.bounds = bounds
        self.nlivepoints = nlivepoints
        self.accuracy = accuracy
    
    def beta_dist(self, N):
        """
        This function generates t_i from a beta distribution. 
            \beta(N,1)
        """
        ti = np.random.beta(N, 1)
        return ti
    
    
#def replace(lpoint, llike, bounds):
    def replace(self, lpoint, llike, hpoint):
        """
        This functions recieves the point with the lowest likelihood and generates one with higher likelihood.
        
        Parameters
        ------------
            lpoint      :   point with lower likelihood.
            llike       :   lower likelihood.
            priormass   :   Prior mass
            priorT      :   Prior transform
        """
        #new_point = lpoint
        new_point = (lpoint+hpoint)/2
        new_loglike = llike
        i=0
        while (new_loglike <= llike):
            i+=1
            #print("it of replace function : ", i)
            #print("finding best likeli")
            #print("new point  = llike", new_point)
            new_point = self.priorTransform(np.random.rand(self.nDims,), self.bounds)  
            #print("new point 1", new_point)
            new_loglike = self.logLike(new_point)       
        print("better point")
        return new_point, new_loglike
    

    def run_sampler(self):             
        #bounds = np.array([0, 10])
        vectors = []
        test = np.random.rand(self.nDims,)
        for i in range(self.nDims):
            vectors.append(self.priorTransform(np.random.rand(self.nDims,), self.bounds))
        
        #evidence is z
        z = 0
        
        #initial prior mass is x0 = 1
        xi = 1
        
        #j iterations
        j = 100
        
        #vector of loglikes
        loglikes = []
        for vector in vectors:
            print(type(vector), np.shape(vector), vector)
            loglikes.append(self.logLike(vector))
        #join loglikes and points
        df = pd.DataFrame()
        df['points'] = vectors
        df['loglikes'] = loglikes


        for i in range(j):
            print("iteration {}".format(i))
            #sort points by loglikes
            df.sort_values('loglikes', inplace = True)
            df.reset_index(drop=True, inplace=True)
            #sleep(0.5)
            #record the lower loglike,  L_i in the skilling paper
            lowerpoint, lowerlike = df.iloc[0]
            #new prior mass X
            xf = np.exp(-i/self.nDims)
            #wi simple
            wi = xi - xf
            z += lowerlike * wi
            #replace lower like point
            print("prior mass {}".format(xf))
            ####I have a problem when the bounds are reduced by the remained prior mass. 
            #If I comment the next line, the code works.
            #self.bounds = xf*self.bounds
            print("bounds remain {}".format(self.bounds))
            newpoint, newlike = self.replace(lowerpoint, lowerlike, self.bounds)
            print("new point {} \n".format(newpoint))
            print("Z : {} ".format(z))
            
            #print("logZ : {} ".format(np.log(z)))
        
            df.iloc[0] = newpoint, newlike
            
            self.print_vals(df, z, xf)
        
    def print_vals(self, df, z, xf):
        print(df['loglikes'].values)
        z += (1/self.nDims)*xf*np.sum(df['loglikes'].values)
        highestpoint, highestlike = df.iloc[self.nDims-1]
        #print("Parameter estimation : \n {}  \n ".format(df['points'].values))
        print("Parameter estimation : \n {}  \n ".format(highestpoint))
        print("Likelihood : \n {}  \n ".format(highestlike))
        print("Bayesian Evidence : {}".format(z))
            #print("log Z : {}".format(np.log(-z)))

