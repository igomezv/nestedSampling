import numpy as np
from scipy.special import logsumexp

class nested:
    def __init__(self, loglike, priorTransform, nlive, ndims, maxiter=50000):
        self.loglike = loglike
        self.priorTransform = priorTransform
        self.ndims = ndims
        self.nlive = nlive
        self.maxiter = maxiter

    def sampling(self, accuracy=0.01):
        #u, x, logx, logw, logz
        upoints = []
        vpoints = []
        logx = []
        logL = []
        logLstar = []
        logw = []
        logLw = []
        logz = []
        dead = []
        # Start nlive points
        for i in range(self.nlive):
            upoints.append(np.random.rand(self.ndims))
            vpoints.append(self.priorTransform(upoints[i]))
            logL.append(self.loglike(vpoints[i]))
            logLstar.append(logL[i])
            logx.append(0)
            logLw.append(0)
            logw.append(1)
            logz.append(1e-300)

        # Begin the nested sampling loop
        clogz = -1e300
        # x_0 = 1 -> logx_0 = 0 x = 1 - exp(-1/N)
        clogx = prevlogx = np.log(1. - np.exp(-1./self.nlive))
        print("clogx", clogx, type(clogx))

        for i in range(self.maxiter):
            # worst is the index of the min loglike
            worst = logL.index(min(logL))
            #print("worst", worst, type(worst))
            clogw = logsumexp([prevlogx, clogx], b=[1, -1])
            #print("clogw", clogw, type(clogw))
            # prevlogx = clogx
            logLw[worst] = clogw + logL[worst]
            #print("logLw[worst]", logLw[worst], type(logLw[worst]))
            # Update Evidence Z
            #print(type(logLw[worst]), logLw[worst])
            clogz = logsumexp([clogz, logLw[worst]])
            logLstar[worst] = logL[worst]
            # Kill worst object in favour of copy of different survivor
            if self.nlive > 1:
                while True:
                    copy = np.random.randint(0, self.nlive)
                    if copy != worst:
                        break

            upoints[worst], vpoints[worst], logL[worst] = upoints[copy], vpoints[copy], logL[copy]
            upoints[worst], vpoints[worst], logL[worst] = self.explore(upoints[worst], logLstar[worst])

            dead.append([upoints[worst], vpoints[worst], logL[worst],
                        logw[worst], logLw[worst], logz[worst]])

            # rlogz logz remain in livepoints or estimate Z as in nestle
            rlogz = np.max(logL) + clogx
            # # dlogz is used to stopping criteria.
            dlogz = np.logaddexp(clogz, rlogz) - clogz
            #dlogz = logsumexp([rlogz, clogz]) - clogz


            print("{}/{} | logz: {:.3f} | dlogz: {:.3f} | logw: {:.3f} "
                  "| logLstar: {:.3f} | v: {}".format(i+1, self.maxiter, clogz,
                                          dlogz, clogw, logL[worst], vpoints[worst]))

            if dlogz < np.log(accuracy):
                print("Stopping criteria!")
                break
            clogx -= 1.0 / self.nlive


    def explore(self, uworst, logLstar):
        step = 0.1
        accept = 0
        reject = 0
        for _ in range(50):
            pu = uworst + step * (2.*np.random.random(self.ndims) - 1.)
            pu -= np.floor(pu)
            pv = self.priorTransform(pu)
            ploglike = self.loglike(pv)
            # Hard likelihood constraint
            if ploglike > logLstar:
                accept += 1
            else:
                reject += 1
            # Refine step-size
            if accept > reject:
                step *= np.exp(1.0 / accept)
            elif accept < reject:
                step /= np.exp(1.0 / reject)
        return pu, pv, ploglike

