from scipy.special import logsumexp
import numpy as np

class Object:
    def __init__(self):
        # self.points = None
        # self.phys_points = None
        self.u = None 
        self.v = None
        self.x = None
        self.y = None
        self.logL = None
        self.logWt = None

class nestedSampling:
    def __init__(self, logLike, priorTransform, nlive, ndims, maxiter):
        self.logLike = logLike
        self.priorTransform = priorTransform
        self.nlive = nlive
        self.ndims = ndims
        self.maxiter = maxiter

    def sampling(self):
        livesamples = []              # Collection of n objects
        samples = []          # livesamplesects stored for posterior results
        H    = 0.0            # Information, initially 0
        logZ =-1e300       # ln(Evidence Z, initially 0)

        # Set prior objects
        for i in range(self.nlive):
            livesamples.append(self.sample_from_prior())

        # Outermost interval of prior mass
        logwidth = np.log(1.0 - np.exp(-1.0 / self.nlive))

        # NESTED SAMPLING LOOP ___________________________________________
        for nest in range(self.maxiter):

            # Worst object in collection, with Weight = width * Likelihood
            worst = 0
            for i in range(1,self.nlive):
                if livesamples[i].logL < livesamples[worst].logL:
                    worst = i

            livesamples[worst].logWt = logwidth + livesamples[worst].logL

            # Update Evidence Z and Information H
            logZnew = logsumexp([logZ, livesamples[worst].logWt])

            H = np.exp(livesamples[worst].logWt - logZnew) * livesamples[worst].logL + \
                np.exp(logZ - logZnew) * (H + logZ) - logZnew
            logZ = logZnew

            # Posterior samples (optional)
            samples.append(livesamples[worst])

            # Kill worst object in favour of copy of different survivor
            if self.nlive>1: # don't kill if n is only 1
                while True:
                    copy = int(self.nlive * np.random.random()) % self.nlive  # force 0 <= copy < n
                    if copy != worst:
                        break

            logLstar = livesamples[worst].logL       # new likelihood constraint
            livesamples[worst] = livesamples[copy]          # overwrite worst object

            # Evolve copied object within constraint
            updated = self.explore(livesamples[worst], logLstar)
            assert(updated != None) # Make sure explore didn't update in-place
            livesamples[worst] = updated

            # Shrink interval
            logwidth -= 1.0 / self.nlive

        # Exit with evidence Z, information H, and optional posterior samples
        sdev_H = H/np.log(2.)
        sdev_logZ = np.sqrt(H/self.nlive)
        results = {"samples":samples, "num_iterations":(nest+1), "logZ":logZ,
                "logZ_sdev":sdev_logZ, "info_nats":H, "info_sdev":sdev_H}

        self.process_results(results)

        return results

    def sample_from_prior(self):
        Obj = Object()
        Obj.u = np.random.random()  # uniform in (0,1)
        Obj.v = np.random.random()  # uniform in (0,1)
        Obj.x, Obj.y = self.priorTransform([Obj.u, Obj.v])
        Obj.logL = self.logLike([Obj.x, Obj.y])
        #Obj.points = np.random.rand(self.ndims)
        # Obj.u = np.random.random()                # uniform in (0,1)
        # Obj.v = np.random.random()                # uniform in (0,1)
        #Obj.phys_points = self.priorTransform(Obj.points)
        # Obj.x = Obj.u * (10-0) + 0
        # Obj.y = Obj.v * (6+2) - 2
        #theta[c]*(bound[1]-bound[0])+bound[0])
        # Obj.x, Obj.y = priorTransform([Obj.u, Obj.v])             # map to x
        # Obj.y = 2.0 * Obj.v                    # map to y
        #Obj.logL = self.logLike(Obj.phys_points)
        return Obj

    def explore(   # Evolve object within likelihood constraint
        self, Obj,       # Object being evolved
        logLstar): # Likelihood constraint L > Lstar

        ret = Object()
        ret.__dict__ = Obj.__dict__.copy()
        step = 0.1   # Initial guess suitable step-size in (0,1)
        accept = 0   # # MCMC acceptances
        reject = 0   # # MCMC rejections
        Try = Object()          # Trial object

        for m in range(20):  # pre-judged number of steps

            # Trial object
            Try.u = ret.u + step * (2.*np.random.random() - 1.)  # |move| < step
            Try.v = ret.v + step * (2.*np.random.random() - 1.)  # |move| < step
            Try.u -= np.floor(Try.u)      # wraparound to stay within (0,1)
            Try.v -= np.floor(Try.v)      # wraparound to stay within (0,1)
            Try.x, Try.y = self.priorTransform([Try.u, Try.v])
            #4.0 * Try.u - 2.0  # map to x
            #Try.y = 2.0 * Try.v        # map to y
            Try.logL = self.logLike([Try.x, Try.y])  # trial likelihood value

            # Accept if and only if within hard likelihood constraint
            if Try.logL > logLstar:
                ret.__dict__ = Try.__dict__.copy()
                accept+=1
            else:
                reject+=1

            # Refine step-size to let acceptance ratio converge around 50%
            if( accept > reject ):   step *= np.exp(1.0 / accept)
            if( accept < reject ):   step /= np.exp(1.0 / reject)
        return ret

    def process_results(self, results):
        (x,xx) = (0.0, 0.0) # 1st and 2nd moments of x
        (y,yy) = (0.0, 0.0) # 1st and 2nd moments of y
        ni = results['num_iterations']
        samples = results['samples']
        logZ = results['logZ']
        for i in range(ni):
            w = np.exp(samples[i].logWt - logZ) # Proportional weight
            x  += w * samples[i].x
            xx += w * samples[i].x * samples[i].x
            y  += w * samples[i].y
            yy += w * samples[i].y * samples[i].y
        logZ_sdev = results['logZ_sdev']
        H = results['info_nats']
        H_sdev = results['info_sdev']
        print("# iterates: %i"%ni)
        print("Evidence: ln(Z) = %g +- %g"%(logZ,logZ_sdev))
        print("Information: H  = %g nats = %g bits"%(H,H/np.log(2.0)))
        #print("mean(x) = %9.4f, stddev(x) = %9.4f"%(x, sqrt(xx-x*x)))
        print("mean(x) = {}, stddev(x) = {}".format(x, np.std(x)))
        print("mean(x) = {}, stddev(x) = {}".format(y, np.std(y)))
        #print("mean(y) = %9.4f, stddev(y) = %9.4f"%(y, sqrt(yy-y*y)))

