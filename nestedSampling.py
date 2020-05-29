from scipy.special import logsumexp
import numpy as np

class Object:
    def __init__(self):
        self.points = None
        self.phys_points = None
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
        livesamples = []
        samples = []
        h    = 0.0
        logz =-1e300

        for _ in range(self.nlive):
            livesamples.append(self.sample_from_prior())

        # Outermost interval of prior mass
        logwidth = np.log(1.0 - np.exp(-1.0 / self.nlive))

        # begin nested sampling loop
        for nest in range(self.maxiter):
            # Worst object
            worst = 0
            for i in range(1,self.nlive):
                if livesamples[i].logL < livesamples[worst].logL:
                    worst = i

            livesamples[worst].logWt = logwidth + livesamples[worst].logL

            # Update Evidence Z and Information h
            logznew = logsumexp([logz, livesamples[worst].logWt])

            h = np.exp(livesamples[worst].logWt - logznew) * livesamples[worst].logL + \
                np.exp(logz - logznew) * (h + logz) - logznew
            logz = logznew

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
            new_sample = self.explore(livesamples[worst], logLstar)
            assert(new_sample != None) # Make sure explore didn't update in-place
            livesamples[worst] = new_sample

            # Shrink interval
            logwidth -= 1.0 / self.nlive

        # Exit with evidence Z, information h, and optional posterior samples
        sdev_h = h/np.log(2.)
        sdev_logz = np.sqrt(h/self.nlive)
        results = {"samples":samples, "num_iterations":(nest+1), "logz":logz,
                "logz_sdev":sdev_logz, "info_nats":h, "info_sdev":sdev_h}

        print("logz:{} h:{}".format(logz, h))

        return results

    def sample_from_prior(self):
        Obj = Object()
        Obj.points = np.random.rand(self.ndims)
        Obj.phys_points = self.priorTransform(Obj.points)
        Obj.logL = self.logLike(Obj.phys_points)
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
            Try.points = ret.points + step * (2.*np.random.random(self.ndims) - 1.)  # |move| < step)
            Try.points -= np.floor(Try.points)      # wraparound to stay within (0,1)
            #Try.v -= np.floor(Try.v)      # wraparound to stay within (0,1)
            Try.phys_points = self.priorTransform(Try.points)
            Try.logL = self.logLike(Try.phys_points)  # trial likelihood value

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
