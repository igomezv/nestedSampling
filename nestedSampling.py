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
        logz = -1e300 # not -np.inf

        for _ in range(self.nlive):
            livesamples.append(self.sample_from_prior())

        # outermost interval of prior mass
        logw = np.log(1.0 - np.exp(-1.0 / self.nlive))

        # begin nested sampling loop
        for nest in range(self.maxiter):
            # Worst object
            worst = 0
            for i in range(1,self.nlive):
                if livesamples[i].logL < livesamples[worst].logL:
                    worst = i

            livesamples[worst].logWt = logw + livesamples[worst].logL

            # Update Evidence Z and Information h
            logznew = logsumexp([logz, livesamples[worst].logWt])

            h = np.exp(livesamples[worst].logWt - logznew) * livesamples[worst].logL + \
                np.exp(logz - logznew) * (h + logz) - logznew
            logz = logznew

            # Posterior samples (optional)
            samples.append(livesamples[worst])

            # Kill worst object in favour of copy of different survivor
            if self.nlive > 1:
                while True:
                    copy = np.random.randint(0, self.nlive)
                    if copy != worst:
                        break

            logLstar = livesamples[worst].logL       # new likelihood constraint
            livesamples[worst] = livesamples[copy]          # overwrite worst object

            # Evolve copied object within constraint
            new_sample = self.explore(livesamples[worst], logLstar)
            assert(new_sample != None) # Make sure explore didn't update in-place
            livesamples[worst] = new_sample

            # Shrink interval # very important!
            logw -= 1.0 / self.nlive

        # Exit with evidence Z, information h, and optional posterior samples
        sdev_h = h/np.log(2.)
        sdev_logz = np.sqrt(h/self.nlive)
        results = {"samples": samples, "n iterations": (nest+1), "logz": logz,
                   "logz_sdev": sdev_logz, "info": h, "info_sdev": sdev_h}

        print("logz:{} +/- {} | h:{} +/- {}".format(logz, sdev_logz, h, sdev_h))
        self.postprocess(samples)
        return results

    def sample_from_prior(self):
        Obj = Object()
        Obj.points = np.random.rand(self.ndims)
        Obj.phys_points = self.priorTransform(Obj.points)
        Obj.logL = self.logLike(Obj.phys_points)
        return Obj

    def explore(self, current_sample, logLstar):
        # need of an auxiliary object!
        aux = Object()
        aux.__dict__ = current_sample.__dict__.copy()
        step = 0.1
        accept = 0
        reject = 0
        propossal = Object()
        for _ in range(50):
            propossal.points = current_sample.points + step * (2.*np.random.random(self.ndims) - 1.)
            propossal.points -= np.floor(propossal.points)
            propossal.phys_points = self.priorTransform(propossal.points)
            propossal.logL = self.logLike(propossal.phys_points)
            # Hard likelihood constraint
            if propossal.logL > logLstar:
                aux.__dict__ = propossal.__dict__.copy()
                # print(propossal == aux)
                accept += 1
            else:
                reject += 1
            # Refine step-size
            if accept > reject:
                step *= np.exp(1.0 / accept)
            elif accept < reject:
                step /= np.exp(1.0 / reject)

        return propossal

    def postprocess(self, samples):
        posterior = []
        weights = []
        loglikes = []
        for sample in samples:
            posterior.append(sample.phys_points)
            weights.append(sample.logWt)
            loglikes.append(sample.logL)
        normws = logsumexp(weights)
        f = open("output.txt", "+w")
        for i, point in enumerate(posterior):
            undesirables = '[ ]'
            strpoint = "{}".format(point).strip(undesirables)
            strpoint = "{}".format(strpoint).replace(',','')
            weightnorm = np.exp(weights[i] - normws)
            strow = "{} {} {}\n".format(weightnorm, loglikes[i], strpoint)
            f.write(strow)
        f.close()
        posterior = np.array(posterior)
        m, n = np.shape(posterior)
        mean = np.mean(posterior, axis=0)
        std = np.std(posterior, axis=0)
        for i in range(n):
            print("{} parameter: {} +/- {}".format(i+1, mean[i], std[i]))


