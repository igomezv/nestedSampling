from nested import nested
from saveDynesty import saveDynestyChain
import numpy as np
import random
from anesthetic import NestedSamples

def priorTransform(theta):
    return theta

def logLike(theta):
    return -1/2/0.1**2 * ((theta-0.5)**2).sum()

s = nested(logLike, priorTransform, nlive=200, ndims=2, outputname="outputs/gaussian")
s.sampling()
samples = NestedSamples(root='outputs/gaussian')
samples.plot_2d([0,1])
