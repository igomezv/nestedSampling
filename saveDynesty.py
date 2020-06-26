import numpy as np

def saveDynestyChain(result, outputname):
    f = open(outputname + '.txt', 'w+')

    weights = np.exp(result['logwt'] - result['logz'][-1])

    postsamples = result.samples

    print('\n Number of posterior samples is {}'.format(postsamples.shape[0]))

    for i, sample in enumerate(postsamples):
        strweights = str(weights[i])
        strlogl = str(result['logl'][i])
        strsamples = str(sample).lstrip('[').rstrip(']')
        row = strweights + ' ' + strlogl + ' ' + strsamples  # + strOLambda
        nrow = " ".join(row.split())
        f.write(nrow + '\n')