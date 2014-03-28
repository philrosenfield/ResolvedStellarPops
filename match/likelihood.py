""" Likelihood used in MATCH """
import numpy as np


def stellar_prob(obs, model, normalize=False):
    '''
    FROM MATCH README
    The quality of the fit is calculated using a Poisson maximum likelihood
    statistic, based on the Poisson equivalent of chi^2.
      2 m                                if (n=0)
      2 [ 0.001 + n * ln(n/0.001) - n ]  if (m<0.001)
      2 [ m + n * ln(n/m) - n ]          otherwise
    m=number of model points; n=number of observed points

    This statistic is based on the Poisson probability function:
       P =  (e ** -m) (m ** n) / (n!),
    Recalling that chi^2 is defined as -2lnP for a Gaussian distribution and
    equals zero where m=n, we treat the Poisson probability in the same
    manner to get the above formula.

    '''
    n = obs
    m = model

    if normalize is True:
        n /= np.sum(n)
        m /= np.sum(m)

    d = 2. * (m + n * np.log(n / m) - n)

    smalln = np.abs(n) < 1e-10
    d[smalln] = 2. * m[smalln]

    smallm = (m < 0.001) & (n != 0)
    d[smallm] = 2. * (0.001 + n[smallm] * np.log(n[smallm] / 0.001) - n[smallm])

    sig = np.sqrt(d) * np.sign(n - m)
    pct_dif = (m - n) / n
    prob = np.sum(d) / float(len(n) - 1)
    return prob, pct_dif, sig
