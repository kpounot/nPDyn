"""Basic analytical convolutions between preset functions."""

import re

import numpy as np

from scipy.special import spherical_jn

from lmfit.lineshapes import (gaussian, lorentzian, pvoigt, voigt)


def getGlobals(params):
    """Helper function to get the global parameters."""

    pGlob = []
    for key in params.keys():
        if not re.match('(\w*_*)\w+_\d+', key):
            pGlob.append(key)

    return pGlob

def conv_lorentzian_lorentzian(left, right, params, **kwargs):
    """Convolution between two Lorentzians.

    .. math::

        a_1 \mathcal{L}_{\sigma_1, center_1}
        \otimes a_2 \mathcal{L}_{sigma_2, center_2} =
            a_1 a_2 . \mathcal{L}_{\sigma_1 + \sigma_2, center_1 + center_2}

    """
    lp = left.make_funcargs(params, kwargs)
    rp = right.make_funcargs(params, kwargs)

    x = lp['x']
    q = lp['q']

    out = []

    lglob = getGlobals(lp)
    rglob = getGlobals(rp)

    for qId, qVal in enumerate(q):
        amplitude = (lp['amplitude'] if 'amplitude' in lglob 
                     else lp['amplitude_%i' % qId])
        amplitude *= (rp['amplitude'] if 'amplitude' in rglob 
                     else rp['amplitude_%i' % qId])
        center = lp['center_%i' % qId] + rp['center_%i' % qId]
        lsigma = (lp['sigma'] * qVal**2 if 'sigma' in lglob 
                  else lp['sigma_%i' % qId])
        rsigma = (rp['sigma'] * qVal**2 if 'sigma' in rglob 
                  else lp['sigma_%i' % qId])
        
        out.append(lorentzian(x, amplitude, center, lsigma + rsigma))

    return np.array(out)

def conv_gaussian_lorentzian(left, right, params, **kwargs):
    """Convolution of a Gaussian and a Lorentzian.

    Results in a Voigt profile as defined in lineshapes.

    """
    # set left to Gaussian component if not so
    if left.func.__name__ == 'lorentzian':
        tmp = right
        right = left
        left = tmp

    lp = left.make_funcargs(params, kwargs)
    rp = right.make_funcargs(params, kwargs)

    x = lp['x']
    q = lp['q']

    out = []

    lglob = getGlobals(lp)
    rglob = getGlobals(rp)

    for qId, qVal in enumerate(q):
        amplitude = (lp['amplitude'] if 'amplitude' in lglob 
                     else lp['amplitude_%i' % qId])
        amplitude *= (rp['amplitude'] if 'amplitude' in rglob 
                     else rp['amplitude_%i' % qId])
        center = lp['center_%i' % qId] + rp['center_%i' % qId]
        sigma = (lp['sigma'] * qVal**2 if 'sigma' in lglob 
                 else lp['sigma_%i' % qId])
        gamma = (rp['sigma'] * qVal**2 if 'sigma' in rglob 
                 else lp['sigma_%i' % qId])
        
        out.append(voigt(x, amplitude, center, sigma, gamma))

    return np.array(out)

def conv_gaussian_jumpdiff(left, right, params, **kwargs):
    """Convolution of a Gaussian and a jump-diffusion Lorentzian.

    Results in a Voigt profile as defined in lineshapes.

    """
    # set left to Gaussian component if not so
    if left.func.__name__ == 'lorentzian':
        tmp = right
        right = left
        left = tmp

    lp = left.make_funcargs(params, kwargs)
    rp = right.make_funcargs(params, kwargs)

    x = lp['x']
    q = lp['q']

    out = []

    lglob = getGlobals(lp)
    rglob = getGlobals(rp)

    for qId, qVal in enumerate(q):
        amplitude = (lp['amplitude'] if 'amplitude' in lglob 
                     else lp['amplitude_%i' % qId])
        amplitude *= (rp['amplitude'] if 'amplitude' in rglob 
                     else rp['amplitude_%i' % qId])
        center = lp['center_%i' % qId] + rp['center_%i' % qId]
        sigma = (lp['sigma'] * qVal**2 if 'sigma' in lglob 
                 else lp['sigma_%i' % qId])
        gamma = (
            rp['sigma'] * qVal**2 / (1 + rp['sigma'] * qVal**2 * rp['tau']) 
            if 'sigma' in rglob else lp['sigma_%i' % qId])
        
        out.append(voigt(x, amplitude, center, sigma, gamma))

    return np.array(out)

def conv_gaussian_rotations(left, right, params, **kwargs):
    """Convolution of a Gaussian and a liquid rotations model.

    """
    # set left to Gaussian component if not so
    if left.func.__name__ == 'rotations':
        tmp = right
        right = left
        left = tmp

    lp = left.make_funcargs(params, kwargs)
    rp = right.make_funcargs(params, kwargs)

    x = lp['x']
    q = lp['q']

    out = []

    lglob = getGlobals(lp)
    rglob = getGlobals(rp)

    for qId, qVal in enumerate(q):
        amplitude = (lp['amplitude'] if 'amplitude' in lglob 
                     else lp['amplitude_%i' % qId])
        amplitude *= (rp['amplitude'] if 'amplitude' in rglob 
                     else rp['amplitude_%i' % qId])
        center = lp['center_%i' % qId] + rp['center_%i' % qId]
        lsigma = (lp['sigma'] * qVal**2 if 'sigma' in lglob 
                 else lp['sigma_%i' % qId])
        bondDist = rp['bondDist']
        rsigma = (rp['sigma'] if 'sigma' in rglob else lp['sigma_%i' % qId])

        eisf = spherical_jn(0, bondDist * qVal)**2 
        eisf *= gaussian(x, amplitude, center, rsigma)
        qisf = np.sum([
            spherical_jn(l, bondDist * qVal)**2 * (2 * l + 1) *
            voigt(x, amplitude, center, lsigma, l * (l + 1) * rsigma)
            for l in range(1, 5)], axis=0)
        
        out.append(eisf + qisf)

    return np.array(out)

def conv_gaussian_gaussian(left, right, params, **kwargs):
    """Convolution between two Gaussians.

    .. math::

        a_1 G_{\sigma_1, center_1} \otimes a_2 G_{\sigma_2, center_2} =
            a_1 a_2 . G_{\sigma_1 + \sigma2, center_1 + center_2}

    """
    lp = left.make_funcargs(params, kwargs)
    rp = right.make_funcargs(params, kwargs)

    x = lp['x']
    q = lp['q']

    out = []

    lglob = getGlobals(lp)
    rglob = getGlobals(rp)

    for qId, qVal in enumerate(q):
        amplitude = (lp['amplitude'] if 'amplitude' in lglob 
                     else lp['amplitude_%i' % qId])
        amplitude *= (rp['amplitude'] if 'amplitude' in rglob 
                     else rp['amplitude_%i' % qId])
        center = lp['center_%i' % qId] + rp['center_%i' % qId]
        lsigma = (lp['sigma'] * qVal**2 if 'sigma' in lglob 
                  else lp['sigma_%i' % qId])
        rsigma = (rp['sigma'] * qVal**2 if 'sigma' in rglob 
                  else lp['sigma_%i' % qId])

        convGauss = amplitude / np.sqrt(2 * np.pi * (lsigma**2 + rsigma**2))
        convGauss *= np.exp(-(x - center)**2 / (2 * lsigma**2 + rsigma**2))

        out.append(convGauss)

    return np.array(out)

def conv_lorentzian_pvoigt(left, right, params, **kwargs):
    """Convolution between a Lorentzian and a pseudo-Voigt profile.

    .. math::

        \\begin{aligned}
        & a_L \\mathcal{L}_{\\sigma_L, center_L} \\otimes
        a_V . p\\mathcal{V}_{\\sigma, center, fraction} =
            
        &\quad a_L a_V \\left[ (1 - fraction)
                \\mathcal{V}_{\\sigma_g, \\sigma, center + center_L}
                + fraction
                    \\mathcal{L}_{\\sigma + \\sigma_L, center + center_L}
            \\right]
        \end{aligned}

    where :math:`p\\mathcal{V}` is the pseudo-Voigt, :math:`\\mathcal{V}` is
    a Voigt profile, :math:`\\sigma_g = \\frac{\\sigma}{\\sqrt{(2 log(2))}}` and
    :math:`\\mathcal{L}` is a Lorentzian.

    """
    # set left to lorentzian component if not so
    if left.func.__name__ == 'pvoigt':
        tmp = right
        right = left
        left = tmp

    lp = left.make_funcargs(params, kwargs)
    rp = right.make_funcargs(params, kwargs)

    x = lp['x']
    q = lp['q']

    out = []

    lglob = getGlobals(lp)
    rglob = getGlobals(rp)

    for qId, qVal in enumerate(q):
        amplitude = (lp['amplitude'] if 'amplitude' in lglob 
                     else lp['amplitude_%i' % qId])
        amplitude *= (rp['amplitude'] if 'amplitude' in rglob 
                     else rp['amplitude_%i' % qId])
        center = lp['center_%i' % qId] + rp['center_%i' % qId]
        rsigma = rp['sigma_%i' % qId]
        sigma_v = rsigma / np.sqrt(2 * np.log(2))
        frac = rp['fraction_%i' % qId]
        lsigma = (lp['sigma'] * qVal**2 if 'sigma' in lglob 
                  else lp['sigma_%i' % qId])
        
        out.append(
            frac * voigt(x, amplitude, center, sigma_v, lsigma) 
            + (1 - frac) * lorentzian(x, amplitude, center, lsigma + rsigma))

    return np.array(out)

def conv_gaussian_pvoigt(left, right, params, **kwargs):
    """Convolution between a Gaussian and a pseudo-Voigt profile.

    .. math::

        \\begin{aligned}
        & a_L G_{\\sigma_G, center_G} \\otimes
        a_V . \\mathcal{pV}_{\\sigma, center, fraction} = 

        & \quad a_G a_V \\left[fraction
                \\mathcal{V}_{\\sigma_G, \\sigma, center + center_G} 
            + (1 - fraction) G_{\\sigma_g + \\sigma_G, center + center_G}
            \\right]
        \\end{aligned}

    where :math:`\\mathcal{pV}` is the pseudo-Voigt, :math:`\\mathcal{V}` is
    a Voigt profile, and 
    :math:`\\sigma_g = \\frac{\\sigma}{\\sqrt{(2 log(2))}}`.

    """
    # set left to gaussian component if not so
    if left.func.__name__ == 'pvoigt':
        tmp = right
        right = left
        left = tmp

    lp = left.make_funcargs(params, kwargs)
    rp = right.make_funcargs(params, kwargs)

    x = lp['x']
    q = lp['q']

    out = []

    lglob = getGlobals(lp)
    rglob = getGlobals(rp)

    for qId, qVal in enumerate(q):
        amplitude = (lp['amplitude'] if 'amplitude' in lglob 
                     else lp['amplitude_%i' % qId])
        amplitude *= (rp['amplitude'] if 'amplitude' in rglob 
                     else rp['amplitude_%i' % qId])
        center = lp['center_%i' % qId] + rp['center_%i' % qId]
        rsigma = rp['sigma_%i' % qId]
        frac = rp['fraction_%i' % qId]
        lsigma = (lp['sigma'] * qVal**2 if 'sigma' in lglob 
                  else lp['sigma_%i' % qId])
        sigma_g = rsigma / np.sqrt(2 * np.log(2)) 
        
        convGauss = amplitude / np.sqrt(2 * np.pi * (sigma_g**2 + lsigma**2))
        convGauss *= np.exp(-(x - center)**2 / (2 * sigma_g**2 + lsigma**2))

        out.append(
            frac * convGauss 
            + (1 - frac) * voigt(x, amplitude, center, lsigma, rsigma))

    return np.array(out)

def conv_jumpdiff_pvoigt(left, right, params, **kwargs):
    """Convolution between the jump diffusion model and a pseudo-Voigt profile.

    """
    # set left to jump_diff component if not so
    if left.func.__name__ == 'pvoigt':
        tmp = right
        right = left
        left = tmp

    lp = left.make_funcargs(params, kwargs)
    rp = right.make_funcargs(params, kwargs)

    x = lp['x']
    q = lp['q']

    out = []

    lglob = getGlobals(lp)
    rglob = getGlobals(rp)

    for qId, qVal in enumerate(q):
        amplitude = (lp['amplitude'] if 'amplitude' in lglob 
                     else lp['amplitude_%i' % qId])
        amplitude *= (rp['amplitude'] if 'amplitude' in rglob 
                     else rp['amplitude_%i' % qId])
        center = lp['center_%i' % qId] + rp['center_%i' % qId]
        frac = rp['fraction_%i' % qId]
        lsigma = (
            lp['sigma'] * qVal**2 / (1 + lp['sigma'] * qVal**2 * lp['tau']) 
            if 'sigma' in lglob else lp['sigma_%i' % qId])
        rsigma = rp['sigma_%i' % qId]
        sigma_v = rsigma / np.sqrt(2 * np.log(2))

        out.append(
            frac * voigt(x, amplitude, center, sigma_v, lsigma) 
            + (1 - frac) * lorentzian(x, amplitude, center, lsigma + rsigma))

    return np.array(out)

def conv_rotations_pvoigt(left, right, params, **kwargs):
    """Convolution between the rotation model and a pseudo-Voigt profile.

    """
    # set left to rotations component if not so
    if left.func.__name__ == 'pvoigt':
        tmp = right
        right = left
        left = tmp

    lp = left.make_funcargs(params, kwargs)
    rp = right.make_funcargs(params, kwargs)

    x = lp['x']
    q = lp['q']

    out = []

    lglob = getGlobals(lp)
    rglob = getGlobals(rp)

    for qId, qVal in enumerate(q):
        amplitude = (lp['amplitude'] if 'amplitude' in lglob 
                     else lp['amplitude_%i' % qId])
        amplitude *= (rp['amplitude'] if 'amplitude' in rglob 
                     else rp['amplitude_%i' % qId])
        center = lp['center_%i' % qId] + rp['center_%i' % qId]
        frac = rp['fraction_%i' % qId]
        lsigma = (lp['sigma'] if 'sigma' in lglob else lp['sigma_%i' % qId])
        rsigma = rp['sigma_%i' % qId]
        sigma_v = rsigma / np.sqrt(2 * np.log(2))

        eisf = spherical_jn(0, bondDist * qVal)**2 
        eisf *= pvoigt(x, amplitude, center, rsigma, frac)
        qisf = np.sum([
            spherical_jn(l, bondDist * qVal)**2 * (2 * l + 1) *
            (frac * 
            voigt(x, amplitude, center, sigma_v, l * (l + 1) * lsigma) *
            (1 - frac) *
            lorentzian(amplitude, center, rsigma + l * (l + 1) * lsigma))
            for l in range(1, 5)], axis=0)

        out.append(eisf + qisf)

    return np.array(out)

def conv_delta(left, right, params, **kwargs):
    """Convolution with a Dirac delta.

    """
    # set left to delta component if not so
    if left.func.__name__ != 'delta':
        tmp = right
        right = left
        left = tmp

    lp = left.make_funcargs(params, kwargs)
    rp = right.make_funcargs(params, kwargs)

    x = lp['x']
    q = lp['q']

    out = []

    lglob = getGlobals(lp)
    rglob = getGlobals(rp)

    for qId, qVal in enumerate(q):
        amplitude = (lp['amplitude'] if 'amplitude' in lglob 
                     else lp['amplitude_%i' % qId])
        amplitude *= (rp['amplitude'] if 'amplitude' in rglob 
                     else rp['amplitude_%i' % qId])
        center = lp['center_%i' % qId] 

        tmp_kws = {'amplitude_%i' % qId: amplitude, 
                   'center_%i' % qId: center}
        
        out.append(right.eval(x=x, q=q, **tmp_kws)[qId])

    return np.array(out)

def conv_linear(left, right, params, **kwargs):
    """Convolution with a linear model.

    Simply returns the linear model itself as it is assumed to
    serve as a background term by default.

    """
    if left.func.__name__ != 'linear':
        tmp = right
        right = left
        left = tmp

    return left.eval(params=params, **kwargs)
