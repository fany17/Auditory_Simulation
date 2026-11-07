"""Independent D18 Ccap algebra/VJP reference; execute only on server2203.

No production imports, model construction, fitting, file access or automatic run.
Caller supplies original TRAIN transformations and frozen D8 coefficients.
These functions cover eligible Ccap windows; record/prefix policy is separate.
This is a reference implementation, not an executed validation receipt.
"""

import numpy as np


def _columns(stat, width):
    mean = np.asarray(stat['mean'], dtype=np.float64).reshape(1, -1)
    std = np.asarray(stat['std'], dtype=np.float64).reshape(1, -1)
    constant = np.asarray(stat['constant'], dtype=bool).reshape(1, -1)
    assert mean.shape == std.shape == constant.shape == (1, width)
    assert np.isfinite(mean).all() and np.isfinite(std).all()
    assert np.all(std > 0)
    inverse = np.where(constant, 0.0, 1.0 / std)
    return mean, std, constant, inverse


def _normalized(value, stat):
    mean, std, constant, inverse = _columns(stat, value.shape[1])
    # Preserve original subtract/divide then explicit constant-column zeroing.
    result = (value - mean) / std
    result[:, constant[0]] = 0.0
    return result, inverse


def ccap_value_and_vjp(latent, head_prediction, statistics, coefficients,
                       intercept, calibration, output_cotangent=None):
    """Return float64 Ccap output and optional analytic input cotangents.

    latent: (B,8,192), with the original channel-major flattening convention.
    head_prediction: (B,192), uncalibrated original neural-head prediction.
    statistics: three dictionaries for raw Z, calibrated P, normalized-Z squared.
    output_cotangent: arbitrary (B,192), not derived from test/VAL targets.
    Returned latent cotangent is the *direct* path through the Ccap features.
    The separate head cotangent must also backpropagate through the actual head;
    this function does not claim to replay the neural head or encoder.
    """
    raw = np.asarray(latent, dtype=np.float64)
    head = np.asarray(head_prediction, dtype=np.float64)
    assert raw.ndim == 3 and raw.shape[1:] == (8, 192)
    assert head.shape == (len(raw), 192) and len(statistics) == 3
    weight = np.asarray(coefficients, dtype=np.float64)
    bias = np.asarray(intercept, dtype=np.float64).reshape(1, -1)
    assert weight.shape == (3264, 192) and bias.shape == (1, 192)
    assert all(np.isfinite(x).all() for x in (raw, head, weight, bias))
    scale = float(calibration['a'])
    shift = float(calibration['b'])
    assert np.isfinite(scale) and np.isfinite(shift)

    u, inv_z = _normalized(raw.reshape(len(raw), 1536), statistics[0])
    baseline = scale * head + shift
    v, inv_p = _normalized(baseline, statistics[1])
    w, inv_square = _normalized(u * u, statistics[2])
    features = np.concatenate((u / np.sqrt(1536.0),
                               v / np.sqrt(192.0),
                               w / np.sqrt(1536.0)), axis=1)
    prediction = baseline + (features @ weight + bias)
    assert np.isfinite(prediction).all()
    if output_cotangent is None:
        return prediction, None

    cotangent = np.asarray(output_cotangent, dtype=np.float64)
    assert cotangent.shape == prediction.shape and np.isfinite(cotangent).all()
    # Three independent feature blocks, then the chain rule through u^2.
    du_linear = (cotangent @ weight[:1536].T) / np.sqrt(1536.0)
    du_square = ((cotangent @ weight[1728:].T) / np.sqrt(1536.0)
                 * inv_square * (2.0 * u))
    latent_cotangent = ((du_linear + du_square) * inv_z).reshape(raw.shape)
    head_cotangent = scale * (cotangent
                             + (cotangent @ weight[1536:1728].T)
                             / np.sqrt(192.0) * inv_p)
    assert np.isfinite(latent_cotangent).all() and np.isfinite(head_cotangent).all()
    return prediction, dict(latent=latent_cotangent, head=head_cotangent)


def central_directional_difference(latent, head_prediction, latent_direction,
                                   head_direction, statistics, coefficients,
                                   intercept, calibration, output_cotangent,
                                   step=1e-5):
    """Caller-bounded synthetic/TRAIN directional check, no new fitting.

    This explicit finite difference is optional and counted separately by the
    caller. The analytic VJP is the independent check against production
    autograd; this check merely verifies the reference's own chain rule.
    """
    assert np.isfinite(step) and step > 0
    z = np.asarray(latent, dtype=np.float64)
    p = np.asarray(head_prediction, dtype=np.float64)
    dz = np.asarray(latent_direction, dtype=np.float64)
    dp = np.asarray(head_direction, dtype=np.float64)
    assert dz.shape == z.shape and dp.shape == p.shape
    common = (statistics, coefficients, intercept, calibration)
    plus, _ = ccap_value_and_vjp(z + step * dz, p + step * dp, *common)
    minus, _ = ccap_value_and_vjp(z - step * dz, p - step * dp, *common)
    _, vjp = ccap_value_and_vjp(z, p, *common,
                               output_cotangent=output_cotangent)
    finite_difference = float(np.sum((plus - minus) * output_cotangent)
                              / (2.0 * step))
    analytic = float(np.sum(vjp['latent'] * dz) + np.sum(vjp['head'] * dp))
    return dict(analytic=analytic, central_difference=finite_difference,
                absolute_difference=abs(analytic - finite_difference), step=step)
