"""D11 observation-state algebra. No paths, loaders, fitting at import or network."""
import numpy as np
import scipy.linalg as la

BURN = 41
STATE_ROLES = ('CurrentCov', 'RecordPrefix', 'PersonTRAIN', 'PastCov')
CACHE_ROLES = STATE_ROLES + ('OlderCov',)
ARMS = ('Add',) + STATE_ROLES
TRIU = np.triu_indices(64, 1)


def stats_finish(sums, squares, n):
    mean = sums / n
    var = squares / n - mean * mean
    assert np.isfinite(mean).all() and np.isfinite(var).all()
    scale = np.maximum(np.maximum(1., abs(squares / n)), mean * mean)
    assert np.all(var >= -1e-10 * scale)
    constant = var <= 0
    std = np.sqrt(np.maximum(var, 0.))
    std[constant] = 1.
    return dict(mean=mean, std=std, constant=constant)


def normalize(x, st):
    y = (x - st['mean']) / st['std']
    y[..., st['constant']] = 0.
    assert np.isfinite(y).all()
    return y


def covariance(x):
    assert x.ndim == 2 and x.shape[1] == 64 and len(x) > 0
    x = np.asarray(x, np.float64)
    center = x - x.mean(0)
    return center.T @ center / len(x)


def descriptor(k, counts=None):
    """Vectorized diagonal log + upper correlation. CAR rank loss is allowed."""
    k = np.asarray(k, np.float64)
    assert k.shape[-2:] == (64, 64) and np.isfinite(k).all()
    diag = np.diagonal(k, axis1=-2, axis2=-1)
    assert np.all(diag >= 0.), 'Region must apply the declared variance roundoff rule'
    safe = np.maximum(diag, 1e-8)
    correlations = k[..., TRIU[0], TRIU[1]] / np.sqrt(
        safe[..., TRIU[0]] * safe[..., TRIU[1]])
    excess = np.maximum(abs(correlations) - 1., 0.)
    assert excess.max(initial=0.) <= 1e-10, 'Invalid correlation; no record exclusion'
    if counts is not None:
        counts['diagonal_floor'] = counts.get('diagonal_floor', 0) + int((diag < 1e-8).sum())
        counts['correlation_roundoff_clip'] = counts.get('correlation_roundoff_clip', 0) + int((excess > 0).sum())
    return np.concatenate((np.log(safe), np.clip(correlations, -1., 1.)), axis=-1)


def intervals(i):
    assert i >= BURN
    start = 96 * i
    union = start - 96
    return dict(CurrentCov=(start, start + 192), RecordPrefix=(0, 1920),
                PastCov=(union - 1920, union),
                OlderCov=(union - 3840, union - 1920))


def record_descriptors(x, windows, person_cov, counts=None):
    """96-sample block moments; only record-local array coordinates."""
    assert x.ndim == 2 and x.shape[1] == 64 and len(x) >= 96 * (windows + 1)
    assert windows > BURN and person_cov.shape == (64, 64)
    blocks = np.asarray(x[:96 * (windows + 1)], np.float64).reshape(windows + 1, 96, 64)
    sums = blocks.sum(1)
    moments = np.einsum('btc,btd->bcd', blocks, blocks, optimize=True)
    sums = np.concatenate((np.zeros((1, 64)), sums.cumsum(0)), axis=0)
    moments = np.concatenate((np.zeros((1, 64, 64)), moments.cumsum(0)), axis=0)
    def region(first, last):
        n = (last - first) * 96
        mean = (sums[last] - sums[first]) / n[..., None] if np.ndim(n) else (sums[last] - sums[first]) / n
        q = (moments[last] - moments[first]) / (n[..., None, None] if np.ndim(n) else n)
        k = q - mean[..., :, None] * mean[..., None, :]
        diag = np.diagonal(k, axis1=-2, axis2=-1).copy()
        uncentered = np.diagonal(q, axis1=-2, axis2=-1)
        scale = np.maximum(np.maximum(1., abs(uncentered)), mean * mean)
        assert np.all(diag >= -1e-10 * scale), 'Negative variance beyond declared tolerance'
        if counts is not None:
            counts['negative_variance_roundoff'] = counts.get('negative_variance_roundoff', 0) + int((diag < 0).sum())
        ids = np.arange(64)
        k[..., ids, ids] = np.maximum(diag, 0.)
        return k
    i = np.arange(BURN, windows)
    current = descriptor(region(i, i + 2), counts)
    past = descriptor(region(i - 21, i - 1), counts)
    older = descriptor(region(i - 41, i - 21), counts)
    prefix = descriptor(region(0, 20), counts)
    person = descriptor(person_cov, counts)
    assert current.shape == past.shape == older.shape == (windows - BURN, 2080)
    return dict(CurrentCov=current, PastCov=past, OlderCov=older,
                RecordPrefix=prefix, PersonTRAIN=person)


def pooled_descriptor_moments(parts, n):
    """Equal roles/windows, including static descriptor multiplicity."""
    sums = np.zeros(2080)
    squares = np.zeros(2080)
    gram = np.zeros((2080, 2080))
    for role in STATE_ROLES:
        x = parts[role]
        if x.ndim == 1:
            sums += n * x
            squares += n * x * x
            gram += n * np.outer(x, x)
        else:
            assert len(x) == n
            sums += x.sum(0)
            squares += (x * x).sum(0)
            gram += x.T @ x
    return sums, squares, gram, n * len(STATE_ROLES)


def pca_from_moments(sums, squares, gram, n, dimensions):
    st = stats_finish(sums, squares, n)
    cov = gram / n - np.outer(st['mean'], st['mean'])
    cov = cov / st['std'][:, None] / st['std'][None, :]
    cov[st['constant'], :] = 0.
    cov[:, st['constant']] = 0.
    return st, principal(cov, dimensions)


def principal(cov, dimensions):
    size = len(cov)
    values, vectors = la.eigh((cov + cov.T) * .5, subset_by_index=(size - dimensions, size - 1), driver='evr')
    values, vectors = values[::-1], vectors[:, ::-1]
    vectors *= np.where(vectors[np.argmax(abs(vectors), axis=0), np.arange(dimensions)] < 0, -1., 1.)[None]
    assert np.isfinite(values).all() and np.isfinite(vectors).all()
    basis = np.zeros_like(vectors)
    good = values > 0
    basis[:, good] = vectors[:, good] / np.sqrt(values[good])
    return dict(basis=basis, eigenvalues=values, positive_components=int(good.sum()))


def project(x, st, pca):
    return normalize(x, st) @ pca['basis']


def features(q, state, product_stats=None):
    if state is None:
        return q / np.sqrt(32)
    assert q.shape == (len(state), 32) and state.shape == (len(q), 8)
    products = (q[:, :, None] * state[:, None, :]).reshape(len(q), 256)
    if product_stats is not None:
        products = normalize(products, product_stats)
    return np.concatenate((q / np.sqrt(32), state / np.sqrt(8), products / 16.), 1)


def ridge(G, H, xm, rm, lam=.01):
    values, vectors = la.eigh((G + G.T) * .5)
    assert values.min() >= -1e-8
    coef = vectors @ ((vectors.T @ H) / (values[:, None] + lam))
    bias = rm - xm @ coef
    error = float(abs((G + lam * np.eye(len(G))) @ coef - H).max())
    assert error <= 1e-7 * max(1., float(abs(H).max()))
    assert np.isfinite(coef).all() and np.isfinite(bias).all()
    return coef, bias, dict(normal_equation_max_abs=error,
                           effective_df=float(np.sum(values / (values + lam))))


def synthetic_check():
    rng = np.random.default_rng(81307)
    x = rng.normal(size=(96 * 51, 64))
    x -= x.mean(1, keepdims=True)  # CAR-singular covariance is a legal input.
    person = covariance(x[:1920])
    parts = record_descriptors(x, 50, person)
    differences = []
    for i in (41, 42, 49):
        for role, (a, b) in intervals(i).items():
            expected = descriptor(covariance(x[a:b]))
            actual = parts[role] if parts[role].ndim == 1 else parts[role][i - BURN]
            differences.append(float(abs(expected - actual).max()))
        assert intervals(i)['PastCov'][1] <= 96 * (i - 1)
        assert intervals(i)['OlderCov'][1] <= intervals(i)['PastCov'][0]
    assert max(differences) < 1e-12
    xx = rng.normal(size=(71, 11)); yy = rng.normal(size=(71, 7))
    xm, ym = xx.mean(0), yy.mean(0)
    xc, yc = xx - xm, yy - ym
    coef, bias, _ = ridge(xc.T @ xc / 71, xc.T @ yc / 71, xm, ym)
    aug = np.column_stack((xx, np.ones(71)))
    penalty = np.column_stack((np.sqrt(71 * .01) * np.eye(11), np.zeros(11)))
    direct = la.lstsq(np.row_stack((aug, penalty)), np.row_stack((yy, np.zeros((11, 7)))))[0]
    error = float(abs(np.row_stack((coef, bias)) - direct).max())
    assert error < 1e-12
    return dict(passed=True, rolling_direct_descriptor_max_abs=max(differences),
                ridge_augmented_lstsq_max_abs=error, singular_CAR_legal=True,
                strict_past_support=True, synthetic_only=True)
