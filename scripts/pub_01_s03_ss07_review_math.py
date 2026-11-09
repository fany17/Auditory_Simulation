"""Independent post-evaluation algebra; scientific execution on2203 only.

No production metric/resampling imports. Frequency matrices implement the
same prespecified crossed bootstrap; quantiles use explicit sorted linear
interpolation. This is not a different inferential protocol.
"""
import numpy as np


def pearson(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    assert x.ndim == y.ndim == 1 and x.shape == y.shape and len(x) > 1
    assert np.isfinite(x).all() and np.isfinite(y).all()
    assert x.max() != x.min() and y.max() != y.min(), 'exact constant: undefined correlation'
    a = x - x.sum() / len(x)
    b = y - y.sum() / len(y)
    denominator = np.sqrt(np.dot(a, a) * np.dot(b, b))
    assert denominator > 0, 'undefined correlation cannot be excluded'
    return float(np.clip(np.dot(a, b) / denominator, -1., 1.))


def signal_metrics(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    error = x - y
    centered = y - y.sum() / len(y)
    return {'r': pearson(x, y), 'MSE_descriptive': float(np.dot(error, error) / len(y)),
            'R2_descriptive': float(1 - np.dot(error, error) / np.dot(centered, centered))}


def support(length):
    assert length >= 192
    weights = np.zeros(length, dtype=np.float64)
    # Explicit Hann formula, rather than calling production fusion/np.hanning.
    hann = .5 - .5 * np.cos(2 * np.pi * np.arange(192) / 191)
    for start in range(0, length - 191, 96):
        weights[start:start + 192] += hann
    return weights > 1e-8


def linear_quantiles(values, probabilities):
    ordered = np.sort(np.asarray(values, dtype=np.float64))
    assert ordered.ndim == 1 and np.isfinite(ordered).all()
    locations = np.asarray(probabilities) * (len(ordered) - 1)
    low = np.floor(locations).astype(int)
    high = np.ceil(locations).astype(int)
    return ordered[low] + (locations - low) * (ordered[high] - ordered[low])


def frequencies(indices, size):
    out = np.zeros((len(indices), size), dtype=np.float64)
    for row, choices in enumerate(indices):
        out[row] = np.bincount(choices, minlength=size) / len(choices)
    return out


def population(values, metadata):
    """Block-to-person averaging as a sparse-in-spirit explicit matrix."""
    assert values.ndim == 3 and values.shape[-1] == len(metadata)
    assert np.isfinite(values).all()
    keys = [(b['record'], b['start']) for b in metadata]
    assert len(set(keys)) == len(keys)
    people = []
    groups = []
    for phase in ['test_seen', 'test_heldout']:
        group = []
        for subject in sorted({b['subject'] for b in metadata if b['phase'] == phase}):
            group.append(len(people)); people.append((phase, subject))
        assert group, 'missing stratum: original weights cannot be renormalized'
        groups.append(group)
    assert len({subject for phase, subject in people}) == len(people)
    averaging = np.zeros((len(people), len(metadata)))
    for p, (phase, subject) in enumerate(people):
        selected = [i for i, b in enumerate(metadata) if (b['phase'], b['subject']) == (phase, subject)]
        averaging[p, selected] = 1 / len(selected)
    assert np.allclose(averaging.sum(0) > 0, True)
    return np.einsum('ksb,pb->ksp', values, averaging, optimize=True), groups, averaging


def crossed(values, metadata, draws=20000):
    person, groups, _ = population(values, metadata)
    k, s, _ = person.shape
    point = np.empty((k, s, 3))
    point[:, :, 1] = person[:, :, groups[0]].sum(2) / len(groups[0])
    point[:, :, 2] = person[:, :, groups[1]].sum(2) / len(groups[1])
    point[:, :, 0] = (2 * point[:, :, 1] + point[:, :, 2]) / 3
    samples = np.empty((k, draws, 3)); rng = np.random.default_rng(20261101)
    for first in range(0, draws, 100):
        count = min(100, draws - first)
        sw = frequencies(rng.integers(s, size=(count, s)), s)
        aw = frequencies(rng.integers(len(groups[0]), size=(count, len(groups[0]))), len(groups[0]))
        bw = frequencies(rng.integers(len(groups[1]), size=(count, len(groups[1]))), len(groups[1]))
        a = np.einsum('ksp,ns,np->kn', person[:, :, groups[0]], sw, aw, optimize=True)
        b = np.einsum('ksp,ns,np->kn', person[:, :, groups[1]], sw, bw, optimize=True)
        samples[:, first:first + count, 1] = a
        samples[:, first:first + count, 2] = b
        samples[:, first:first + count, 0] = (2 * a + b) / 3
    return point, samples


def family_sensitivity(values, metadata, draws=20000):
    _, groups, averaging = population(values, metadata)
    families = sorted({b['source_family'] for b in metadata}); assert all(families)
    family_ids = np.array([families.index(b['source_family']) for b in metadata])
    k, s, blocks = values.shape
    samples = np.empty((k, draws, 3)); rng = np.random.default_rng(20261103)
    for first in range(0, draws, 100):
        count = min(100, draws - first)
        sw = frequencies(rng.integers(s, size=(count, s)), s)
        pw = np.maximum(rng.exponential(size=(count, len(averaging))), np.finfo(float).tiny)
        fw = np.maximum(rng.exponential(size=(count, len(families))), np.finfo(float).tiny)
        outputs = []
        for group in groups:
            weights = np.zeros((count, blocks))
            for p in group:
                indices = np.flatnonzero(averaging[p])
                local = fw[:, family_ids[indices]]
                weights[:, indices] = (local / local.sum(1)[:, None]
                    * (pw[:, p] / pw[:, group].sum(1))[:, None])
            outputs.append(np.einsum('ksb,ns,nb->kn', values, sw, weights, optimize=True))
        samples[:, first:first + count] = np.stack([(2 * outputs[0] + outputs[1]) / 3,
                                                   outputs[0], outputs[1]], axis=2)
    return samples


def external(values, draws=20000):
    assert values.ndim == 4 and values.shape[1:] == (5, 18, 60)
    assert np.isfinite(values).all()
    person = values.sum(3) / 60
    samples = np.empty((len(values), draws)); rng = np.random.default_rng(20261105)
    for first in range(0, draws, 100):
        count = min(100, draws - first)
        sw = frequencies(rng.integers(5, size=(count, 5)), 5)
        pw = frequencies(rng.integers(18, size=(count, 18)), 18)
        samples[:, first:first + count] = np.einsum('ksp,ns,np->kn', person, sw, pw, optimize=True)
    return person, samples
