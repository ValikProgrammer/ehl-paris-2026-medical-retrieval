from __future__ import annotations

"""Candidate matching methods for the synthetic-d2 harness.

Each method takes the harness `data` dict:
    train_q, train_t : (Ntr, G, G, G)  clean downsampled dataset1 pairs (fit on these)
    eval_q,  eval_t  : (Nev, G, G, G)  independently deformed eval pairs (score these)
and returns an (Nev, Nev) score matrix whose diagonal is the true match.
"""

import numpy as np
from scipy.ndimage import affine_transform, map_coordinates, sobel, zoom
from scipy.optimize import minimize
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import normalize


# --- shared helpers -----------------------------------------------------------


def _l2(x: np.ndarray) -> np.ndarray:
    x = x - x.mean(axis=-1, keepdims=True)
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-6)


def _fit_pca_ridge(qf: np.ndarray, tf: np.ndarray, components: int, alpha: float):
    n = min(components, qf.shape[0] - 1, qf.shape[1], tf.shape[1])
    q_pca = PCA(n_components=n, whiten=True, random_state=20260627).fit(qf)
    t_pca = PCA(n_components=n, whiten=True, random_state=20260627).fit(tf)
    ridge = Ridge(alpha=alpha).fit(q_pca.transform(qf), t_pca.transform(tf))
    return q_pca, t_pca, ridge


def _pca_ridge_scores(train_qf, train_tf, eval_qf, eval_tf, components=128, alpha=100.0):
    q_pca, t_pca, ridge = _fit_pca_ridge(train_qf, train_tf, components, alpha)
    qz = normalize(ridge.predict(q_pca.transform(eval_qf)))
    tz = normalize(t_pca.transform(eval_tf))
    return (qz @ tz.T).astype(np.float32)


# --- feature extractors (operate on a GxGxG array) ----------------------------


def flat_feature(vol: np.ndarray) -> np.ndarray:
    return _l2(vol.reshape(-1).astype(np.float32))


def canonical_array(vol: np.ndarray, size: int) -> np.ndarray:
    """PCA-axis canonicalization of one volume (array version of canonical_pca)."""
    mask = vol > 0.05
    coords = np.column_stack(np.where(mask))
    if len(coords) < 32:
        coords = np.column_stack(np.where(np.ones_like(vol, dtype=bool)))
    center = coords.mean(axis=0)
    cov = np.cov((coords - center).T)
    values, vectors = np.linalg.eigh(cov)
    vectors = vectors[:, np.argsort(values)[::-1]]
    projections = (coords - center) @ vectors
    scales = np.maximum(np.percentile(np.abs(projections), 99.0, axis=0), 1.0)
    for axis in range(3):
        if float(np.mean(projections[:, axis] ** 3)) < 0:
            vectors[:, axis] *= -1
    grid = np.stack(
        np.meshgrid(*[np.linspace(-1.0, 1.0, size) for _ in range(3)], indexing="ij"), axis=-1
    )
    physical = center + (grid.reshape(-1, 3) * scales) @ vectors.T
    sampled = map_coordinates(
        vol, [physical[:, 0], physical[:, 1], physical[:, 2]], order=1, mode="constant", cval=0.0
    ).reshape(size, size, size)
    edge = np.sqrt(sum(sobel(sampled, axis=a) ** 2 for a in range(3)))
    return _l2(np.concatenate([sampled.reshape(-1), edge.reshape(-1)]).astype(np.float32))


def invariant_feature(vol: np.ndarray, n_int=48, n_rad=24, n_grad=32) -> np.ndarray:
    """Rigid-invariant descriptor: intensity hist + radial intensity profile +
    gradient-magnitude hist + mask-covariance eigenvalues. Survives rotation/
    translation by construction; elastic warp perturbs only mildly."""
    mask = vol > 0.05
    fg = vol[mask]
    if fg.size < 32:
        fg = vol.reshape(-1)
    parts = []

    # intensity histogram over foreground
    hist, _ = np.histogram(fg, bins=n_int, range=(0.0, 1.0), density=True)
    parts.append(hist.astype(np.float32))

    # radial intensity profile around foreground centroid
    coords = np.column_stack(np.where(mask)).astype(np.float32)
    if len(coords) < 32:
        coords = np.column_stack(np.where(np.ones_like(vol, dtype=bool))).astype(np.float32)
    center = coords.mean(axis=0)
    rad = np.linalg.norm(coords - center, axis=1)
    rmax = float(rad.max()) + 1e-6
    bins = np.clip((rad / rmax * n_rad).astype(int), 0, n_rad - 1)
    vals = vol[mask] if mask.sum() == len(coords) else vol.reshape(-1)
    prof = np.zeros(n_rad, dtype=np.float32)
    cnt = np.zeros(n_rad, dtype=np.float32)
    np.add.at(prof, bins, vals)
    np.add.at(cnt, bins, 1.0)
    parts.append(prof / (cnt + 1e-6))

    # gradient magnitude histogram
    grad = np.sqrt(sum(sobel(vol, axis=a) ** 2 for a in range(3)))
    ghist, _ = np.histogram(grad[mask], bins=n_grad, range=(0.0, float(grad.max()) + 1e-6), density=True)
    parts.append(ghist.astype(np.float32))

    # mask covariance eigenvalues (rotation-invariant shape), normalized
    cov = np.cov((coords - center).T)
    eig = np.sort(np.linalg.eigvalsh(cov))[::-1]
    eig = eig / (eig.sum() + 1e-6)
    parts.append(eig.astype(np.float32))

    return _l2(np.concatenate(parts))


# --- methods ------------------------------------------------------------------


def m_raw_grid(data):
    q = np.stack([flat_feature(v) for v in data["eval_q"]])
    t = np.stack([flat_feature(v) for v in data["eval_t"]])
    return (q @ t.T).astype(np.float32)


def m_pca_ridge_grid(data):
    tq = np.stack([flat_feature(v) for v in data["train_q"]])
    tt = np.stack([flat_feature(v) for v in data["train_t"]])
    eq = np.stack([flat_feature(v) for v in data["eval_q"]])
    et = np.stack([flat_feature(v) for v in data["eval_t"]])
    return _pca_ridge_scores(tq, tt, eq, et)


def _canonical_scores(data, size, components=128):
    s = min(size, data["grid"])
    tq = np.stack([canonical_array(v, s) for v in data["train_q"]])
    tt = np.stack([canonical_array(v, s) for v in data["train_t"]])
    eq = np.stack([canonical_array(v, s) for v in data["eval_q"]])
    et = np.stack([canonical_array(v, s) for v in data["eval_t"]])
    return _pca_ridge_scores(tq, tt, eq, et, components=components)


def m_canonical(data):
    return _canonical_scores(data, 24)


def _zscore_rows(s):
    return (s - s.mean(axis=1, keepdims=True)) / (s.std(axis=1, keepdims=True) + 1e-6)


def m_canonical_ms(data):
    """Multi-scale canonical: rank-fuse z-scored scores from several grid sizes."""
    total = np.zeros((data["eval_q"].shape[0],) * 2, dtype=np.float32)
    for size in (16, 24, 32):
        if size <= data["grid"]:
            total += _zscore_rows(_canonical_scores(data, size))
    return total


def m_canonical_inv(data):
    """Blend canonical spatial signal with pose-invariant descriptors."""
    return _zscore_rows(m_canonical(data)) + 0.5 * _zscore_rows(m_invariant(data))


def m_invariant(data):
    tq = np.stack([invariant_feature(v) for v in data["train_q"]])
    tt = np.stack([invariant_feature(v) for v in data["train_t"]])
    eq = np.stack([invariant_feature(v) for v in data["eval_q"]])
    et = np.stack([invariant_feature(v) for v in data["eval_t"]])
    return _pca_ridge_scores(tq, tt, eq, et, components=min(64, tq.shape[1]))


# --- registration re-rank -----------------------------------------------------


def _edge_map(vol: np.ndarray) -> np.ndarray:
    e = np.sqrt(sum(sobel(vol, axis=a) ** 2 for a in range(3)))
    m = e.max()
    return (e / m).astype(np.float32) if m > 1e-6 else e


def _rigid_matrix(params: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ax, ay, az = params[:3]
    rot = np.eye(3)
    for axis, ang in enumerate((ax, ay, az)):
        c, s = np.cos(ang), np.sin(ang)
        r = np.eye(3)
        a, b = [i for i in range(3) if i != axis]
        r[a, a], r[a, b], r[b, a], r[b, b] = c, -s, s, c
        rot = rot @ r
    return rot, params[3:6]


def _apply_rigid(vol: np.ndarray, params: np.ndarray) -> np.ndarray:
    rot, shift = _rigid_matrix(params)
    center = (np.asarray(vol.shape) - 1) / 2.0
    offset = center - rot @ center + shift
    return affine_transform(vol, rot, offset=offset, order=1, mode="constant", cval=0.0)


def _ncc(a: np.ndarray, b: np.ndarray) -> float:
    fg = (a > 0.02) | (b > 0.02)
    if fg.sum() < 16:
        return 0.0
    x, y = a[fg], b[fg]
    x = x - x.mean()
    y = y - y.mean()
    denom = (np.linalg.norm(x) * np.linalg.norm(y)) + 1e-6
    return float((x @ y) / denom)


def register_score(edge_q: np.ndarray, edge_t: np.ndarray, reg_size: int = 24) -> float:
    """Best edge-NCC after rigid alignment of target onto query (cross-modal safe:
    edges mark anatomical boundaries present in both T1 and T2)."""
    zq = zoom(edge_q, reg_size / edge_q.shape[0], order=1) if edge_q.shape[0] != reg_size else edge_q
    zt = zoom(edge_t, reg_size / edge_t.shape[0], order=1) if edge_t.shape[0] != reg_size else edge_t

    def neg_ncc(params):
        return -_ncc(zq, _apply_rigid(zt, params))

    best = neg_ncc(np.zeros(6))
    for start in (np.zeros(6), np.array([0.3, 0, 0, 0, 0, 0]), np.array([0, 0.3, 0, 0, 0, 0])):
        res = minimize(neg_ncc, start, method="Powell", options={"maxiter": 60, "xtol": 0.05, "ftol": 0.01})
        best = min(best, res.fun)
    return -best


def m_register(data, topk: int = 8, reg_size: int = 24):
    """Canonical prefilter -> rigid edge-registration re-rank of top-K candidates."""
    base = m_canonical(data)
    eq_edge = [_edge_map(v) for v in data["eval_q"]]
    et_edge = [_edge_map(v) for v in data["eval_t"]]
    n = base.shape[0]
    reg = np.full_like(base, -1.0)
    for qi in range(n):
        cand = np.argsort(-base[qi])[:topk]
        for ti in cand:
            reg[qi, ti] = register_score(eq_edge[qi], et_edge[ti], reg_size)
        if (qi + 1) % 10 == 0:
            print(f"  registered {qi + 1}/{n} queries", flush=True)
    # combine: registration dominates among prefiltered, base breaks ties elsewhere
    out = base * 0.01 + np.where(reg >= 0, reg, -1.0)
    return out.astype(np.float32)


# --- template normalization (O(N) intramodal registration) --------------------


def _register_to_template(vol, template, opt_size=20):
    """Rigidly align vol to template (same modality -> intensity NCC). Returns
    vol resampled into template frame at its native grid."""
    g = vol.shape[0]
    zt = zoom(template, opt_size / template.shape[0], order=1)
    zv = zoom(vol, opt_size / g, order=1)

    def neg(params):
        return -_ncc(zt, _apply_rigid(zv, params))

    best_p, best_f = np.zeros(6), neg(np.zeros(6))
    for start in (np.zeros(6), np.array([0.25, 0, 0, 0, 0, 0]),
                  np.array([0, 0.25, 0, 0, 0, 0]), np.array([0, 0, 0.25, 0, 0, 0])):
        res = minimize(neg, start, method="Powell", options={"maxiter": 80, "xtol": 0.03, "ftol": 0.005})
        if res.fun < best_f:
            best_f, best_p = res.fun, res.x
    scaled = best_p.copy()
    scaled[3:6] *= g / opt_size  # rescale translation to native grid
    return _apply_rigid(vol, scaled)


def _register_affine_to_template(vol, template, opt_size=24, rigid_init=None):
    """Affine (12-param) alignment to template, initialized from the rigid fit.
    Params = [rigid6, a00..a22 perturbation as 6 off-diagonal/scale terms]."""
    g = vol.shape[0]
    zt = zoom(template, opt_size / template.shape[0], order=1)
    zv = zoom(vol, opt_size / g, order=1)
    center = (np.asarray(zv.shape) - 1) / 2.0

    rigid = rigid_init if rigid_init is not None else np.zeros(6)
    rot0, _ = _rigid_matrix(rigid)

    def apply(mat, shift):
        offset = center - mat @ center + shift
        return affine_transform(zv, mat, offset=offset, order=1, mode="constant", cval=0.0)

    # optimize a 9-dof linear matrix (init = rigid rotation) + 3 translation
    def neg(p):
        mat = rot0 + p[:9].reshape(3, 3)
        return -_ncc(zt, apply(mat, rigid[3:6] * opt_size / g + p[9:12]))

    x0 = np.zeros(12)
    res = minimize(neg, x0, method="Powell", options={"maxiter": 120, "xtol": 0.02, "ftol": 0.005})
    mat = rot0 + res.x[:9].reshape(3, 3)
    shift = rigid[3:6] * opt_size / g + res.x[9:12]
    # apply at native grid
    cfull = (np.asarray(vol.shape) - 1) / 2.0
    sh = shift * g / opt_size
    offset = cfull - mat @ cfull + sh
    return affine_transform(vol, mat, offset=offset, order=1, mode="constant", cval=0.0)


def _best_rigid_params(vol, template, opt_size=20):
    g = vol.shape[0]
    zt = zoom(template, opt_size / template.shape[0], order=1)
    zv = zoom(vol, opt_size / g, order=1)

    def neg(params):
        return -_ncc(zt, _apply_rigid(zv, params))

    best_p, best_f = np.zeros(6), neg(np.zeros(6))
    for start in (np.zeros(6), np.array([0.25, 0, 0, 0, 0, 0]),
                  np.array([0, 0.25, 0, 0, 0, 0]), np.array([0, 0, 0.25, 0, 0, 0])):
        res = minimize(neg, start, method="Powell", options={"maxiter": 80, "xtol": 0.03, "ftol": 0.005})
        if res.fun < best_f:
            best_f, best_p = res.fun, res.x
    best_p = best_p.copy()
    best_p[3:6] *= g / opt_size
    return best_p


def m_template_affine(data):
    t1 = data["train_q"].mean(axis=0); t2 = data["train_t"].mean(axis=0)
    tq = np.stack([flat_feature(v) for v in data["train_q"]])
    tt = np.stack([flat_feature(v) for v in data["train_t"]])
    def emb(vol, tpl):
        rig = _best_rigid_params(vol, tpl)
        return flat_feature(_register_affine_to_template(vol, tpl, rigid_init=rig))
    eq = np.stack([emb(v, t1) for v in data["eval_q"]]); print("  affine q done", flush=True)
    et = np.stack([emb(v, t2) for v in data["eval_t"]]); print("  affine t done", flush=True)
    return _pca_ridge_scores(tq, tt, eq, et)


def mind_descriptor(vol, dilation=1, patch=1):
    """MIND: Modality-Independent Neighbourhood Descriptor (Heinrich 2012).
    Per-voxel 6-channel self-similarity field; near-identical across T1/T2 of one
    subject, and locally tolerant to smooth deformation. Pure numpy/scipy."""
    from scipy.ndimage import uniform_filter
    d = dilation
    offsets = [(d,0,0),(-d,0,0),(0,d,0),(0,-d,0),(0,0,d),(0,0,-d)]
    size = 2*patch+1
    Dp = []
    for o in offsets:
        shifted = np.roll(vol, shift=o, axis=(0,1,2))
        Dp.append(uniform_filter((vol-shifted)**2, size=size).astype(np.float32))
    Dp = np.stack(Dp)                      # (6,D,H,W)
    V = np.mean(Dp, axis=0) + 1e-6
    mind = np.exp(-Dp / V)
    mind /= (mind.max(axis=0, keepdims=True) + 1e-6)
    return mind.astype(np.float32)         # (6,D,H,W)


def mind_feature(vol):
    """Flattened, L2-normalized MIND field as a retrieval feature."""
    m = mind_descriptor(vol)
    return _l2(m.reshape(-1))


def m_template_mind(data):
    t1 = data["train_q"].mean(axis=0); t2 = data["train_t"].mean(axis=0)
    tq = np.stack([mind_feature(v) for v in data["train_q"]])
    tt = np.stack([mind_feature(v) for v in data["train_t"]])
    eq = np.stack([mind_feature(_register_to_template(v, t1)) for v in data["eval_q"]])
    print("  mind q done", flush=True)
    et = np.stack([mind_feature(_register_to_template(v, t2)) for v in data["eval_t"]])
    print("  mind t done", flush=True)
    return _pca_ridge_scores(tq, tt, eq, et)


def sliceview_feature(vol, n_slices=16, ds=12):
    """Multi-direction slice descriptor: along each of the 3 axes take n_slices
    evenly-spaced slices, downsample each to ds x ds, concatenate all. Captures
    structure from multiple viewing directions (a classical, no-torch stand-in for
    Raptor-style multi-view slice features)."""
    from scipy.ndimage import zoom as _zoom
    g = vol.shape[0]
    parts = []
    for axis in range(3):
        idx = np.linspace(0, vol.shape[axis]-1, n_slices).astype(int)
        sl = np.take(vol, idx, axis=axis)                    # (n_slices, X, Y) along moved axis
        sl = np.moveaxis(sl, axis, 0)
        for s in sl:
            parts.append(_zoom(s, ds/np.array(s.shape), order=1).reshape(-1))
    return _l2(np.concatenate(parts).astype(np.float32))


def m_template_sliceview(data):
    t1 = data["train_q"].mean(axis=0); t2 = data["train_t"].mean(axis=0)
    tq = np.stack([sliceview_feature(v) for v in data["train_q"]])
    tt = np.stack([sliceview_feature(v) for v in data["train_t"]])
    eq = np.stack([sliceview_feature(_register_to_template(v, t1)) for v in data["eval_q"]])
    print("  sliceview q done", flush=True)
    et = np.stack([sliceview_feature(_register_to_template(v, t2)) for v in data["eval_t"]])
    print("  sliceview t done", flush=True)
    return _pca_ridge_scores(tq, tt, eq, et)


def rich_feature(vol):
    """Post-registration multi-channel feature: intensity + gradient-magnitude
    (edges) + a half-scale intensity map. More discriminative than raw intensity."""
    edge = np.sqrt(sum(sobel(vol, axis=a) ** 2 for a in range(3))).astype(np.float32)
    half = zoom(vol, 0.5, order=1).astype(np.float32)
    parts = [_l2(vol.reshape(-1)), _l2(edge.reshape(-1)), _l2(half.reshape(-1))]
    return np.concatenate(parts).astype(np.float32)


def m_template_rich(data):
    t1 = data["train_q"].mean(axis=0); t2 = data["train_t"].mean(axis=0)
    tq = np.stack([rich_feature(v) for v in data["train_q"]])
    tt = np.stack([rich_feature(v) for v in data["train_t"]])
    eq = np.stack([rich_feature(_register_to_template(v, t1)) for v in data["eval_q"]])
    print("  rich q done", flush=True)
    et = np.stack([rich_feature(_register_to_template(v, t2)) for v in data["eval_t"]])
    print("  rich t done", flush=True)
    return _pca_ridge_scores(tq, tt, eq, et)


def _pca_axes(vol):
    mask = vol > 0.05
    coords = np.column_stack(np.where(mask)).astype(np.float64)
    if len(coords) < 32:
        coords = np.column_stack(np.where(np.ones_like(vol, dtype=bool))).astype(np.float64)
    center = coords.mean(axis=0)
    cov = np.cov((coords - center).T)
    vals, vecs = np.linalg.eigh(cov)
    vecs = vecs[:, np.argsort(vals)[::-1]]
    if np.linalg.det(vecs) < 0:
        vecs[:, 2] *= -1
    return center, vecs


def _register_robust(vol, template, tpl_axes, opt_size=24):
    """Rigid NCC alignment with PCA-axis initialization + sign-flip multistarts.
    Better than identity-only starts when the pose rotation is large."""
    g = vol.shape[0]
    zt = zoom(template, opt_size / template.shape[0], order=1)
    zv = zoom(vol, opt_size / g, order=1)
    center = (np.asarray(zv.shape) - 1) / 2.0
    _, Vv = _pca_axes(zv)
    Vt = tpl_axes

    def apply_mat(M, shift):
        offset = center - M @ center + shift
        return affine_transform(zv, M, offset=offset, order=1, mode="constant", cval=0.0)

    # PCA-axis init rotations (4 sign combos preserving det=+1)
    inits = []
    for s in ([1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]):
        R = Vt @ np.diag(s) @ Vv.T
        if np.linalg.det(R) > 0:
            inits.append(R)
    inits.append(np.eye(3))  # identity fallback

    best_vol, best_f = None, np.inf
    for R0 in inits:
        def neg(p):
            M = _rigid_matrix(p)[0] @ R0
            return -_ncc(zt, apply_mat(M, p[3:6]))
        res = minimize(neg, np.zeros(6), method="Powell",
                       options={"maxiter": 60, "xtol": 0.03, "ftol": 0.005})
        if res.fun < best_f:
            best_f = res.fun
            Mres = _rigid_matrix(res.x)[0] @ R0
            shift = res.x[3:6]
            best_vol = (Mres, shift)
    # apply at native grid
    M, shift = best_vol
    cfull = (np.asarray(vol.shape) - 1) / 2.0
    sh = shift * g / opt_size
    offset = cfull - M @ cfull + sh
    return affine_transform(vol, M, offset=offset, order=1, mode="constant", cval=0.0)


def m_template_robust(data):
    t1 = data["train_q"].mean(axis=0); t2 = data["train_t"].mean(axis=0)
    _, a1 = _pca_axes(zoom(t1, 24 / t1.shape[0], order=1))
    _, a2 = _pca_axes(zoom(t2, 24 / t2.shape[0], order=1))
    tq = np.stack([flat_feature(v) for v in data["train_q"]])
    tt = np.stack([flat_feature(v) for v in data["train_t"]])
    eq = np.stack([flat_feature(_register_robust(v, t1, a1)) for v in data["eval_q"]])
    print("  robust q done", flush=True)
    et = np.stack([flat_feature(_register_robust(v, t2, a2)) for v in data["eval_t"]])
    print("  robust t done", flush=True)
    return _pca_ridge_scores(tq, tt, eq, et)


def m_template(data):
    t1_tpl = data["train_q"].mean(axis=0)
    t2_tpl = data["train_t"].mean(axis=0)
    tq = np.stack([flat_feature(v) for v in data["train_q"]])
    tt = np.stack([flat_feature(v) for v in data["train_t"]])
    eq = np.stack([flat_feature(_register_to_template(v, t1_tpl)) for v in data["eval_q"]])
    print("  template-normalized eval queries", flush=True)
    et = np.stack([flat_feature(_register_to_template(v, t2_tpl)) for v in data["eval_t"]])
    print("  template-normalized eval targets", flush=True)
    return _pca_ridge_scores(tq, tt, eq, et)


def m_template_canon(data):
    """Rank-fuse template normalization with canonical."""
    return _zscore_rows(m_template(data)) + _zscore_rows(m_canonical(data))


def _contrast_jitter(vol, rng):
    """Random gamma + gain on the normalized [0,1] foreground (modality robustness)."""
    gamma = float(np.exp(rng.uniform(-0.4, 0.4)))
    gain = float(np.exp(rng.uniform(-0.2, 0.2)))
    out = np.clip(gain * np.power(np.clip(vol, 0, 1), gamma), 0, 1).astype(np.float32)
    return out


def m_template_augfit(data, k_aug=2):
    """Template normalization, but PCA/Ridge is fit on clean train pairs PLUS k
    geom+contrast augmented copies, so the cross-modal map sees deformation."""
    from synthetic_d2_eval import deform, _rng_for

    t1 = data["train_q"].mean(axis=0); t2 = data["train_t"].mean(axis=0)
    dkw = dict(max_rot_deg=12, max_shift=6, elastic_sigma=8, elastic_alpha=3)

    qf, tf = [], []
    for v in data["train_q"]:
        qf.append(flat_feature(v))
    for v in data["train_t"]:
        tf.append(flat_feature(v))
    for a in range(k_aug):
        for i, v in enumerate(data["train_q"]):
            w = _contrast_jitter(deform(v, _rng_for(1000 + a, f"q{i}"), **dkw), _rng_for(2000 + a, f"q{i}"))
            qf.append(flat_feature(_register_to_template(w, t1)))
        for i, v in enumerate(data["train_t"]):
            w = _contrast_jitter(deform(v, _rng_for(3000 + a, f"t{i}"), **dkw), _rng_for(4000 + a, f"t{i}"))
            tf.append(flat_feature(_register_to_template(w, t2)))
        print(f"  augfit built aug copy {a+1}/{k_aug}", flush=True)
    qf = np.stack(qf); tf = np.stack(tf)
    q_pca, t_pca, ridge = _fit_pca_ridge(qf, tf, 128, 100.0)

    eq = np.stack([flat_feature(_register_to_template(v, t1)) for v in data["eval_q"]])
    et = np.stack([flat_feature(_register_to_template(v, t2)) for v in data["eval_t"]])
    qz = normalize(ridge.predict(q_pca.transform(eq)))
    tz = normalize(t_pca.transform(et))
    return (qz @ tz.T).astype(np.float32)


METHODS = {
    "raw_grid": m_raw_grid,
    "pca_ridge_grid": m_pca_ridge_grid,
    "canonical": m_canonical,
    "canonical_ms": m_canonical_ms,
    "canonical_inv": m_canonical_inv,
    "invariant": m_invariant,
    "register": m_register,
    "template": m_template,
    "template_affine": m_template_affine,
    "template_augfit": m_template_augfit,
    "template_robust": m_template_robust,
    "template_rich": m_template_rich,
    "template_mind": m_template_mind,
    "template_sliceview": m_template_sliceview,
    "template_canon": m_template_canon,
}
