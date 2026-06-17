"""
Synthetic experiment for DPP-NEPv vs.\ L1 / simplex relaxations.

Two regimes are run and saved as separate figures.

Easy regime  (synthetic.pdf, synthetic.png)
-------------------------------------------
- 5 anchor points on a circle of radius 5 (equally spaced).
- 200 Gaussian noise points, zero-mean and std=1.
All three methods recover the 5 anchors -- sanity check.

Hard regime  (synthetic_hard.pdf, synthetic_hard.png)
-----------------------------------------------------
- 5 anchor CLUSTERS on the same circle: each cluster has 1 center
  point + 20 near-duplicates (Gaussian jitter, std=0.08).
- 100 broad Gaussian noise points (std=1).
- n = 5 + 5*20 + 100 = 205; k = 5.
The DPP-MAP optimum is "one representative per cluster" (5 covered).
The L1 / softmax relaxations distribute soft weight across the
nearly-identical cluster-mates, and top-k rounding ends up picking
multiple items from a single cluster -- missing one or more anchor
groups. DPP-NEPv's orthogonal-subspace structure is immune to this
redundancy because adding a duplicate yields a near-singular Gram
block, which the SCF eigensolver naturally avoids.

Methods (all select k=5)
------------------------
(a) DPP-NEPv      -- Stiefel / L2 relaxation, Algorithm 1 (this paper).
(b) Softmax ext.  -- Gillenwater et al. (2012),
                       tilde F(x) = log det( I + diag(x) (L - I) )
                     over the capped simplex { x in [0,1]^n : 1^T x = k }.
(c) D-optimal L1  -- Nikolov 2015 / Singh 2020,
                       G(x) = log det( Phi^T diag(x) Phi + lambda I )
                     on the same polytope.
"""

import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Kernel and small linear-algebra utilities
# ---------------------------------------------------------------------------

def build_rbf_kernel(X, gamma=None):
    """RBF kernel  L_ij = exp(-gamma * ||x_i - x_j||^2)."""
    sq = np.sum(X ** 2, axis=1)
    D2 = sq[:, None] + sq[None, :] - 2.0 * X @ X.T
    D2 = np.maximum(D2, 0.0)
    if gamma is None:
        med = np.median(D2[D2 > 0])
        gamma = 1.0 / med
    return np.exp(-gamma * D2), gamma


def logdet_psd(M):
    """Numerically stable log-det of (nearly) PSD M via Cholesky."""
    n = M.shape[0]
    eps = 1e-10 * (np.trace(M) / max(1, n) + 1.0)
    R = np.linalg.cholesky(M + eps * np.eye(n))
    return 2.0 * np.sum(np.log(np.diag(R)))


def polar_retract(Y):
    """Polar retraction onto Stiefel: Y -> U V^T from thin SVD."""
    U, _, Vt = np.linalg.svd(Y, full_matrices=False)
    return U @ Vt


def factorize_kernel(L, tol=1e-10):
    """L = Phi Phi^T via eigendecomposition; keeps positive eigenpairs only."""
    w, U = np.linalg.eigh(L)
    mask = w > tol * max(1.0, w.max())
    return U[:, mask] * np.sqrt(w[mask])


# ---------------------------------------------------------------------------
# (a) DPP-NEPv  -- SCF for  H(V) V = V Lambda   (Algorithm 1 in the paper)
# ---------------------------------------------------------------------------

def scf_dpp_nepv(L, k, n_iter=200, tol=1e-9, alpha=1.0, sigma=0.0):
    n = L.shape[0]
    _, U = np.linalg.eigh(L)
    V = U[:, -k:]                                # spectral warm start
    history = []
    for _ in range(n_iter):
        G = V.T @ L @ V
        G_inv = np.linalg.inv(G)
        VtL = V.T @ L
        LP = L @ V @ G_inv @ VtL                 # L * P(V)
        H = 0.5 * (LP + LP.T) + sigma * (V @ V.T)
        _, U = np.linalg.eigh(H)
        V_tilde = U[:, -k:]
        V_next = polar_retract(alpha * V_tilde + (1.0 - alpha) * V)
        S = np.linalg.svd(V.T @ V_next, compute_uv=False)
        sin_t = np.sqrt(np.maximum(0.0, 1.0 - S ** 2))
        diff = float(np.linalg.norm(sin_t))
        history.append(logdet_psd(V_next.T @ L @ V_next))
        V = V_next
        if diff < tol:
            break
    return V, history


def round_to_subset(V_star, L, k, pool_factor=6, n_restarts=10):
    """Leverage-score-guided greedy DPP-MAP + full 1-swap refinement."""
    n = V_star.shape[0]
    lev = np.sum(V_star ** 2, axis=1)
    pool = np.argsort(-lev)[: min(pool_factor * k, n)]

    def greedy(start):
        S = [int(start)]
        while len(S) < k:
            best_val, best_i = -np.inf, None
            for i in pool:
                i = int(i)
                if i in S:
                    continue
                val = logdet_psd(L[np.ix_(S + [i], S + [i])])
                if val > best_val:
                    best_val, best_i = val, i
            S.append(best_i)
        return S

    best_S, best_val = None, -np.inf
    for s in pool[: min(n_restarts, len(pool))]:
        S = greedy(int(s))
        val = logdet_psd(L[np.ix_(S, S)])
        if val > best_val:
            best_val, best_S = val, S
    S = best_S

    for _ in range(20):
        improved = False
        cur = logdet_psd(L[np.ix_(S, S)])
        for j in range(len(S)):
            best_val, best_cand = cur, None
            for cand in range(n):
                if cand in S:
                    continue
                S_try = S[:j] + [cand] + S[j + 1:]
                val = logdet_psd(L[np.ix_(S_try, S_try)])
                if val > best_val + 1e-9:
                    best_val, best_cand = val, cand
            if best_cand is not None:
                S = S[:j] + [best_cand] + S[j + 1:]
                cur = best_val
                improved = True
        if not improved:
            break
    return sorted(S)


# ---------------------------------------------------------------------------
# (b) Softmax extension and (c) D-optimal L1 relaxation
# Both maximized over the capped simplex { x in [0,1]^n : 1^T x = k }.
# ---------------------------------------------------------------------------

def project_capped_simplex(v, k, n_bisect=80):
    """Euclidean projection onto { x in [0,1]^n : 1^T x = k } via bisection."""
    n = len(v)
    if k <= 0:
        return np.zeros(n)
    if k >= n:
        return np.ones(n)
    lo = v.min() - 1.0
    hi = v.max() + 1.0
    for _ in range(n_bisect):
        tau = 0.5 * (lo + hi)
        s = float(np.clip(v - tau, 0.0, 1.0).sum())
        if s > k:
            lo = tau
        else:
            hi = tau
    return np.clip(v - 0.5 * (lo + hi), 0.0, 1.0)


def softmax_extension(L):
    """Closures (value, gradient) for  tilde F(x) = log det(I + Dx (L - I))."""
    n = L.shape[0]
    LmI = L - np.eye(n)

    def value(x):
        M = np.eye(n) + x[:, None] * LmI
        sign, val = np.linalg.slogdet(M)
        return val if sign > 0 else -np.inf

    def grad(x):
        M = np.eye(n) + x[:, None] * LmI
        Minv = np.linalg.solve(M, np.eye(n))
        # d/dx_i log det M = ( (L - I) M^{-1} )_{ii}
        return np.einsum("ij,ji->i", LmI, Minv)

    return value, grad


def doptimal_design(Phi, lam=1e-3):
    """Closures for  G(x) = log det( Phi^T diag(x) Phi + lambda I )."""
    d = Phi.shape[1]
    Id = np.eye(d)

    def value(x):
        G = Phi.T @ (x[:, None] * Phi) + lam * Id
        sign, val = np.linalg.slogdet(G)
        return val if sign > 0 else -np.inf

    def grad(x):
        G = Phi.T @ (x[:, None] * Phi) + lam * Id
        G_inv = np.linalg.inv(G)
        # d/dx_i G = a_i^T G^{-1} a_i  with a_i = Phi[i].
        return np.einsum("ij,jk,ik->i", Phi, G_inv, Phi)

    return value, grad


def projected_gradient_ascent(value_fn, grad_fn, n, k,
                              n_iter=600, lr0=0.5, tol=1e-10):
    """Backtracking projected gradient ascent on the capped simplex."""
    x = np.full(n, k / n)
    val = value_fn(x)
    lr = lr0
    for _ in range(n_iter):
        g = grad_fn(x)
        step = lr
        improved = False
        prev = val
        for _bt in range(30):
            x_try = project_capped_simplex(x + step * g, k)
            v_try = value_fn(x_try)
            if v_try > val + 1e-12:
                x = x_try
                val = v_try
                improved = True
                lr = min(lr * 1.5, 10.0)
                break
            step *= 0.5
        if not improved:
            break
        if abs(val - prev) < tol:
            break
    return x, val


def topk_round(x, k):
    """Round a fractional x in [0,1]^n to a size-k indicator (top-k)."""
    return sorted(int(i) for i in np.argsort(-x)[:k])


# ---------------------------------------------------------------------------
# Data-generation helpers
# ---------------------------------------------------------------------------

def make_easy(rng, n_anchor=5, n_noise=200, radius=5.0, noise_std=1.0):
    """Easy regime: isolated anchors + diffuse Gaussian noise.

    Returns (X, cluster_id) with cluster_id[i] in {0..n_anchor-1} for the
    anchor centers and -1 for the noise points.
    """
    angles = np.linspace(0.0, 2.0 * np.pi, n_anchor, endpoint=False)
    anchors = np.stack([radius * np.cos(angles),
                        radius * np.sin(angles)], axis=1)
    noise = rng.normal(0.0, noise_std, size=(n_noise, 2))
    X = np.vstack([anchors, noise])
    cluster_id = np.concatenate([
        np.arange(n_anchor),
        -np.ones(n_noise, dtype=int),
    ])
    return X, cluster_id


def make_uniform(rng, n=1000, low=-5.0, high=5.0):
    """Uniform regime: ``n`` points drawn i.i.d. from U([low, high]^2).

    No cluster structure -- the "right answer" for diverse selection is
    points spread out across the square; we score it by the average
    pairwise distance among the selected points.
    """
    X = rng.uniform(low=low, high=high, size=(n, 2))
    return X


def make_grid_clusters(rng, n_rows=3, n_cols=5, n_per_cluster=30,
                       spacing=4.0, cluster_std=0.5):
    """Grid regime: cluster means on a regular ``n_rows`` x ``n_cols``
    lattice in the 2-D plane (equally spaced both horizontally and
    vertically), with ``n_per_cluster`` Gaussian samples around each.

    Layout
    ------
    Means sit on grid points
        ( (j - (n_cols-1)/2) * spacing ,  (i - (n_rows-1)/2) * spacing )
    for i = 0..n_rows-1, j = 0..n_cols-1, so the grid is centered at the
    origin. Default 3x5 = 15 means with adjacent spacing 4.

    Returns
    -------
    X          : (n_rows * n_cols * n_per_cluster, 2) sample array.
    cluster_id : integer cluster label for each point.
    means      : (n_rows * n_cols, 2) underlying grid centers (not
                 included in X; for plotting reference only).
    """
    xs = (np.arange(n_cols) - (n_cols - 1) / 2.0) * spacing
    ys = (np.arange(n_rows) - (n_rows - 1) / 2.0) * spacing
    means = np.array([[x, y] for y in ys for x in xs])  # row-major
    pts, ids = [], []
    for c, mu in enumerate(means):
        pts.append(mu + rng.normal(0.0, cluster_std,
                                   size=(n_per_cluster, 2)))
        ids.extend([c] * n_per_cluster)
    X = np.vstack(pts)
    cluster_id = np.asarray(ids, dtype=int)
    return X, cluster_id, means


def make_hard(rng, n_anchor=5, m_per_anchor=20, n_noise=100,
              radius=5.0, jitter=0.08, noise_std=1.0):
    """Hard regime: anchor CLUSTERS (1 center + m near-duplicates each).

    Cluster id assignment:
        index 0..n_anchor-1                   -> centers, cluster_id = c
        index n_anchor..n_anchor+n_anchor*m-1 -> duplicates, cluster_id = c
        index ...n-1                          -> noise, cluster_id = -1
    """
    angles = np.linspace(0.0, 2.0 * np.pi, n_anchor, endpoint=False)
    centers = np.stack([radius * np.cos(angles),
                        radius * np.sin(angles)], axis=1)
    dup_points = []
    dup_cluster = []
    for c in range(n_anchor):
        pts = centers[c] + rng.normal(0.0, jitter, size=(m_per_anchor, 2))
        dup_points.append(pts)
        dup_cluster.extend([c] * m_per_anchor)
    dup_points = np.vstack(dup_points)
    noise = rng.normal(0.0, noise_std, size=(n_noise, 2))
    X = np.vstack([centers, dup_points, noise])
    cluster_id = np.concatenate([
        np.arange(n_anchor),
        np.asarray(dup_cluster, dtype=int),
        -np.ones(n_noise, dtype=int),
    ])
    return X, cluster_id


# ---------------------------------------------------------------------------
# Run all three methods on a given instance.
# ---------------------------------------------------------------------------

def run_all_methods(X, k, gamma=None, verbose_label="",
                    pga_iter_sm=600, pga_iter_do=600,
                    scf_n_iter=200):
    n = X.shape[0]
    L, gamma_used = build_rbf_kernel(X, gamma=gamma)
    Phi = factorize_kernel(L)

    V_star, history = scf_dpp_nepv(L, k=k, n_iter=scf_n_iter, tol=1e-10)
    S_nepv = round_to_subset(V_star, L, k=k)

    sm_val, sm_grad = softmax_extension(L)
    x_sm, fstar_sm = projected_gradient_ascent(sm_val, sm_grad,
                                               n=n, k=k,
                                               n_iter=pga_iter_sm)
    S_sm = topk_round(x_sm, k)

    do_val, do_grad = doptimal_design(Phi, lam=1e-3)
    x_do, fstar_do = projected_gradient_ascent(do_val, do_grad,
                                               n=n, k=k,
                                               n_iter=pga_iter_do)
    S_do = topk_round(x_do, k)

    if verbose_label:
        print(f"[{verbose_label}] gamma={gamma_used:.4f}  "
              f"SCF iters={len(history)}")
    return {
        "L": L, "Phi": Phi, "gamma": gamma_used,
        "S_nepv": S_nepv, "S_sm": S_sm, "S_do": S_do,
        "x_sm": x_sm, "x_do": x_do,
        "fstar_sm": fstar_sm, "fstar_do": fstar_do,
        "fstar_nepv": history[-1], "scf_iters": len(history),
    }


def cluster_coverage(S, cluster_id):
    """Number of distinct non-noise cluster ids hit by S."""
    return len({int(cluster_id[i]) for i in S if cluster_id[i] >= 0})


def pairwise_dists(X, S):
    """Return the flattened upper-triangle of pairwise distances among
    points in subset ``S``."""
    P = X[np.asarray(S)]
    diff = P[:, None, :] - P[None, :, :]
    D = np.sqrt((diff ** 2).sum(-1))
    iu = np.triu_indices(len(P), k=1)
    return D[iu]


def avg_pairwise_dist(X, S):
    """Mean Euclidean distance among all C(|S|,2) pairs in subset S.

    Higher = more spread-out on average (a coarse diversity metric).
    """
    return float(pairwise_dists(X, S).mean())


def min_pairwise_dist(X, S):
    """Smallest Euclidean distance among any pair in subset S.

    Higher = more uniform spacing (no near-duplicates). Better proxy for
    "well-spread coverage" than the mean, which can be inflated by
    clumping points at distant boundaries.
    """
    return float(pairwise_dists(X, S).min())


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _panel(ax, X, cluster_id, S, k, title, radius=5.0,
           xy_lim=7.0, point_size=22, cluster_marker_size=44,
           selected_size=360, reference="circle", grid_means=None,
           background_label="Noise"):
    is_noise = cluster_id < 0
    noise = X[is_noise]

    ax.scatter(noise[:, 0], noise[:, 1], s=point_size, c="#b0b0b0",
               label=(f"{background_label} ({len(noise)} pts)"
                      if len(noise) else None),
               zorder=1)
    # Cluster points, color-coded by cluster. tab20 has 20 distinguishable
    # colors, enough for our regimes (up to 15 clusters).
    cluster_ids_present = np.unique(cluster_id[~is_noise])
    palette = plt.get_cmap("tab20")
    for idx, c in enumerate(cluster_ids_present):
        mask = cluster_id == c
        ax.scatter(X[mask, 0], X[mask, 1],
                   s=cluster_marker_size,
                   c=[palette(int(c) % 20)],
                   marker="x", linewidths=1.6,
                   label=("Cluster point" if idx == 0 else None),
                   zorder=2)
    sel = np.array(S)
    ax.scatter(X[sel, 0], X[sel, 1], s=selected_size, facecolors="none",
               edgecolors="#ff7f0e", linewidths=3.0,
               label=fr"Selected ($k{{=}}{k}$)", zorder=3)
    if reference == "circle":
        th = np.linspace(0.0, 2.0 * np.pi, 256)
        ax.plot(radius * np.cos(th), radius * np.sin(th), "--",
                color="#1f77b4", alpha=0.35, lw=1.2)
    elif reference == "grid" and grid_means is not None:
        ax.scatter(grid_means[:, 0], grid_means[:, 1], s=60,
                   facecolors="none", edgecolors="#1f77b4",
                   linewidths=1.0, alpha=0.45, zorder=0)

    ax.set_aspect("equal")
    ax.set_xlim(-xy_lim, xy_lim)
    ax.set_ylim(-xy_lim, xy_lim)
    ax.set_xlabel(r"$x_1$", fontsize=14)
    ax.set_ylabel(r"$x_2$", fontsize=14)
    ax.tick_params(labelsize=12)
    ax.set_title(title, fontsize=15)
    ax.grid(alpha=0.25)


def plot_three(X, cluster_id, results, out_stem, regime_title,
               k=5, radius=5.0, xy_lim=7.0,
               point_size=22, cluster_marker_size=44, selected_size=360,
               reference="circle", grid_means=None, panel_titles=None,
               background_label="Noise"):
    """2+1 layout: (a) and (b) on top, (c) centered below. Each panel is
    large enough to read individually; horizontal alignment is dropped
    in favor of bigger per-panel area.

    If ``panel_titles`` is given (length-3 list of strings), it overrides
    the default "X/Y clusters covered" titles.
    """
    has_clusters = (cluster_id >= 0).any()
    if has_clusters:
        n_anchor = int(cluster_id[cluster_id >= 0].max()) + 1
    else:
        n_anchor = 0
    cov = lambda S: cluster_coverage(S, cluster_id)

    fig = plt.figure(figsize=(14, 13.5))
    gs = fig.add_gridspec(
        2, 4,
        hspace=0.30, wspace=0.45,
        left=0.07, right=0.97, top=0.93, bottom=0.08,
    )
    ax_a = fig.add_subplot(gs[0, 0:2])     # top-left: two columns of four
    ax_b = fig.add_subplot(gs[0, 2:4])     # top-right
    ax_c = fig.add_subplot(gs[1, 1:3])     # bottom row, centered

    panel_kwargs = dict(
        k=k, radius=radius, xy_lim=xy_lim,
        point_size=point_size,
        cluster_marker_size=cluster_marker_size,
        selected_size=selected_size,
        reference=reference, grid_means=grid_means,
        background_label=background_label,
    )
    if panel_titles is None:
        panel_titles = [
            f"(a) DPP-NEPv (ours) -- "
            f"{cov(results['S_nepv'])}/{n_anchor} clusters covered",
            f"(b) Softmax extension -- "
            f"{cov(results['S_sm'])}/{n_anchor} clusters covered",
            f"(c) D-optimal $L_1$ -- "
            f"{cov(results['S_do'])}/{n_anchor} clusters covered",
        ]
    _panel(ax_a, X, cluster_id, results["S_nepv"],
           title=panel_titles[0], **panel_kwargs)
    _panel(ax_b, X, cluster_id, results["S_sm"],
           title=panel_titles[1], **panel_kwargs)
    _panel(ax_c, X, cluster_id, results["S_do"],
           title=panel_titles[2], **panel_kwargs)

    fig.suptitle(regime_title, fontsize=16, y=0.985)
    handles, labels = ax_a.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels),
               fontsize=13, frameon=True, bbox_to_anchor=(0.5, 0.005))
    fig.savefig(f"{out_stem}.pdf", bbox_inches="tight")
    fig.savefig(f"{out_stem}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def _report(name, S, cluster_id, L, extra=""):
    cov = cluster_coverage(S, cluster_id)
    n_cluster = int(cluster_id[cluster_id >= 0].max()) + 1
    ld = logdet_psd(L[np.ix_(S, S)])
    print(f"  {name:20s}  coverage {cov}/{n_cluster}   "
          f"logdet(L_S) = {ld:+.4f}   S = {S}   {extra}")


def main():
    rng = np.random.default_rng(42)
    k = 5

    # =========================
    # Easy regime
    # =========================
    print("\n=== Easy regime: isolated anchors + diffuse noise ===")
    X_easy, cid_easy = make_easy(rng, n_anchor=5, n_noise=200,
                                 radius=5.0, noise_std=1.0)
    res_easy = run_all_methods(X_easy, k=k, verbose_label="easy")
    _report("DPP-NEPv (ours)", res_easy["S_nepv"], cid_easy, res_easy["L"],
            f"f(V*) = {res_easy['fstar_nepv']:+.4f}")
    _report("Softmax extension", res_easy["S_sm"], cid_easy, res_easy["L"],
            f"tilde F(x*) = {res_easy['fstar_sm']:+.4f}")
    _report("D-optimal L1", res_easy["S_do"], cid_easy, res_easy["L"],
            f"G(x*) = {res_easy['fstar_do']:+.4f}")
    plot_three(X_easy, cid_easy, res_easy, "synthetic",
               regime_title="Easy regime: 5 anchors on circle "
                            "+ 200 Gaussian noise points",
               k=k, radius=5.0, xy_lim=7.0)

    # =========================
    # Hard regime (DPP-NEPv should visibly win)
    # =========================
    # The kernel bandwidth gamma is set explicitly. The median heuristic
    # on this instance gives a small gamma that washes out the per-cluster
    # structure; a moderate fixed gamma keeps within-cluster kernel ~ 1
    # and between-cluster kernel ~ 0, which is the regime where the L1
    # relaxations exhibit their pitfalls (rank-one / correlated kernels).
    print("\n=== Hard regime: 5 anchor clusters with redundancy + noise ===")
    rng_hard = np.random.default_rng(7)
    X_hard, cid_hard = make_hard(rng_hard, n_anchor=5, m_per_anchor=20,
                                 n_noise=100, radius=5.0,
                                 jitter=0.08, noise_std=1.0)
    res_hard = run_all_methods(X_hard, k=k, gamma=0.5,
                               verbose_label="hard")
    _report("DPP-NEPv (ours)", res_hard["S_nepv"], cid_hard, res_hard["L"],
            f"f(V*) = {res_hard['fstar_nepv']:+.4f}")
    _report("Softmax extension", res_hard["S_sm"], cid_hard, res_hard["L"],
            f"tilde F(x*) = {res_hard['fstar_sm']:+.4f}")
    _report("D-optimal L1", res_hard["S_do"], cid_hard, res_hard["L"],
            f"G(x*) = {res_hard['fstar_do']:+.4f}")
    plot_three(X_hard, cid_hard, res_hard, "synthetic_hard",
               regime_title="Hard regime: each anchor is a cluster of "
                            "21 near-duplicates (redundancy)",
               k=k, radius=5.0, xy_lim=7.0)

    # =========================
    # 15-cluster grid regime: 3 x 5 lattice of means, 30 samples each
    # =========================
    # 15 means on a regular 3 x 5 grid (equal spacing 4 in both x and y).
    # n = 15 * 30 = 450, k = 15. No separate "noise". The DPP-MAP optimum
    # is one point per cluster; failure of a method shows up as multiple
    # selections in the same cluster (and therefore at least one missed
    # cluster).
    print("\n=== 15-cluster grid regime: 3x5 lattice, 30 samples each ===")
    rng_15 = np.random.default_rng(11)
    X_15, cid_15, means_15 = make_grid_clusters(
        rng_15, n_rows=3, n_cols=5, n_per_cluster=30,
        spacing=4.0, cluster_std=0.5,
    )
    k_15 = 15
    # Fixed gamma so within-cluster pairs have kernel ~ 0.78 and
    # nearest-neighbor cluster pairs have kernel ~ 0. With the median
    # heuristic the cross-cluster distances would dominate and the kernel
    # would lose sensitivity.
    res_15 = run_all_methods(X_15, k=k_15, gamma=0.5,
                             verbose_label="15-grid")
    _report("DPP-NEPv (ours)", res_15["S_nepv"], cid_15, res_15["L"],
            f"f(V*) = {res_15['fstar_nepv']:+.4f}")
    _report("Softmax extension", res_15["S_sm"], cid_15, res_15["L"],
            f"tilde F(x*) = {res_15['fstar_sm']:+.4f}")
    _report("D-optimal L1", res_15["S_do"], cid_15, res_15["L"],
            f"G(x*) = {res_15['fstar_do']:+.4f}")
    # Grid extent: x in [-8, 8], y in [-4, 4]; use square 12x12 frame.
    plot_three(X_15, cid_15, res_15, "synthetic_15",
               regime_title="15-cluster grid regime: 3x5 lattice of means "
                            "(spacing 4), 30 samples per mean",
               k=k_15, radius=10.0, xy_lim=11.0,
               point_size=14, cluster_marker_size=26, selected_size=240,
               reference="grid", grid_means=means_15)

    # =========================
    # Uniform regime: 1000 i.i.d. uniform points in [-5, 5]^2, k = 15
    # =========================
    # No cluster structure -- we score each method by the average pairwise
    # distance among its selected points. Higher = more spread out =
    # more diverse selection. A random reference baseline is included for
    # context.
    #
    # PGA iteration counts are reduced compared to the cluster regimes
    # because each value / gradient evaluation on the softmax extension is
    # O(n^3) at n = 1000 (Cholesky on a 1000x1000 matrix per eval).
    print("\n=== Uniform regime: 1000 uniform points in [-5,5]^2 ===")
    rng_u = np.random.default_rng(2026)
    X_u = make_uniform(rng_u, n=1000, low=-5.0, high=5.0)
    k_u = 15
    res_u = run_all_methods(X_u, k=k_u, verbose_label="uniform",
                            pga_iter_sm=120, pga_iter_do=300)

    def metrics(S):
        return dict(
            avg=avg_pairwise_dist(X_u, S),
            mn=min_pairwise_dist(X_u, S),
            ld=logdet_psd(res_u["L"][np.ix_(S, S)]),
        )

    m_nepv = metrics(res_u["S_nepv"])
    m_sm = metrics(res_u["S_sm"])
    m_do = metrics(res_u["S_do"])

    # Random baseline: 200 random size-k subsets.
    rng_rand = np.random.default_rng(0)
    avg_rand_list, min_rand_list = [], []
    for _ in range(200):
        S_rand = rng_rand.choice(len(X_u), size=k_u, replace=False)
        avg_rand_list.append(avg_pairwise_dist(X_u, S_rand))
        min_rand_list.append(min_pairwise_dist(X_u, S_rand))
    avg_rand = float(np.mean(avg_rand_list))
    min_rand = float(np.mean(min_rand_list))

    print(f"  {'method':22s}  {'avg dist':>9s}  {'min dist':>9s}  "
          f"{'logdet(L_S)':>12s}")
    print(f"  {'random (200 draws)':22s}  {avg_rand:9.4f}  "
          f"{min_rand:9.4f}  {'-':>12s}")
    print(f"  {'DPP-NEPv (ours)':22s}  {m_nepv['avg']:9.4f}  "
          f"{m_nepv['mn']:9.4f}  {m_nepv['ld']:+12.4f}")
    print(f"  {'Softmax extension':22s}  {m_sm['avg']:9.4f}  "
          f"{m_sm['mn']:9.4f}  {m_sm['ld']:+12.4f}")
    print(f"  {'D-optimal L1':22s}  {m_do['avg']:9.4f}  "
          f"{m_do['mn']:9.4f}  {m_do['ld']:+12.4f}")

    # All points treated as "background" (no clusters in uniform data).
    cid_u = np.full(len(X_u), -1, dtype=int)
    panel_titles_u = [
        (f"(a) DPP-NEPv (ours)\n"
         f"avg dist = {m_nepv['avg']:.3f},  min dist = {m_nepv['mn']:.3f}"),
        (f"(b) Softmax extension\n"
         f"avg dist = {m_sm['avg']:.3f},  min dist = {m_sm['mn']:.3f}"),
        (fr"(c) D-optimal $L_1$" + "\n"
         f"avg dist = {m_do['avg']:.3f},  min dist = {m_do['mn']:.3f}"),
    ]
    plot_three(
        X_u, cid_u, res_u, "synthetic_uniform",
        regime_title=(r"Uniform regime: 1000 i.i.d.\ points in "
                      r"$[-5,5]^2$, $k=15$  "
                      f"(random baseline: avg dist $\\approx$ {avg_rand:.2f},  "
                      f"min dist $\\approx$ {min_rand:.2f})"),
        k=k_u, radius=5.0, xy_lim=5.5,
        point_size=10, selected_size=240,
        reference="none", panel_titles=panel_titles_u,
        background_label="Data",
    )

    print("\nsaved: synthetic.{pdf,png}, synthetic_hard.{pdf,png}, "
          "synthetic_15.{pdf,png}, synthetic_uniform.{pdf,png}")


if __name__ == "__main__":
    main()
