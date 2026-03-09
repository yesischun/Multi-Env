"""
Scene distribution difference toolkit

Provides:
- static feature extraction from environment config dicts
- building per-feature distributions from many env configs or sampled episodes
- per-feature distance metrics (Wasserstein for continuous, JS for categorical)
- aggregated weighted difference score and per-feature report

Usage example:
    from scene_distribution import build_static_distributions, compare_distributions

    distA = build_static_distributions(list_of_env_configs_A)
    distB = build_static_distributions(list_of_env_configs_B)
    report = compare_distributions(distA, distB)

Notes:
- This module prefers scipy for numeric metrics (wasserstein, jensenshannon). If scipy is not available,
  it falls back to simple histogram-based L1 distances.
- For dynamic features (from rollouts) collect arrays like average speed, collision flags, etc., and place
  them under the `dynamic` key when building distributions.
"""

from typing import List, Dict, Any, Tuple
import numpy as np

# Try to import scipy utilities; fall back gracefully
try:
    from scipy.stats import wasserstein_distance
    from scipy.spatial.distance import jensenshannon
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False


def extract_static_features(env_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract a fixed set of static features from an environment config dict.
    Expected keys in env_cfg (common in this repo):
      - 'map_config' (may contain lane num/width, generate type)
      - 'traffic_density' (float)
      - 'num_scenarios' (int)
      - 'custom_dist' (dict of road-type -> weight)
      - other keys are tolerated

    Returns dict of simple scalars / small vectors suitable for distribution building.
    """
    features = {}

    # map type / generator
    map_cfg = env_cfg.get('map_config', {})
    gen_type = map_cfg.get('GENERATE_TYPE') or map_cfg.get('generate_type') or map_cfg.get('type')
    features['map_type'] = str(gen_type)

    # lane count and width
    lane_num = map_cfg.get('LANE_NUM') or map_cfg.get('lane_num') or map_cfg.get('lane_count')
    features['lane_num'] = int(lane_num) if lane_num is not None else None
    lane_width = map_cfg.get('LANE_WIDTH') or map_cfg.get('lane_width')
    features['lane_width'] = float(lane_width) if lane_width is not None else None

    # traffic density
    features['traffic_density'] = float(env_cfg.get('traffic_density', 0.0))

    # num scenarios
    features['num_scenarios'] = int(env_cfg.get('num_scenarios', 0))

    # custom dist over scenario types (categorical distribution)
    custom = env_cfg.get('custom_dist', {}) or {}
    # normalize to vector
    total = sum(custom.values()) if len(custom) > 0 else 0.0
    features['custom_dist'] = {k: float(v) / total if total > 0 else float(v) for k, v in custom.items()}

    # other lightweight indicators
    features['use_render'] = bool(env_cfg.get('use_render', False))

    return features


def build_static_distributions(env_cfgs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Build per-feature distributions from a list of environment config dicts.
    Returns a dictionary mapping feature name -> representation:
      - continuous features: numpy array of values
      - categorical (map_type, custom_dist): frequency dict or matrix
    """
    feats = [extract_static_features(c) for c in env_cfgs]

    # continuous: traffic_density, lane_width, num_scenarios
    traffic = np.array([f['traffic_density'] for f in feats], dtype=float)
    lane_widths = np.array([f['lane_width'] if f['lane_width'] is not None else np.nan for f in feats], dtype=float)
    lane_nums = np.array([f['lane_num'] if f['lane_num'] is not None else np.nan for f in feats], dtype=float)
    num_scenarios = np.array([f['num_scenarios'] for f in feats], dtype=float)

    # categorical map_type
    map_types = [f['map_type'] for f in feats]
    unique_map_types, counts = np.unique(map_types, return_counts=True)
    map_type_freq = dict(zip(unique_map_types.tolist(), (counts / counts.sum()).tolist()))

    # custom_dist: aggregate keys into unified vector over union of keys
    all_keys = set()
    for f in feats:
        all_keys.update(f['custom_dist'].keys())
    all_keys = sorted(list(all_keys))
    custom_matrix = np.zeros((len(feats), len(all_keys)), dtype=float)
    for i, f in enumerate(feats):
        for j, k in enumerate(all_keys):
            custom_matrix[i, j] = f['custom_dist'].get(k, 0.0)

    result = {
        'traffic_density': traffic,
        'lane_widths': lane_widths[~np.isnan(lane_widths)],
        'lane_nums': lane_nums[~np.isnan(lane_nums)],
        'num_scenarios': num_scenarios,
        'map_type_freq': map_type_freq,
        'custom_keys': all_keys,
        'custom_matrix': custom_matrix,
        'count': len(feats)
    }

    return result


def _js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon divergence for probability vectors (base e)."""
    # Ensure numpy arrays and small epsilon
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    eps = 1e-12
    p = p + eps
    q = q + eps
    p /= p.sum()
    q /= q.sum()
    m = 0.5 * (p + q)
    # KL(p||m) + KL(q||m)
    kl = lambda a, b: np.sum(a * np.log(a / b))
    return 0.5 * (kl(p, m) + kl(q, m))


def compare_continuous(a: np.ndarray, b: np.ndarray) -> Dict[str, float]:
    """
    Compute different continuous-feature distances and return a small dict of metrics.
    Metrics: wasserstein (if available), ks_stat (if scipy available), l1_hist
    """
    res = {}
    if a.size == 0 and b.size == 0:
        return {'wasserstein': 0.0, 'ks_stat': 0.0, 'l1_hist': 0.0}
    if SCIPY_AVAILABLE:
        try:
            res['wasserstein'] = float(wasserstein_distance(a, b))
        except Exception:
            res['wasserstein'] = float(np.abs(a.mean() - b.mean()))
    else:
        # fallback to difference in means
        res['wasserstein'] = float(np.abs(a.mean() - b.mean())) if a.size and b.size else float(np.nan)

    # l1 over histograms
    try:
        bins = 20
        ab = np.concatenate([a, b]) if (a.size and b.size) else (a if a.size else b)
        if ab.size == 0:
            res['l1_hist'] = 0.0
        else:
            hist_range = (np.nanmin(ab), np.nanmax(ab)) if ab.size > 0 else (0.0, 1.0)
            ha, _ = np.histogram(a, bins=bins, range=hist_range, density=True)
            hb, _ = np.histogram(b, bins=bins, range=hist_range, density=True)
            res['l1_hist'] = float(0.5 * np.sum(np.abs(ha - hb)))
    except Exception:
        res['l1_hist'] = float(np.nan)

    return res


def compare_categorical_freq(freqA: Dict[str, float], freqB: Dict[str, float]) -> Dict[str, float]:
    """
    Compare two categorical frequency dicts over possibly different keys.
    Returns JS divergence and L1 distance.
    """
    keys = sorted(set(list(freqA.keys()) + list(freqB.keys())))
    pa = np.array([freqA.get(k, 0.0) for k in keys], dtype=float)
    pb = np.array([freqB.get(k, 0.0) for k in keys], dtype=float)
    if SCIPY_AVAILABLE:
        try:
            js = float(jensenshannon(pa, pb, base=np.e) ** 2)  # scipy returns sqrt(JS)
        except Exception:
            js = float(_js_divergence(pa, pb))
    else:
        js = float(_js_divergence(pa, pb))
    l1 = float(0.5 * np.sum(np.abs(pa - pb)))
    return {'js': js, 'l1': l1}


def compare_custom_matrix(matA: np.ndarray, matB: np.ndarray, keysA: List[str], keysB: List[str]) -> Dict[str, Any]:
    """
    Compare aggregated custom_dist matrices. We average rows to get empirical categorical distributions.
    """
    # unify keys
    all_keys = sorted(list(set(keysA + keysB)))
    def expand_matrix(mat, keys, all_keys):
        # mat shape (n, k)
        if mat.size == 0:
            return np.zeros((0, len(all_keys)), dtype=float)
        key_idx = {k: i for i, k in enumerate(keys)}
        out = np.zeros((mat.shape[0], len(all_keys)), dtype=float)
        for j, k in enumerate(all_keys):
            if k in key_idx:
                out[:, j] = mat[:, key_idx[k]]
        return out

    Aexp = expand_matrix(matA, keysA, all_keys)
    Bexp = expand_matrix(matB, keysB, all_keys)
    # average across rows to get empirical prob vectors
    pa = Aexp.mean(axis=0) if Aexp.size else np.zeros(len(all_keys), dtype=float)
    pb = Bexp.mean(axis=0) if Bexp.size else np.zeros(len(all_keys), dtype=float)
    # normalize
    if pa.sum() > 0:
        pa = pa / pa.sum()
    if pb.sum() > 0:
        pb = pb / pb.sum()
    cat_res = compare_categorical_freq(dict(zip(all_keys, pa.tolist())), dict(zip(all_keys, pb.tolist())))
    cat_res['keys'] = all_keys
    return cat_res


def compare_distributions(distA: Dict[str, Any], distB: Dict[str, Any], weights: Dict[str, float] = None) -> Dict[str, Any]:
    """
    Compare two distribution objects returned by `build_static_distributions`.
    Returns per-feature metrics and an aggregated weighted score in [0, +inf) (lower = more similar).

    weights: optional dict to weight features; default weights are reasonable defaults.
    """
    if weights is None:
        weights = {
            'traffic_density': 1.0,
            'lane_widths': 0.5,
            'lane_nums': 0.5,
            'num_scenarios': 0.2,
            'map_type': 1.0,
            'custom_dist': 1.0
        }

    report = {}

    # traffic_density
    report['traffic_density'] = compare_continuous(distA['traffic_density'], distB['traffic_density'])

    # lane_widths
    report['lane_widths'] = compare_continuous(distA['lane_widths'], distB['lane_widths'])

    # lane_nums
    report['lane_nums'] = compare_continuous(distA['lane_nums'], distB['lane_nums'])

    # num_scenarios (scalar distribution)
    report['num_scenarios'] = compare_continuous(distA['num_scenarios'], distB['num_scenarios'])

    # map types (categorical frequency)
    report['map_type'] = compare_categorical_freq(distA['map_type_freq'], distB['map_type_freq'])

    # custom_dist (matrix)
    report['custom_dist'] = compare_custom_matrix(distA['custom_matrix'], distB['custom_matrix'], distA['custom_keys'], distB['custom_keys'])

    # aggregate score (simple weighted sum using JS or wasserstein when present)
    # choose for each feature a scalar distance:
    def pick_scalar(feature_report, prefer='wasserstein'):
        if prefer == 'wasserstein' and 'wasserstein' in feature_report:
            return float(feature_report['wasserstein'])
        # fallback to l1_hist or l1 or js
        for k in ['l1_hist', 'l1', 'js']:
            if k in feature_report:
                return float(feature_report[k])
        # last resort
        vals = [v for v in feature_report.values() if isinstance(v, (int, float, np.floating))]
        return float(vals[0]) if vals else 0.0

    scalar_sum = 0.0
    total_w = 0.0
    scalar_sum += weights['traffic_density'] * pick_scalar(report['traffic_density'])
    total_w += weights['traffic_density']
    scalar_sum += weights['lane_widths'] * pick_scalar(report['lane_widths'])
    total_w += weights['lane_widths']
    scalar_sum += weights['lane_nums'] * pick_scalar(report['lane_nums'])
    total_w += weights['lane_nums']
    scalar_sum += weights['num_scenarios'] * pick_scalar(report['num_scenarios'])
    total_w += weights['num_scenarios']
    # map type use js
    scalar_sum += weights['map_type'] * report['map_type'].get('js', report['map_type'].get('l1', 0.0))
    total_w += weights['map_type']
    # custom_dist use js
    scalar_sum += weights['custom_dist'] * report['custom_dist'].get('js', report['custom_dist'].get('l1', 0.0))
    total_w += weights['custom_dist']

    aggregated = scalar_sum / total_w if total_w > 0 else float('nan')

    out = {
        'per_feature': report,
        'weights': weights,
        'aggregate_distance': aggregated
    }
    return out


# Quick demo using fake env configs
if __name__ == '__main__':
    cfgA = [{
        'map_config': {'GENERATE_TYPE': 'BIG_BLOCK_NUM', 'LANE_NUM': 3, 'LANE_WIDTH': 3.5},
        'traffic_density': 0.1,
        'num_scenarios': 100,
        'custom_dist': {'Straight': 0.6, 'StdInterSection': 0.4}
    } for _ in range(50)]

    cfgB = [{
        'map_config': {'GENERATE_TYPE': 'BIG_BLOCK_NUM', 'LANE_NUM': 2, 'LANE_WIDTH': 3.0},
        'traffic_density': 0.05,
        'num_scenarios': 80,
        'custom_dist': {'Straight': 1.0}
    } for _ in range(50)]

    dA = build_static_distributions(cfgA)
    dB = build_static_distributions(cfgB)
    r = compare_distributions(dA, dB)
    import json
    print(json.dumps(r['aggregate_distance'], indent=2))
