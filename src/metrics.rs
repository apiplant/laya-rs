//! Numpy-free port of `rl_common`'s reward + calibration/ranking metrics.

/// Strictly proper reward: log score + spherical score for every question type, plus a ranked
/// probability score penalty for ordinal (`score`) questions. Mirrors `rl_common.proper_reward`,
/// operating on a single item's already-softmaxed distribution `q` and its (possibly soft) target.
pub fn proper_reward(q: &[f32], target: &[f32], is_score: bool, w_sph: f32, w_rps: f32, log_floor: f32) -> f32 {
    let k = q.len();
    debug_assert_eq!(k, target.len());
    let logq: Vec<f32> = q.iter().map(|&v| v.max(1e-12).ln().max(log_floor)).collect();
    let log_score: f32 = target.iter().zip(&logq).map(|(&t, &lq)| t * lq).sum();
    let qnorm: f32 = q.iter().map(|&v| v * v).sum::<f32>().sqrt().max(1e-9);
    let sph: f32 = target.iter().zip(q).map(|(&t, &v)| t * v).sum::<f32>() / qnorm;
    let mut r = log_score + w_sph * sph;
    if is_score && k >= 2 {
        let mut cdf_q = 0f32;
        let mut cdf_t = 0f32;
        let mut rps = 0f32;
        for i in 0..k {
            cdf_q += q[i];
            cdf_t += target[i];
            rps += (cdf_q - cdf_t).powi(2);
        }
        rps /= (k - 1) as f32;
        r -= w_rps * rps;
    }
    r
}

pub fn confidence_from_probs(p: &[f32]) -> f32 {
    let k = p.len();
    if k < 2 {
        return 1.0;
    }
    let ent: f32 = -p.iter().map(|&v| v * v.max(1e-12).ln()).sum::<f32>();
    1.0 - ent / (k as f32).ln()
}

/// Expected calibration error over 15 equal-width confidence bins.
pub fn ece_score(conf: &[f32], correct: &[f32], bins: usize) -> f32 {
    if conf.is_empty() {
        return f32::NAN;
    }
    let n = conf.len() as f32;
    let mut e = 0f32;
    for b in 0..bins {
        let lo = b as f32 / bins as f32;
        let hi = (b + 1) as f32 / bins as f32;
        let sel: Vec<usize> = (0..conf.len()).filter(|&i| conf[i] > lo && conf[i] <= hi).collect();
        if !sel.is_empty() {
            let mean_conf: f32 = sel.iter().map(|&i| conf[i]).sum::<f32>() / sel.len() as f32;
            let mean_correct: f32 = sel.iter().map(|&i| correct[i]).sum::<f32>() / sel.len() as f32;
            e += (sel.len() as f32 / n) * (mean_conf - mean_correct).abs();
        }
    }
    e
}

/// Area under the ROC curve, ties averaged.
pub fn auroc(scores: &[f32], labels: &[u8]) -> f32 {
    let n = scores.len();
    let n_pos = labels.iter().filter(|&&l| l == 1).count();
    let n_neg = n - n_pos;
    if n_pos == 0 || n_neg == 0 {
        return f32::NAN;
    }
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| scores[a].partial_cmp(&scores[b]).unwrap());
    let mut ranks = vec![0f32; n];
    let mut i = 0;
    while i < n {
        let mut j = i;
        while j + 1 < n && scores[order[j + 1]] == scores[order[i]] {
            j += 1;
        }
        let avg_rank = (i + j) as f32 / 2.0 + 1.0;
        for &idx in &order[i..=j] {
            ranks[idx] = avg_rank;
        }
        i = j + 1;
    }
    let sum_pos_ranks: f32 = (0..n).filter(|&i| labels[i] == 1).map(|i| ranks[i]).sum();
    let np = n_pos as f32;
    let nn = n_neg as f32;
    (sum_pos_ranks - np * (np + 1.0) / 2.0) / (np * nn)
}

pub fn spearman(a: &[f32], b: &[f32]) -> f32 {
    if a.len() < 3 {
        return f32::NAN;
    }
    let rank = |v: &[f32]| -> Vec<f32> {
        let mut order: Vec<usize> = (0..v.len()).collect();
        order.sort_by(|&i, &j| v[i].partial_cmp(&v[j]).unwrap());
        let mut r = vec![0f32; v.len()];
        for (pos, &idx) in order.iter().enumerate() {
            r[idx] = pos as f32;
        }
        r
    };
    let ra = rank(a);
    let rb = rank(b);
    let mean = |v: &[f32]| v.iter().sum::<f32>() / v.len() as f32;
    let (ma, mb) = (mean(&ra), mean(&rb));
    let std = |v: &[f32], m: f32| (v.iter().map(|&x| (x - m).powi(2)).sum::<f32>() / v.len() as f32).sqrt();
    let (sa, sb) = (std(&ra, ma), std(&rb, mb));
    if sa == 0.0 || sb == 0.0 {
        return f32::NAN;
    }
    let cov: f32 = ra.iter().zip(&rb).map(|(&x, &y)| (x - ma) * (y - mb)).sum::<f32>() / ra.len() as f32;
    cov / (sa * sb)
}

/// Area under the risk-coverage curve (lower is better).
pub fn aurc(conf: &[f32], correct: &[f32]) -> f32 {
    if conf.is_empty() {
        return f32::NAN;
    }
    let n = conf.len();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| conf[b].partial_cmp(&conf[a]).unwrap());
    let mut cum_err = 0f32;
    let mut sum = 0f32;
    for (i, &idx) in order.iter().enumerate() {
        cum_err += 1.0 - correct[idx];
        sum += cum_err / (i + 1) as f32;
    }
    sum / n as f32
}
