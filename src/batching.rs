//! Port of `rl_common`'s record -> model-input pipeline: `encode_record`, `collate_items`,
//! `pack_groups`, `make_token_batches`, `episode_prefix_lengths`, `temp_bucket`.

use rand::seq::SliceRandom;
use rand::Rng;
use serde::Deserialize;
use serde_json::Value;

use crate::schema::{build_sequence, render_options, QType, Question, SpecialTokens};
use tokenizers::Tokenizer;

/// One model-ready training item: a token sequence with option-marker positions and a
/// (possibly soft) target distribution over those options.
#[derive(Debug, Clone)]
pub struct Item {
    pub ids: Vec<u32>,
    pub markers: Vec<usize>,
    pub qtype: QType,
    pub target: Vec<f32>,
    pub label: i64, // -1 if unknown (e.g. episodes before the outcome, or unlabeled `score`)
    pub episode: bool,
    pub ep_step: usize,
    pub ep_len: usize,
    pub rec_uid: i64,
}

/// A stored training record: either a single-turn state with one or more typed questions, or a
/// multi-turn conversation episode (`kind: "episode"`) evaluated at several prefix lengths.
#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum Record {
    Episode {
        #[serde(rename = "kind")]
        _kind: String,
        ep: Episode,
        qs: Vec<RawQuestion>,
        #[serde(default)]
        src: String,
    },
    Plain {
        state: Value,
        qs: Vec<QuestionWithLabel>,
        #[serde(default)]
        src: String,
    },
}

#[derive(Debug, Deserialize)]
pub struct Episode {
    pub ctx: Value,
    pub turns: Vec<Value>,
    pub y: f32, // outcome in [0,1] (churn / escalation / etc.)
}

#[derive(Debug, Deserialize)]
pub struct RawQuestion {
    #[serde(rename = "type")]
    pub t: String,
    pub instructions: Value,
    #[serde(default)]
    pub criteria: Option<Value>,
}

#[derive(Debug, Deserialize)]
pub struct QuestionWithLabel {
    #[serde(flatten)]
    pub q: RawQuestion,
    pub y: Option<usize>,
    #[serde(default)]
    pub soft: Option<Vec<f32>>,
}

pub fn raw_question_to_question(rq: &RawQuestion) -> Question {
    let qtype = match rq.t.as_str() {
        "choice" => QType::Choice,
        "score" => QType::Score,
        "noul" => QType::Noul,
        other => panic!("unknown question type {other:?}"),
    };
    let instructions = match &rq.instructions {
        Value::String(s) => s.clone(),
        other => other.to_string(),
    };
    let mut q = Question {
        qtype,
        instructions,
        choice_criteria: vec![],
        score_criteria: vec![],
        noul_true: None,
        noul_false: None,
    };
    match (&qtype, &rq.criteria) {
        (QType::Choice, Some(Value::Object(map))) => {
            q.choice_criteria = map.iter().map(|(k, v)| (k.clone(), v.as_str().map(|s| s.to_string()))).collect();
        }
        (QType::Choice, Some(Value::Array(arr))) => {
            q.choice_criteria = arr.iter().filter_map(|v| v.as_str()).map(|s| (s.to_string(), None)).collect();
        }
        (QType::Score, Some(Value::Array(arr))) => {
            q.score_criteria = arr.iter().filter_map(|v| v.as_str()).map(|s| s.to_string()).collect();
        }
        (QType::Noul, Some(Value::Object(map))) => {
            q.noul_true = map.get("true").and_then(|v| v.as_str()).map(|s| s.to_string());
            q.noul_false = map.get("false").and_then(|v| v.as_str()).map(|s| s.to_string());
        }
        _ => {}
    }
    q
}

/// `rl_common.episode_prefix_lengths`.
pub fn episode_prefix_lengths(n_turns: usize, max_prefixes: usize) -> Vec<usize> {
    if n_turns <= max_prefixes {
        return (1..=n_turns).collect();
    }
    let mut set = std::collections::BTreeSet::new();
    for i in 0..max_prefixes {
        let x = 1.0 + i as f64 * (n_turns as f64 - 1.0) / (max_prefixes as f64 - 1.0).max(1.0);
        set.insert(x.round() as usize);
    }
    set.into_iter().collect()
}

/// `rl_common.encode_record`. `rng` shuffles option order for non-`score` questions when `train`.
pub fn encode_record(
    tok: &Tokenizer,
    special: &SpecialTokens,
    rec: &Record,
    max_len: usize,
    head_max_len: usize,
    max_prefixes: usize,
    rec_uid: i64,
    rng: Option<&mut impl Rng>,
    train: bool,
) -> Vec<Item> {
    match rec {
        Record::Episode { ep, qs, .. } => {
            let q = raw_question_to_question(&qs[0]);
            let lens = episode_prefix_lengths(ep.turns.len(), max_prefixes);
            let ep_len = lens.len();
            let mut items = Vec::with_capacity(ep_len);
            for (step, &t) in lens.iter().enumerate() {
                let mut state = ep.ctx.clone();
                if let Value::Object(ref mut map) = state {
                    map.insert("conversation".to_string(), Value::Array(ep.turns[..t].to_vec()));
                }
                let built = build_sequence(tok, special, &state, &q, max_len, head_max_len);
                if built.markers.len() != 2 {
                    continue;
                }
                items.push(Item {
                    ids: built.ids,
                    markers: built.markers,
                    qtype: QType::Noul,
                    target: vec![1.0 - ep.y, ep.y],
                    label: ep.y.round() as i64,
                    episode: true,
                    ep_step: step,
                    ep_len,
                    rec_uid,
                });
            }
            items
        }
        Record::Plain { state, qs, .. } => {
            let mut items = Vec::with_capacity(qs.len());
            let mut rng = rng;
            for qwl in qs {
                let q = raw_question_to_question(&qwl.q);
                let k = render_options(&q).len();
                let mut order: Vec<usize> = (0..k).collect();
                if train && q.qtype != QType::Score {
                    if let Some(r) = rng.as_deref_mut() {
                        order.shuffle(r);
                    }
                }
                let target_base: Vec<f32> = if let Some(soft) = &qwl.soft {
                    soft.clone()
                } else {
                    let y = qwl.y.unwrap_or(0);
                    (0..k).map(|i| if i == y { 1.0 } else { 0.0 }).collect()
                };

                // Reorder the question's options to match `order`, matching `build_sequence`'s
                // `option_order` parameter (python builds markers directly in shuffled order).
                let q_reordered = reorder_question(&q, &order);
                let built = build_sequence(tok, special, state, &q_reordered, max_len, head_max_len);
                if built.markers.len() != k {
                    continue; // options did not fit; skip rather than train on a truncated answer space
                }
                let target: Vec<f32> = order.iter().map(|&i| target_base[i]).collect();
                let label = qwl.y.map(|y| order.iter().position(|&i| i == y).unwrap() as i64).unwrap_or(-1);
                items.push(Item {
                    ids: built.ids,
                    markers: built.markers,
                    qtype: q.qtype,
                    target,
                    label,
                    episode: false,
                    ep_step: 0,
                    ep_len: 1,
                    rec_uid,
                });
            }
            items
        }
    }
}

fn reorder_question(q: &Question, order: &[usize]) -> Question {
    let mut r = q.clone();
    match q.qtype {
        QType::Choice => r.choice_criteria = order.iter().map(|&i| q.choice_criteria[i].clone()).collect(),
        QType::Score => {} // score options are never shuffled (ordinal)
        QType::Noul => {}  // always exactly [false, true]
    }
    r
}

/// Padded batch tensors' raw ingredients (kept as plain Vecs so callers decide device/dtype).
pub struct Collated {
    pub ids: Vec<u32>,          // [n, l] row-major
    pub attention_mask: Vec<i64>, // [n, l]
    pub marker_pos: Vec<i64>,   // [n, kmax]
    pub marker_mask: Vec<f32>,  // [n, kmax], 1.0 = real marker
    pub target: Vec<f32>,       // [n, kmax]
    pub qtype: Vec<i64>,        // [n]
    pub label: Vec<i64>,        // [n]
    pub n: usize,
    pub l: usize,
    pub kmax: usize,
}

pub fn collate_items(items: &[&Item], pad_id: u32) -> Collated {
    let n = items.len();
    let l = items.iter().map(|it| it.ids.len()).max().unwrap_or(0);
    let kmax = items.iter().map(|it| it.markers.len()).max().unwrap_or(0);
    let mut ids = vec![pad_id; n * l];
    let mut attention_mask = vec![0i64; n * l];
    let mut marker_pos = vec![0i64; n * kmax];
    let mut marker_mask = vec![0f32; n * kmax];
    let mut target = vec![0f32; n * kmax];
    let mut qtype = Vec::with_capacity(n);
    let mut label = Vec::with_capacity(n);
    for (r, it) in items.iter().enumerate() {
        for (c, &id) in it.ids.iter().enumerate() {
            ids[r * l + c] = id;
            attention_mask[r * l + c] = 1;
        }
        for (c, &m) in it.markers.iter().enumerate() {
            marker_pos[r * kmax + c] = m as i64;
            marker_mask[r * kmax + c] = 1.0;
        }
        for (c, &t) in it.target.iter().enumerate() {
            target[r * kmax + c] = t;
        }
        qtype.push(it.qtype.as_index());
        label.push(it.label);
    }
    Collated { ids, attention_mask, marker_pos, marker_mask, target, qtype, label, n, l, kmax }
}

/// `rl_common.pack_groups`: split sampled record groups into sub-batches under a padded-token
/// budget, keeping every item of a record (all its questions, or all its episode prefixes) together.
pub fn pack_groups(mut groups: Vec<Vec<Item>>, max_tokens: usize, max_seqs: usize) -> Vec<Vec<Vec<Item>>> {
    groups.retain(|g| !g.is_empty());
    groups.sort_by_key(|g| g.iter().map(|it| it.ids.len()).max().unwrap_or(0));
    let mut subs = Vec::new();
    let mut cur: Vec<Vec<Item>> = Vec::new();
    let (mut cur_max, mut cur_n) = (0usize, 0usize);
    for g in groups {
        let g_max = g.iter().map(|it| it.ids.len()).max().unwrap_or(0);
        let g_n = g.len();
        if g_max * g_n > max_tokens {
            let step = (max_tokens / g_max.max(1)).max(1);
            let mut s = 0;
            while s < g_n {
                let end = (s + step).min(g_n);
                subs.push(vec![g[s..end].to_vec()]);
                s = end;
            }
            continue;
        }
        let new_max = cur_max.max(g_max);
        let new_n = cur_n + g_n;
        if !cur.is_empty() && (new_max * new_n > max_tokens || new_n > max_seqs) {
            subs.push(std::mem::take(&mut cur));
            cur_max = g_max;
            cur_n = g_n;
        } else {
            cur_max = new_max;
            cur_n = new_n;
        }
        cur.push(g);
    }
    if !cur.is_empty() {
        subs.push(cur);
    }
    subs
}

/// `rl_common.make_token_batches`: length-bucketed batches of record indices under a padded-token
/// budget, shuffled at both the chunk and final-batch level.
pub fn make_token_batches(
    lengths: &[usize],
    nseq: &[usize],
    max_tokens: usize,
    max_seqs: usize,
    rng: &mut impl Rng,
    chunk: usize,
) -> Vec<Vec<usize>> {
    let mut order: Vec<usize> = (0..lengths.len()).collect();
    order.shuffle(rng);
    let mut batches: Vec<Vec<usize>> = Vec::new();
    for chunk_slice in order.chunks(chunk) {
        let mut part = chunk_slice.to_vec();
        part.sort_by_key(|&i| lengths[i]);
        let mut cur = Vec::new();
        let (mut cur_max, mut cur_n) = (0usize, 0usize);
        for &i in &part {
            let (ln, ns) = (lengths[i], nseq[i]);
            let new_max = cur_max.max(ln);
            let new_n = cur_n + ns;
            if !cur.is_empty() && (new_max * new_n > max_tokens || new_n > max_seqs) {
                batches.push(std::mem::take(&mut cur));
                cur_max = ln;
                cur_n = ns;
            } else {
                cur_max = new_max;
                cur_n = new_n;
            }
            cur.push(i);
        }
        if !cur.is_empty() {
            batches.push(cur);
        }
    }
    batches.shuffle(rng);
    batches
}

/// `rl_common.temp_bucket`: key for per-cardinality temperature fitting.
pub fn temp_bucket(qtype: QType, k: usize) -> String {
    let size = if k <= 2 { "2" } else if k <= 5 { "3-5" } else if k <= 10 { "6-10" } else { "11+" };
    format!("{}:{}", qtype.as_str(), size)
}

/// `rl_common.td_lambda_targets`: TD(lambda) soft targets for conversation prefixes.
/// `p_true` is P(outcome=true) predicted at each item (same order as `items`); `items` must all
/// share `rec_uid` grouping and be indexable by the same order as `p_true`.
pub fn td_lambda_targets(items: &[Item], p_true: &[f32], lam: f32) -> Vec<[f32; 2]> {
    let mut targets: Vec<[f32; 2]> = items.iter().map(|it| [it.target[0], it.target[1]]).collect();
    let mut groups: std::collections::HashMap<i64, Vec<usize>> = std::collections::HashMap::new();
    for (i, it) in items.iter().enumerate() {
        if it.episode {
            groups.entry(it.rec_uid).or_default().push(i);
        }
    }
    for (_g, mut idx) in groups {
        idx.sort_by_key(|&i| items[i].ep_step);
        let last = *idx.last().unwrap();
        let mut g = targets[last][1];
        for j in (0..idx.len()).rev() {
            if j < idx.len() - 1 {
                g = (1.0 - lam) * p_true[idx[j + 1]] + lam * g;
            }
            targets[idx[j]] = [1.0 - g, g];
        }
    }
    targets
}
