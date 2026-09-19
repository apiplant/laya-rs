//! Port of `rl_common.render_options` / `build_sequence`: turns a typed question + a state
//! (arbitrary JSON) into a token sequence with `[MASK]` markers, one per option.

use serde_json::Value;
use tokenizers::Tokenizer;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QType {
    Choice,
    Score,
    Noul,
}

impl QType {
    pub fn as_index(self) -> i64 {
        match self {
            QType::Choice => 0,
            QType::Score => 1,
            QType::Noul => 2,
        }
    }
    pub fn as_str(self) -> &'static str {
        match self {
            QType::Choice => "choice",
            QType::Score => "score",
            QType::Noul => "noul",
        }
    }
}

/// A typed question: choice options keep insertion order (key -> optional description);
/// score criteria are ordinal level descriptions in order; noul may carry {"true": .., "false": ..}.
#[derive(Debug, Clone)]
pub struct Question {
    pub qtype: QType,
    pub instructions: String,
    pub choice_criteria: Vec<(String, Option<String>)>,
    pub score_criteria: Vec<String>,
    pub noul_true: Option<String>,
    pub noul_false: Option<String>,
}

pub fn render_options(q: &Question) -> Vec<String> {
    match q.qtype {
        QType::Choice => q
            .choice_criteria
            .iter()
            .map(|(k, v)| match v {
                Some(v) if !v.is_empty() => format!("{}: {}", k, v),
                _ => k.clone(),
            })
            .collect(),
        QType::Score => q
            .score_criteria
            .iter()
            .enumerate()
            .map(|(i, c)| format!("level {}: {}", i, c))
            .collect(),
        QType::Noul => {
            let f = q.noul_false.clone().unwrap_or_else(|| "no, the statement does not hold".to_string());
            let t = q.noul_true.clone().unwrap_or_else(|| "yes, the statement holds".to_string());
            vec![format!("false: {}", f), format!("true: {}", t)]
        }
    }
}

pub fn serialize_state(state: &Value) -> String {
    match state {
        Value::String(s) => s.clone(),
        other => serde_json::to_string(other).unwrap_or_default(),
    }
}

pub struct BuiltSequence {
    pub ids: Vec<u32>,
    pub markers: Vec<usize>,
}

/// The subset of a tokenizer's `tokenizer_config.json` we need: real special-token strings
/// (ModernBERT uses `[CLS]`/`[SEP]`/`[MASK]`/`[PAD]`, mmBERT uses `<bos>`/`<eos>`/`<mask>`/`<pad>`).
pub struct SpecialTokens {
    pub cls: String,
    pub sep: String,
    pub mask: String,
    pub pad: String,
}

/// [CLS] <type> question: instructions [SEP] [MASK] opt0 [MASK] opt1 ... [SEP] state [SEP]
pub fn build_sequence(
    tok: &Tokenizer,
    special: &SpecialTokens,
    state: &Value,
    q: &Question,
    max_len: usize,
    head_max_len: usize,
) -> BuiltSequence {
    let mask_str = special.mask.as_str();
    let mask_id = tok.token_to_id(mask_str).unwrap_or_else(|| panic!("tokenizer missing mask token {mask_str:?}"));
    let cls_id = tok.token_to_id(&special.cls).unwrap_or_else(|| panic!("tokenizer missing cls token {:?}", special.cls));
    let sep_id = tok.token_to_id(&special.sep).unwrap_or_else(|| panic!("tokenizer missing sep token {:?}", special.sep));

    let opts = render_options(q);
    let ins = q.instructions.replace(mask_str, " ");
    let head_text = format!("{} question: {}", q.qtype.as_str(), ins);
    let mut head_ids = encode_ids(tok, &head_text);

    let mut opt_ids: Vec<Vec<u32>> = Vec::with_capacity(opts.len());
    for opt in &opts {
        let text = format!(" {}", opt.replace(mask_str, " "));
        let mut ids = encode_ids(tok, &text);
        ids.truncate(48);
        let mut full = vec![mask_id];
        full.extend(ids.drain(..));
        opt_ids.push(full);
    }

    let mut opt_len_sum: usize = opt_ids.iter().map(|o| o.len()).sum();
    let mut opt_budget = head_max_len as i64 - opt_len_sum as i64;
    if opt_budget < 16 {
        let per = ((head_max_len as i64 - 16) / (opt_ids.len().max(1) as i64)).max(4) as usize;
        for o in opt_ids.iter_mut() {
            if o.len() > per {
                o.truncate(per);
            }
        }
        opt_len_sum = opt_ids.iter().map(|o| o.len()).sum();
        opt_budget = head_max_len as i64 - opt_len_sum as i64;
    }
    let head_keep = opt_budget.max(8) as usize;
    if head_ids.len() > head_keep {
        head_ids.truncate(head_keep);
    }

    let mut ids: Vec<u32> = Vec::with_capacity(max_len);
    ids.push(cls_id);
    ids.extend(head_ids);
    ids.push(sep_id);

    let mut markers = Vec::with_capacity(opt_ids.len());
    for o in &opt_ids {
        markers.push(ids.len());
        ids.extend(o.iter().copied());
    }
    ids.push(sep_id);

    let room = max_len.saturating_sub(ids.len() + 1);
    let state_text = serialize_state(state).replace(mask_str, " ");
    let mut state_ids = encode_ids(tok, &state_text);
    if state_ids.len() > room {
        state_ids.truncate(room);
    }
    ids.extend(state_ids);
    ids.push(sep_id);

    ids.truncate(max_len);
    markers.retain(|&m| m < max_len);

    BuiltSequence { ids, markers }
}

fn encode_ids(tok: &Tokenizer, text: &str) -> Vec<u32> {
    tok.encode(text, false).map(|e| e.get_ids().to_vec()).unwrap_or_default()
}
