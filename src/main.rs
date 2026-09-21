use std::collections::BTreeMap;
use std::path::PathBuf;

use clap::{Parser, Subcommand};
use serde_json::{json, Value};

use laya::agent::answer_to_json;
use laya::batching::raw_question_to_question;
use laya::{model_path, route, Checkpoint, QType, Question, RLAgent, RlcdConfig, Trainer};

/// The `model_path::VariantDef` key routed to for a checkpoint (see `router::Checkpoint`).
fn variant_key(checkpoint: Checkpoint) -> &'static str {
    match checkpoint {
        Checkpoint::English => "typed-decisions",
        Checkpoint::Multilingual => "multilingual",
    }
}

/// Rust reimplementation of Laya's typed-decision inference + RLCD training.
// clap derives the app name from `CARGO_PKG_NAME` by default, which is the
// crate/package name `laya-rs` — pinned explicitly so `laya --version` keeps
// reporting the binary name, not the package name. `version` isn't added
// automatically without clap's `cargo` feature, so it's spelled out too.
#[derive(Parser)]
#[command(name = "laya", version)]
struct Args {
    #[command(subcommand)]
    command: Option<Command>,

    /// Root of a `convaiinnovations/laya`-shaped checkpoint family — a local clone of that
    /// hub repo, or any directory laid out the same way (a `typed-decisions/` and a
    /// `multilingual/` subfolder, each a full checkpoint) — checked for the routed
    /// checkpoint before falling back to a standalone per-variant download. Optional; with
    /// neither this nor `--model`/`LAYA_MODEL` set, checkpoints are downloaded into
    /// `~/.cache/laya-rs` on first use (see `laya::model_path`).
    #[arg(long, env = "LAYA_MODELS_ROOT", global = true)]
    models_root: Option<PathBuf>,

    /// State text/body to run the router + example questions against.
    #[arg(default_value = "We were billed twice for March. Please refund the duplicate.")]
    body: String,
}

#[derive(Subcommand)]
enum Command {
    /// Ask a single `choice` question against a state string (quick manual smoke test).
    Ask {
        /// Checkpoint directory to use as-is, bypassing routing/`--models-root` entirely.
        #[arg(long, env = "LAYA_MODEL")]
        model: Option<PathBuf>,
        #[arg(long)]
        state: String,
        #[arg(long)]
        question: String,
        #[arg(long = "option")]
        options: Vec<String>,
    },
    /// Answer a batch of typed questions against one state
    /// ({"state": ..., "questions": {qid: {type, instructions, criteria}}})
    /// and write {qid: {...answer}} JSON to `output`.
    ///
    /// This is the same state+questions/answers shape `WasmAgent::ask` uses.
    /// A jev-questions-style file ({section: {state, questions}}) is not
    /// read directly — batch multi-section files with `scripts/jev_batch.py`,
    /// which calls this subcommand once per section.
    Answer {
        input: String,
        output: String,
        /// Checkpoint directory to use as-is, bypassing `--model-variant`/`--models-root`.
        #[arg(long, env = "LAYA_MODEL")]
        model: Option<PathBuf>,
        /// Which `model_path::VARIANTS` checkpoint to use when `--model` isn't given.
        #[arg(long, default_value = "typed-decisions")]
        model_variant: String,
        #[arg(long)]
        only: Option<String>,
    },
    /// Run RLCD training over a JSONL dataset (see `batching::Record` for the line schema).
    Train {
        model_dir: String,
        dataset: String,
        #[arg(long, default_value_t = 1)]
        epochs: usize,
        #[arg(long, default_value_t = 1e-5)]
        lr: f64,
        #[arg(long, default_value_t = 8)]
        group_size: usize,
        #[arg(long, default_value_t = 1.0)]
        sigma: f32,
        #[arg(long)]
        save_to: Option<String>,
    },
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    if let Some(Command::Ask { model, state, question, options }) = &args.command {
        let checkpoint = route(state);
        let dir = model_path::resolve(variant_key(checkpoint), model.clone(), Some(variant_key(checkpoint)), args.models_root.as_deref())?;
        eprintln!("routed to: {:?} ({})", checkpoint, dir.display());
        let agent = RLAgent::load(&dir)?;
        let q = Question {
            qtype: QType::Choice,
            instructions: question.clone(),
            choice_criteria: options.iter().map(|o| (o.clone(), None)).collect(),
            score_criteria: vec![],
            noul_true: None,
            noul_false: None,
        };
        let answers = agent.system_one(&json!(state), &[("answer".to_string(), q)])?;
        for (_, answer) in answers {
            if let laya::Answer::Choice { choice, probabilities, confidence, act_probability } = answer {
                println!("choice={choice} confidence={confidence:.4} act_p={act_probability:.4}");
                for (k, v) in probabilities {
                    println!("    {k}: {v:.4}");
                }
            }
        }
        return Ok(());
    }

    if let Some(Command::Answer { input, output, model, model_variant, only }) = &args.command {
        let raw: Value = serde_json::from_str(&std::fs::read_to_string(input)?)?;
        let dir = model_path::resolve(model_variant, model.clone(), Some(model_variant), args.models_root.as_deref())?;
        let agent = RLAgent::load(&dir)?;

        let state = raw.get("state").ok_or_else(|| anyhow::anyhow!("missing state"))?.clone();
        let questions_obj =
            raw.get("questions").and_then(|v| v.as_object()).ok_or_else(|| anyhow::anyhow!("missing questions"))?;

        let mut questions: Vec<(String, Question)> = Vec::with_capacity(questions_obj.len());
        for (qid, qval) in questions_obj {
            if let Some(o) = only {
                if o != qid {
                    continue;
                }
            }
            let raw_q = serde_json::from_value(qval.clone())?;
            questions.push((qid.clone(), raw_question_to_question(&raw_q)));
        }
        anyhow::ensure!(!questions.is_empty(), "no questions to answer");

        let t0 = std::time::Instant::now();
        let answers = agent.system_one(&state, &questions)?;
        let elapsed_ms = t0.elapsed().as_secs_f64() * 1e3;
        eprintln!("{} questions in {elapsed_ms:.2} ms ({:.2} ms/question)", questions.len(), elapsed_ms / questions.len() as f64);

        let out: BTreeMap<String, Value> = answers.into_iter().map(|(qid, answer)| (qid, answer_to_json(answer))).collect();
        std::fs::write(output, serde_json::to_string_pretty(&out)?)?;
        println!("wrote {output}");
        return Ok(());
    }

    if let Some(Command::Train { model_dir, dataset, epochs, lr, group_size, sigma, save_to }) = args.command {
        let rlcd = RlcdConfig { lr, group_size, noise_sigma: sigma, ..Default::default() };
        let mut trainer = Trainer::load(&model_dir, &rlcd)?;
        trainer.train_jsonl(&dataset, epochs, &rlcd)?;
        let out = save_to.unwrap_or_else(|| format!("{model_dir}/model.trained.safetensors"));
        trainer.save(&out)?;
        println!("saved trained weights to {out}");
        return Ok(());
    }

    let checkpoint = route(&args.body);
    let model_dir = model_path::resolve(variant_key(checkpoint), None, Some(variant_key(checkpoint)), args.models_root.as_deref())?;
    println!("routed to: {:?} ({})", checkpoint, model_dir.display());

    let agent = RLAgent::load(&model_dir)?;

    let state = json!({
        "subject": "Duplicate charge on invoice 4411",
        "body": args.body,
    });

    let questions = vec![
        (
            "department".to_string(),
            Question {
                qtype: QType::Choice,
                instructions: "Which team should handle this?".to_string(),
                choice_criteria: vec![
                    ("billing".to_string(), Some("invoices, payments, refunds".to_string())),
                    ("technical".to_string(), Some("bugs and outages".to_string())),
                    ("sales".to_string(), Some("pricing".to_string())),
                ],
                score_criteria: vec![],
                noul_true: None,
                noul_false: None,
            },
        ),
        (
            "urgency".to_string(),
            Question {
                qtype: QType::Score,
                instructions: "How urgent is this?".to_string(),
                choice_criteria: vec![],
                score_criteria: vec!["not urgent".to_string(), "soon".to_string(), "blocking".to_string()],
                noul_true: None,
                noul_false: None,
            },
        ),
        (
            "churn_risk".to_string(),
            Question {
                qtype: QType::Noul,
                instructions: "Does the user threaten to cancel?".to_string(),
                choice_criteria: vec![],
                score_criteria: vec![],
                noul_true: None,
                noul_false: None,
            },
        ),
    ];

    let answers = agent.system_one(&state, &questions)?;
    for (qid, answer) in answers {
        match answer {
            laya::Answer::Choice { choice, probabilities, confidence, act_probability } => {
                println!("{qid}: choice={choice} confidence={confidence:.4} act_p={act_probability:.4}");
                for (k, v) in probabilities {
                    println!("    {k}: {v:.4}");
                }
            }
            laya::Answer::Score { score, legend, probabilities, confidence, act_probability } => {
                println!("{qid}: score={score:.4} confidence={confidence:.4} act_p={act_probability:.4}");
                for (level, p) in legend.iter().zip(probabilities.iter()) {
                    println!("    {level}: {p:.4}");
                }
            }
            laya::Answer::Noul { noul, act_probability } => {
                println!("{qid}: noul={noul:.4} act_p={act_probability:.4}");
            }
        }
    }

    Ok(())
}
