use std::collections::BTreeMap;

use clap::{Parser, Subcommand};
use serde_json::{json, Value};

use laya::batching::raw_question_to_question;
use laya::{route, Answer, Checkpoint, QType, Question, RLAgent, RlcdConfig, Trainer};

/// Rust reimplementation of Laya's typed-decision inference + RLCD training.
#[derive(Parser)]
struct Args {
    #[command(subcommand)]
    command: Option<Command>,

    /// Root of the model family (contains model.safetensors/tokenizer/ at top level, plus
    /// multilingual/ and typed-decisions/ subfolders), e.g. /mnt/extra/ai/laya
    #[arg(default_value = "/mnt/extra/ai/laya")]
    models_root: String,

    /// State text/body to run the router + example questions against.
    #[arg(default_value = "We were billed twice for March. Please refund the duplicate.")]
    body: String,
}

#[derive(Subcommand)]
enum Command {
    /// Ask a single `choice` question against a state string (quick manual smoke test).
    Ask {
        #[arg(long, default_value = "/mnt/extra/ai/laya")]
        model_dir: String,
        #[arg(long)]
        state: String,
        #[arg(long)]
        question: String,
        #[arg(long = "option")]
        options: Vec<String>,
    },
    /// Answer a jev-questions-style file ({section: {state, questions}}) and write
    /// gliner-answers-style JSON ({section: {qid: {...answer}}}) to `output`.
    Jev {
        input: String,
        output: String,
        #[arg(long, default_value = "/mnt/extra/ai/laya/typed-decisions")]
        model_dir: String,
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

    if let Some(Command::Ask { model_dir, state, question, options }) = &args.command {
        let checkpoint = route(state);
        let dir = match checkpoint {
            Checkpoint::English => model_dir.clone(),
            Checkpoint::Multilingual => format!("{model_dir}/multilingual"),
        };
        eprintln!("routed to: {:?} ({dir})", checkpoint);
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

    if let Some(Command::Jev { input, output, model_dir, only }) = &args.command {
        let raw: Value = serde_json::from_str(&std::fs::read_to_string(input)?)?;
        let agent = RLAgent::load(model_dir)?;
        let sections = raw.as_object().ok_or_else(|| anyhow::anyhow!("expected top-level object"))?;

        let mut out: BTreeMap<String, BTreeMap<String, Value>> = BTreeMap::new();
        for (section, body) in sections {
            let state = body.get("state").ok_or_else(|| anyhow::anyhow!("{section}: missing state"))?.clone();
            let questions_obj = body.get("questions").and_then(|v| v.as_object()).ok_or_else(|| anyhow::anyhow!("{section}: missing questions"))?;

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
            if questions.is_empty() {
                continue;
            }

            let answers = agent.system_one(&state, &questions)?;
            let mut section_out = BTreeMap::new();
            for (qid, answer) in answers {
                let v = match answer {
                    Answer::Choice { choice, probabilities, confidence, act_probability } => json!({
                        "type": "choice", "choice": choice, "confidence": confidence, "act_probability": act_probability,
                        "probabilities": probabilities.into_iter().collect::<BTreeMap<_, _>>(),
                    }),
                    Answer::Score { score, legend, probabilities, confidence, act_probability } => json!({
                        "type": "score", "score": score, "confidence": confidence, "act_probability": act_probability,
                        "legend": legend.iter().enumerate().map(|(i, c)| (i.to_string(), c.clone())).collect::<BTreeMap<_, _>>(),
                        "probabilities": probabilities.iter().enumerate().map(|(i, p)| (i.to_string(), *p)).collect::<BTreeMap<_, _>>(),
                    }),
                    Answer::Noul { noul, act_probability } => json!({ "type": "noul", "noul": noul, "act_probability": act_probability }),
                };
                section_out.insert(qid, v);
            }
            eprintln!("{section}: answered {} questions", section_out.len());
            out.insert(section.clone(), section_out);
        }

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
    let model_dir = match checkpoint {
        Checkpoint::English => args.models_root.clone(),
        Checkpoint::Multilingual => format!("{}/multilingual", args.models_root),
    };
    println!("routed to: {:?} ({})", checkpoint, model_dir);

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
