/**
 * Feature: spec/features/strict-jev-classifier-request-schema-parse-validate.feature
 *
 * This test file validates the acceptance criteria defined in the feature file.
 * Scenarios map directly to Gherkin scenarios.
 *
 * All scenarios are weight-free: `parse_classifier_request` is pure over
 * bytes (no model, no IO). The "422 error" of the Gherkin steps is the
 * schema-error that this module returns (a `Vec<RequestError>`); the 422 HTTP
 * status mapping itself is JEV-004's error envelope (the router folds every
 * `RequestError` into the protocol envelope with code 422).
 */

use laya::server::request::{parse_classifier_request, ClassifierRequest, Context, RequestError};
use laya::QType;

fn parse(body: &str) -> Result<ClassifierRequest, Vec<RequestError>> {
    parse_classifier_request(body.as_bytes())
}

/// A choice-question body with exactly `n` criteria keys under question id `route`.
fn choice_body_with(n: usize) -> String {
    let keys: Vec<String> = (0..n).map(|i| format!("\"c{}\": null", i)).collect();
    let joined = keys.join(",");
    String::from(
        "{\"model\":\"m\",\"state\":\"x\",\"questions\":{\"route\":{\"type\":\"choice\",\"instructions\":\"Which one?\",\"criteria\":{",
    )
    + &joined
    + "}}}}"
}

/// Scenario: Valid mixed batch with unknown top-level fields parses cleanly
#[test]
fn valid_mixed_batch_with_unknown_top_level_fields_parses_cleanly() {
    // @step Given a request body with model "convaiinnovations/laya", state "We were billed twice for March. Please refund the duplicate.", and questions route (choice: billing/technical), urgency (score: 3 levels), refund (noul) plus the unknown top-level field stream: true
    let body = r#"{
      "model": "convaiinnovations/laya",
      "state": "We were billed twice for March. Please refund the duplicate.",
      "stream": true,
      "temperature": 0.7,
      "max_tokens": 128,
      "questions": {
        "route": {"type": "choice", "instructions": "Which team should handle this?",
                  "criteria": {"billing": "invoices, payments, refunds", "technical": null}},
        "urgency": {"type": "score", "instructions": "How urgent is this?",
                    "criteria": ["not urgent", "soon", "blocking"]},
        "refund": {"type": "noul", "instructions": "Does the customer explicitly request a refund?"}
      }
    }"#;

    // @step When the body is parsed against the Jev v1 classifier schema
    let req = parse(body).expect("valid mixed batch must parse (unknown top-level fields are ignored)");

    // @step Then parsing succeeds with the unknown top-level field ignored
    assert_eq!(req.model, "convaiinnovations/laya");
    match &req.context {
        Context::State(v) => assert_eq!(
            v,
            &serde_json::json!("We were billed twice for March. Please refund the duplicate.")
        ),
        Context::Messages(_) => panic!("expected a state context"),
    }

    // @step And the request keeps model "convaiinnovations/laya", the state context, and all three questions in request order
    let ids: Vec<&str> = req.questions.iter().map(|(id, _)| id.as_str()).collect();
    assert_eq!(ids, vec!["route", "urgency", "refund"], "request order must be preserved");
    assert_eq!(req.questions[0].1.qtype, QType::Choice);
    assert_eq!(req.questions[1].1.qtype, QType::Score);
    assert_eq!(req.questions[2].1.qtype, QType::Noul);
    let route = &req.questions[0].1;
    assert_eq!(
        route.choice_criteria,
        vec![
            ("billing".to_string(), Some("invoices, payments, refunds".to_string())),
            ("technical".to_string(), None),
        ]
    );
    assert_eq!(
        req.questions[1].1.score_criteria,
        vec!["not urgent", "soon", "blocking"]
    );
}

/// Scenario: Context XOR violations are rejected
#[test]
fn context_xor_violations_are_rejected() {
    // @step Given a request body that sets both state and messages (or both to null)
    let both = r#"{
      "model": "m",
      "state": "x",
      "messages": [{"role": "user", "content": "x"}],
      "questions": {"q1": {"type": "noul", "instructions": "x?"}}
    }"#;
    let both_null = r#"{
      "model": "m",
      "state": null,
      "messages": null,
      "questions": {"q1": {"type": "noul", "instructions": "x?"}}
    }"#;

    // @step When the body is parsed against the Jev v1 classifier schema
    let e_both = parse(both).expect_err("state AND messages must fail");
    let e_null = parse(both_null).expect_err("both-null context must fail");

    // @step Then parsing fails with a 422 error whose message is 'Provide exactly one of state or messages'
    assert!(
        e_both.iter().any(|e| e.message == "Provide exactly one of state or messages"),
        "state+messages: {e_both:?}"
    );
    assert!(
        e_null.iter().any(|e| e.message == "Provide exactly one of state or messages"),
        "both null: {e_null:?}"
    );

    // @step And the same body with state: {} (empty object) and one valid question instead parses successfully
    let empty_obj = r#"{
      "model": "m",
      "state": {},
      "questions": {"q1": {"type": "choice", "instructions": "x?", "criteria": {"a": null, "b": null}}}
    }"#;
    let req = parse(empty_obj).expect("an empty JSON object is a valid state");
    match req.context {
        Context::State(v) => assert_eq!(v, serde_json::json!({})),
        Context::Messages(_) => panic!("expected a state context"),
    }
}

/// Scenario: Non-text chat messages and reserved fields are rejected
#[test]
fn non_text_chat_messages_and_reserved_fields_are_rejected() {
    // @step Given a messages request whose message content is an image part array, or whose role is 'tool', or which carries a 'name' field
    let image = r#"{"model":"m","messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"https://example.com/a.png"}}]}],"questions":{"q1":{"type":"noul","instructions":"x?"}}}"#;
    let tool_role = r#"{"model":"m","messages":[{"role":"tool","content":"x"}],"questions":{"q1":{"type":"noul","instructions":"x?"}}}"#;
    let extra_field = r#"{"model":"m","messages":[{"role":"user","content":"x","name":"bot"}],"questions":{"q1":{"type":"noul","instructions":"x?"}}}"#;

    // @step When the body is parsed against the Jev v1 classifier schema
    let e_img = parse(image).expect_err("image-part content must fail (text messages only)");
    let e_tool = parse(tool_role).expect_err("tool role must fail (text messages only)");
    let e_extra = parse(extra_field).expect_err("extra message field must fail");

    // @step Then parsing fails with a 422 error identifying the offending message (text messages only)
    assert!(
        e_img.iter().any(|e| e.param.starts_with("messages[0].content")),
        "image content: {e_img:?}"
    );
    assert!(e_tool.iter().any(|e| e.param == "messages[0].role"), "tool role: {e_tool:?}");
    assert!(e_extra.iter().any(|e| e.param == "messages[0].name"), "extra field: {e_extra:?}");

    // @step And a request with options: {raw_logits: true} or a nonempty tools array also fails with a 422
    let raw_logits = r#"{"model":"m","state":"x","options":{"raw_logits":true},"questions":{"q1":{"type":"noul","instructions":"x?"}}}"#;
    let tools = r#"{"model":"m","state":"x","tools":[{"type":"function"}],"questions":{"q1":{"type":"noul","instructions":"x?"}}}"#;
    assert!(parse(raw_logits).is_err(), "options.raw_logits=true must be rejected (laya backend has no raw-logit diagnostics)");
    assert!(parse(tools).is_err(), "a nonempty tools array must be rejected");

    // @step And the same request with an empty tools: [] is accepted
    let empty_tools = r#"{"model":"m","state":"x","tools":[],"mm_processor_kwargs":{},"media_io_kwargs":{},"questions":{"q1":{"type":"noul","instructions":"x?"}}}"#;
    let req = parse(empty_tools).expect("empty reserved containers are accepted (accepted but no effect)");
    assert_eq!(req.model, "m");
}

/// Scenario: Choice criteria cardinality is enforced at 2-50
#[test]
fn choice_criteria_cardinality_is_enforced_at_2_50() {
    // @step Given a choice question with 51 criteria keys
    let body51 = choice_body_with(51);

    // @step When the body is parsed against the Jev v1 classifier schema
    let e = parse(&body51).expect_err("51 criteria keys must fail");

    // @step Then parsing fails with a 422 error whose param names the question id
    assert!(
        e.iter().any(|e| e.param == "questions.route.criteria"),
        "param must name the question id: {e:?}"
    );

    // @step And the same question with 50 criteria keys parses successfully
    let req = parse(&choice_body_with(50)).expect("50 criteria keys is the schema maximum and must parse");
    assert_eq!(req.questions[0].1.choice_criteria.len(), 50);
    // and 1 key is below the minimum
    let e_one = parse(&choice_body_with(1)).expect_err("a single candidate must fail");
    assert!(e_one.iter().any(|e| e.param == "questions.route.criteria"), "one key: {e_one:?}");

    // @step And a noul question whose criteria has only a 'true' key also parses successfully
    let noul_true_only = r#"{"model":"m","state":"x","questions":{"q1":{"type":"noul","instructions":"x?","criteria":{"true":"affirmative"}}}}"#;
    let req = parse(noul_true_only).expect("a noul with only a 'true' key must parse");
    let q = &req.questions[0].1;
    assert_eq!(q.noul_true.as_deref(), Some("affirmative"));
    assert_eq!(q.noul_false, None);
}

/// Scenario: Unknown question fields and noul criterion keys are rejected
#[test]
fn unknown_question_fields_and_noul_criterion_keys_are_rejected() {
    // @step Given a question object carrying a field other than type/instructions/criteria (e.g. "context")
    let body = r#"{"model":"m","state":"x","questions":{"route":{"type":"choice","instructions":"x?","context":"y","criteria":{"a":null,"b":null}}}}"#;

    // @step When the body is parsed against the Jev v1 classifier schema
    let e = parse(body).expect_err("an unknown question field must fail");

    // @step Then parsing fails with a 422 error whose param is the dotted path into that question
    assert!(
        e.iter().any(|e| e.param == "questions.route.context"),
        "param must be the dotted path into the question: {e:?}"
    );

    // @step And a noul question whose criteria contains a 'maybe' key also fails with a 422
    let noul_maybe = r#"{"model":"m","state":"x","questions":{"q1":{"type":"noul","instructions":"x?","criteria":{"maybe":"50%"}}}}"#;
    let e2 = parse(noul_maybe).expect_err("an unknown noul criterion key must fail");
    assert!(
        e2.iter().any(|e| e.param == "questions.q1.criteria"),
        "param must name the criteria of the noul question: {e2:?}"
    );
}

/// Scenario: State values that are not string/object/array are rejected
#[test]
fn state_values_that_are_not_string_object_array_are_rejected() {
    // @step Given a request body whose state is a bare number (or a boolean)
    let num = r#"{"model":"m","state":42,"questions":{"q1":{"type":"noul","instructions":"x?"}}}"#;
    let bool = r#"{"model":"m","state":true,"questions":{"q1":{"type":"noul","instructions":"x?"}}}"#;

    // @step When the body is parsed against the Jev v1 classifier schema
    let e_num = parse(num).expect_err("a bare number state must fail");
    let e_bool = parse(bool).expect_err("a bare boolean state must fail");

    // @step Then parsing fails with a 422 error naming the state field
    assert!(e_num.iter().any(|e| e.param == "state"), "number: {e_num:?}");
    assert!(e_bool.iter().any(|e| e.param == "state"), "boolean: {e_bool:?}");

    // @step And state: "" and state: [] with one valid question both parse successfully
    let empty_str = r#"{"model":"m","state":"","questions":{"q1":{"type":"noul","instructions":"x?"}}}"#;
    let empty_arr = r#"{"model":"m","state":[],"questions":{"q1":{"type":"noul","instructions":"x?"}}}"#;
    assert!(parse(empty_str).is_ok(), "an empty string state is valid");
    assert!(parse(empty_arr).is_ok(), "an empty array state is valid");
}

/// Scenario: Non-string instructions and criteria are canonicalized
#[test]
fn non_string_instructions_and_criteria_are_canonicalized() {
    // @step Given a choice question whose instructions is a JSON object {"topic": "routing"} and whose candidate description is an array
    let body = r#"{
      "model": "m",
      "state": "x",
      "questions": {
        "route": {
          "type": "choice",
          "instructions": {"topic": "routing", "b": 2, "a": 1},
          "criteria": {"billing": ["first", "second"], "technical": null}
        }
      }
    }"#;

    // @step When the body is parsed against the Jev v1 classifier schema
    let req = parse(body).expect("non-string entries must parse");

    // @step Then parsing succeeds and the object/array entries are serialized with the protocol canonical() deterministic JSON (compact, object keys sorted) into the question's prompt text
    let q = &req.questions[0].1;
    assert_eq!(
        q.instructions, r#"{"a":1,"b":2,"topic":"routing"}"#,
        "object instructions must be canonicalized (compact, object keys sorted)"
    );
    assert_eq!(
        q.choice_criteria,
        vec![
            ("billing".to_string(), Some(r#"["first","second"]"#.to_string())),
            ("technical".to_string(), None),
        ],
        "array criterion descriptions must be canonicalized; null stays None"
    );
    // a score with structured levels canonicalizes each level too
    let score = r#"{"model":"m","state":"x","questions":{"u":{"type":"score","instructions":"urgent?","criteria":[{"level":"low"},null]}}}"#;
    let req = parse(score).expect("structured score criteria must parse");
    assert_eq!(
        req.questions[0].1.score_criteria,
        vec![r#"{"level":"low"}"#.to_string(), "null".to_string()]
    );
}
