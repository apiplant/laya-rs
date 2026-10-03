//! `GET /openapi.json` (JEV-004): the machine-readable protocol contract.
//!
//! A hand-written static OpenAPI 3.1 document (no codegen dependency)
//! describing the classifier endpoints, the request/response schemas, and
//! the protocol error envelope. The content mirrors
//! `docs/jev-server/protocol-reference.md` — it is the reference's schema
//! rendered as an OpenAPI document, not a generated artifact. The only
//! dynamic field is the loaded model id (`__MODEL__` placeholder), which
//! the request body's `model` field must match.

use ntex::web::types::State;
use ntex::web::HttpResponse;

use crate::server::state::ServerState;

pub async fn openapi(state: State<ServerState>) -> HttpResponse {
    HttpResponse::Ok().content_type("application/json").body(openapi_document(state.model_name()))
}

fn openapi_document(model: &str) -> String {
    // The model id sits inside JSON string literals; escape it via
    // serde_json so a hostile model name can't break the document.
    let quoted = serde_json::to_string(model).expect("string serializes");
    OPENAPI.replace("__MODEL__", &quoted[1..quoted.len() - 1])
}

const OPENAPI: &str = r##"{
  "openapi": "3.1.0",
  "info": {
    "title": "Laya classifier (Jev/Simple-Jev v1 protocol)",
    "description": "Serves a convaiinnovations/laya checkpoint behind the open Jev/Simple-Jev v1 classifier contract. POST /v1/systemone is an exact alias of POST /v1/classifier. Requests execute serially against the model (one in-flight forward + a bounded queue; full queue → 429 with Retry-After: 1).",
    "version": "1.0.0"
  },
  "servers": [{ "url": "/" }],
  "paths": {
    "/v1/classifier": {
      "post": {
        "operationId": "classify",
        "summary": "Answer a batch of typed questions (choice/score/noul)",
        "requestBody": {
          "required": true,
          "content": {
            "application/json": { "schema": { "$ref": "#/components/schemas/classifier_request" } }
          }
        },
        "responses": {
          "200": {
            "description": "Typed answers + token usage",
            "content": {
              "application/json": { "schema": { "$ref": "#/components/schemas/classifier_response" } }
            }
          },
          "422": {
            "description": "Invalid JSON/schema, unknown model, invalid context combination, unsupported chat/media/tool input, branch/token limits, or a body over 1 MiB",
            "content": {
              "application/json": { "schema": { "$ref": "#/components/schemas/error_envelope" } }
            }
          },
          "429": {
            "description": "Request queue full",
            "headers": { "Retry-After": { "schema": { "type": "string" } } },
            "content": {
              "application/json": {
                "schema": {
                  "type": "object",
                  "properties": { "detail": { "const": "Scoring queue is full" } }
                }
              }
            }
          },
          "500": {
            "description": "Unhandled runtime failure (e.g. model execution error)",
            "content": {
              "application/json": {
                "schema": {
                  "type": "object",
                  "properties": { "detail": { "const": "internal error" } }
                }
              }
            }
          }
        }
      }
    },
    "/v1/systemone": {
      "post": {
        "operationId": "classifyAlias",
        "summary": "Exact alias of /v1/classifier (identical behavior and response shape)",
        "requestBody": {
          "required": true,
          "content": {
            "application/json": { "schema": { "$ref": "#/components/schemas/classifier_request" } }
          }
        },
        "responses": { "$ref": "#/paths/~1v1~1classifier/post/responses" }
      }
    },
    "/health": {
      "get": {
        "operationId": "health",
        "summary": "Readiness (no inference)",
        "responses": {
          "200": {
            "description": "Ready",
            "content": {
              "application/json": {
                "schema": {
                  "type": "object",
                  "properties": {
                    "status": { "const": "ready" },
                    "model": { "type": "string", "description": "The loaded model id (currently __MODEL__)." }
                  },
                  "required": ["status", "model"]
                }
              }
            }
          }
        }
      }
    }
  },
  "components": {
    "schemas": {
      "classifier_request": {
        "type": "object",
        "properties": {
          "model": { "type": "string", "minLength": 1, "description": "Must equal the loaded model id (__MODEL__)." },
          "state": { "description": "Shared context: a string, JSON object, or array. Exactly one of state/messages must be supplied (null means absent; empty string/object/array still count as supplied)." },
          "messages": {
            "type": "array",
            "minItems": 1,
            "items": {
              "type": "object",
              "properties": {
                "role": { "enum": ["system", "developer", "user", "assistant"] },
                "content": { "type": "string" }
              },
              "required": ["role", "content"],
              "additionalProperties": false
            },
            "description": "Text-only chat history (laya-rs rejects image/audio/tool-call content and extra fields)."
          },
          "questions": {
            "type": "object",
            "minProperties": 1,
            "maxProperties": 256,
            "additionalProperties": {
              "oneOf": [
                { "$ref": "#/components/schemas/choice_question" },
                { "$ref": "#/components/schemas/score_question" },
                { "$ref": "#/components/schemas/noul_question" }
              ]
            },
            "description": "Mapping of question ids to question definitions, answered in one batched forward."
          },
          "options": {
            "type": "object",
            "properties": { "raw_logits": { "type": "boolean", "enum": [false], "description": "raw_logits diagnostics are not supported by the laya backend (true → 422)." } },
            "additionalProperties": false
          }
        },
        "required": ["model", "questions"]
      },
      "choice_question": {
        "type": "object",
        "properties": {
          "type": { "const": "choice" },
          "instructions": { "anyOf": [{ "type": "string" }, { "type": "object" }, { "type": "array" }, { "type": "null" }] },
          "criteria": {
            "type": "object",
            "minProperties": 2,
            "maxProperties": 50,
            "additionalProperties": { "anyOf": [{ "type": "string" }, { "type": "object" }, { "type": "array" }, { "type": "null" }] }
          }
        },
        "required": ["type", "instructions", "criteria"],
        "additionalProperties": false
      },
      "score_question": {
        "type": "object",
        "properties": {
          "type": { "const": "score" },
          "instructions": { "anyOf": [{ "type": "string" }, { "type": "object" }, { "type": "array" }, { "type": "null" }] },
          "criteria": {
            "type": "array",
            "minItems": 2,
            "maxItems": 50,
            "items": { "anyOf": [{ "type": "string" }, { "type": "object" }, { "type": "array" }, { "type": "null" }] }
          }
        },
        "required": ["type", "instructions", "criteria"],
        "additionalProperties": false
      },
      "noul_question": {
        "type": "object",
        "properties": {
          "type": { "const": "noul" },
          "instructions": { "anyOf": [{ "type": "string" }, { "type": "object" }, { "type": "array" }, { "type": "null" }] },
          "criteria": {
            "type": "object",
            "properties": {
              "true": { "anyOf": [{ "type": "string" }, { "type": "object" }, { "type": "array" }, { "type": "null" }] },
              "false": { "anyOf": [{ "type": "string" }, { "type": "object" }, { "type": "array" }, { "type": "null" }] }
            },
            "additionalProperties": false
          }
        },
        "required": ["type", "instructions"],
        "additionalProperties": false
      },
      "classifier_response": {
        "type": "object",
        "properties": {
          "model": { "type": "string" },
          "answers": {
            "type": "object",
            "description": "Mapping of question ids to laya-native protocol answers, in the request's insertion order.",
            "additionalProperties": {
              "oneOf": [
                {
                  "type": "object",
                  "properties": {
                    "type": { "const": "choice" },
                    "choice": { "type": "string" },
                    "confidence": { "type": "number" },
                    "probabilities": {
                      "type": "object",
                      "additionalProperties": { "type": "number" },
                      "description": "Candidate keys in the request's insertion order."
                    }
                  },
                  "required": ["type", "choice", "confidence", "probabilities"]
                },
                {
                  "type": "object",
                  "properties": {
                    "type": { "const": "score" },
                    "score": { "type": "number" },
                    "confidence": { "type": "number" },
                    "probabilities": { "type": "object", "additionalProperties": { "type": "number" } },
                    "legend": { "type": "object", "additionalProperties": { "type": "string" } }
                  },
                  "required": ["type", "score", "confidence", "probabilities", "legend"]
                },
                {
                  "type": "object",
                  "properties": {
                    "type": { "const": "noul" },
                    "noul": { "type": "number", "description": "Native P(true) in [0, 1]; no confidence field." }
                  },
                  "required": ["type", "noul"],
                  "additionalProperties": false
                }
              ]
            }
          },
          "usage": { "$ref": "#/components/schemas/usage" }
        },
        "required": ["model", "answers", "usage"]
      },
      "usage": {
        "type": "object",
        "properties": {
          "input_tokens": { "type": "integer", "minimum": 0 },
          "output_tokens": { "type": "integer", "const": 0, "description": "Always 0 — no tokens are sampled." }
        },
        "required": ["input_tokens", "output_tokens"]
      },
      "error_envelope": {
        "type": "object",
        "properties": {
          "error": {
            "type": "object",
            "properties": {
              "message": { "type": "string", "description": "Summary; extra validation errors beyond 10 details are folded here." },
              "type": { "const": "invalid_request_error" },
              "code": { "const": 422 },
              "param": { "type": "string", "description": "Dotted field path (first detail's path); array indices as [i]." },
              "details": {
                "type": "array",
                "maxItems": 10,
                "items": {
                  "type": "object",
                  "properties": {
                    "param": { "type": "string" },
                    "message": { "type": "string" },
                    "type": { "type": "string" }
                  },
                  "required": ["param", "message", "type"]
                }
              }
            },
            "required": ["message", "type", "code", "param", "details"]
          }
        },
        "required": ["error"]
      }
    }
  }
}"##;
