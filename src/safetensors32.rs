//! A `model.safetensors` reader that works on 32-bit targets (wasm32).
//!
//! The `safetensors` crate validates each tensor by computing its size in
//! *bits* through `usize` arithmetic: `nelements * dtype.bitsize()`. On a
//! 32-bit target that overflows for any tensor with more than ~134M f32
//! elements — which ModernBERT-large's input embedding alone exceeds (its
//! vocab embedding is on the order of 50k x 1024 = 51M elements per tensor,
//! and other checkpoints in the family go larger). Loading then fails with
//! "overflow computing buffer size from shape and/or element type", even
//! though the tensor itself fits in memory fine.
//!
//! The format itself is simple, so this reads it directly with `u64` math:
//! an 8-byte little-endian header length, that many bytes of JSON header
//! (`{"name": {"dtype", "shape", "data_offsets": [start, end]}, ...}`), then
//! the tensor data, whose offsets are relative to the end of the header.

use std::collections::HashMap;

use anyhow::{bail, Context, Result};
use candle_core::{DType, Device, Tensor};
use serde::Deserialize;

#[derive(Deserialize)]
struct TensorInfo {
    dtype: String,
    shape: Vec<u64>,
    data_offsets: (u64, u64),
}

fn dtype_of(name: &str) -> Result<DType> {
    Ok(match name {
        "U8" | "BOOL" => DType::U8,
        "U32" => DType::U32,
        "I16" => DType::I16,
        "I32" => DType::I32,
        "I64" => DType::I64,
        "BF16" => DType::BF16,
        "F16" => DType::F16,
        "F32" => DType::F32,
        "F64" => DType::F64,
        other => bail!("unsupported safetensors dtype {other:?}"),
    })
}

/// Reads every tensor out of `buffer` onto `device`. Tensors keep their
/// stored dtype; the caller converts (see [`candle_nn::VarBuilder`]).
pub fn load_buffer(buffer: &[u8], device: &Device) -> Result<HashMap<String, Tensor>> {
    if buffer.len() < 8 {
        bail!("safetensors buffer is too short to hold a header length");
    }
    let header_len = u64::from_le_bytes(buffer[..8].try_into().expect("8 bytes")) as usize;
    let data_start = 8usize.checked_add(header_len).context("safetensors header length overflows")?;
    if data_start > buffer.len() {
        bail!("safetensors header ({header_len} bytes) runs past the end of the file");
    }

    let header: HashMap<String, serde_json::Value> =
        serde_json::from_slice(&buffer[8..data_start]).context("parsing the safetensors header")?;
    let data = &buffer[data_start..];

    let mut tensors = HashMap::with_capacity(header.len());
    for (name, value) in header {
        if name == "__metadata__" {
            continue;
        }
        let info: TensorInfo =
            serde_json::from_value(value).with_context(|| format!("parsing header entry for tensor {name:?}"))?;
        let (start, end) = info.data_offsets;
        if end < start || end > data.len() as u64 {
            bail!("tensor {name:?} has data offsets {start}..{end} outside the {} byte data block", data.len());
        }
        // The whole point: all of this stays in u64, so a tensor larger than
        // `u32::MAX` bits doesn't wrap on wasm32.
        let dtype = dtype_of(&info.dtype)?;
        let elements = info.shape.iter().try_fold(1u64, |acc, d| acc.checked_mul(*d)).with_context(|| {
            format!("tensor {name:?} shape {:?} overflows an element count", info.shape)
        })?;
        let expected = elements
            .checked_mul(dtype.size_in_bytes() as u64)
            .with_context(|| format!("tensor {name:?} is too large to address"))?;
        if end - start != expected {
            bail!("tensor {name:?} spans {} bytes but its shape/dtype need {expected}", end - start);
        }
        let shape: Vec<usize> = info
            .shape
            .iter()
            .map(|d| usize::try_from(*d).with_context(|| format!("tensor {name:?} dimension {d} does not fit in usize")))
            .collect::<Result<_>>()?;
        let (start, end) = (start as usize, end as usize);
        let tensor = Tensor::from_raw_buffer(&data[start..end], dtype, &shape, device)
            .with_context(|| format!("reading tensor {name:?}"))?;
        tensors.insert(name, tensor);
    }
    Ok(tensors)
}
