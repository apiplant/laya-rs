//! Two fused CUDA kernels for the elementwise work that dominates what's left of the forward
//! pass once the GEMMs and LayerNorm are on their fast paths.
//!
//! Both are memory-bound and were being executed as chains of separate candle ops, each of which
//! reads and writes the whole tensor:
//!
//! - **RoPE** was `mul, narrow, narrow, neg, cat, mul, add` — 6 kernels and ~5 round trips
//!   through a 14.7MB tensor, per tensor, per layer (measured 0.81ms/call vs a ~0.37ms
//!   bandwidth bound).
//! - **GeGLU** was `gelu, mul` over two strided halves — 3 round trips (0.55ms vs ~0.24ms).
//!
//! Fusing each into one kernel gets them to one read plus one write. Compiled at runtime with
//! NVRTC on first use and cached on the device, so there's no build-time CUDA dependency (unlike
//! the `flash-attn` feature) — this module is part of the plain `cuda` build.

#[cfg(feature = "cuda")]
use candle_core::backend::BackendStorage;
#[cfg(feature = "cuda")]
use candle_core::cuda_backend::cudarc::driver::{LaunchConfig, PushKernelArg};
#[cfg(feature = "cuda")]
use candle_core::cuda_backend::WrapErr;
// Always available: candle-core re-exports a dummy `CudaStorage`/`CudaDevice` pair without the
// `cuda` feature (see its `dummy_cuda_backend`), just without the methods `cuda_fwd` below needs
// — those calls are what's actually gated, not the type itself.
use candle_core::{CpuStorage, CudaStorage, DType, Layout, Result, Shape, Tensor};

// The kernel source and its NVRTC compilation are only reachable through `cuda_fwd` below, which
// only exists with the `cuda` feature — `candle_core::cuda_backend` itself doesn't exist without
// it, so this whole block has to be gated rather than just the call sites.
#[cfg(feature = "cuda")]
const KERNELS: &str = r#"
#include <cuda_fp16.h>

// out = x * cos + rotate_half(x) * sin, over x laid out as [B, S, H, D] (row-major) with
// cos/sin as [S, D]. rotate_half swaps the halves of the last dim and negates the first:
// for d < D/2 the partner is x[d + D/2] negated, for d >= D/2 it's x[d - D/2] as-is.
// Flat index i decomposes as ((b*S + s)*H + h)*D + d, so d = i % D and s = (i / (H*D)) % S.
extern "C" __global__ void rope_f16(
    const __half* __restrict__ x, const __half* __restrict__ cos, const __half* __restrict__ sin,
    __half* __restrict__ out, const int n, const int hd, const int d, const int s_count)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int di = i % d;
    const int si = (i / hd) % s_count;
    const int half_d = d >> 1;
    const int partner = (di < half_d) ? (i + half_d) : (i - half_d);
    const float rot = (di < half_d) ? -__half2float(x[partner]) : __half2float(x[partner]);
    const float c = __half2float(cos[si * d + di]);
    const float s = __half2float(sin[si * d + di]);
    out[i] = __float2half(__half2float(x[i]) * c + rot * s);
}

// out[n, i] = gelu(x[n, i]) * x[n, I + i], with x laid out as [N, 2I] row-major.
extern "C" __global__ void geglu_f16(
    const __half* __restrict__ x, __half* __restrict__ out, const int n, const int inter)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int row = i / inter;
    const int col = i % inter;
    const int base = row * 2 * inter;
    const float g = __half2float(x[base + col]);
    const float u = __half2float(x[base + inter + col]);
    // exact erf-based gelu, matching candle's gelu_erf and torch's default GELU
    const float gelu = 0.5f * g * (1.0f + erff(g * 0.7071067811865476f));
    out[i] = __float2half(gelu * u);
}
"#;

#[cfg(feature = "cuda")]
fn ptx() -> &'static str {
    use candle_core::cuda_backend::cudarc::nvrtc::safe::{compile_ptx_with_opts, CompileOptions};
    use std::sync::OnceLock;
    static PTX: OnceLock<String> = OnceLock::new();
    PTX.get_or_init(|| {
        // NVRTC has no default include path, so `cuda_fp16.h` has to be pointed at explicitly.
        // CUDA_PATH/CUDA_HOME if set, else the usual install locations.
        let mut include_paths: Vec<String> = ["CUDA_PATH", "CUDA_HOME", "CUDA_ROOT"]
            .iter()
            .filter_map(|v| std::env::var(v).ok())
            .map(|p| format!("{p}/include"))
            .collect();
        include_paths.push("/opt/cuda/include".to_string());
        include_paths.push("/usr/local/cuda/include".to_string());
        include_paths.retain(|p| std::path::Path::new(p).join("cuda_fp16.h").exists());

        let opts = CompileOptions { include_paths, use_fast_math: Some(true), ..Default::default() };
        compile_ptx_with_opts(KERNELS, opts)
            .expect("compiling laya fused kernels")
            .to_src()
    })
}

/// `x`: [B, S, H, D] f16 contiguous; `cos`/`sin`: [S, D] f16 contiguous.
struct RopeOp;

impl candle_core::CustomOp3 for RopeOp {
    fn name(&self) -> &'static str {
        "laya-rope"
    }

    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout, _: &CpuStorage, _: &Layout, _: &CpuStorage, _: &Layout) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("laya-rope: cuda only")
    }

    #[cfg(not(feature = "cuda"))]
    fn cuda_fwd(
        &self,
        _x: &CudaStorage, _xl: &Layout,
        _cos: &CudaStorage, _cl: &Layout,
        _sin: &CudaStorage, _sl: &Layout,
    ) -> Result<(CudaStorage, Shape)> {
        candle_core::bail!("laya-rope: built without the `cuda` feature")
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        x: &CudaStorage, xl: &Layout,
        cos: &CudaStorage, _cl: &Layout,
        sin: &CudaStorage, _sl: &Layout,
    ) -> Result<(CudaStorage, Shape)> {
        let dev = x.device().clone();
        let (_b, s, h, d) = xl.shape().dims4()?; // [B, S, H, D]
        let n = xl.shape().elem_count();
        let x = x.as_cuda_slice::<half::f16>()?.slice(xl.start_offset()..);
        let cos = cos.as_cuda_slice::<half::f16>()?;
        let sin = sin.as_cuda_slice::<half::f16>()?;
        let out = unsafe { dev.alloc::<half::f16>(n) }?;

        let func = dev.get_or_load_custom_func("rope_f16", "laya_fused", ptx())?;
        let cfg = LaunchConfig::for_num_elems(n as u32);
        let (hd, dd, sc) = ((h * d) as i32, d as i32, s as i32);
        let n_i = n as i32;
        let mut b = func.builder();
        b.arg(&x);
        b.arg(cos);
        b.arg(sin);
        b.arg(&out);
        b.arg(&n_i);
        b.arg(&hd);
        b.arg(&dd);
        b.arg(&sc);
        unsafe { b.launch(cfg) }.w()?;
        Ok((CudaStorage::wrap_cuda_slice(out, dev), xl.shape().clone()))
    }
}

/// RoPE in one kernel. `x` is `[B, S, H, D]`, `cos`/`sin` are `[S, D]` (per-token already, so the
/// packed/unpadded path just passes its position-indexed tables).
pub fn rope(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    x.apply_op3(cos, sin, RopeOp)
}

struct GegluOp;

impl candle_core::CustomOp1 for GegluOp {
    fn name(&self) -> &'static str {
        "laya-geglu"
    }

    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("laya-geglu: cuda only")
    }

    #[cfg(not(feature = "cuda"))]
    fn cuda_fwd(&self, _x: &CudaStorage, _xl: &Layout) -> Result<(CudaStorage, Shape)> {
        candle_core::bail!("laya-geglu: built without the `cuda` feature")
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(&self, x: &CudaStorage, xl: &Layout) -> Result<(CudaStorage, Shape)> {
        let dev = x.device().clone();
        let dims = xl.shape().dims();
        let two_i = dims[dims.len() - 1];
        let inter = two_i / 2;
        let rows: usize = dims[..dims.len() - 1].iter().product();
        let n = rows * inter;
        let x = x.as_cuda_slice::<half::f16>()?.slice(xl.start_offset()..);
        let out = unsafe { dev.alloc::<half::f16>(n) }?;

        let func = dev.get_or_load_custom_func("geglu_f16", "laya_fused", ptx())?;
        let cfg = LaunchConfig::for_num_elems(n as u32);
        let (n_i, inter_i) = (n as i32, inter as i32);
        let mut b = func.builder();
        b.arg(&x);
        b.arg(&out);
        b.arg(&n_i);
        b.arg(&inter_i);
        unsafe { b.launch(cfg) }.w()?;

        let mut shape = dims[..dims.len() - 1].to_vec();
        shape.push(inter);
        Ok((CudaStorage::wrap_cuda_slice(out, dev), Shape::from(shape)))
    }
}

/// `gelu(x[.., :I]) * x[.., I:]` in one kernel, for `x` shaped `[.., 2I]`.
pub fn geglu(x: &Tensor) -> Result<Tensor> {
    x.apply_op1_no_bwd(&GegluOp)
}

/// Whether the fused path applies: CUDA device and F16 (the kernels are f16-specific).
pub fn usable(t: &Tensor) -> bool {
    t.device().is_cuda() && t.dtype() == DType::F16
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, D};

    /// Reference RoPE: the candle op chain these kernels replace.
    fn rope_ref(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
        let last = x.dim(D::Minus1)?;
        let half = last / 2;
        let x1 = x.narrow(D::Minus1, 0, half)?;
        let x2 = x.narrow(D::Minus1, half, half)?;
        let rot = Tensor::cat(&[&x2.neg()?, &x1], D::Minus1)?;
        x.broadcast_mul(cos)? + rot.broadcast_mul(sin)?
    }

    fn geglu_ref(x: &Tensor) -> Result<Tensor> {
        let last = x.dim(D::Minus1)?;
        let half = last / 2;
        let gate = x.narrow(D::Minus1, 0, half)?.gelu_erf()?;
        let up = x.narrow(D::Minus1, half, half)?;
        gate * up
    }

    fn max_abs_diff(a: &Tensor, b: &Tensor) -> Result<f32> {
        let d = (a.to_dtype(DType::F32)? - b.to_dtype(DType::F32)?)?.abs()?;
        d.flatten_all()?.max(0)?.to_scalar::<f32>()
    }

    #[test]
    fn fused_rope_matches_reference() -> Result<()> {
        let dev = match Device::cuda_if_available(0) {
            Ok(d) if d.is_cuda() => d,
            _ => return Ok(()), // no GPU in this environment; nothing to check
        };
        let (b, s, h, d) = (3usize, 37usize, 4usize, 64usize);
        let x = Tensor::randn(0f32, 1f32, (b, s, h, d), &dev)?.to_dtype(DType::F16)?;
        let cos = Tensor::randn(0f32, 1f32, (s, d), &dev)?.to_dtype(DType::F16)?;
        let sin = Tensor::randn(0f32, 1f32, (s, d), &dev)?.to_dtype(DType::F16)?;

        let got = rope(&x, &cos, &sin)?;
        let want = rope_ref(&x, &cos.reshape((1, s, 1, d))?, &sin.reshape((1, s, 1, d))?)?;
        assert_eq!(got.dims(), want.dims());
        let diff = max_abs_diff(&got, &want)?;
        assert!(diff < 0.01, "fused rope diverges from reference: max abs diff {diff}");
        Ok(())
    }

    #[test]
    fn fused_geglu_matches_reference() -> Result<()> {
        let dev = match Device::cuda_if_available(0) {
            Ok(d) if d.is_cuda() => d,
            _ => return Ok(()),
        };
        let x = Tensor::randn(0f32, 1f32, (5usize, 29usize, 128usize), &dev)?.to_dtype(DType::F16)?;
        let got = geglu(&x)?;
        let want = geglu_ref(&x)?;
        assert_eq!(got.dims(), want.dims());
        let diff = max_abs_diff(&got, &want)?;
        assert!(diff < 0.01, "fused geglu diverges from reference: max abs diff {diff}");
        Ok(())
    }
}
