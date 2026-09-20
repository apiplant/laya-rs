use candle_core::{DType, Device, Tensor, D};
use std::time::Instant;

fn bench<F: FnMut() -> anyhow::Result<()>>(name: &str, dev: &Device, per_layer: f64, mut f: F) -> anyhow::Result<f64> {
    for _ in 0..5 { f()?; }
    dev.synchronize()?;
    let n = 30;
    let t0 = Instant::now();
    for _ in 0..n { f()?; }
    dev.synchronize()?;
    let ms = t0.elapsed().as_secs_f64() * 1e3 / n as f64;
    println!("{name:<28} {ms:7.3} ms/call   x{per_layer:>4.0} = {:7.2} ms", ms * per_layer);
    Ok(ms * per_layer)
}

fn main() -> anyhow::Result<()> {
    let dev = Device::cuda_if_available(0)?;
    let (b, s, h, nh, hd) = (7usize, 1024usize, 1024usize, 16usize, 64usize);
    let f16 = DType::F16;
    let x = Tensor::randn(0f32, 1f32, (b, s, h), &dev)?.to_dtype(f16)?;
    let w_qkv = Tensor::randn(0f32, 1f32, (3 * h, h), &dev)?.to_dtype(f16)?;
    let w_o = Tensor::randn(0f32, 1f32, (h, h), &dev)?.to_dtype(f16)?;
    let w_i = Tensor::randn(0f32, 1f32, (5248, h), &dev)?.to_dtype(f16)?;
    let w_wo = Tensor::randn(0f32, 1f32, (h, 2624), &dev)?.to_dtype(f16)?;
    let x_ff = Tensor::randn(0f32, 1f32, (b, s, 2624), &dev)?.to_dtype(f16)?;
    let qkv_big = Tensor::randn(0f32, 1f32, (b, s, 5248), &dev)?.to_dtype(f16)?;
    let qh = Tensor::randn(0f32, 1f32, (b, s, nh, hd), &dev)?.to_dtype(f16)?;
    let cos = Tensor::randn(0f32, 1f32, (1, s, 1, hd), &dev)?.to_dtype(f16)?;
    let lnw = Tensor::randn(0f32, 1f32, h, &dev)?.to_dtype(f16)?;
    let lnb = Tensor::zeros(h, f16, &dev)?;
    let ln = candle_nn::LayerNorm::new(lnw, lnb, 1e-5);
    use candle_nn::Module;

    let mut total = 0.0;
    total += bench("gemm wqkv [7168,1024]x3072", &dev, 28.0, || { x.reshape((b*s, h))?.matmul(&w_qkv.t()?)?; Ok(()) })?;
    total += bench("gemm attn.wo 1024x1024", &dev, 28.0, || { x.reshape((b*s, h))?.matmul(&w_o.t()?)?; Ok(()) })?;
    total += bench("gemm mlp.wi 1024x5248", &dev, 28.0, || { x.reshape((b*s, h))?.matmul(&w_i.t()?)?; Ok(()) })?;
    total += bench("gemm mlp.wo 2624x1024", &dev, 28.0, || { x_ff.reshape((b*s, 2624))?.matmul(&w_wo.t()?)?; Ok(()) })?;
    total += bench("flash global", &dev, 10.0, || {
        candle_flash_attn::flash_attn_windowed(&qh, &qh, &qh, 0.125, None, None)?; Ok(()) })?;
    total += bench("flash local(64)", &dev, 18.0, || {
        candle_flash_attn::flash_attn_windowed(&qh, &qh, &qh, 0.125, Some(64), Some(64))?; Ok(()) })?;
    total += bench("rope (q+k)", &dev, 28.0, || {
        for _ in 0..2 {
            let a = qh.broadcast_mul(&cos)?;
            let last = qh.dim(D::Minus1)?;
            let x1 = qh.narrow(D::Minus1, 0, last / 2)?;
            let x2 = qh.narrow(D::Minus1, last / 2, last / 2)?;
            let r = Tensor::cat(&[&x2.neg()?, &x1], D::Minus1)?;
            let _ = (a + r.broadcast_mul(&cos)?)?;
        }
        Ok(()) })?;
    total += bench("layernorm [7,1024,1024]", &dev, 56.0, || { ln.forward(&x)?; Ok(()) })?;
    total += bench("gelu_gate [7,1024,5248]", &dev, 28.0, || {
        let g = qkv_big.narrow(D::Minus1, 0, 2624)?;
        let u = qkv_big.narrow(D::Minus1, 2624, 2624)?;
        let _ = (g.gelu_erf()? * u)?; Ok(()) })?;
    total += bench("residual add", &dev, 56.0, || { (&x + &x)?; Ok(()) })?;
    println!("\n{:<28} {:>28} {total:7.2} ms", "SUM (unfused reference)", "");

    println!("\nfused kernels (src/fused.rs) replacing the two rows above:");
    let cos2 = cos.reshape((s, hd))?;
    let sin2 = cos2.clone();
    let mut fused_total = 0.0;
    fused_total += bench("  rope fused (q+k)", &dev, 28.0, || {
        laya::fused::rope(&qh, &cos2, &sin2)?;
        laya::fused::rope(&qh, &cos2, &sin2)?;
        Ok(()) })?;
    fused_total += bench("  geglu fused", &dev, 28.0, || { laya::fused::geglu(&qkv_big)?; Ok(()) })?;
    println!("{:<28} {:>28} {fused_total:7.2} ms  (was {:.2} ms)", "  subtotal", "", 22.79 + 15.53);
    Ok(())
}
