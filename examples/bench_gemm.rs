use candle_core::{DType, Device, Tensor};
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    let device = Device::cuda_if_available(0)?;
    let x = Tensor::randn(0f32, 1f32, (7168, 1024), &device)?.to_dtype(DType::F16)?;
    let w = Tensor::randn(0f32, 1f32, (5248, 1024), &device)?.to_dtype(DType::F16)?;

    for _ in 0..5 {
        let _y = x.matmul(&w.t()?)?;
    }
    device.synchronize()?;

    let n = 30;
    let t0 = Instant::now();
    for _ in 0..n {
        let _y = x.matmul(&w.t()?)?;
    }
    device.synchronize()?;
    let dt_ms = t0.elapsed().as_secs_f64() * 1e3 / n as f64;
    let flops = 2.0 * 7168.0 * 1024.0 * 5248.0;
    let tflops = flops / (dt_ms / 1e3) / 1e12;
    println!("candle F16: {dt_ms:.3} ms/iter, {tflops:.1} TFLOPS");
    Ok(())
}
