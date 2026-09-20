use candle_core::{DType, Device, Tensor};
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    let device = Device::cuda_if_available(0)?;
    let x = Tensor::randn(0f32, 1f32, (7, 1024, 1024), &device)?.to_dtype(DType::F16)?;
    device.synchronize()?;

    let n = 500;
    let t0 = Instant::now();
    for _ in 0..n {
        let y = (&x * 2.0)?;
        let _y = (y + 1.0)?;
    }
    device.synchronize()?;
    let us = t0.elapsed().as_secs_f64() * 1e6 / n as f64;
    println!("candle: {us:.1} us/pair-of-small-ops");
    Ok(())
}
