use candle_core::{DType, Device, Tensor};
use laya::modernbert::ModernBertConfig;
use laya::decision_model::DecisionModel;
use candle_nn::VarBuilder;
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    let dir = std::path::Path::new("/mnt/extra/ai/laya/typed-decisions");
    let device = Device::cuda_if_available(0)?;
    let cfg: ModernBertConfig =
        serde_json::from_str(&std::fs::read_to_string(dir.join("encoder").join("config.json"))?)?;
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&[dir.join("model.safetensors")], DType::F16, &device)?
    };
    let model = DecisionModel::load(cfg, 2, 2, vb)?;

    let (b, s) = (7usize, 1024usize);
    let ids: Vec<i64> = (0..b * s).map(|i| (100 + (i % 4900)) as i64).collect();
    let input_ids = Tensor::from_vec(ids, (b, s), &device)?;
    let attention_mask = Tensor::from_vec(vec![1i64; b * s], (b, s), &device)?;
    let markers: Vec<Vec<usize>> = (0..b).map(|_| (0..10).collect()).collect();
    let qtypes = vec![0i64; b];

    for _ in 0..3 {
        model.forward(&input_ids, &attention_mask, &markers, &qtypes)?;
    }
    device.synchronize()?;
    let n = 20;
    let t0 = Instant::now();
    for _ in 0..n {
        model.forward(&input_ids, &attention_mask, &markers, &qtypes)?;
    }
    device.synchronize()?;
    println!("candle model.forward b={b} s={s}: {:.2} ms", t0.elapsed().as_secs_f64() * 1e3 / n as f64);
    Ok(())
}
