use std::env;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use transcribe_rs::accel;
use transcribe_rs::onnx::parakeet::{ParakeetModel, ParakeetParams, TimestampGranularity};
use transcribe_rs::onnx::Quantization;

struct Args {
    accelerator: accel::OrtAccelerator,
    model_dir: PathBuf,
    wav: PathBuf,
    runs: usize,
    warmups: usize,
    granularity: TimestampGranularity,
}

fn main() -> Result<()> {
    let args = Args::parse()?;

    println!(
        "bench_parakeet accelerator={} runs={} warmups={} granularity={:?}",
        args.accelerator, args.runs, args.warmups, args.granularity
    );
    println!("model_dir={}", args.model_dir.display());
    println!("wav={}", args.wav.display());

    let samples = read_wav_samples(&args.wav)?;
    let audio_seconds = samples.len() as f64 / 16_000.0;
    println!(
        "audio_samples={} audio_seconds={:.3}",
        samples.len(),
        audio_seconds
    );

    accel::set_ort_accelerator(args.accelerator);

    let load_start = Instant::now();
    let mut model = ParakeetModel::load(&args.model_dir, &Quantization::Int8)
        .with_context(|| format!("failed to load model from {}", args.model_dir.display()))?;
    println!("load_ms={}", load_start.elapsed().as_millis());

    let params = ParakeetParams {
        timestamp_granularity: Some(args.granularity),
        ..Default::default()
    };

    for i in 0..args.warmups {
        let started = Instant::now();
        let result = model.transcribe_with(&samples, &params)?;
        println!(
            "warmup_run={} inference_ms={} chars={} words={}",
            i + 1,
            started.elapsed().as_millis(),
            result.text.chars().count(),
            count_words(&result.text)
        );
    }

    let mut timings = Vec::with_capacity(args.runs);
    let mut last_text = String::new();
    for i in 0..args.runs {
        let started = Instant::now();
        let result = model.transcribe_with(&samples, &params)?;
        let elapsed = started.elapsed();
        timings.push(elapsed);
        last_text = result.text;
        println!(
            "run={} inference_ms={} realtime_factor={:.3}",
            i + 1,
            elapsed.as_millis(),
            elapsed.as_secs_f64() / audio_seconds
        );
    }

    print_summary(&timings, audio_seconds);
    println!("transcript={}", last_text.replace('\n', " "));

    Ok(())
}

impl Args {
    fn parse() -> Result<Self> {
        let mut accelerator = None;
        let mut model_dir = None;
        let mut wav = None;
        let mut runs = 5;
        let mut warmups = 1;
        let mut granularity = TimestampGranularity::Segment;

        let mut args = env::args().skip(1);
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--accelerator" => {
                    let value = args.next().context("--accelerator requires a value")?;
                    accelerator = Some(parse_accelerator(&value)?);
                }
                "--model-dir" => {
                    model_dir = Some(PathBuf::from(
                        args.next().context("--model-dir requires a path")?,
                    ));
                }
                "--wav" => {
                    wav = Some(PathBuf::from(args.next().context("--wav requires a path")?));
                }
                "--runs" => {
                    runs = args
                        .next()
                        .context("--runs requires a value")?
                        .parse()
                        .context("--runs must be a positive integer")?;
                }
                "--warmups" => {
                    warmups = args
                        .next()
                        .context("--warmups requires a value")?
                        .parse()
                        .context("--warmups must be a non-negative integer")?;
                }
                "--granularity" => {
                    let value = args.next().context("--granularity requires a value")?;
                    granularity = parse_granularity(&value)?;
                }
                "--help" | "-h" => {
                    print_help();
                    std::process::exit(0);
                }
                other => anyhow::bail!("unknown argument: {other}"),
            }
        }

        let accelerator = accelerator.context("missing --accelerator")?;
        let model_dir = model_dir.context("missing --model-dir")?;
        let wav = wav.context("missing --wav")?;

        if runs == 0 {
            anyhow::bail!("--runs must be greater than 0");
        }

        Ok(Self {
            accelerator,
            model_dir,
            wav,
            runs,
            warmups,
            granularity,
        })
    }
}

fn print_help() {
    println!(
        "Usage: bench_parakeet --accelerator <cpu|auto> --model-dir <dir> --wav <file> [--runs 5] [--warmups 1] [--granularity segment|word|token]"
    );
}

fn parse_accelerator(value: &str) -> Result<accel::OrtAccelerator> {
    match value {
        "auto" => Ok(accel::OrtAccelerator::Auto),
        "cpu" => Ok(accel::OrtAccelerator::CpuOnly),
        other => anyhow::bail!("unsupported accelerator for this benchmark: {other}"),
    }
}

fn parse_granularity(value: &str) -> Result<TimestampGranularity> {
    match value {
        "segment" => Ok(TimestampGranularity::Segment),
        "word" => Ok(TimestampGranularity::Word),
        "token" => Ok(TimestampGranularity::Token),
        other => anyhow::bail!("unsupported timestamp granularity: {other}"),
    }
}

fn read_wav_samples(path: &PathBuf) -> Result<Vec<f32>> {
    let mut reader = hound::WavReader::open(path)
        .with_context(|| format!("failed to open {}", path.display()))?;
    let spec = reader.spec();

    if spec.channels != 1 || spec.sample_rate != 16_000 {
        anyhow::bail!(
            "expected mono 16 kHz WAV, got channels={} sample_rate={}",
            spec.channels,
            spec.sample_rate
        );
    }

    match (spec.sample_format, spec.bits_per_sample) {
        (hound::SampleFormat::Int, 16) => reader
            .samples::<i16>()
            .map(|sample| sample.map(|value| value as f32 / i16::MAX as f32))
            .collect::<Result<Vec<_>, _>>()
            .context("failed to read 16-bit PCM samples"),
        (hound::SampleFormat::Float, 32) => reader
            .samples::<f32>()
            .collect::<Result<Vec<_>, _>>()
            .context("failed to read 32-bit float samples"),
        _ => anyhow::bail!(
            "unsupported WAV format: {:?} {} bits",
            spec.sample_format,
            spec.bits_per_sample
        ),
    }
}

fn print_summary(timings: &[Duration], audio_seconds: f64) {
    let mut values: Vec<f64> = timings.iter().map(Duration::as_secs_f64).collect();
    values.sort_by(f64::total_cmp);

    let sum: f64 = values.iter().sum();
    let avg = sum / values.len() as f64;
    let min = values[0];
    let max = values[values.len() - 1];
    let median = values[values.len() / 2];

    println!(
        "summary avg_ms={} median_ms={} min_ms={} max_ms={} avg_realtime_factor={:.3}",
        (avg * 1000.0).round() as u64,
        (median * 1000.0).round() as u64,
        (min * 1000.0).round() as u64,
        (max * 1000.0).round() as u64,
        avg / audio_seconds
    );
}

fn count_words(text: &str) -> usize {
    text.split_whitespace().count()
}
