# Phonon-2 Core ML in Handy Beta

Phonon is a selectable, bundled English model on Apple silicon with macOS 15
or newer. It uses the Apple Neural Engine for its encoder and the CPU for its
decoder. The model selector reports Phonon by name and keeps Parakeet available
for other languages. Bundled model files are immutable and cannot be deleted
from the model selector.

The existing transcription manager owns a persistent Swift helper. Startup
loads every encoder window before reporting readiness. On this M1 Pro, Apple
Neural Engine preparation has recurred after idle unloading and taken minutes;
a compiled-model cache does not guarantee a quick reload.

Phonon prepares in the background while dictation uses a cached Parakeet V3
backup. Recordings never wait for Neural Engine preparation. Once ready,
Phonon takes over and the backup instance is released. Phonon remains the
selected primary throughout. Preparation has a 20-minute safety limit and
can be cancelled; transcription requests retain their separate short timeout.
A prepared Phonon instance stays resident through normal idle periods until
a model switch or application exit. The explicit Immediately unload setting
still applies. There is no Python dependency for the Phonon runtime and no
audio leaves the Mac.

Requests use the same 16 kHz mono float32 framing as the MLX sidecar. A worker
thread bounds the entire write/read operation with a timeout, including a
blocked pipe. Failed helpers are killed and reaped; the next request can
restart them. Requests have a 30-second limit. Cmd+K stays registered during
processing and interrupts Phonon startup/request waits; cancellation does not
start a fallback or paste late text. A failed transcription tries an already
downloaded Parakeet V3 model before returning an error. Phonon errors and
fallback use appear in logs.

## Build

`bun run tauri build --bundles app` prepares the native helper and model before
building the frontend. `bun run prepare:phonon` stages them separately. The
preparer requires Swift 6 and either the Hugging Face `hf` CLI or `uv`, and
skips other platforms. `HANDY_PHONON_MODEL_DIR` can point to an existing model
download. All required files are checked against pinned SHA-256 values before
they enter the application bundle.

The Swift package is pinned to Fermion's release 1.1.2, commit
`bf59e574639b060e30ee1a3f9c0e003fed5744c8`. The model is pinned to revision
`1143812fd4236522232a8c33e3115f0577e3cd7d`. Generated model assets and native
build outputs are ignored by Git.

## Verify

Run the Rust protocol/timeout tests with:

```sh
cargo test --manifest-path src-tauri/Cargo.toml --release phonon::tests --lib
```

`scripts/check-phonon-sidecar.py` checks the production helper against the
model repository's public `ci` fixtures, a recording longer than 35 seconds,
silence, invalid input recovery, and clean shutdown. Supply `--helper`,
`--model`, and `--fixtures`; its Python environment needs numpy and soundfile.

The app's `--transcribe-file WAV --model phonon-2-coreml --json` command tests
the complete Rust transcription path. Its reported backend must be
`coreml-neural-engine`. Without an explicit `--model`, the selected Phonon
model uses the same background preparation path as interactive dictation;
`onnx-parakeet-backup` identifies the temporary backup honestly. These tests do not measure microphone, hotkey, or paste
latency and do not establish accuracy on arbitrary dictations.

## Attribution

Phonon-2 is Fermion Research's derivative of NVIDIA Parakeet TDT 0.6B v3.
Weights are CC-BY-4.0; the native runtime is Apache-2.0. Their NOTICE and license
files accompany the model under `resources/phonon-coreml/model` in the app.

Sources: [model and attribution](https://huggingface.co/FermionResearch/Phonon-2-CoreML)
and [native runtime](https://github.com/fermionresearch/phonon-coreml).
