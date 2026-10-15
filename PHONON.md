# Phonon-2 Core ML in Handy Beta

Phonon is a selectable, bundled English model on Apple silicon with macOS 15
or newer. It uses the Apple Neural Engine for its encoder and the CPU for its
decoder. The model selector reports Phonon by name and keeps Parakeet available
for other languages. Bundled model files are immutable and cannot be deleted
from the model selector.

The existing transcription manager owns a persistent Swift helper. Loading,
switching, and idle unloading use the normal manager lifecycle. Startup loads
every encoder window before reporting readiness; fresh Neural Engine setup
can take several minutes. Model-loading events cover that preparation. There
is no Python dependency for the Phonon runtime and no audio leaves the Mac.

Requests use the same 16 kHz mono float32 framing as the MLX sidecar. A worker
thread bounds the entire write/read operation with a timeout, including a
blocked pipe. Failed helpers are killed and reaped; the next request can
restart them. A failed transcription tries an already downloaded Parakeet V3
model before returning an error. Phonon errors and fallback use appear in logs.

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
`coreml-neural-engine`. These tests do not measure microphone, hotkey, or paste
latency and do not establish accuracy on arbitrary dictations.

## Attribution

Phonon-2 is Fermion Research's derivative of NVIDIA Parakeet TDT 0.6B v3.
Weights are CC-BY-4.0; the native runtime is Apache-2.0. Their NOTICE and license
files accompany the model under `resources/phonon-coreml/model` in the app.

Sources: [model and attribution](https://huggingface.co/FermionResearch/Phonon-2-CoreML)
and [native runtime](https://github.com/fermionresearch/phonon-coreml).
