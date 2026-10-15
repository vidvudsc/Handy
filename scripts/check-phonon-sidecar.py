"""Exercise the shipped helper with public references, long audio, and bad PCM.

Requires numpy and soundfile. Audio stays local. --recording optionally checks
additional Handy WAVs for a nonempty, repeatable response without claiming a
human reference for them.
"""
import argparse
import json
from pathlib import Path
import re
import select
import struct
import subprocess

import numpy as np
import soundfile as sf


def normalized(text):
    return re.sub(r"[^\w\s]", "", text.lower().replace("’", "'").replace("-", " ")).split()


def response(process):
    if not select.select([process.stdout], [], [], 1200)[0]:
        raise TimeoutError("Phonon did not respond within twenty minutes")
    line = process.stdout.readline()
    if not line:
        raise RuntimeError("Phonon exited without a response")
    return json.loads(line)


def request(process, samples):
    payload = np.asarray(samples, dtype="<f4").tobytes()
    process.stdin.write(struct.pack("<Q", len(payload)))
    process.stdin.write(payload)
    process.stdin.flush()
    return response(process)


def read_audio(path):
    samples, rate = sf.read(path, dtype="float32")
    if rate != 16000 or samples.ndim != 1:
        raise ValueError(f"Expected 16 kHz mono audio: {path}")
    return samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--helper", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--recording", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    proc = subprocess.Popen([str(args.helper.resolve()), str(args.model.resolve())],
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    report = {}
    try:
        assert response(proc).get("ready") is True
        references = json.loads((args.fixtures / "expected_texts.json").read_text())
        clips = []
        for name, reference in references.items():
            samples = read_audio(args.fixtures / "clips" / f"{name}.flac")
            clips.append(samples)
            result = request(proc, samples)
            assert normalized(result["text"]) == normalized(reference), (name, result)
            report[name] = result
        assert request(proc, np.zeros(16000 * 4, dtype=np.float32))["text"] == ""
        # A rejected request must not break the persistent protocol.
        assert "error" in request(proc, np.array([np.nan], dtype=np.float32))
        assert normalized(request(proc, clips[0])["text"]) == normalized(next(iter(references.values())))
        long_audio = np.concatenate(clips * 2)
        long_text = request(proc, long_audio)["text"]
        reference_words = normalized(" ".join(list(references.values()) * 2))
        hypothesis = normalized(long_text)
        ending = normalized(list(references.values())[-1])
        assert hypothesis[-len(ending):] == ending, "Long audio lost its final sentence"
        assert len(hypothesis) >= len(reference_words) * 0.9, "Long audio lost substantial speech"
        report["long_audio"] = {"seconds": len(long_audio) / 16000, "words": len(hypothesis), "text": long_text}
        for path in args.recording:
            audio = read_audio(path)
            first, second = request(proc, audio), request(proc, audio)
            assert first["text"] and first["text"] == second["text"]
            report[path.name] = first
        proc.stdin.close()
        assert proc.wait(timeout=5) == 0, "Helper did not exit cleanly on stdin EOF"
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
    text = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
