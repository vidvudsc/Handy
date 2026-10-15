/** Build and stage the pinned, offline Phonon engine for Apple silicon. */
import { createHash } from "node:crypto";
import {
  chmod,
  copyFile,
  mkdir,
  mkdtemp,
  readFile,
  rename,
  rm,
} from "node:fs/promises";
import { homedir } from "node:os";
import { join, resolve } from "node:path";

if (process.platform !== "darwin" || process.arch !== "arm64") process.exit(0);

const root = resolve(import.meta.dir, "..");
const modelDir =
  process.env.HANDY_PHONON_MODEL_DIR ||
  join(homedir(), ".cache/handy-benchmarks/Phonon-2-CoreML");
const resources = join(root, "src-tauri/resources/phonon-coreml");
const swiftPackage = join(root, "src-tauri/swift/phonon-sidecar");
const revision = "1143812fd4236522232a8c33e3115f0577e3cd7d";
const files: Record<string, string> = {
  "Phonon-2.mlpackage/Data/com.apple.CoreML/model.mlmodel":
    "4f6790ec94fe4429b10397c013563d0a51119953ff6d3da466bbcb23a25dee2a",
  "Phonon-2.mlpackage/Data/com.apple.CoreML/weights/weight.bin":
    "93aa991318a00ed492fdbb724e7bca4e6a69c6cf671ab4f14a14ef6101e7648b",
  "Phonon-2.mlpackage/Manifest.json":
    "6ebc9309aea0e16bda9c7558521fef6307fac0afba9a5b4959fea45ed7cd0429",
  "decoder.bin":
    "36fa7202dda85c603af60d930ab88d7e9f0d2a619490aff46f5384db924cfa70",
  "manifest.json":
    "ab3fb5dfd3fc07b1d8a24379d04418f458d0443ba6359d3a5ab2f098a7d64089",
  NOTICE: "b468a23a1ce2c5181ea4050432bb68713a6468ce2e452927228d24f3d56226be",
  "LICENSE-CODE-Apache-2.0.txt":
    "cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30",
  "LICENSE-WEIGHTS-CC-BY-4.0.txt":
    "9ba9550ad48438d0836ddab3da480b3b69ffa0aac7b7878b5a0039e7ab429411",
};

function run(command: string[], cwd = root) {
  const result = Bun.spawnSync(command, {
    cwd,
    stdout: "inherit",
    stderr: "inherit",
  });
  if (result.exitCode !== 0)
    throw new Error(`Command failed: ${command[0]} (${result.exitCode})`);
}

async function checksum(path: string): Promise<string | undefined> {
  try {
    return createHash("sha256")
      .update(await readFile(path))
      .digest("hex");
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === "ENOENT") return undefined;
    throw error;
  }
}

const missing = (
  await Promise.all(
    Object.keys(files).map(
      async (file) => [file, await checksum(join(modelDir, file))] as const,
    ),
  )
).some(([file, hash]) => hash !== files[file]);
if (missing) {
  const cli = Bun.which("hf")
    ? ["hf"]
    : Bun.which("uvx")
      ? ["uvx", "--from", "huggingface_hub", "hf"]
      : undefined;
  if (!cli)
    throw new Error(
      "Install the Hugging Face hf CLI (or uv) to prepare Phonon Core ML.",
    );
  run([
    ...cli,
    "download",
    "FermionResearch/Phonon-2-CoreML",
    "--revision",
    revision,
    "--local-dir",
    modelDir,
    "--include",
    ...Object.keys(files),
  ]);
}
for (const [file, expected] of Object.entries(files)) {
  if ((await checksum(join(modelDir, file))) !== expected)
    throw new Error(`Phonon checksum mismatch: ${file}`);
  const target = join(resources, "model", file);
  if ((await checksum(target)) !== expected) {
    await mkdir(resolve(target, ".."), { recursive: true });
    await copyFile(join(modelDir, file), target);
  }
}
run(["swift", "build", "--package-path", swiftPackage, "-c", "release"]);
await mkdir(resources, { recursive: true });
for (const file of ["LICENSE", "NOTICE"]) {
  const target = join(resources, `${file}-RUNTIME`);
  try {
    await chmod(target, 0o644);
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code !== "ENOENT") throw error;
  }
  await copyFile(
    join(swiftPackage, ".build/checkouts/phonon-coreml", file),
    target,
  );
  // SPM checkouts are read-only; macOS bundling must be able to strip xattrs.
  await chmod(target, 0o644);
}
// An existing dev/helper process may still map the previous executable.
// Sign a separate inode, then replace it atomically only when it changes.
const staging = await mkdtemp(join(resources, ".stage-"));
try {
  const stagedBinary = join(staging, "handy-phonon");
  const binary = join(resources, "handy-phonon");
  await copyFile(
    join(swiftPackage, ".build/release/handy-phonon"),
    stagedBinary,
  );
  run(["/usr/bin/codesign", "--force", "--sign", "-", stagedBinary]);
  if ((await checksum(stagedBinary)) !== (await checksum(binary))) {
    await rename(stagedBinary, binary);
  }
} finally {
  await rm(staging, { recursive: true, force: true });
}
console.log("Phonon Core ML helper and verified model are staged.");
