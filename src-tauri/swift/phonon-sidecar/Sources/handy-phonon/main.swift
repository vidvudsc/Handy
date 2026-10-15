import Darwin
import Foundation
import PhononCoreML

func respond(_ value: [String: Any]) throws {
    var data = try JSONSerialization.data(withJSONObject: value, options: [.sortedKeys])
    data.append(10)
    try FileHandle.standardOutput.write(contentsOf: data)
}

func readExactly(_ count: Int) throws -> Data? {
    var result = Data()
    while result.count < count {
        guard let part = try FileHandle.standardInput.read(upToCount: count - result.count), !part.isEmpty else {
            if result.isEmpty { return nil }
            throw NSError(domain: "HandyPhonon", code: 1, userInfo: [NSLocalizedDescriptionKey: "Truncated audio request"])
        }
        result.append(part)
    }
    return result
}

guard CommandLine.arguments.count == 2 else {
    FileHandle.standardError.write(Data("Usage: handy-phonon MODEL_DIRECTORY\n".utf8))
    exit(2)
}

do {
    // Core ML preparation can outlive an abrupt host exit. Stop promptly when
    // the owning app exits, including before the request loop becomes ready.
    let parent = getppid()
    let parentWatch = DispatchSource.makeTimerSource(queue: .global())
    parentWatch.schedule(deadline: .now() + .milliseconds(500), repeating: .milliseconds(500))
    parentWatch.setEventHandler { @Sendable in if getppid() != parent { exit(0) } }
    parentWatch.resume()
    defer { parentWatch.cancel() }
    var options = Transcriber.Options()
    options.progress = { phase, seconds in
        FileHandle.standardError.write(Data(String(format: "[phonon] %.2fs %@\n", seconds, phase).utf8))
    }
    let model = try Transcriber(bundle: URL(fileURLWithPath: CommandLine.arguments[1]), options: options)
    // Prepare every shape before reporting readiness. A first dictation must
    // never pay the Neural Engine's one-time compilation cost.
    try model.preloadAllFunctions()
    try respond(["ready": true])
    while true {
        // Foundation's read buffers also need draining between requests.
        let keepGoing = try autoreleasepool { () throws -> Bool in
            guard let header = try readExactly(8) else { return false }
            let count = header.withUnsafeBytes { $0.loadUnaligned(as: UInt64.self).littleEndian }
            guard count > 0, count <= 64 * 1024 * 1024, count % 4 == 0,
                  let audio = try readExactly(Int(count)) else {
                throw NSError(domain: "HandyPhonon", code: 2, userInfo: [NSLocalizedDescriptionKey: "Invalid PCM request size"])
            }
            var samples = [Float](repeating: 0, count: Int(count) / 4)
            _ = samples.withUnsafeMutableBytes { audio.copyBytes(to: $0) }
            let started = ProcessInfo.processInfo.systemUptime
            do {
                guard samples.allSatisfy({ $0.isFinite }) else {
                    throw NSError(domain: "HandyPhonon", code: 3, userInfo: [NSLocalizedDescriptionKey: "Invalid audio samples"])
                }
                // Preserve the full recording: the library handles recordings
                // beyond 35 seconds with overlapping windows at pauses.
                let text = try model.transcribe(samples).text
                try respond(["text": text, "ms": (ProcessInfo.processInfo.systemUptime - started) * 1000])
            } catch {
                try respond(["error": error.localizedDescription])
            }
            return true
        }
        if !keepGoing { break }
    }
} catch {
    try? respond(["error": error.localizedDescription])
    exit(1)
}
