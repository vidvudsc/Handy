// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "HandyPhonon",
    platforms: [.macOS(.v15)],
    dependencies: [
        .package(url: "https://github.com/fermionresearch/phonon-coreml", revision: "bf59e574639b060e30ee1a3f9c0e003fed5744c8")
    ],
    targets: [
        .executableTarget(name: "handy-phonon", dependencies: [.product(name: "PhononCoreML", package: "phonon-coreml")])
    ]
)
