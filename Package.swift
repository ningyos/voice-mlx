// swift-tools-version: 5.10
import PackageDescription

let package = Package(
    name: "voice-mlx",
    platforms: [.macOS("15.0")],
    products: [
        .library(name: "VoiceMLX", targets: ["VoiceMLX"]),
        .library(name: "SpeakerSim", targets: ["SpeakerSim"]),
        .library(name: "VoiceAudio", targets: ["VoiceAudio"]),
        .executable(name: "voice-bench", targets: ["VoiceBench"]),
    ],
    dependencies: [
        // A range, not exact: dollcore embeds this package next to mlx-swift-lm, which needs
        // mlx-swift 0.32.x. Standalone builds stay on the resolved 0.31.6 (Package.resolved).
        .package(url: "https://github.com/ml-explore/mlx-swift", "0.31.6"..<"0.33.0"),
        .package(url: "https://github.com/huggingface/swift-transformers", exact: "1.4.0"),
    ],
    targets: [
        .target(name: "VoiceAudio"),
        .target(
            name: "VoiceMLX",
            dependencies: [
                "VoiceAudio",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift"),
                .product(name: "Hub", package: "swift-transformers"),
            ]
        ),
        .target(
            name: "SpeakerSim",
            dependencies: [
                "VoiceAudio",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
            ]
        ),
        .executableTarget(
            name: "VoiceBench",
            dependencies: [
                "VoiceMLX", "SpeakerSim", "VoiceAudio",
                .product(name: "MLXRandom", package: "mlx-swift"),
            ]
        ),
        .testTarget(name: "VoiceAudioTests", dependencies: ["VoiceAudio"], exclude: ["Fixtures"]),
        .testTarget(name: "VoiceMLXTests", dependencies: ["VoiceMLX", "VoiceAudio"]),
        .testTarget(
            name: "SpeakerSimTests", dependencies: ["SpeakerSim", "VoiceAudio"],
            resources: [.copy("Fixtures")]),
    ]
)
