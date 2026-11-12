// swift-tools-version: 5.10
import PackageDescription

let package = Package(
    name: "omnivoice-mlx",
    platforms: [.macOS("15.0")],
    products: [
        .library(name: "OmniVoiceMLX", targets: ["OmniVoiceMLX"]),
        .library(name: "OmniVoiceSim", targets: ["OmniVoiceSim"]),
        .library(name: "OmniVoiceAudio", targets: ["OmniVoiceAudio"]),
        .executable(name: "omnivoice-bench", targets: ["OmniVoiceBench"]),
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift", exact: "0.31.6"),
        .package(url: "https://github.com/huggingface/swift-transformers", exact: "1.4.0"),
    ],
    targets: [
        .target(name: "OmniVoiceAudio"),
        .target(
            name: "OmniVoiceMLX",
            dependencies: [
                "OmniVoiceAudio",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift"),
                .product(name: "Hub", package: "swift-transformers"),
                .product(name: "Tokenizers", package: "swift-transformers"),
            ]
        ),
        .target(
            name: "OmniVoiceSim",
            dependencies: [
                "OmniVoiceAudio",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
            ]
        ),
        .executableTarget(
            name: "OmniVoiceBench",
            dependencies: ["OmniVoiceMLX", "OmniVoiceSim", "OmniVoiceAudio"]
        ),
        .testTarget(name: "OmniVoiceAudioTests", dependencies: ["OmniVoiceAudio"], exclude: ["Fixtures"]),
        .testTarget(name: "OmniVoiceMLXTests", dependencies: ["OmniVoiceMLX", "OmniVoiceAudio"]),
        .testTarget(
            name: "OmniVoiceSimTests", dependencies: ["OmniVoiceSim", "OmniVoiceAudio"],
            resources: [.copy("Fixtures")]),
    ]
)
