import Foundation
import MLX
import MLXNN

/// Build a Linear layer that is quantized when bits > 0, plain when bits == 0.
/// QuantizedLinear inherits from Linear, so the return type is always Linear and
/// caller code can store it in a single `@ModuleInfo var x: Linear` field.
public func makeMaybeQuantizedLinear(
    _ inputDimensions: Int,
    _ outputDimensions: Int,
    bias: Bool,
    groupSize: Int,
    bits: Int
) -> Linear {
    if bits > 0 {
        return QuantizedLinear.preQuantized(inputDimensions, outputDimensions, bias: bias,
                                            groupSize: groupSize, bits: bits)
    } else {
        return Linear(inputDimensions, outputDimensions, bias: bias)
    }
}

/// Generic weight loading utilities shared between ASR and TTS
public enum CommonWeightLoader {

    /// Apply weights to a Linear (or QuantizedLinear, since the latter inherits from
    /// the former). When the layer is a QuantizedLinear, `.scales`/`.biases` are wired
    /// in addition to `.weight`. For plain Linear those keys are absent in the
    /// safetensors (bf16/fp32 model) and only `.weight` (+ optional `.bias`) apply.
    public static func applyQuantizedLinearWeights(
        to linear: Linear,
        prefix: String,
        from weights: [String: MLXArray]
    ) {
        var params: [String: NestedItem<String, MLXArray>] = [:]

        if let weight = weights["\(prefix).weight"] {
            params["weight"] = .value(weight)
        }
        if linear is QuantizedLinear {
            if let scales = weights["\(prefix).scales"] {
                params["scales"] = .value(scales)
            }
            if let biases = weights["\(prefix).biases"] {
                params["biases"] = .value(biases)
            }
        }
        // Regular linear bias (separate from quantization biases)
        if let bias = weights["\(prefix).bias"] {
            params["bias"] = .value(bias)
        }

        if !params.isEmpty {
            linear.update(parameters: ModuleParameters(values: params))
        }
    }

    public static func applyRMSNormWeights(
        to norm: RMSNorm,
        prefix: String,
        from weights: [String: MLXArray]
    ) {
        var params: [String: NestedItem<String, MLXArray>] = [:]

        if let weight = weights["\(prefix).weight"] {
            params["weight"] = .value(weight)
        }

        if !params.isEmpty {
            norm.update(parameters: ModuleParameters(values: params))
        }
    }

    public static func applyLinearWeights(
        to linear: Linear,
        prefix: String,
        from weights: [String: MLXArray]
    ) {
        var params: [String: NestedItem<String, MLXArray>] = [:]

        if let weight = weights["\(prefix).weight"] {
            params["weight"] = .value(weight)
        }
        if let bias = weights["\(prefix).bias"] {
            params["bias"] = .value(bias)
        }

        if !params.isEmpty {
            linear.update(parameters: ModuleParameters(values: params))
        }
    }

    public static func applyLayerNormWeights(
        to layerNorm: LayerNorm,
        prefix: String,
        from weights: [String: MLXArray]
    ) {
        var params: [String: NestedItem<String, MLXArray>] = [:]

        if let weight = weights["\(prefix).weight"] {
            params["weight"] = .value(weight)
        }
        if let bias = weights["\(prefix).bias"] {
            params["bias"] = .value(bias)
        }

        if !params.isEmpty {
            layerNorm.update(parameters: ModuleParameters(values: params))
        }
    }

    public static func applyEmbeddingWeights(
        to embedding: Embedding,
        prefix: String,
        from weights: [String: MLXArray]
    ) {
        var params: [String: NestedItem<String, MLXArray>] = [:]

        if let weight = weights["\(prefix).weight"] {
            params["weight"] = .value(weight)
        }

        if !params.isEmpty {
            embedding.update(parameters: ModuleParameters(values: params))
        }
    }

    public static func applyConv1dWeights(
        to conv: Conv1d,
        prefix: String,
        from weights: [String: MLXArray],
        transpose: Bool = false
    ) {
        var params: [String: NestedItem<String, MLXArray>] = [:]

        if let weight = weights["\(prefix).weight"] {
            // PyTorch Conv1d: [out, in, kernel] -> MLX Conv1d: [out, kernel, in]
            let w = transpose ? weight.transposed(0, 2, 1) : weight
            params["weight"] = .value(w)
        }
        if let bias = weights["\(prefix).bias"] {
            params["bias"] = .value(bias)
        }

        if !params.isEmpty {
            conv.update(parameters: ModuleParameters(values: params))
        }
    }

    public static func applyConvTransposed1dWeights(
        to conv: ConvTransposed1d,
        prefix: String,
        from weights: [String: MLXArray],
        transpose: Bool = false
    ) {
        var params: [String: NestedItem<String, MLXArray>] = [:]

        if let weight = weights["\(prefix).weight"] {
            // PyTorch ConvTranspose1d: [in, out, kernel] -> MLX ConvTransposed1d: [out, kernel, in]
            let w = transpose ? weight.transposed(1, 2, 0) : weight
            params["weight"] = .value(w)
        }
        if let bias = weights["\(prefix).bias"] {
            params["bias"] = .value(bias)
        }

        if !params.isEmpty {
            conv.update(parameters: ModuleParameters(values: params))
        }
    }

    /// Apply QuantizedMLP weights (SwiGLU)
    public static func applyQuantizedMLPWeights(
        to mlp: QuantizedMLP,
        prefix: String,
        from weights: [String: MLXArray]
    ) {
        applyQuantizedLinearWeights(to: mlp.gateProj, prefix: "\(prefix).gate_proj", from: weights)
        applyQuantizedLinearWeights(to: mlp.upProj, prefix: "\(prefix).up_proj", from: weights)
        applyQuantizedLinearWeights(to: mlp.downProj, prefix: "\(prefix).down_proj", from: weights)
    }
}

extension QuantizedLinear {
    /// A quantized linear whose weight/scales/biases are zero placeholders to be filled
    /// from safetensors (speech-swift `MLXCommon/PreQuantizedLinear.swift`).
    static func preQuantized(
        _ inputDimensions: Int, _ outputDimensions: Int, bias: Bool = false, groupSize: Int = 64, bits: Int = 4
    ) -> QuantizedLinear {
        let layer = QuantizedLinear(
            weight: MLXArray.zeros([outputDimensions, inputDimensions * bits / 32], dtype: .uint32),
            bias: bias ? MLXArray.zeros([outputDimensions], dtype: .bfloat16) : nil,
            scales: MLXArray.zeros([outputDimensions, inputDimensions / groupSize], dtype: .bfloat16),
            biases: MLXArray.zeros([outputDimensions, inputDimensions / groupSize], dtype: .bfloat16),
            groupSize: groupSize,
            bits: bits)
        layer.freeze()
        return layer
    }
}
