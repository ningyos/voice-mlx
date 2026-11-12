import Foundation

/// Sampling parameters. Defaults are the official `qwen_tts` defaults
/// (`generate_config.json` of Qwen3-TTS-12Hz-0.6B), except `maxNewTokens`.
public struct SamplingOptions: Sendable, Equatable {
    /// Talker (first codebook): multinomial sampling when true, argmax otherwise.
    public var doSample = true
    public var temperature: Float = 0.9
    public var topK = 50
    public var topP: Float = 1.0
    /// HF-style repetition penalty over the talker's previously generated first-codebook
    /// tokens (applied in greedy mode too, like HF `generate`).
    public var repetitionPenalty: Float = 1.05
    /// Code predictor (codebooks 1–15).
    public var subtalkerDoSample = true
    public var subtalkerTemperature: Float = 0.9
    public var subtalkerTopK = 50
    /// Codec frames (12.5 per second) generated at most. The official
    /// `generate_config.json` says 8192 (~11 min); 2048 (~164 s) is the `qwen_tts` hard default.
    public var maxNewTokens = 2048
    /// EOS is masked for the first `minNewTokens` frames (official: 2).
    public var minNewTokens = 2

    public init() {}

    /// Deterministic decoding for both talker and code predictor.
    public static var greedy: SamplingOptions {
        var s = SamplingOptions()
        s.doSample = false
        s.subtalkerDoSample = false
        return s
    }

    func validate() throws {
        if doSample && !(temperature > 0) { throw VoiceError.invalidArgument("temperature must be > 0 when sampling") }
        if subtalkerDoSample && !(subtalkerTemperature > 0) {
            throw VoiceError.invalidArgument("subtalkerTemperature must be > 0 when sampling")
        }
        if !(topP > 0 && topP <= 1) { throw VoiceError.invalidArgument("topP must be in (0, 1]") }
        if topK < 0 || subtalkerTopK < 0 { throw VoiceError.invalidArgument("topK must be >= 0") }
        if !(repetitionPenalty > 0) { throw VoiceError.invalidArgument("repetitionPenalty must be > 0") }
        if maxNewTokens < 1 { throw VoiceError.invalidArgument("maxNewTokens must be >= 1") }
        if minNewTokens < 0 { throw VoiceError.invalidArgument("minNewTokens must be >= 0") }
    }
}

/// How the target text enters the talker (`non_streaming_mode` in `qwen_tts`).
public enum TextFeed: Sendable, Equatable {
    /// Text tokens are fed one per generated frame after the prompt
    /// (`non_streaming_mode=False`; official default for voice cloning).
    case interleaved
    /// The whole text is in the prompt (`non_streaming_mode=True`; official default for
    /// `generate_custom_voice`).
    case upfront
}

public struct GenerationOptions: Sendable {
    public var sampling = SamplingOptions()
    /// `nil` draws a fresh seed (reported in the result) so any run can be replayed.
    public var seed: UInt64?
    /// `nil` = the official default for the path (ICL: `.interleaved`, speaker: `.upfront`).
    public var textFeed: TextFeed?
    /// Codec decode chunking (`chunked_decode(chunk_size=300, left_context_size=25)` in
    /// the reference).
    public var decodeChunkFrames = 300
    public var decodeLeftContextFrames = 25
    /// ICL: only the last `referenceDecodeContextFrames` reference frames are prepended for
    /// the decode (the reference prepends all of them). `nil` = all (reference behaviour).
    public var referenceDecodeContextFrames: Int?
    /// Evaluate and time every stage separately (adds GPU syncs; for profiling only).
    public var profile = false

    public init() {}
}

/// Chunking of the streaming variant.
public struct StreamingOptions: Sendable {
    /// Frames generated before the first chunk is decoded (12.5 frames = 1 s).
    public var firstChunkFrames = 4
    /// Frames per later chunk.
    public var chunkFrames = 12
    /// Previously generated (or reference) frames re-decoded as left context per chunk.
    public var leftContextFrames = 25

    public init() {}
}
