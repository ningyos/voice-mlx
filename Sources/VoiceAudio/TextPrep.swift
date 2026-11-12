import Foundation

/// Port of the text helpers OmniVoice applies to reference transcripts
/// (`omnivoice/utils/text.py`).
public enum TextPrep {
    /// `END_PUNCTUATION` from `omnivoice/utils/text.py`. The Python set also contains the
    /// two-character "……", which can never equal a single trailing character.
    static let endPunctuation: Set<Unicode.Scalar> = [
        ";", ":", ",", ".", "!", "?", "…", ")", "]", "}", "\"", "'",
        "\u{201C}", "\u{201D}", "\u{2018}", "\u{2019}",
        "；", "：", "，", "。", "！", "？", "、", "）", "】",
    ]

    /// Code points for which Python's `str.isspace()` is true (what `str.strip()` removes).
    static let pythonWhitespace: Set<Unicode.Scalar> = Set(
        ([0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x1C, 0x1D, 0x1E, 0x1F, 0x20, 0x85, 0xA0, 0x1680]
            + Array(0x2000 ... 0x200A)
            + [0x2028, 0x2029, 0x202F, 0x205F, 0x3000])
            .map { Unicode.Scalar(UInt32($0))! })

    /// `add_punctuation`: strip Python whitespace; if the last code point is not end
    /// punctuation, append "。" when any CJK unified ideograph (U+4E00–U+9FFF) occurs,
    /// otherwise ".". Empty (after stripping) text is returned as is.
    public static func addPunctuation(_ text: String) -> String {
        let scalars = Array(text.unicodeScalars)
        guard let first = scalars.firstIndex(where: { !pythonWhitespace.contains($0) }),
              let last = scalars.lastIndex(where: { !pythonWhitespace.contains($0) })
        else { return "" }
        var view = String.UnicodeScalarView(scalars[first ... last])
        if !endPunctuation.contains(scalars[last]) {
            let isChinese = view.contains { (0x4E00 ... 0x9FFF).contains($0.value) }
            view.append(isChinese ? "。" : ".")
        }
        return String(view)
    }
}
