//! Sentence previews and lossless UTF-16-bounded pages for Telegram cards.

pub(super) fn preview_end(text: &str) -> usize {
    let mut sentence_count = 0;
    let chars: Vec<(usize, char)> = text.char_indices().collect();
    let mut i = 0;
    while i < chars.len() {
        let (start, ch) = chars[i];
        if !matches!(ch, '.' | '!' | '?' | '…') {
            i += 1;
            continue;
        }
        if ch == '.'
            && i > 0
            && i + 1 < chars.len()
            && chars[i - 1].1.is_ascii_digit()
            && chars[i + 1].1.is_ascii_digit()
        {
            i += 1;
            continue;
        }
        let mut end = i + 1;
        while end < chars.len() && matches!(chars[end].1, '.' | '!' | '?' | '…') {
            end += 1;
        }
        while end < chars.len() && matches!(chars[end].1, '"' | '\'' | '»' | '”' | ')' | ']' | '}')
        {
            end += 1;
        }
        if end < chars.len() && !chars[end].1.is_whitespace() {
            i = end;
            continue;
        }
        if ch == '.' && is_abbreviation(&text[..start]) {
            i = end;
            continue;
        }
        sentence_count += 1;
        if sentence_count == 3 {
            return chars.get(end).map_or(text.len(), |(byte, _)| *byte);
        }
        i = end;
    }
    text.len()
}

fn is_abbreviation(prefix: &str) -> bool {
    let token = prefix
        .rsplit(|ch: char| ch.is_whitespace() || matches!(ch, '(' | '[' | '«'))
        .next()
        .unwrap_or("");
    let token = token.trim_matches(|ch: char| matches!(ch, '"' | '\''));
    let lower = token.to_ascii_lowercase();
    let common = matches!(
        lower.as_str(),
        "mr" | "mrs" | "ms" | "dr" | "prof" | "sr" | "jr" | "st" | "vs" | "etc" | "e.g" | "i.e"
    );
    if common {
        return true;
    }
    if token.chars().count() == 1 && token.chars().all(char::is_uppercase) {
        let prior = prefix[..prefix.len() - token.len()].trim_end();
        let prior_token = prior
            .split_whitespace()
            .last()
            .unwrap_or("")
            .trim_end_matches('.')
            .to_ascii_lowercase();
        return !matches!(prior_token.as_str(), "mr" | "mrs" | "ms" | "dr" | "prof");
    }
    false
}

pub(super) fn pages(text: &str, max_utf16: usize) -> Vec<&str> {
    assert!(max_utf16 > 0);
    if text.is_empty() {
        return vec![""];
    }
    let mut out = Vec::new();
    let mut offset = 0;
    while offset < text.len() {
        let rest = &text[offset..];
        let mut units = 0;
        let mut fit_end = 0;
        let mut whitespace_end = 0;
        for (byte, ch) in rest.char_indices() {
            if units + ch.len_utf16() > max_utf16 {
                break;
            }
            units += ch.len_utf16();
            fit_end = byte + ch.len_utf8();
            if ch.is_whitespace() {
                whitespace_end = fit_end;
            }
        }
        // Page budgets are always much larger than one scalar in production.
        let end = if fit_end == rest.len() || whitespace_end == 0 {
            fit_end
        } else {
            whitespace_end
        };
        assert!(end > 0);
        out.push(&rest[..end]);
        offset += end;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn three_sentences_handle_quotes_abbreviations_decimals_and_fragments() {
        assert_eq!(preview_end(""), 0);
        assert_eq!(preview_end("One. Two."), "One. Two.".len());
        let sample = "Dr. A met Mr. B. It scored 8.5! «Really?!» Fourth.";
        assert_eq!(
            &sample[..preview_end(sample)],
            "Dr. A met Mr. B. It scored 8.5! «Really?!»"
        );
        let sample = "One. Two? Third without punctuation Fourth";
        assert_eq!(preview_end(sample), sample.len());
    }

    #[test]
    fn pages_are_scalar_safe_and_lossless() {
        let original = " 🌸 word\n".repeat(100);
        let chunks = pages(&original, 17);
        assert_eq!(chunks.concat(), original);
        assert!(chunks
            .iter()
            .all(|chunk| chunk.encode_utf16().count() <= 17));
        let original = "🌸".repeat(20);
        assert_eq!(pages(&original, 3).concat(), original);
    }
}
