mod common;

use chonkie::{Chunk, Overlap, TokenChunker, Tokenizer};
use common::Verdict;

const CHUNK_SIZES: [usize; 5] = [1, 2, 7, 64, 512];
const OVERLAPS: [Overlap; 3] = [
    Overlap::Tokens(0),
    Overlap::Fraction(0.25),
    Overlap::Fraction(0.5),
];

/// What a rule check can use besides the text and its chunks.
struct Case<'a> {
    chunk_size: usize,
    overlap: usize,
    chunker: &'a mut TokenChunker,
    /// A tokenizer the chunker never used, to count chunks independently.
    counter: &'a mut Tokenizer,
    /// Builds a new chunker, with a new tokenizer, configured the same way.
    fresh: &'a dyn Fn() -> TokenChunker,
    /// One fresh chunker per configuration, kept across texts by rule 4.
    other: &'a mut Option<TokenChunker>,
    corpus: &'a [String],
}

/// Runs `check` on every tokenizer, chunk size, overlap and corpus text, and
/// fails listing every case that broke the rule.
fn every_case(rule: &str, mut check: impl FnMut(&str, &[Chunk], Case) -> Verdict) {
    let corpus = common::corpus();
    let mut failures = Vec::new();
    for (name, make) in common::tokenizers() {
        let mut counter = make();
        for overlap in OVERLAPS {
            for chunk_size in CHUNK_SIZES {
                let tokens = match overlap {
                    Overlap::Tokens(n) => n,
                    Overlap::Fraction(f) => (f * chunk_size as f64) as usize,
                };
                let fresh = || {
                    TokenChunker::new(make())
                        .chunk_size(chunk_size)
                        .chunk_overlap(overlap)
                };
                let mut chunker = fresh();
                let mut other = None;
                for (i, text) in corpus.iter().enumerate() {
                    let chunks = chunker.chunk(text).unwrap();
                    let case = Case {
                        chunk_size,
                        overlap: tokens,
                        chunker: &mut chunker,
                        counter: &mut counter,
                        fresh: &fresh,
                        other: &mut other,
                        corpus: &corpus,
                    };
                    if let Err(why) = check(text, &chunks, case) {
                        failures.push(format!(
                            "{name}, {overlap:?}, chunk_size {chunk_size}, corpus text {i}: {why}"
                        ));
                    }
                }
            }
        }
    }
    assert!(
        failures.is_empty(),
        "{rule} broke in {} case(s):\n{}",
        failures.len(),
        failures.join("\n")
    );
}

#[test]
fn rule_1_chunks_rebuild_the_source() {
    every_case("rule 1", |text, chunks, case| match case.overlap {
        0 => common::rebuilds_source(text, chunks),
        _ => Ok(()),
    });
}

#[test]
fn rule_2_offsets_slice_the_source() {
    every_case("rule 2", |text, chunks, case| match case.overlap {
        0 => common::offsets_slice_source(text, chunks),
        n => overlaps_slice_source(text, chunks, case.counter, n),
    });
}

#[test]
fn rule_3_token_counts_are_exact_and_fit() {
    every_case("rule 3", |_, chunks, case| {
        common::token_counts_exact(chunks, case.counter, case.chunk_size)
    });
}

#[test]
fn rule_4_same_input_gives_same_chunks() {
    // Again with the same chunker, whose tokenizer cache is now warm, and with
    // a second one that saw the whole corpus in reverse first, so no state
    // carried between calls can change a chunk. Building that one once per
    // configuration, not once per text, keeps tokenizer loads off the clock.
    every_case("rule 4", |text, chunks, case| {
        common::same_chunks(chunks, &case.chunker.chunk(text).unwrap())?;
        let other = case.other.get_or_insert_with(|| {
            let mut other = (case.fresh)();
            for text in case.corpus.iter().rev() {
                other.chunk(text).unwrap();
            }
            other
        });
        common::same_chunks(chunks, &other.chunk(text).unwrap())
    });
}

#[test]
fn rule_5_no_dropped_chars_or_split_codepoints() {
    every_case("rule 5", |text, chunks, case| match case.overlap {
        0 => common::keeps_every_char(text, chunks),
        _ => {
            let edges = chunks.iter().flat_map(|c| [c.start, c.end]);
            match edges.into_iter().find(|&at| !text.is_char_boundary(at)) {
                Some(at) => Err(format!("a chunk edge at byte {at} is inside a codepoint")),
                None => Ok(()),
            }
        }
    });
}

/// Rule 2 with overlap: each chunk slices the source, starts after the one
/// before it and no later than where that one ended, and repeats at most
/// `overlap` tokens of it. The chunks still cover the source end to end.
fn overlaps_slice_source(
    source: &str,
    chunks: &[Chunk],
    counter: &mut Tokenizer,
    overlap: usize,
) -> Verdict {
    let (mut start, mut end) = (None, 0);
    for (i, chunk) in chunks.iter().enumerate() {
        if source.get(chunk.start..chunk.end) != Some(chunk.text) {
            return Err(format!(
                "chunk {i}: source[{}..{}] is not its text",
                chunk.start, chunk.end
            ));
        }
        let fresh = start.is_none_or(|s| chunk.start > s);
        if !fresh || chunk.start > end || (i == 0 && chunk.start != 0) || chunk.end <= end {
            return Err(format!(
                "chunk {i} at {}..{} doesn't follow on from ..{end}",
                chunk.start, chunk.end
            ));
        }
        let repeated = counter.count(&source[chunk.start..end]).unwrap();
        if repeated > overlap {
            return Err(format!(
                "chunk {i} repeats {repeated} tokens, over chunk_overlap {overlap}"
            ));
        }
        (start, end) = (Some(chunk.start), chunk.end);
    }
    match end == source.len() {
        true => Ok(()),
        false => Err(format!(
            "the last chunk ends at {end}, the source at {}",
            source.len()
        )),
    }
}

#[test]
fn windows_hold_chunk_size_tokens() {
    let chunks = TokenChunker::new(Tokenizer::characters())
        .chunk_size(4)
        .chunk("abcdefghij")
        .unwrap();
    let texts: Vec<&str> = chunks.iter().map(|c| c.text).collect();
    assert_eq!(texts, ["abcd", "efgh", "ij"]);
}

#[test]
fn overlap_counts_inside_chunk_size() {
    for overlap in [Overlap::Tokens(2), Overlap::Fraction(0.5)] {
        let chunks = TokenChunker::new(Tokenizer::characters())
            .chunk_size(4)
            .chunk_overlap(overlap)
            .chunk("abcdefghij")
            .unwrap();
        let texts: Vec<&str> = chunks.iter().map(|c| c.text).collect();
        assert_eq!(texts, ["abcd", "cdef", "efgh", "ghij"]);
    }
}

#[test]
fn empty_text_has_no_chunks() {
    assert!(
        TokenChunker::new(Tokenizer::characters())
            .chunk("")
            .unwrap()
            .is_empty()
    );
}

#[test]
fn rejects_bad_config() {
    let bad = [
        (0, Overlap::Tokens(0)),
        (4, Overlap::Tokens(4)),
        (4, Overlap::Fraction(1.0)),
        (4, Overlap::Fraction(-0.1)),
    ];
    for (size, overlap) in bad {
        let result = TokenChunker::new(Tokenizer::characters())
            .chunk_size(size)
            .chunk_overlap(overlap)
            .chunk("hi");
        assert!(result.is_err(), "chunk_size {size}, {overlap:?}");
    }
}
