use crate::{Chunk, Error, Result, Tokenizer};

/// How much of the previous chunk the next one starts with.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Overlap {
    /// Up to this many tokens.
    Tokens(usize),
    /// Up to this fraction of `chunk_size`, rounded down. Must be in `[0, 1)`.
    Fraction(f64),
}

impl Default for Overlap {
    fn default() -> Self {
        Overlap::Tokens(0)
    }
}

/// Windows of up to `chunk_size` tokens. With overlap, each chunk starts
/// with up to `chunk_overlap` tokens from the end of the one before, counted
/// inside `chunk_size`.
///
/// Chunks slice the source, so a window that ends inside a character ends
/// before it instead. Tokenizers that rewrite text, whose tokens don't map
/// back to bytes, cut at character boundaries instead of token ends.
/// Where chunks may end, as byte offsets, and how many tokens of the whole
/// text come before each.
struct Bounds {
    at: Vec<usize>,
    tokens: Vec<usize>,
}

pub struct TokenChunker {
    tokenizer: Tokenizer,
    chunk_size: usize,
    overlap: Overlap,
    /// Decoded byte length of each token id, filled in as ids are seen.
    lens: Vec<u32>,
}

impl Default for TokenChunker {
    fn default() -> Self {
        Self::new(Tokenizer::default())
    }
}

impl TokenChunker {
    pub fn new(tokenizer: impl Into<Tokenizer>) -> Self {
        Self {
            tokenizer: tokenizer.into(),
            chunk_size: 2048,
            overlap: Overlap::default(),
            lens: Vec::new(),
        }
    }

    pub fn chunk_size(mut self, chunk_size: usize) -> Self {
        self.chunk_size = chunk_size;
        self
    }

    pub fn chunk_overlap(mut self, overlap: Overlap) -> Self {
        self.overlap = overlap;
        self
    }

    pub fn tokenizer(&mut self) -> &mut Tokenizer {
        self.lens.clear();
        &mut self.tokenizer
    }

    pub fn chunk<'a>(&mut self, text: &'a str) -> Result<Vec<Chunk<'a>>> {
        let overlap = self.overlap_tokens()?;
        let mut chunks = Vec::new();
        if text.is_empty() {
            return Ok(chunks);
        }
        let Bounds { at, tokens } = self.bounds(text)?;
        let tokens_to = |i: Option<usize>| i.map_or(0, |i| tokens[i]);
        // The chunk starts at `start`, just after bound `before` if any, and
        // must reach past bound `next - 1`, where the previous chunk ended.
        let (mut start, mut before, mut next) = (0, None, 0);
        loop {
            let budget = tokens_to(before) + self.chunk_size;
            let guess = tokens.partition_point(|&t| t <= budget).saturating_sub(1);
            let (idx, n) = match self.last_fit(text, start, &at, next, guess)? {
                Some(found) => found,
                // The overlap leaves no room for the next unit, so drop it.
                None if before.is_some_and(|b| b + 1 < next) => {
                    (start, before) = (at[next - 1], Some(next - 1));
                    continue;
                }
                // A single unit over budget still becomes its own chunk.
                None => (next, self.tokenizer.count(&text[start..at[next]])?),
            };
            let end = at[idx];
            chunks.push(Chunk::new(text, start, end, n));
            if end == text.len() {
                return Ok(chunks);
            }
            let from = before.map_or(0, |b| b + 1);
            let tail = (from..idx).find(|&j| tokens[idx] - tokens[j] <= overlap);
            before = match tail {
                Some(j) => self.first_tail(text, end, &at, j, idx, overlap)?,
                None => None,
            }
            .or(Some(idx));
            start = at[before.unwrap()];
            next = idx + 1;
        }
    }

    fn overlap_tokens(&self) -> Result<usize> {
        if self.chunk_size == 0 {
            return Err(Error::Config("chunk_size must be at least 1".into()));
        }
        let n = match self.overlap {
            Overlap::Tokens(n) => n,
            Overlap::Fraction(f) if (0.0..1.0).contains(&f) => {
                (f * self.chunk_size as f64) as usize
            }
            Overlap::Fraction(f) => {
                return Err(Error::Config(format!(
                    "chunk_overlap fraction {f} is not in [0, 1)"
                )));
            }
        };
        if n >= self.chunk_size {
            return Err(Error::Config(
                "chunk_overlap must be less than chunk_size".into(),
            ));
        }
        Ok(n)
    }

    /// Where a chunk may end: after each token, moved back to a character
    /// boundary, or after each character when tokens don't map to bytes.
    fn bounds(&mut self, text: &str) -> Result<Bounds> {
        let Some(ends) = self.tokenizer.token_ends(text, &mut self.lens)? else {
            let at: Vec<usize> = text.char_indices().map(|(i, c)| i + c.len_utf8()).collect();
            let tokens = (1..=at.len()).collect();
            return Ok(Bounds { at, tokens });
        };
        let mut bounds = Bounds {
            at: Vec::with_capacity(ends.len()),
            tokens: Vec::with_capacity(ends.len()),
        };
        for (i, mut end) in ends.into_iter().enumerate() {
            while !text.is_char_boundary(end) {
                end -= 1;
            }
            if end > bounds.at.last().copied().unwrap_or(0) {
                bounds.at.push(end);
                bounds.tokens.push(i + 1);
            }
        }
        Ok(bounds)
    }

    /// The last bound at or after `first` whose chunk from `start` fits
    /// `chunk_size`, with its count. Gallops up or down from `guess`, then bisects. A
    /// guess that fills `chunk_size` exactly is taken as is.
    fn last_fit(
        &mut self,
        text: &str,
        start: usize,
        bounds: &[usize],
        first: usize,
        guess: usize,
    ) -> Result<Option<(usize, usize)>> {
        let size = self.chunk_size;
        let count = |i: usize, tok: &mut Tokenizer| tok.count(&text[start..bounds[i]]);
        let last = bounds.len() - 1;
        let mut best;
        let (mut lo, mut hi);
        let g = guess.clamp(first, last);
        let n = count(g, &mut self.tokenizer)?;
        if n == size || (n < size && g == last) {
            return Ok(Some((g, n)));
        }
        if n < size {
            best = Some((g, n));
            let mut step = 1;
            loop {
                let probe = best.map_or(g, |b: (usize, usize)| b.0) + step;
                if probe > last {
                    (lo, hi) = (best.unwrap().0 + 1, last);
                    break;
                }
                let n = count(probe, &mut self.tokenizer)?;
                if n > size {
                    (lo, hi) = (best.unwrap().0 + 1, probe - 1);
                    break;
                }
                best = Some((probe, n));
                step *= 2;
            }
        } else {
            let (mut fail, mut step) = (g, 1);
            loop {
                if fail == first {
                    return Ok(None);
                }
                let probe = fail.saturating_sub(step).max(first);
                let n = count(probe, &mut self.tokenizer)?;
                if n <= size {
                    best = Some((probe, n));
                    (lo, hi) = (probe + 1, fail - 1);
                    break;
                }
                (fail, step) = (probe, step * 2);
            }
        }
        while lo <= hi {
            let mid = (lo + hi) / 2;
            let n = count(mid, &mut self.tokenizer)?;
            if n <= size {
                best = Some((mid, n));
                lo = mid + 1;
            } else if mid == 0 {
                break;
            } else {
                hi = mid - 1;
            }
        }
        Ok(best)
    }

    /// The first bound from `guess` up to `idx` after which the rest of the
    /// chunk, up to `end`, has at most `overlap` tokens.
    fn first_tail(
        &mut self,
        text: &str,
        end: usize,
        at: &[usize],
        guess: usize,
        idx: usize,
        overlap: usize,
    ) -> Result<Option<usize>> {
        for j in guess..idx {
            if self.tokenizer.count(&text[at[j]..end])? <= overlap {
                return Ok(Some(j));
            }
        }
        Ok(None)
    }
}
