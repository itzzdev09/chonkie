# 🦛 How the token chunker works

The token chunker cuts text into windows of up to `chunk_size` tokens, optionally starting each window with the last few tokens of the one before. It's the simplest chunker, and in v2 it holds the same guarantees as the others: every chunk is a slice of your text with an exact `token_count`.

## ✂️ The algorithm

1. **Find the bounds.** Encode the text once, and decode each token to get its length in bytes. The running sum gives the byte offset where each token ends. A token can end inside a character (gpt2 spends two tokens on `’`), so that offset moves back to the start of the character. These offsets are the only places a chunk may end.
2. **Fill a window.** From the chunk's start, guess the bound `chunk_size` tokens ahead, then count the text up to it. If the count is exactly `chunk_size`, that's the chunk. Otherwise gallop up or down from the guess and bisect to the last bound that fits.
3. **Step back for overlap.** With overlap, the next chunk starts at the earliest bound after which the rest of this chunk has at most `chunk_overlap` tokens. If that leaves no room to reach past this chunk, the next chunk starts with no overlap instead, so every chunk adds new text.
4. **Repeat** until a chunk reaches the end.

`chunk_overlap` takes tokens (`Overlap::Tokens(64)`) or a fraction of `chunk_size` (`Overlap::Fraction(0.25)`, rounded down), and counts inside `chunk_size`: a 512-token chunk with 64 tokens of overlap has 448 new tokens. It must be less than `chunk_size`.

Why count a window instead of trusting the token positions? Because the positions come from encoding the whole text, and a window's own text can encode differently at its edges. An edge that moved back to a character boundary is one example: the window loses part of a token, and its first token may encode differently once it starts the text. Counting each chunk's own text is what makes `token_count` exact. The guess is right nearly every time: on OpenWebText it takes about 1.04 counts per chunk.

Some tokenizers rewrite text before encoding it (BERT lowercases and strips accents), so their tokens don't decode back to your bytes. Then the bounds are character boundaries instead, and the same gallop and bisect finds the longest window that fits.

## ⚡ Numbers

1MB of OpenWebText, gpt2, one core, no overlap. **Warm** is the best of 3 passes, **cold** is the first.

| `chunk_size` | v2 warm | v2 cold | v1.7 + tiktoken |
| ------------ | ------: | ------: | --------------: |
| 128 | 80 MB/s | 13 MB/s | 8.9 MB/s |
| 512 | 110 MB/s | 12 MB/s | 11.1 MB/s |
| 2048 | 83 MB/s | 12 MB/s | 12.3 MB/s |

Cold, both are bound by the tokenizer encoding every word for the first time. Warm, v2 encodes the text about twice: once to find the bounds and once to count the chunks.

## 🔀 Differences from 1.7

- **Chunks slice the source.** 1.7 decoded each window of token ids back into text. When a window ended inside a character, the text got a `�` and was one character off from then on, so 1,724 of its 1,754 chunks at `chunk_size=128` had a `start_index` and `end_index` that didn't match their text. v2 ends windows on character boundaries, so `text[start..end]` is always the chunk.
- **Windows can hold fewer tokens.** A window that would end inside a character, or whose slice encodes to more tokens than it had in the whole text, ends a bound earlier. On the corpus above, v2 made 1,755 chunks at `chunk_size=128` against 1.7's 1,754.
- **Overlap is the same idea.** `chunk_overlap` works as in 1.7: tokens or a fraction, inside `chunk_size`.

## ✅ How the guarantees are tested

`crates/chonkie/tests/token.rs` reuses the recursive chunker's harness ([how it works](recursive.md#-how-the-guarantees-are-tested)): every tokenizer, chunk sizes 1, 2, 7, 64 and 512, and the same texts, each with no overlap, `Fraction(0.25)` and `Fraction(0.5)`. Rule 1 holds with overlap 0. With overlap, rule 2 checks instead that each chunk slices the source, starts after the chunk before it and no later than where it ended, repeats at most `chunk_overlap` tokens of it, and that the chunks reach the end. Rule 5 checks that no chunk edge splits a character. Smaller tests pin down the windows and overlap on a character tokenizer and reject bad settings.
