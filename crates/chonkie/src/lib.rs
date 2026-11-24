//! Chonkie: small, fast, correct text chunking.
//!
//! Every chunk borrows its text from the input. `start` and `end` are byte
//! offsets, so `&text[chunk.start..chunk.end] == chunk.text`.

mod delim;
mod error;
mod recursive;
mod token;
mod tokenizer;
mod types;

pub use error::Error;
pub use recursive::{Include, Level, RecursiveChunker};
pub use token::{Overlap, TokenChunker};
pub use tokenizer::{DEFAULT_TOKENIZER, Tokenizer};
pub use types::Chunk;

pub type Result<T> = std::result::Result<T, Error>;
