use std::io;

use thiserror::Error;

use crate::text_splitter::TextSplitterError;

#[derive(Error, Debug)]
pub enum LoaderError {
    #[error("Error loading document: {0}")]
    LoadDocumentError(String),

    #[error("{0}")]
    TextSplitterError(#[from] TextSplitterError),

    #[error(transparent)]
    IOError(#[from] io::Error),

    #[error(transparent)]
    CSVError(#[from] csv::Error),

    #[error(transparent)]
    LoPdfError(#[from] lopdf::Error),

    #[error(transparent)]
    ReadabilityError(#[from] readability::error::Error),

    #[error(transparent)]
    JoinError(#[from] tokio::task::JoinError),

    #[error(transparent)]
    DiscoveryError(#[from] gix::discover::Error),

    #[error("Error: {0}")]
    OtherError(String),
}
