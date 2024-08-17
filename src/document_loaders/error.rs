use std::{io, string::FromUtf8Error};

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
    FromUtf8Error(#[from] FromUtf8Error),

    #[error(transparent)]
    CSVError(#[from] csv::Error),

    #[cfg(feature = "lopdf")]
    #[error(transparent)]
    LoPdfError(#[from] lopdf::Error),

    #[cfg(feature = "pdf-extract")]
    #[error(transparent)]
    PdfExtractOutputError(#[from] pdf_extract::OutputError),

    #[error(transparent)]
    ReadabilityError(#[from] readability::error::Error),

    #[error(transparent)]
    JoinError(#[from] tokio::task::JoinError),

    #[cfg(feature = "git")]
    #[error(transparent)]
    DiscoveryError(#[from] gix::discover::Error),

    #[error("Error: {0}")]
    OtherError(String),
}
