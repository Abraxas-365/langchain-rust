use std::fmt::Display;

pub struct FormattedVec<T>(pub Vec<T>)
where
    T: Display;

impl<T> Display for FormattedVec<T>
where
    T: Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if !self.0.is_empty() {
            write!(
                f,
                "{}",
                self.0
                    .iter()
                    .map(|r| r.to_string())
                    .collect::<Vec<_>>()
                    .join("\n---\n")
            )?;
        } else {
            write!(f, "No results found, try adjusting your search query")?;
        }
        Ok(())
    }
}
