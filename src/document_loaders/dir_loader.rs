use async_recursion::async_recursion;
use std::{path::Path, pin::Pin};
use tokio::fs;
use regex::Regex;

use super::LoaderError;

#[derive(Debug, Clone, Default)]
pub struct DirLoaderOptions {
    pub glob: Option<String>,
    pub suffixes: Option<Vec<String>>,
    pub exclude_files: Option<Vec<Regex>>,
    pub exclude_dirs: Option<Vec<Regex>>,
}

/// Recursively list all files in a directory
#[async_recursion]
pub async fn list_files_in_path(
    dir_path: &Path,
    files: &mut Vec<String>,
    opts: &DirLoaderOptions
) -> Result<Pin<Box<()>>, LoaderError> {
    if dir_path.is_file() {
        files.push(dir_path.to_string_lossy().to_string());
        return Ok(Box::pin(()));
    }
    if !dir_path.is_dir() {
        return Err(LoaderError::OtherError(format!(
            "Path is not a directory: {:?}",
            dir_path
        )));
    }
    let mut reader = fs::read_dir(dir_path).await.unwrap();
    while let Some(entry) = reader.next_entry().await.unwrap() {
        let path = entry.path();
        if path.is_file() {
            files.push(path.to_string_lossy().to_string());
        } else if path.is_dir() {
            let file_name = path.file_name().unwrap().to_str().unwrap_or("Invalid dir name");
            if let Some(exclude_list) = &opts.exclude_dirs {
                // Check if the path matches any of the regex patterns
                let is_excluded = exclude_list.iter().any(|regex| regex.is_match(file_name));
                if is_excluded {
                    continue;
                }
            }

            list_files_in_path(&path, files, opts).await.unwrap();
        }
    }
    Ok(Box::pin(()))
}

/// Find files in a directory that match the given options
pub async fn find_files_with_extension(folder_path: &str, opts: &DirLoaderOptions) -> Vec<String> {
    let mut matching_files = Vec::new();
    let folder_path = Path::new(folder_path);
    let mut all_files: Vec<String> = Vec::new();

    list_files_in_path(folder_path, &mut all_files, &opts.clone())
        .await
        .unwrap();

    for file_name in all_files {
        let path_str = file_name;

        // check if the file has the required extension
        if let Some(suffixes) = &opts.suffixes {
            let mut has_suffix = false;
            for suffix in suffixes {
                if path_str.ends_with(suffix) {
                    has_suffix = true;
                    break;
                }
            }
            if !has_suffix {
                continue;
            }
        }

        if let Some(exclude_list) = &opts.exclude_files {
            // Check if the path matches any of the regex patterns
            let is_excluded = exclude_list.iter().any(|regex| regex.is_match(&path_str));

            if is_excluded {
                continue;  // Skip this path if it matches any of the regex patterns
            }
        }

        // check if the file matches the glob pattern
        if let Some(glob_pattern) = &opts.glob {
            let glob = glob::Pattern::new(glob_pattern).unwrap();
            if !glob.matches(&path_str) {
                continue;
            }
        }

        matching_files.push(path_str);
    }

    matching_files
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;

    #[tokio::test]
    async fn test_find_files_with_extension() {
        // Create a temporary directory for testing
        let temp_dir = env::temp_dir().join("dir_loader_test_dir");

        if temp_dir.exists() {
            fs::remove_dir_all(&temp_dir)
                .await
                .expect("Failed to remove existing directory");
        }

        fs::create_dir(&temp_dir)
            .await
            .expect("Failed to create temporary directory");
        // Create some files with different extensions
        let file_paths = [
            temp_dir.as_path().join("file1.txt"),
            temp_dir.as_path().join("file2.txt"),
            temp_dir.as_path().join("file3.md"),
            temp_dir.as_path().join("file4.txt"),
        ];

        // Write some content to the files
        for path in &file_paths {
            let content = "Hello, world!";
            std::fs::write(path, content).expect("Failed to write file");
        }

        // Call the function to find files with the ".txt" extension
        let found_files = find_files_with_extension(
            temp_dir.as_path().to_str().unwrap(),
            &DirLoaderOptions {
                glob: None,
                suffixes: Some(vec![".txt".to_string()]),
                exclude_dirs: None,
                exclude_files: None
            },
        )
            .await
            .into_iter()
            .collect::<Vec<_>>();

        // Expecting to find 3 files with ".txt" extension
        assert_eq!(found_files.len(), 3);
        // Expecting each file name to contain ".txt" extension
        for file in &found_files {
            assert!(file.ends_with(".txt"));
        }
        assert!(found_files.contains(&temp_dir.join("file1.txt").to_string_lossy().to_string()),);
        assert!(found_files.contains(&temp_dir.join("file2.txt").to_string_lossy().to_string()),);
        assert!(found_files.contains(&temp_dir.join("file4.txt").to_string_lossy().to_string()),);

        // Clean up: remove the temporary directory and its contents
        fs::remove_dir_all(&temp_dir)
            .await
            .expect("Failed to remove temporary directory");
    }
}
