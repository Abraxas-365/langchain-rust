use std::{collections::HashSet, error::Error, fmt};

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
pub enum Dialect {
    #[serde(rename = "mysql")]
    MySQL,
    #[serde(rename = "sqlite")]
    SQLite,
    #[serde(rename = "postgresql")]
    PostgreSQL,
}

impl fmt::Display for Dialect {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Dialect::MySQL => write!(f, "mysql"),
            Dialect::SQLite => write!(f, "sqlite"),
            Dialect::PostgreSQL => write!(f, "postgresql"),
        }
    }
}

#[async_trait]
pub trait Engine: Send + Sync {
    // Dialect returns the dialect(e.g. mysql, sqlite, postgre) of the database.
    fn dialect(&self) -> Dialect;
    // Query executes the query and returns the columns and results.
    async fn query(&self, query: &str) -> Result<(Vec<String>, Vec<Vec<String>>), Box<dyn Error>>;
    // TableNames returns all the table names of the database.
    async fn table_names(&self) -> Result<Vec<String>, Box<dyn Error>>;
    // TableInfo returns the table information of the database.
    // Typically, it returns the CREATE TABLE statement.
    async fn table_info(&self, tables: &str) -> Result<String, Box<dyn Error>>;
    // Close closes the database.
    fn close(&self) -> Result<(), Box<dyn Error>>;
}

pub struct SQLDatabase {
    pub engine: Box<dyn Engine>,
    pub sample_rows_number: i32,
    pub all_tables: HashSet<String>,
}

pub struct SQLDatabaseBuilder {
    engine: Box<dyn Engine>,
    sample_rows_number: i32,
    ignore_tables: HashSet<String>,
}

impl SQLDatabaseBuilder {
    pub fn new<E>(engine: E) -> Self
    where
        E: Engine + 'static,
    {
        SQLDatabaseBuilder {
            engine: Box::new(engine),
            sample_rows_number: 3, // Default value
            ignore_tables: HashSet::new(),
        }
    }

    // Function to set custom number of sample rows
    pub fn custom_sample_rows_number(mut self, number: i32) -> Self {
        self.sample_rows_number = number;
        self
    }

    // Function to set tables to ignore
    pub fn ignore_tables(mut self, ignore_tables: HashSet<String>) -> Self {
        self.ignore_tables = ignore_tables;
        self
    }

    // Function to build the SQLDatabase instance
    pub async fn build(self) -> Result<SQLDatabase, Box<dyn Error>> {
        let table_names_result = self.engine.table_names().await;

        // Handle potential error from table_names call
        let table_names = match table_names_result {
            Ok(names) => names,
            Err(error) => {
                return Err(error);
            }
        };

        // Filter out ignored tables
        let all_tables: HashSet<String> = table_names
            .into_iter()
            .filter(|name| !self.ignore_tables.contains(name))
            .collect();

        Ok(SQLDatabase {
            engine: self.engine,
            sample_rows_number: self.sample_rows_number,
            all_tables,
        })
    }
}

impl SQLDatabase {
    pub fn dialect(&self) -> Dialect {
        self.engine.dialect()
    }

    pub fn table_names(&self) -> Vec<String> {
        self.all_tables.iter().cloned().collect()
    }

    pub async fn table_info(&self, tables: &[String]) -> Result<String, Box<dyn Error>> {
        let mut tables: HashSet<String> = tables.iter().cloned().collect();
        if tables.is_empty() {
            tables = self.all_tables.clone();
        }
        let mut info = String::new();
        for table in tables {
            let table_info = self.engine.table_info(&table).await?;
            info.push_str(&table_info);
            info.push_str("\n\n");

            if self.sample_rows_number > 0 {
                let sample_rows = self.sample_rows(&table).await?;
                info.push_str("/*\n");
                info.push_str(&sample_rows);
                info.push_str("*/ \n\n");
            }
        }
        Ok(info)
    }

    pub async fn query(&self, query: &str) -> Result<String, Box<dyn Error>> {
        let (cols, results) = self.engine.query(query).await?;
        let mut str = cols.join("\t") + "\n";
        for row in results {
            str += &row.join("\t");
            str.push('\n');
        }
        Ok(str)
    }

    pub fn close(&self) -> Result<(), Box<dyn Error>> {
        self.engine.close()
    }

    pub async fn sample_rows(&self, table: &str) -> Result<String, Box<dyn Error>> {
        let query = format!("SELECT * FROM {} LIMIT {}", table, self.sample_rows_number);
        self.query(&query).await
    }
}
