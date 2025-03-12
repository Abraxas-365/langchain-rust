use async_trait::async_trait;
use sqlx::{postgres::PgPoolOptions, Column, Pool, Postgres, Row, TypeInfo};
use std::error::Error;

use crate::tools::{Dialect, Engine};

pub struct PostgreSQLEngine {
    pool: Pool<Postgres>,
}

impl PostgreSQLEngine {
    pub async fn new(dsn: &str) -> Result<Self, Box<dyn Error>> {
        let pool = PgPoolOptions::new().max_connections(5).connect(dsn).await?;

        Ok(PostgreSQLEngine { pool })
    }
}

impl From<PostgreSQLEngine> for Box<dyn Engine> {
    fn from(val: PostgreSQLEngine) -> Self {
        Box::new(val)
    }
}

#[async_trait]
impl Engine for PostgreSQLEngine {
    fn dialect(&self) -> Dialect {
        Dialect::PostgreSQL
    }

    async fn query(&self, query: &str) -> Result<(Vec<String>, Vec<Vec<String>>), Box<dyn Error>> {
        let rows = sqlx::query(query).fetch_all(&self.pool).await?;

        let mut cols = vec![];
        let mut results = vec![];

        if let Some(row) = rows.first() {
            cols = row
                .columns()
                .iter()
                .map(|col| col.name().to_string())
                .collect();
        }

        for row in rows {
            let mut result = Vec::with_capacity(cols.len());
            for index in 0..cols.len() {
                let column_type = row.columns()[index].type_info().name();

                let value_str = match column_type {
                    "TEXT[]" => {
                        // Fetch the TEXT[] column as a vector of strings
                        match row.try_get::<Vec<String>, _>(index) {
                            Ok(array) => format!("{:?}", array), // Format the vector as a string
                            Err(_) => "N/A".to_string(),
                        }
                    }
                    _ => {
                        // For other types, attempt to get them as strings
                        match row.try_get::<&str, _>(index) {
                            Ok(str_val) => str_val.to_string(),
                            Err(_) => {
                                // Fallback for types that cannot be directly converted to string
                                "N/A".to_string()
                            }
                        }
                    }
                };

                result.push(value_str);
            }
            results.push(result);
        }

        Ok((cols, results))
    }

    async fn table_names(&self) -> Result<Vec<String>, Box<dyn Error>> {
        let query =
            "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'";
        let rows = sqlx::query(query).fetch_all(&self.pool).await?;

        let table_names = rows
            .into_iter()
            .map(|row| row.get::<String, &str>("table_name"))
            .collect();

        Ok(table_names)
    }

    async fn table_info(&self, table: &str) -> Result<String, Box<dyn Error>> {
        let query =
            "SELECT column_name, data_type FROM information_schema.columns WHERE table_name = $1"
                .to_string();
        let rows = sqlx::query(&query)
            .bind(table)
            .fetch_all(&self.pool)
            .await?;

        // Simplified representation of table info, similar to a CREATE TABLE statement
        let info = rows
            .into_iter()
            .map(|row| {
                format!(
                    "{} {}",
                    row.get::<String, &str>("column_name"),
                    row.get::<String, &str>("data_type")
                )
            })
            .collect::<Vec<_>>()
            .join(", ");

        Ok(format!("CREATE TABLE {} ({})", table, info))
    }

    fn close(&self) -> Result<(), Box<dyn Error>> {
        // sqlx Pool is automatically closed when it goes out of scope
        Ok(())
    }
}
