//! Id lookups shared by the sqlite-vec and sqlite-vss stores. Both keep documents in a
//! plain `{table}` (rowid, text, metadata, text_embedding) and vectors in a virtual
//! index table keyed by the same rowid; the ids `add_documents` returns are rowids.

use std::{collections::HashMap, error::Error};

use serde_json::{json, Value};
use sqlx::{sqlite::SqliteRow, AssertSqlSafe, Pool, Row, Sqlite};

use crate::{
    schemas::Document,
    vectorstore::{vectorstore::in_id_order, VecStoreOptions},
};

/// The rowids in `ids` as one JSON array (bound once, read with `json_each`, so there's
/// no limit on bound parameters). Ids that aren't integers can't name a row and are
/// dropped; `None` when nothing is left.
pub(crate) fn rowid_param(ids: &[String]) -> Option<String> {
    let rowids: Vec<i64> = ids.iter().filter_map(|id| id.parse().ok()).collect();
    (!rowids.is_empty()).then(|| json!(rowids).to_string())
}

/// Id lookups go straight to rows: there is nothing for `filters`/`name_space` to narrow.
pub(crate) fn reject_id_options(opt: &VecStoreOptions<Value>) -> Result<(), Box<dyn Error>> {
    if opt.filters.is_some() || opt.name_space.is_some() {
        return Err("sqlite delete/get_by_ids don't support filters or name_space".into());
    }
    Ok(())
}

/// The `text`/`metadata` columns of a row as a document (score 0).
fn row_document(row: &SqliteRow, id: String) -> Result<Document, sqlx::Error> {
    let page_content: String = row.try_get("text")?;
    let metadata = match row.try_get::<Option<Value>, _>("metadata")? {
        Some(Value::Object(obj)) => obj.into_iter().collect(),
        _ => HashMap::new(),
    };
    Ok(Document {
        page_content,
        metadata,
        score: 0.0,
        id: Some(id),
    })
}

/// Rows of `table` by rowid, in request order.
pub(crate) async fn rows_by_id(
    pool: &Pool<Sqlite>,
    table: &str,
    ids: &[String],
) -> Result<Vec<Document>, Box<dyn Error>> {
    let Some(rowids) = rowid_param(ids) else {
        return Ok(Vec::new());
    };
    let rows = sqlx::query(AssertSqlSafe(format!(
        "SELECT rowid, text, metadata FROM {table} \
         WHERE rowid IN (SELECT value FROM json_each(?))"
    )))
    .bind(rowids)
    .fetch_all(pool)
    .await?;
    let docs = rows
        .iter()
        .map(|row| {
            let rowid: i64 = row.try_get("rowid")?;
            row_document(row, rowid.to_string())
        })
        .collect::<Result<Vec<_>, sqlx::Error>>()?;
    // Ask with the canonical form ("7" for "007") so matches line up.
    let wanted: Vec<String> = ids
        .iter()
        .map(|id| {
            id.parse::<i64>()
                .map_or_else(|_| id.clone(), |n| n.to_string())
        })
        .collect();
    Ok(in_id_order(&wanted, docs))
}

/// Deletes rows of `table` and their vectors in `index_table`, in one transaction.
pub(crate) async fn delete_rows(
    pool: &Pool<Sqlite>,
    table: &str,
    index_table: &str,
    ids: &[String],
) -> Result<(), Box<dyn Error>> {
    let Some(rowids) = rowid_param(ids) else {
        return Ok(());
    };
    let mut tx = pool.begin().await?;
    for t in [index_table, table] {
        sqlx::query(AssertSqlSafe(format!(
            "DELETE FROM {t} WHERE rowid IN (SELECT value FROM json_each(?))"
        )))
        .bind(&rowids)
        .execute(&mut *tx)
        .await?;
    }
    tx.commit().await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use sqlx::sqlite::SqlitePoolOptions;

    use super::*;

    /// The documents table as both stores create it, plus a plain table standing in
    /// for the vector index (same rowids), so no extension is needed.
    async fn pool() -> Pool<Sqlite> {
        let pool = SqlitePoolOptions::new()
            .max_connections(1)
            .connect("sqlite::memory:")
            .await
            .unwrap();
        for sql in [
            "CREATE TABLE docs (rowid INTEGER PRIMARY KEY AUTOINCREMENT, text TEXT, \
             metadata BLOB, text_embedding BLOB)",
            "CREATE TABLE idx_docs (rowid INTEGER PRIMARY KEY, text_embedding BLOB)",
            "CREATE TRIGGER t AFTER INSERT ON docs BEGIN \
             INSERT INTO idx_docs(rowid, text_embedding) VALUES (new.rowid, new.text_embedding); END",
        ] {
            sqlx::query(sql).execute(&pool).await.unwrap();
        }
        for (text, metadata) in [
            ("alpha", Some(json!({"k": 1}))),
            ("beta", None),
            ("gamma", Some(json!({"k": 3}))),
        ] {
            sqlx::query("INSERT INTO docs (text, metadata, text_embedding) VALUES (?, ?, '[]')")
                .bind(text)
                .bind(metadata)
                .execute(&pool)
                .await
                .unwrap();
        }
        pool
    }

    async fn count(pool: &Pool<Sqlite>, table: &str) -> i64 {
        sqlx::query_scalar(AssertSqlSafe(format!("SELECT count(*) FROM {table}")))
            .fetch_one(pool)
            .await
            .unwrap()
    }

    fn ids(ids: &[&str]) -> Vec<String> {
        ids.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn rowid_param_keeps_integers_only() {
        assert_eq!(
            rowid_param(&ids(&["3", "x", "007", "-1"])).as_deref(),
            Some("[3,7,-1]")
        );
        assert_eq!(rowid_param(&ids(&["x", ""])), None);
        assert_eq!(rowid_param(&[]), None);
    }

    #[tokio::test]
    async fn rows_by_id_in_request_order() {
        let pool = pool().await;
        let docs = rows_by_id(&pool, "docs", &ids(&["3", "9", "x", "01", "2"]))
            .await
            .unwrap();
        let got: Vec<_> = docs
            .iter()
            .map(|d| (d.id.as_deref().unwrap(), d.page_content.as_str()))
            .collect();
        assert_eq!(got, [("3", "gamma"), ("1", "alpha"), ("2", "beta")]);
        assert_eq!(docs[0].metadata["k"], json!(3));
        assert!(docs[2].metadata.is_empty(), "NULL metadata");
        assert!(docs.iter().all(|d| d.score == 0.0));
        assert!(rows_by_id(&pool, "docs", &ids(&["x"]))
            .await
            .unwrap()
            .is_empty());
    }

    #[tokio::test]
    async fn delete_rows_removes_documents_and_vectors() {
        let pool = pool().await;
        delete_rows(&pool, "docs", "idx_docs", &ids(&["1", "3", "42", "x"]))
            .await
            .unwrap();
        assert_eq!(count(&pool, "docs").await, 1);
        assert_eq!(count(&pool, "idx_docs").await, 1);
        let left = rows_by_id(&pool, "docs", &ids(&["1", "2", "3"]))
            .await
            .unwrap();
        assert_eq!(left.len(), 1);
        assert_eq!(left[0].page_content, "beta");
        // Nothing usable: no statement runs (a missing table would fail otherwise).
        delete_rows(&pool, "nope", "nope", &ids(&["x"]))
            .await
            .unwrap();
    }

    #[tokio::test]
    async fn delete_rows_is_atomic() {
        let pool = pool().await;
        // The index table delete succeeds, the documents one fails: nothing is kept.
        let err = delete_rows(&pool, "missing_table", "idx_docs", &ids(&["1"])).await;
        assert!(err.is_err());
        assert_eq!(count(&pool, "idx_docs").await, 3);
    }

    #[test]
    fn id_lookups_reject_scopes() {
        assert!(reject_id_options(&VecStoreOptions::default()).is_ok());
        assert!(reject_id_options(&VecStoreOptions::default().with_name_space("t")).is_err());
        assert!(
            reject_id_options(&VecStoreOptions::default().with_filters(json!({"k": 1}))).is_err()
        );
    }
}
