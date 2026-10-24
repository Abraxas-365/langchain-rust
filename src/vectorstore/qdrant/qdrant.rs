use async_trait::async_trait;
use qdrant_client::qdrant::point_id::PointIdOptions;
use qdrant_client::qdrant::{
    DeletePointsBuilder, Filter, GetPointsBuilder, PointId, PointStruct, SearchPointsBuilder,
    UpsertPointsBuilder,
};
use qdrant_client::Payload;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::error::Error;
use std::num::IntErrorKind;
use std::sync::Arc;

pub use qdrant_client::Qdrant;

// Re-export now deprecated client with old name to prevent breakage for users
#[deprecated(note = "use `Qdrant` instead")]
pub use qdrant_client::Qdrant as QdrantClient;

use crate::{
    embedding::embedder_trait::Embedder,
    schemas::Document,
    vectorstore::{vectorstore::in_id_order, VecStoreOptions, VectorStore},
};
use uuid::Uuid;

pub struct Store {
    pub client: Qdrant,
    pub embedder: Arc<dyn Embedder>,
    pub collection_name: String,
    pub content_field: String,
    pub metadata_field: String,
    pub search_filter: Option<Filter>,
}

type QdrantOptions = VecStoreOptions<Value>;

struct ResolvedPointId {
    point_id: PointId,
    /// Canonical form actually stored on the point (lowercase UUID or decimal `u64`).
    canonical: String,
}

/// Parse a caller-supplied Qdrant id. `None` is handled by [`resolve_document_id`].
fn parse_supplied_id(raw: &str) -> Result<ResolvedPointId, Box<dyn Error>> {
    if raw.is_empty() {
        return Err("document id must not be empty".into());
    }
    if let Ok(uuid) = Uuid::parse_str(raw) {
        let canonical = uuid.to_string();
        return Ok(ResolvedPointId {
            point_id: PointId {
                point_id_options: Some(PointIdOptions::Uuid(canonical.clone())),
            },
            canonical,
        });
    }
    if raw.starts_with('-') {
        return Err(format!(
            "document id `{raw}` is negative; Qdrant ids must be a UUID or an unsigned integer"
        )
        .into());
    }
    match raw.parse::<u64>() {
        Ok(n) => Ok(ResolvedPointId {
            point_id: PointId {
                point_id_options: Some(PointIdOptions::Num(n)),
            },
            canonical: n.to_string(),
        }),
        Err(err) if err.kind() == &IntErrorKind::PosOverflow => {
            Err(format!("document id `{raw}` overflows u64").into())
        }
        Err(_) => Err(format!("document id `{raw}` is not a UUID or an unsigned integer").into()),
    }
}

fn resolve_document_id(id: Option<&str>) -> Result<ResolvedPointId, Box<dyn Error>> {
    match id {
        Some(id) => parse_supplied_id(id),
        None => {
            let canonical = Uuid::new_v4().to_string();
            Ok(ResolvedPointId {
                point_id: PointId {
                    point_id_options: Some(PointIdOptions::Uuid(canonical.clone())),
                },
                canonical,
            })
        }
    }
}

fn validate_supplied_ids(docs: &[Document]) -> Result<(), Box<dyn Error>> {
    for doc in docs {
        if let Some(id) = &doc.id {
            parse_supplied_id(id)?;
        }
    }
    Ok(())
}

/// Build the points that `add_documents` upserts and the canonical ids it returns.
///
/// Generated ids are created once here so the point and the returned id match.
/// Vector and document counts are checked before any point is built.
fn points_for_upsert(
    docs: &[Document],
    vectors: &[Vec<f64>],
    content_field: &str,
    metadata_field: &str,
    extra_payload: Option<Value>,
) -> Result<(Vec<PointStruct>, Vec<String>), Box<dyn Error>> {
    if vectors.len() != docs.len() {
        return Err(Box::new(std::io::Error::other(
            "Number of vectors and documents do not match",
        )));
    }

    let mut points = Vec::with_capacity(docs.len());
    let mut ids = Vec::with_capacity(docs.len());
    for (doc, vector) in docs.iter().zip(vectors.iter()) {
        let resolved = resolve_document_id(doc.id.as_deref())?;
        let mut payload = json!({
            content_field: doc.page_content,
            metadata_field: doc.metadata,
        });
        if let Some(extra) = extra_payload.clone() {
            if let (Value::Object(ref mut base_map), Value::Object(extra_map)) =
                (&mut payload, extra)
            {
                base_map.extend(extra_map);
            }
        }
        let vector: Vec<f32> = vector.iter().copied().map(|f| f as f32).collect();
        let point = PointStruct::new(resolved.point_id, vector, Payload::try_from(payload)?);
        points.push(point);
        ids.push(resolved.canonical);
    }
    Ok((points, ids))
}

/// Qdrant rejects a batch that repeats a point id. Keep the last document for each id.
fn last_point_wins(points: Vec<PointStruct>) -> Vec<PointStruct> {
    let mut positions: HashMap<PointId, usize> = HashMap::new();
    let mut kept = Vec::with_capacity(points.len());
    for point in points {
        let Some(id) = point.id.clone() else {
            kept.push(point);
            continue;
        };
        if let Some(&index) = positions.get(&id) {
            kept[index] = point;
        } else {
            positions.insert(id, kept.len());
            kept.push(point);
        }
    }
    kept
}

fn point_id_to_string(id: &PointId) -> Option<String> {
    match &id.point_id_options {
        Some(PointIdOptions::Num(n)) => Some(n.to_string()),
        Some(PointIdOptions::Uuid(s)) => Some(s.clone()),
        None => None,
    }
}

#[async_trait]
impl VectorStore for Store {
    type Options = QdrantOptions;

    /// Add documents to the store.
    /// Returns a list of document IDs added to the Qdrant collection.
    async fn add_documents(
        &self,
        docs: &[Document],
        opt: &QdrantOptions,
    ) -> Result<Vec<String>, Box<dyn Error>> {
        // Invalid ids fail before embedding or upsert.
        validate_supplied_ids(docs)?;

        let embedder = opt.embedder.as_ref().unwrap_or(&self.embedder);
        let texts: Vec<String> = docs.iter().map(|d| d.page_content.clone()).collect();
        let vectors = embedder.embed_documents(&texts).await?;
        // One conversion feeds both the points and the ids returned to the caller.
        let (points, ids) = points_for_upsert(
            docs,
            &vectors,
            &self.content_field,
            &self.metadata_field,
            opt.filters.clone(),
        )?;
        let points = last_point_wins(points);

        self.client
            .upsert_points(UpsertPointsBuilder::new(&self.collection_name, points).wait(true))
            .await?;

        Ok(ids)
    }

    /// Perform a similarity search on the store.
    /// Returns a list of documents similar to the query.
    async fn similarity_search(
        &self,
        query: &str,
        limit: usize,
        opt: &QdrantOptions,
    ) -> Result<Vec<Document>, Box<dyn Error>> {
        if opt.name_space.is_some() {
            return Err("Qdrant doesn't support namespaces".into());
        }

        if opt.filters.is_some() {
            return Err(
                "'qdrant_client' doesn't support 'serde_json::Value' filters. 
            Use `search_filter` when constructing VectorStore instead"
                    .into(),
            );
        }

        let embedder = opt.embedder.as_ref().unwrap_or(&self.embedder);
        let query_vector: Vec<f32> = embedder
            .embed_query(query)
            .await?
            .into_iter()
            .map(|f| f as f32)
            .collect();

        let mut operation =
            SearchPointsBuilder::new(&self.collection_name, query_vector, limit as u64)
                .with_payload(true);
        if let Some(score_threshold) = opt.score_threshold {
            operation = operation.score_threshold(score_threshold);
        }
        if let Some(filter) = &self.search_filter {
            operation = operation.filter(filter.clone());
        }
        let results = self.client.search_points(operation).await?;

        let documents = results
            .result
            .into_iter()
            .map(|scored_point| {
                let mut doc = self.payload_to_document(scored_point.payload);
                doc.score = scored_point.score as f64;
                doc.id = scored_point.id.as_ref().and_then(point_id_to_string);
                doc
            })
            .collect();

        Ok(documents)
    }

    async fn delete(&self, ids: &[String], opt: &QdrantOptions) -> Result<(), Box<dyn Error>> {
        let (point_ids, _) = point_ids_for_lookup(ids, opt)?;
        if point_ids.is_empty() {
            return Ok(());
        }
        self.client
            .delete_points(
                DeletePointsBuilder::new(&self.collection_name)
                    .points(point_ids)
                    .wait(true),
            )
            .await?;
        Ok(())
    }

    async fn get_by_ids(
        &self,
        ids: &[String],
        opt: &QdrantOptions,
    ) -> Result<Vec<Document>, Box<dyn Error>> {
        let (point_ids, canonical) = point_ids_for_lookup(ids, opt)?;
        if point_ids.is_empty() {
            return Ok(vec![]);
        }
        let found = self
            .client
            .get_points(GetPointsBuilder::new(&self.collection_name, point_ids).with_payload(true))
            .await?;
        let docs = found
            .result
            .into_iter()
            .map(|point| {
                let mut doc = self.payload_to_document(point.payload);
                doc.id = point.id.as_ref().and_then(point_id_to_string);
                doc
            })
            .collect();
        Ok(in_id_order(&canonical, docs))
    }
}

impl Store {
    fn payload_to_document(
        &self,
        payload: HashMap<String, qdrant_client::qdrant::Value>,
    ) -> Document {
        let page_content = payload
            .get(&self.content_field)
            .and_then(|v| v.as_str())
            .cloned()
            .unwrap_or_default();
        let metadata = payload
            .get(&self.metadata_field)
            .and_then(|v| serde_json::from_value(v.clone().into_json()).ok())
            .unwrap_or_default();
        Document::new(page_content).with_metadata(metadata)
    }
}

/// Point ids to look up or delete, plus the canonical id strings they are stored under
/// (an upper-case UUID matches its lower-case point). Namespaces don't exist in Qdrant.
fn point_ids_for_lookup(
    ids: &[String],
    opt: &QdrantOptions,
) -> Result<(Vec<PointId>, Vec<String>), Box<dyn Error>> {
    if opt.name_space.is_some() {
        return Err("Qdrant doesn't support namespaces".into());
    }
    let resolved = ids
        .iter()
        .map(|id| parse_supplied_id(id))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(resolved
        .into_iter()
        .map(|r| (r.point_id, r.canonical))
        .unzip())
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;

    use async_trait::async_trait;
    use qdrant_client::qdrant::point_id::PointIdOptions;
    use qdrant_client::Qdrant;

    use super::{last_point_wins, point_id_to_string, points_for_upsert, Store};
    use crate::embedding::{Embedder, EmbedderError};
    use crate::schemas::Document;
    use crate::test_utils::FakeEmbedder;
    use crate::vectorstore::qdrant::StoreBuilder;
    use crate::vectorstore::{VecStoreOptions, VectorStore};

    fn point_kind(point: &qdrant_client::qdrant::PointStruct) -> PointIdOptions {
        point
            .id
            .as_ref()
            .and_then(|id| id.point_id_options.clone())
            .unwrap()
    }

    #[test]
    fn converts_document_ids_into_the_points_that_are_upserted() {
        let uuid = "550e8400-e29b-41d4-a716-446655440000";
        let docs = vec![
            Document::new("uuid-doc").with_id("550E8400-E29B-41D4-A716-446655440000"),
            Document::new("zero").with_id("0"),
            Document::new("max").with_id(u64::MAX.to_string()),
            Document::new("generated-a"),
            Document::new("generated-b"),
        ];
        let vectors = vec![vec![0.1], vec![0.2], vec![0.3], vec![0.4], vec![0.5]];
        let (points, ids) =
            points_for_upsert(&docs, &vectors, "page_content", "metadata", None).unwrap();

        assert_eq!(points.len(), ids.len());
        assert_eq!(ids[0], uuid);
        assert_eq!(
            point_kind(&points[0]),
            PointIdOptions::Uuid(uuid.to_string())
        );
        assert_eq!(
            point_id_to_string(points[0].id.as_ref().unwrap()).as_deref(),
            Some(uuid)
        );

        assert_eq!(ids[1], "0");
        assert_eq!(point_kind(&points[1]), PointIdOptions::Num(0));
        assert_eq!(ids[2], u64::MAX.to_string());
        assert_eq!(point_kind(&points[2]), PointIdOptions::Num(u64::MAX));

        assert_ne!(ids[3], ids[4]);
        uuid::Uuid::parse_str(&ids[3]).unwrap();
        uuid::Uuid::parse_str(&ids[4]).unwrap();
        assert_eq!(
            point_id_to_string(points[3].id.as_ref().unwrap()).as_deref(),
            Some(ids[3].as_str())
        );
        assert_eq!(
            point_id_to_string(points[4].id.as_ref().unwrap()).as_deref(),
            Some(ids[4].as_str())
        );
        assert!(matches!(point_kind(&points[3]), PointIdOptions::Uuid(_)));
    }

    #[test]
    fn rejects_malformed_ids_and_vector_count_mismatch() {
        let doc = |id: &str| Document::new("a").with_id(id);
        let overflow = (u64::MAX as u128 + 1).to_string();
        for bad in ["", "-1", overflow.as_str(), "not-an-id", "1.5"] {
            let err =
                points_for_upsert(&[doc(bad)], &[vec![1.0]], "page_content", "metadata", None)
                    .unwrap_err()
                    .to_string();
            assert!(!err.is_empty(), "{bad}");
        }
        let err = points_for_upsert(
            &[Document::new("a"), Document::new("b")],
            &[vec![1.0]],
            "page_content",
            "metadata",
            None,
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("do not match"), "{err}");
    }

    #[test]
    fn duplicate_ids_keep_the_last_point() {
        let docs = vec![
            Document::new("first").with_id("7"),
            Document::new("second").with_id("7"),
        ];
        let (points, ids) = points_for_upsert(
            &docs,
            &[vec![1.0], vec![2.0]],
            "page_content",
            "metadata",
            None,
        )
        .unwrap();
        assert_eq!(ids, vec!["7".to_string(), "7".to_string()]);
        let kept = last_point_wins(points);
        assert_eq!(kept.len(), 1);
        assert_eq!(kept[0].payload["page_content"].to_string(), "\"second\"");
        assert_eq!(point_kind(&kept[0]), PointIdOptions::Num(7));
    }

    struct GuardEmbedder {
        called: Arc<AtomicBool>,
        vectors: Vec<Vec<f64>>,
    }

    #[async_trait]
    impl Embedder for GuardEmbedder {
        async fn embed_documents(&self, _: &[String]) -> Result<Vec<Vec<f64>>, EmbedderError> {
            self.called.store(true, Ordering::SeqCst);
            if self.vectors.is_empty() {
                Err(EmbedderError::FastEmbedError("embed failed".into()))
            } else {
                Ok(self.vectors.clone())
            }
        }

        async fn embed_query(&self, _: &str) -> Result<Vec<f64>, EmbedderError> {
            Err(EmbedderError::FastEmbedError("embed failed".into()))
        }
    }

    fn offline_store(embedder: impl Embedder + 'static) -> Store {
        let client = Qdrant::from_url("http://127.0.0.1:1")
            .skip_compatibility_check()
            .build()
            .unwrap();
        Store {
            client,
            embedder: Arc::new(embedder),
            collection_name: "unused".to_string(),
            content_field: "page_content".to_string(),
            metadata_field: "metadata".to_string(),
            search_filter: None,
        }
    }

    #[tokio::test]
    async fn malformed_ids_fail_before_embedding_and_count_mismatch_before_upsert() {
        let called = Arc::new(AtomicBool::new(false));
        let store = offline_store(GuardEmbedder {
            called: called.clone(),
            vectors: Vec::new(),
        });
        let err = store
            .add_documents(
                &[Document::new("a").with_id("")],
                &VecStoreOptions::default(),
            )
            .await
            .unwrap_err()
            .to_string();
        assert!(err.contains("empty"), "{err}");
        assert!(!called.load(Ordering::SeqCst));

        let called = Arc::new(AtomicBool::new(false));
        let store = offline_store(GuardEmbedder {
            called: called.clone(),
            vectors: vec![vec![1.0]],
        });
        let err = store
            .add_documents(
                &[Document::new("a"), Document::new("b")],
                &VecStoreOptions::default(),
            )
            .await
            .unwrap_err()
            .to_string();
        assert!(err.contains("do not match"), "{err}");
        assert!(called.load(Ordering::SeqCst));

        let store = offline_store(GuardEmbedder {
            called: Arc::new(AtomicBool::new(false)),
            vectors: Vec::new(),
        });
        let err = store
            .add_documents(&[Document::new("a")], &VecStoreOptions::default())
            .await
            .unwrap_err()
            .to_string();
        assert!(err.contains("embed failed"), "{err}");
    }

    /// Upsert the same numeric and UUID ids twice and read back one updated point each.
    ///
    /// ```text
    /// QDRANT_TEST_URL=http://localhost:6334 \
    /// cargo test --features qdrant --lib -- --ignored qdrant_custom_id_replaces_point
    /// ```
    #[tokio::test]
    #[ignore]
    async fn qdrant_custom_id_replaces_point() {
        let url = std::env::var("QDRANT_TEST_URL")
            .unwrap_or_else(|_| "http://localhost:6334".to_string());
        let client = Qdrant::from_url(&url)
            .skip_compatibility_check()
            .build()
            .unwrap();
        let embedder = FakeEmbedder::new(
            3,
            &[
                ("first", vec![1.0, 0.0, 0.0]),
                ("second", vec![0.0, 1.0, 0.0]),
                ("uuid-body", vec![0.0, 0.0, 1.0]),
                ("query-num", vec![0.0, 1.0, 0.0]),
                ("query-uuid", vec![0.0, 0.0, 1.0]),
            ],
        );
        let collection = format!("id-test-{}", uuid::Uuid::new_v4());
        let store = StoreBuilder::new()
            .client(client)
            .embedder(embedder)
            .collection_name(&collection)
            .recreate_collection(true)
            .build()
            .await
            .unwrap();

        let outcome = qdrant_round_trip(&store).await;
        let _ = store.client.delete_collection(collection).await;
        outcome.unwrap();
    }

    async fn qdrant_round_trip(store: &Store) -> Result<(), Box<dyn std::error::Error>> {
        let opts = VecStoreOptions::default();
        let uuid = "550e8400-e29b-41d4-a716-446655440000";
        let first = store
            .add_documents(
                &[
                    Document::new("first").with_id("42"),
                    Document::new("uuid-body").with_id(uuid),
                    Document::new("generated"),
                ],
                &opts,
            )
            .await?;
        assert_eq!(first[0], "42");
        assert_eq!(first[1], uuid);
        uuid::Uuid::parse_str(&first[2])?;
        assert_ne!(first[2], uuid);

        let second = store
            .add_documents(&[Document::new("second").with_id("42")], &opts)
            .await?;
        assert_eq!(second, vec!["42".to_string()]);

        let by_num = store.similarity_search("query-num", 10, &opts).await?;
        let updated: Vec<_> = by_num
            .iter()
            .filter(|doc| doc.id.as_deref() == Some("42"))
            .collect();
        if updated.len() != 1 {
            return Err(format!("expected one point with id 42, got {updated:?}").into());
        }
        if updated[0].page_content != "second" {
            return Err(format!("numeric point content was {}", updated[0].page_content).into());
        }

        let by_uuid = store.similarity_search("query-uuid", 10, &opts).await?;
        let uuid_hits: Vec<_> = by_uuid
            .iter()
            .filter(|doc| doc.id.as_deref() == Some(uuid))
            .collect();
        if uuid_hits.len() != 1 {
            return Err(format!("expected one uuid point, got {by_uuid:?}").into());
        }
        if uuid_hits[0].page_content != "uuid-body" {
            return Err(format!("uuid point content was {}", uuid_hits[0].page_content).into());
        }

        let generated = by_num
            .iter()
            .chain(by_uuid.iter())
            .find(|doc| doc.id.as_deref() == Some(first[2].as_str()));
        if generated.is_none() {
            return Err(format!("generated id {} was not stored", first[2]).into());
        }

        // get_by_ids: request order, upper-case UUID, missing id skipped.
        let ask = [uuid.to_uppercase(), "7".to_string(), "42".to_string()];
        let got = store.get_by_ids(&ask, &opts).await?;
        let got_ids: Vec<_> = got.iter().map(|d| d.id.clone().unwrap()).collect();
        assert_eq!(got_ids, [uuid.to_string(), "42".to_string()]);
        assert_eq!(got[1].page_content, "second");
        assert!(store.get_by_ids(&[], &opts).await?.is_empty());
        assert!(store
            .get_by_ids(&["not-an-id".to_string()], &opts)
            .await
            .is_err());

        store
            .delete(&["42".to_string(), "7".to_string()], &opts)
            .await?;
        assert!(store
            .get_by_ids(&["42".to_string()], &opts)
            .await?
            .is_empty());
        assert_eq!(
            store.get_by_ids(&[uuid.to_string()], &opts).await?.len(),
            1,
            "other points kept"
        );
        Ok(())
    }
}
