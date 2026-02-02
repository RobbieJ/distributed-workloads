"""Feast feature definitions for RAG wiki passages with Milvus vector store."""

from datetime import timedelta

from feast import Entity, FeatureView, Field, FileSource, ValueType
from feast.data_format import ParquetFormat
from feast.types import Array, Float32, String

# Entity: unique passage identifier
wiki_passage = Entity(
    name="passage_id",
    join_keys=["passage_id"],
    value_type=ValueType.STRING,
    description="Unique ID of a Wikipedia passage",
)

parquet_file_path = "data/wiki_dpr.parquet"

# Offline source: Parquet file with passage data
wiki_dpr_source = FileSource(
    name="wiki_dpr_source",
    file_format=ParquetFormat(),
    path=parquet_file_path,
    timestamp_field="event_timestamp",
)

# Feature view: passage text + embedding vector
wiki_passage_feature_view = FeatureView(
    name="wiki_passages",
    entities=[wiki_passage],
    ttl=timedelta(days=1),
    schema=[
        Field(
            name="passage_text",
            dtype=String,
            description="Content of the Wikipedia passage",
        ),
        Field(
            name="embedding",
            dtype=Array(Float32),
            description="DPR embedding vector",
            vector_index=True,
            vector_length=768,
            vector_search_metric="COSINE",
        ),
    ],
    online=True,
    source=wiki_dpr_source,
    description="Wikipedia passage content and embeddings for RAG",
)
