"""Feast feature definitions for driver stats example."""

from datetime import timedelta

from feast import Entity, FeatureView, Field, FileSource
from feast.types import Float32, Int32

driver = Entity(name="driver", join_keys=["driver_id"])

driver_stats_source = FileSource(
    name="driver_hourly_stats_source",
    path="data/driver_stats.parquet",
    timestamp_field="event_timestamp",
    created_timestamp_column="created",
)

driver_hourly_stats = FeatureView(
    name="driver_hourly_stats",
    entities=[driver],
    ttl=timedelta(days=1),
    schema=[
        Field(name="conv_rate", dtype=Float32),
        Field(name="acc_rate", dtype=Float32),
        Field(name="avg_daily_trips", dtype=Int32),
    ],
    online=True,
    source=driver_stats_source,
)
