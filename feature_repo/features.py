from datetime import timedelta
from feast import Entity, FeatureView, Field, FileSource, ValueType
from feast.types import Float32, String

# 1️⃣ Define FileSource for offline store
iris_source = FileSource(
    path="../data/processed/iris_features.parquet",
    timestamp_field="event_timestamp"
)

# 2️⃣ Define Entity (primary key)
iris_entity = Entity(
    name="iris_id",
    join_keys=["iris_id"],
    value_type=ValueType.INT64,
    description="Unique identifier for each Iris sample",
)

# 3️⃣ Define FeatureView
iris_feature_view = FeatureView(
    name="iris_features",
    entities=[iris_entity],
    ttl=timedelta(days=1),
    schema=[
        Field(name="sepal_length", dtype=Float32),
        Field(name="sepal_width", dtype=Float32),
        Field(name="petal_length", dtype=Float32),
        Field(name="petal_width", dtype=Float32),
        Field(name="petal_area", dtype=Float32),
        Field(name="sepal_area", dtype=Float32),
        Field(name="petal_sepal_ratio", dtype=Float32),
        Field(name="species", dtype=String),
        Field(name="species_encoded", dtype=Float32),
    ],
    source=iris_source,
    online=True,  # enable for online materialization
)
