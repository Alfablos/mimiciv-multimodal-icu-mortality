from mmim.generator.utils import sha256str

from pydantic import BaseModel, Field, computed_field, ConfigDict

_general_model_config = ConfigDict(strict=True, frozen=True)

type SchemaSpecV1 = dict[str, dict[str, str]]


class GeneratorCodeSpec(BaseModel):
    model_config = _general_model_config
    git_sha: str
    git_ref: str


class QueriesSpecV1(BaseModel):
    model_config = _general_model_config
    images_query: str
    cohort_query: str
    features_query: str

    @computed_field
    @property
    def images_query_sha256(self):
        return sha256str(self.images_query)

    @computed_field
    @property
    def cohort_query_sha256(self):
        return sha256str(self.cohort_query)

    @computed_field
    @property
    def features_query_sha256(self):
        return sha256str(self.features_query)


class SplitSummaryV1(BaseModel):
    model_config = _general_model_config

    total: int
    positives: int
    negatives: int
    prevalence: float


class LeakageCheckV1(BaseModel):
    model_config = _general_model_config

    train_val_are_disjoint: bool
    train_test_are_disjoint: bool
    val_test_are_disjoint: bool

    def is_passed(self):
        return all(
            [
                self.train_val_are_disjoint,
                self.train_test_are_disjoint,
                self.val_test_are_disjoint,
            ]
        )


class SplitsSpecV1(BaseModel):
    model_config = _general_model_config

    strategy: str
    random_seed: int
    train: SplitSummaryV1
    validation: SplitSummaryV1
    test: SplitSummaryV1
    leakage_checks: LeakageCheckV1


class TabularDataSpec(BaseModel):
    model_config = _general_model_config
    pass


class ImageDataSpec(BaseModel):
    model_config = _general_model_config
    pass


class DataSpecV1(BaseModel):
    model_config = _general_model_config
    pass


class DatasetDefaultsV1(BaseModel):
    model_config = _general_model_config
    loss_pos_weight: float


class ManifestV1(BaseModel):
    model_config = _general_model_config

    manifest_version: str = "v1"
    dataset_version: str
    schema_version: str
    prediction_time: str
    lookback_window_hours: int
    sources: list[str] = Field(default_factory=list, gt=1)
    generator_code: GeneratorCodeSpec
    queries: QueriesSpecV1
    splits: SplitsSpecV1
    data_prefix: str
    data: DataSpecV1
    defaults: DatasetDefaultsV1
