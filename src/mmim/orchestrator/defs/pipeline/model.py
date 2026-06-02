from pydantic import BaseModel
from typing import Literal

from mmim.generator.manifest import ManifestV1


class DatasetManifestOutput(BaseModel):
    source: Literal["generated", "existing"]
    manifest: ManifestV1
    manifest_uri: str
    output_dir: str | None = None
    lakefs_ref: str | None = None
