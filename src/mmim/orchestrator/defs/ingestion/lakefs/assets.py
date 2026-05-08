import json
from typing import Any
import dagster as dg
import pandas as pd
from io import StringIO

from orchestrator.defs.ingestion.lakefs.resources import LakeFSResource


class LakeFSRepoRef(dg.Config):
    repo: str
    ref: str


@dg.asset
def get_dataset_files(
    context: dg.AssetExecutionContext, lakefs: LakeFSResource, config: LakeFSRepoRef
):
    files = lakefs.get_text_files_from_ref(
        repo=config.repo,
        ref=config.ref,
        requested_files=[
            "multimodal-icu-mortality-24h/v001/ds_train.csv",
            "multimodal-icu-mortality-24h/v001/ds_val.csv",
            "multimodal-icu-mortality-24h/v001/stats.json",
            "multimodal-icu-mortality-24h/v001/schema.json",
            "multimodal-icu-mortality-24h/v001/manifest.json",
        ],
    )
    result = {}
    for fname, content in files.items():
        if fname.endswith(".csv"):
            value = pd.read_csv(StringIO(content))
            if fname.endswith("ds_train.csv"):
                result["ds_train"] = value
                continue
            elif fname.endswith("ds_val.csv"):
                result["ds_val"] = value
                continue
            else:
                raise NameError(f"No such file should exist: {fname}")
        elif fname.endswith(".json"):
            value = json.loads(content)
            if fname.endswith("manifest.json"):
                result["manifest"] = value
                continue
            if fname.endswith("schema.json"):
                result["schema"] = value
                continue
            if fname.endswith("stats.json"):
                result["stats"] = value
                continue
        else:
            raise NameError(f"No such file should exist: {fname}")

    schema: dict[str, Any] = result["schema"]
    context.log.info(schema)

    return result


@dg.asset
def test_lakefs_connection(
    context: dg.AssetExecutionContext,
    lakefs: LakeFSResource,
) -> str:
    client = lakefs.get_client()

    context.log.info(f"LakeFS endpoint: {client.config.host}")
    context.add_output_metadata({"host": client.config.host})

    return "ok"
