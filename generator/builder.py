import json
import os
from sklearn.model_selection import train_test_split
import pandas as pd
import lakefs
from lakefs.client import Client
import duckdb
from duckdb import DuckDBPyConnection

from .utils import (
    get_local_repo,
    find_paths,
    dataset_summary,
    sha256str,
    df_schema,
    leakage_check,
)

default_datasets_dir = "./"

core_features_allowed_missing = [
    "glucose",
    "lactate",
    "creatinine",
    "heart_rate",
    "blood_pressure",
    "resp_rate",
    "temp_f",
    "spO2",
]
allowed_missing = [
    a
    for f in core_features_allowed_missing
    for a in [f"{f}_min", f"{f}_max", f"{f}_mean"]
]
mandatory = ["gender", "age", "hospital_expire_flag"]
to_onehot = ["gender"]
continuous_variables = [
    "age",
    "glucose_min",
    "glucose_max",
    "glucose_mean",
    "lactate_min",
    "lactate_max",
    "lactate_mean",
    "creatinine_min",
    "creatinine_max",
    "creatinine_mean",
    "heart_rate_mean",
    "heart_rate_min",
    "heart_rate_max",
    "blood_pressure_mean",
    "blood_pressure_min",
    "blood_pressure_max",
    "resp_rate_mean",
    "resp_rate_min",
    "resp_rate_max",
    "temp_f_mean",
    "temp_f_min",
    "temp_f_max",
    "spO2_mean",
    "spO2_min",
    "spO2_max",
]

label = "hospital_expire_flag"
default_dataset_version = "v001"


def build(args):
    duckdb_db = args.database_path
    metadata_file = args.metadata_file

    git_sha = os.getenv("GIT_SHA")
    git_ref = os.getenv("GIT_REF")

    if (not git_ref or git_ref == "") or (not git_sha or git_sha == ""):
        repo = get_local_repo()

    if git_ref is None or git_ref == "":
        git_ref = repo.head.ref.name
    if git_sha is None or git_sha == "":
        git_sha = repo.head.commit.hexsha

    lakefs_host = os.getenv("LAKEFS_HOST")

    lakefs_username = os.getenv("LAKEFS_USERNAME")
    if (lakefs_username is None or lakefs_username == "") and lakefs_host is not None:
        raise ValueError("The variable LAKEFS_USERNAME MUST be set.")

    lakefs_password = os.getenv("LAKEFS_PASSWORD")
    if (lakefs_password is None or lakefs_password == "") and lakefs_host is not None:
        raise ValueError("The variable LAKEFS_PASSWORD MUST be set.")

    lakefs_repository = os.getenv("LAKEFS_REPOSITORY")
    if (
        lakefs_repository is None or lakefs_repository == ""
    ) and lakefs_host is not None:
        raise ValueError("The variable LAKEFS_REPOSITORY MUST be set.")

    output_dir = os.getenv("DATASET_OUTPUT_DIR", default_datasets_dir).rstrip("/") + "/"

    dataset_version = os.getenv("DATASET_VERSION", default_dataset_version)

    not_found = find_paths([duckdb_db, metadata_file])
    if len(not_found) != 0:
        raise FileNotFoundError(f"Unable to find files: {', '.join(not_found)}")

    lakefs_branch = (
        "build_" + git_ref.replace("/", "-").replace("_", "-") + "-" + git_sha[:9]
    )
    print(f"Will push to LakeFS branch: {lakefs_branch}")

    dconn = duckdb.connect(database=duckdb_db, read_only=True)
    images_manifest, images_query = build_images_manifest(
        connection=dconn, metadata_file=metadata_file
    )
    cohort, cohort_query = build_cohort(
        connection=dconn, images_manifest=images_manifest
    )

    features, labs_n_vitals_query = build_features(connection=dconn, cohort=cohort)
    initial_dataset = cohort.join(
        features.set_index("subject_id").drop(
            ["stay_id", "gender", "age", "hospital_expire_flag"], axis=1
        ),  # stay_id and others already exists in cohort
        on="subject_id",
        how="left",
    )
    print(
        "Done querying the DB, I'm now left with the following columns:",
        initial_dataset.columns.to_list(),
        "\n",
    )

    print("About to check for null values. ")
    for feat in core_features_allowed_missing:
        if not (
            initial_dataset[initial_dataset[f"{feat}_min"].isna()].shape[0]
            == initial_dataset[initial_dataset[f"{feat}_max"].isna()].shape[0]
            == initial_dataset[initial_dataset[f"{feat}_mean"].isna()].shape[0]
        ):
            raise ValueError(
                "The same number of null values must be shared by min_ max_ and mean_ columns, this is a bug!"
            )
    print("Check passed.\n")
    print("Recap:")
    for c in initial_dataset.columns:
        print(
            "  ",
            c,
            "=>",
            initial_dataset[c].dtype,
            "| ONEHOT" if c in to_onehot else "",
        )

    train_ds, other_ds = train_test_split(
        initial_dataset, test_size=0.2, shuffle=True, random_state=42
    )
    val_ds, test_ds = train_test_split(
        other_ds, test_size=0.5, shuffle=True, random_state=42
    )

    medians: dict[str, float] = dict()

    for col in train_ds.columns:
        if train_ds[col].isna().any() and pd.api.types.is_numeric_dtype(train_ds[col]):
            if col in allowed_missing:
                median = float(train_ds[col].median())
                medians[col] = median
            else:
                print(
                    f"Skipping computing median value for {col} as it's not allowed to be missing."
                )

    print("Preparing datasets...")
    train_ds = prepare_set(train_ds, medians=medians)
    val_ds = prepare_set(val_ds, medians=medians)
    test_ds = prepare_set(test_ds, medians=medians)

    print("Splitting features and labels...")
    X_train: pd.DataFrame = train_ds.drop(label, axis=1)
    # Y_train: pd.Series = train_ds['hospital_expire_flag']  # noqa: F841
    # X_val: pd.DataFrame = val_ds.drop('hospital_expire_flag', axis=1)  # noqa: F841
    # Y_val: pd.Series = val_ds['hospital_expire_flag']  # noqa: F841
    # X_test: pd.DataFrame = test_ds.drop('hospital_expire_flag', axis=1)  # noqa: F841
    # Y_test: pd.Series = test_ds['hospital_expire_flag']  # noqa: F841

    print("Computing training set statistics...")
    stats = {  # noqa: F841
        "mean": X_train[continuous_variables].mean().to_dict(),
        "std": X_train[continuous_variables].std().to_dict(),
    }

    train_ds_csv = train_ds.to_csv(index=False)
    val_ds_csv = val_ds.to_csv(index=False)
    test_ds_csv = test_ds.to_csv(index=False)
    json_stats = json.dumps(stats)

    queries = {
        "images_query_sha": sha256str(images_query),
        "images_query": images_query,
        "cohort_query_sha": sha256str(cohort_query),
        "cohort_query": cohort_query,
        "features_query_sha": sha256str(labs_n_vitals_query),
        "features_query": labs_n_vitals_query,
    }

    manifest = {
        "dataset": "multimodal-icu-mortality-24",
        "dataset_version": "v001",
        "prediction_time": "icu_intime",
        "lookback_window_hours": 24,
        "source": ["MIMIC-IV", "MIMIC-ED", "MIMIC-CXR", "MIMIC-CXR-JPG"],
        "code": {
            "git_sha": git_sha,
            "git_ref": git_ref,
        },
        "queries": queries,
        "splits": {
            "strategy": "first_stay_per_subject_firstcxr_random_split",
            "random_seed": 42,
            "train": dataset_summary(train_ds, label),
            "validation": dataset_summary(val_ds, label),
            "test": dataset_summary(test_ds, label),
            "leakage_checks": leakage_check(train_ds, val_ds, test_ds),
        },
        "files": {
            "ds_train.csv": {"sha256": sha256str(train_ds_csv)},
            "ds_val.csv": {"sha256": sha256str(val_ds_csv)},
            "ds_test.csv": {"sha256": sha256str(test_ds_csv)},
            "stats.json": {"sha256": sha256str(json_stats)},
        },
    }

    schema = df_schema(
        train_ds, label_column=label, id_columns=["subject_id", "study_id", "dicom_id"]
    )

    # Write files according to variables! $TRAINING_DATASET_FILE, $VALIDATION_DATASET_FILE, $DATASET_STATS_FILE and a ds_test.csv
    train_ds.to_csv(output_dir + "ds_train.csv", index=False, lineterminator="\n")
    val_ds.to_csv(output_dir + "ds_val.csv", index=False, lineterminator="\n")
    test_ds.to_csv(output_dir + "ds_test.csv", index=False, lineterminator="\n")

    with open(output_dir + "stats.json", "w") as f:
        json.dump(stats, f, indent=2, sort_keys=True)

    with open(output_dir + "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)

    with open(output_dir + "schema.json", "w") as f:
        json.dump(schema, f, indent=2, sort_keys=True)

    if lakefs_host:
        lakefs_prefix = (
            "multimodal-icu-mortality-24h/" + dataset_version.rstrip("/") + "/"
        )
        lclient = Client(
            host=lakefs_host, username=lakefs_username, password=lakefs_password
        )
        lrepo = lakefs.Repository(lakefs_repository, client=lclient)

        # create a build/${code-ref}-${code-sha}
        branch = lrepo.branch(lakefs_branch).create(
            source_reference="master", exist_ok=True
        )

        branch.object(lakefs_prefix + "ds_train.csv").upload(data=train_ds_csv)
        branch.object(lakefs_prefix + "ds_val.csv").upload(data=val_ds_csv)
        branch.object(lakefs_prefix + "ds_test.csv").upload(data=test_ds_csv)
        branch.object(lakefs_prefix + "stats.json").upload(data=json_stats)
        branch.object(lakefs_prefix + "manifest.json").upload(
            data=json.dumps(manifest, indent=2, sort_keys=True)
        )
        branch.object(lakefs_prefix + "schema.json").upload(
            data=json.dumps(schema, indent=True, sort_keys=True)
        )

        commit = branch.commit(
            message=f"Generated by {git_ref}/{git_sha}",
            metadata={"builder_ref": git_ref, "builder_sha": git_sha},
        )

        print(f"Successfully committed on {lakefs_branch} with commit id {commit.id}")


def prepare_set(df: pd.DataFrame, medians: dict[str, float]) -> pd.DataFrame:
    final_df = df.drop(["icu_intime", "stay_id", "hadm_id"], axis=1)
    # Encoding gender assuming M and F are the only values in the dataset (no 'f', 'm' or others)
    # Encoding M as 0, F as 1
    if not final_df["gender"].isin([0, 1]).any():
        assert final_df["gender"].isin(["M", "F"]).all(), (
            "M and F are not the only values for gender. Adjust your encoding algorithm accordingly."
        )
        final_df["gender"] = final_df["gender"].map({"M": 0, "F": 1})

    # Set missing values to the median of the column
    # after creating the column {feat}_missing
    for feat in core_features_allowed_missing:
        f_max = f"{feat}_max"
        # f_min, f_max_f_mean are GUARANTEED to miss at the same time by an assertion above
        if final_df[f_max].isna().any():
            final_df[f"{feat}_missing"] = final_df[f_max].isna().astype(int)

    for col in final_df.columns:
        if final_df[col].isna().any():
            if col not in allowed_missing:
                raise ValueError(
                    f"Values in {col} are not allowed to be missing because they can't be computed."
                )

            final_df[col] = final_df[col].fillna(medians[col])

    return final_df


def build_images_manifest(
    connection: DuckDBPyConnection, metadata_file: str
) -> tuple[pd.DataFrame, str]:
    images_q = f"""
    WITH cxr_meta_parsed AS (
	SELECT
		subject_id,
		dicom_id,
        study_id,
        -- Cannot cast StudyTime to string directly: even ignoring the decimal part 80556.875 would become "80556" while the %H format needs 08 for the hour, not 8 (080556)
		strptime (
			StudyDate || ' ' || lpad(string_split(StudyTime, '.')[1], 6, '0'),
			'%Y%m%d %H%M%S'
		) as cxr_time
        FROM read_csv_auto(
    		'{metadata_file}',
    		types={{'StudyDate': 'VARCHAR', 'StudyTime': 'VARCHAR'}}, -- StudyDate and StudyTime are parsed as DOUBLEs by DuckDB
    		quote='"'                                                 -- some columns have spaces
      	)
        WHERE
		ViewPosition IN ('AP', 'PA')
    ),
    stays_by_patient AS (
	SELECT
		subject_id,
		stay_id,
		intime,
		row_number() OVER(PARTITION BY subject_id ORDER BY intime ASC) as stay_number
        FROM mimiciv_icu.icustays
    ), -- => stay_id is unique, patient is not, but each stay now has a number (for each patient)
    valid_stays AS (
	SELECT
		s.subject_id,
            s.stay_id,
            s.intime,
            c.study_id,
            c.dicom_id,
            c.cxr_time,
            -- time between image (as dicom_id) and ICU admission. Used later to select which image to take if multiples
            -- No ABS(): I'm setting cxr_time <= T0 in the WHERE clause; (intime - cxr_time) cannot be negative
            date_diff('seconds', s.intime, c.cxr_time) as time_diff_cxr
        FROM stays_by_patient AS s
        INNER JOIN cxr_meta_parsed AS c
        ON CAST(s.subject_id AS VARCHAR) = CAST(c.subject_id AS VARCHAR)
        WHERE
    		s.stay_number = 1
    		AND c.cxr_time >= (s.intime - INTERVAL 24 HOURS)  -- only after T=-24h
    		AND c.cxr_time <= s.intime                        -- only before T=0
    ),
    deduplicated AS (
	SELECT
		subject_id,
		stay_id,
		study_id,
		dicom_id,
		intime AS icu_intime,
		-- numbering CXR images using how close to ICU intime they are for each patient and stay
		row_number() OVER(PARTITION BY subject_id, stay_id ORDER BY time_diff_cxr) AS cxr_number
	FROM valid_stays
    )
    SELECT * EXCLUDE(cxr_number) FROM deduplicated
	WHERE cxr_number = 1;
    """
    return connection.query(images_q).df(), images_q


def build_cohort(
    connection: DuckDBPyConnection, images_manifest: pd.DataFrame
) -> tuple[pd.DataFrame, str]:
    # images_manifest is used in this query, DuckDB supports
    # referencing pandas dataframes in queries directly (see at the end)
    cohort_q = """
    WITH stays AS (
	SELECT
		stay_id,
		subject_id,
		hadm_id,
        intime
        FROM mimiciv_icu.icustays
    ),
    admissions AS (
	SELECT
		hadm_id,
		subject_id,
        admittime,
        hospital_expire_flag
	FROM mimiciv_hosp.admissions
    ),
    patients AS (
	SELECT
		subject_id,
		gender,
          	anchor_age,
          	anchor_year
        FROM mimiciv_hosp.patients
    )
    SELECT
    	s.subject_id,
    	m.stay_id,
        m.dicom_id,
        m.study_id,
    	s.hadm_id,
    	s.intime AS icu_intime,
    	p.gender,
    	p.anchor_age + (EXTRACT(YEAR FROM a.admittime) - p.anchor_year) AS age,
    	a.hospital_expire_flag
    FROM images_manifest AS m
    INNER JOIN stays AS s
    	ON s.stay_id = m.stay_id
    INNER JOIN admissions AS a
    	ON a.hadm_id = s.hadm_id
    INNER JOIN patients as p
    	ON p.subject_id = s.subject_id
    ;
    """

    return connection.query(cohort_q).df(), cohort_q


def build_features(
    connection: DuckDBPyConnection, cohort: pd.DataFrame
) -> tuple[pd.DataFrame, str]:
    # TODO: parametrize

    labs_n_vitals_q = """
    WITH gt AS (
        SELECT stay_id, subject_id, icu_intime, gender, age, hospital_expire_flag
        FROM cohort
    ),
    aggregated_labs AS (
	SELECT

		g.stay_id,
		min(CASE WHEN l.itemid IN (50809, 50931) THEN l.valuenum END) as glucose_min,
		max(CASE WHEN l.itemid IN (50809, 50931) THEN l.valuenum END) as glucose_max,
		avg(CASE WHEN l.itemid IN (50809, 50931) THEN l.valuenum END) as glucose_mean,

		min(CASE WHEN l.itemid = 50813 THEN l.valuenum END) as lactate_min,
		max(CASE WHEN l.itemid = 50813 THEN l.valuenum END) as lactate_max,
		avg(CASE WHEN l.itemid = 50813 THEN l.valuenum END) as lactate_mean,

		min(CASE WHEN l.itemid = 50912 THEN l.valuenum END) as creatinine_min,
		max(CASE WHEN l.itemid = 50912 THEN l.valuenum END) as creatinine_max,
		avg(CASE WHEN l.itemid = 50912 THEN l.valuenum END) as creatinine_mean

	FROM gt AS g
	INNER JOIN mimiciv_hosp.labevents AS l
		ON g.subject_id = l.subject_id
        WHERE
      		l.itemid IN (50809, 50931, 50813, 50912)
		AND l.valuenum IS NOT NULL
		AND l.charttime <= g.icu_intime
		AND l.charttime >= (g.icu_intime - INTERVAL 24 HOURS)
	GROUP BY g.stay_id
    ),
    aggregated_icu_vitals AS (
	SELECT

        	g.stay_id,
            min(CASE WHEN v.itemid = 220045 THEN v.valuenum END) AS hr_min_icu,
            max(CASE WHEN v.itemid = 220045 THEN v.valuenum END) AS hr_max_icu,
            avg(CASE WHEN v.itemid = 220045 THEN v.valuenum END) AS hr_mean_icu,
		min(CASE WHEN v.itemid = 220181 THEN v.valuenum END) AS bp_min_icu,
		avg(CASE WHEN v.itemid = 220181 THEN v.valuenum END) AS bp_mean_icu,
		max(CASE WHEN v.itemid = 220181 THEN v.valuenum END) AS bp_max_icu,
		min(CASE WHEN v.itemid = 220210 THEN v.valuenum END) AS resprate_min_icu,
		avg(CASE WHEN v.itemid = 220210 THEN v.valuenum END) AS resprate_mean_icu,
		max(CASE WHEN v.itemid = 220210 THEN v.valuenum END) AS resprate_max_icu,
		min(CASE WHEN v.itemid = 223761 THEN v.valuenum END) AS temp_f_min_icu,
		avg(CASE WHEN v.itemid = 223761 THEN v.valuenum END) AS temp_f_mean_icu,
		max(CASE WHEN v.itemid = 223761 THEN v.valuenum END) AS temp_f_max_icu,
		min(CASE WHEN v.itemid = 220277 THEN v.valuenum END) AS spO2_min_icu,
		avg(CASE WHEN v.itemid = 220277 THEN v.valuenum END) AS spO2_mean_icu,
		max(CASE WHEN v.itemid = 220277 THEN v.valuenum END) AS spO2_max_icu

	FROM gt AS g
	INNER JOIN mimiciv_icu.chartevents AS v
	ON g.stay_id = v.stay_id
	WHERE
      		v.valuenum IS NOT NULL
		AND v.charttime >= (g.icu_intime - INTERVAL 24 HOURS)
        	AND v.charttime <= g.icu_intime
	GROUP BY g.stay_id
    ),
    aggregated_ed_vitals AS (
	SELECT

		g.stay_id,
		min(v.heartrate) AS hr_min_ed,
		max(v.heartrate) AS hr_max_ed,
		avg(v.heartrate) AS hr_mean_ed,
		min(v.sbp) AS bp_min_ed,
		max(v.sbp) AS bp_max_ed,
		avg(v.sbp) AS bp_mean_ed,
		min(v.resprate) AS resprate_min_ed,
		max(v.resprate) AS resprate_max_ed,
		avg(v.resprate) AS resprate_mean_ed,
		min(v.temperature) AS temp_f_min_ed,
		max(v.temperature) AS temp_f_max_ed,
		avg(v.temperature) AS temp_f_mean_ed,
		min(v.o2sat) AS spO2_min_ed,
		max(v.o2sat) AS spO2_max_ed,
		avg(v.o2sat) AS spO2_mean_ed

	FROM gt AS g
	INNER JOIN mimiciv_ed.vitalsign AS v
		ON g.subject_id = v.subject_id
	WHERE
		v.charttime >= (g.icu_intime - INTERVAL 24 HOURS)
		AND v.charttime <= g.icu_intime
	GROUP BY g.stay_id
    )
    SELECT
	g.subject_id,
        g.stay_id,
        g.gender,
      	g.age,
      	g.hospital_expire_flag,
        l.* EXCLUDE(stay_id),

        -- coalesce(a, b, c, ...) returns the first non-NULL value
        -- Preferring ED here since averages hopefully come from a longer monitoring in ED
        -- rather than in ICU before T=0
        coalesce(ev.hr_mean_ed, iv.hr_mean_icu) AS heart_rate_mean,
        least(iv.hr_min_icu, ev.hr_min_ed) AS heart_rate_min,
        greatest(iv.hr_max_icu, ev.hr_max_ed) AS heart_rate_max,
        coalesce(ev.bp_mean_ed, iv.bp_mean_icu) AS blood_pressure_mean,
        least(iv.bp_min_icu, ev.bp_min_ed) AS blood_pressure_min,
        greatest(iv.bp_max_icu, ev.bp_max_ed) AS blood_pressure_max,
        coalesce(ev.resprate_mean_ed, iv.resprate_mean_icu) AS resp_rate_mean,
        least(iv.resprate_min_icu, ev.resprate_min_ed) AS resp_rate_min,
        greatest(iv.resprate_max_icu, ev.resprate_max_ed) AS resp_rate_max,
        coalesce(ev.temp_f_mean_ed, iv.temp_f_mean_icu) AS temp_f_mean,
        least(iv.temp_f_min_icu, ev.temp_f_min_ed) AS temp_f_min,
        greatest(iv.temp_f_max_icu, ev.temp_f_max_ed) AS temp_f_max,
        coalesce(ev.spO2_mean_ed, iv.spO2_mean_icu) AS spO2_mean,
        least(iv.spO2_min_icu, ev.spO2_min_ed) AS spO2_min,
        greatest(iv.spO2_max_icu, ev.spO2_max_ed) AS spO2_max

    FROM gt AS g
    LEFT JOIN aggregated_labs AS l
	ON g.stay_id = l.stay_id
    LEFT JOIN aggregated_icu_vitals AS iv
	ON g.stay_id = iv.stay_id
    LEFT JOIN aggregated_ed_vitals AS ev
	ON g.stay_id = ev.stay_id
    """

    return connection.query(labs_n_vitals_q).df(), labs_n_vitals_q
