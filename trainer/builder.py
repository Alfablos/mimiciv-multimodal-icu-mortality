import json
import os
from sklearn.model_selection import train_test_split
import pandas as pd
import duckdb
from duckdb import DuckDBPyConnection

from .utils import find_paths

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


def build(args):
    duckdb_db = args.database_path
    images_basedir = args.images_basedir
    metadata_file = args.metadata_file
    not_found = find_paths([duckdb_db, images_basedir, metadata_file])
    if len(not_found) != 0:
        raise FileNotFoundError(f"Unable to find files: {', '.join(not_found)}")

    dconn = duckdb.connect(database=duckdb_db, read_only=True)
    images_manifest = build_images_manifest(
        connection=dconn, metadata_file=metadata_file
    )
    cohort = build_cohort(connection=dconn, images_manifest=images_manifest)

    features = build_features(connection=dconn, cohort=cohort)
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
    val_ds, test_ds = train_test_split(other_ds, test_size=0.1)

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
    X_train: pd.DataFrame = train_ds.drop("hospital_expire_flag", axis=1)
    # Y_train: pd.DataFrame = train_ds["hospital_expire_flag"]  # noqa: F841
    # X_val: pd.DataFrame = val_ds.drop("hospital_expire_flag", axis=1)  # noqa: F841
    # Y_val: pd.DataFrame = val_ds["hospital_expire_flag"]  # noqa: F841
    # X_test: pd.DataFrame = test_ds.drop("hospital_expire_flag", axis=1)  # noqa: F841
    # Y_test: pd.DataFrame = test_ds["hospital_expire_flag"]  # noqa: F841

    print("Computing training set statistics...")
    stats = {  # noqa: F841
        "mean": X_train[continuous_variables].mean().to_dict(),
        "std": X_train[continuous_variables].std().to_dict(),
    }

    # Write files according to variables! $TRAINING_DATASET_FILE, $VALIDATION_DATASET_FILE, $DATASET_STATS_FILE and a ds_test.csv
    train_ds.to_csv(
        os.getenv("TRAINING_DATASET_FILE", default_datasets_dir + "ds_train.csv"),
        index=False,
    )
    with open(
        os.getenv("DATASET_STATS_FILE", default_datasets_dir + "stats.json"), "w"
    ) as f:
        json.dump(stats, f)
    val_ds.to_csv(
        os.getenv("VALIDATION_DATASET_FILE", default_datasets_dir + "ds_val.csv"),
        index=False,
    )
    test_ds.to_csv(
        os.getenv("TEST_DATASET_FILE", default_datasets_dir + "ds_test.csv"),
        index=False,
    )


def prepare_set(df: pd.DataFrame, medians: dict[str, float]):
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


def build_images_manifest(connection: DuckDBPyConnection, metadata_file: str):
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
    return connection.query(images_q).df()


def build_cohort(connection: DuckDBPyConnection, images_manifest: str) -> pd.DataFrame:
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

    return connection.query(cohort_q).df()


def build_features(
    connection: DuckDBPyConnection, cohort: pd.DataFrame
) -> pd.DataFrame:
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

    return connection.query(labs_n_vitals_q).df()
