from argparse import ArgumentParser

from .builder import build


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.set_defaults(func=parser.print_help)

    commands = parser.add_subparsers()

    build_cmd = commands.add_parser(
        name="build-dataset",
        help="""Build a dataset from a DuckDB database and an image directory.
                It will generate: $TRAINING_DATASET_FILE, $VALIDATION_DATASET_FILE, $DATASET_STATS_FILE
                and a ds_test.csv (test data from the same distribution)
             """,
        aliases=["build", "build-ds", "dataset", "ds"],
    )
    build_cmd.add_argument(
        "-d",
        "--database-path",
        "--build",
        "--dbpath",
        "--db",
        "--database",
        "--db-path",
        required=True,
        help="The path to the DuckDB database file. IMPORTANT: the db should contain both MIMIC-IV and MIMIC-ED",
    )

    build_cmd.add_argument(
        "-m",
        "--metadata-file",
        "--metadata",
        "--xcr-metadata-file",
        required=True,
        help="""The file containing the metadata for MIMIC-CXR,
                that is actually stored in the MIMIC-CXR-JPG dataset.""",
    )

    build_cmd.set_defaults(func=build)

    args = parser.parse_args()
    args.func(args)
