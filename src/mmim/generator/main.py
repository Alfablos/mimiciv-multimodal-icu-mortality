from os import cpu_count
from argparse import ArgumentParser

from .builder import build_cli


def main():
    parser = ArgumentParser()
    parser.set_defaults(func=parser.print_help)

    parser.add_argument("--debug", type=bool, default=False)

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

    build_cmd.add_argument(
        "-i",
        "--images-base-dir",
        "--images-basedir",
        "--images-dir",
        required=True,
        help="""
        The path to the root of the images dataset and its alias (no spaces allowed in the alias).
        Format: ./path/to/dir@alias. Example: /home/myuser/datasets/mimic-images@mimic-cxr-jpg
        """,
    )
    build_cmd.add_argument(
        "-o", "--output-dir", "--out-dir", "--out", type=str, default="out"
    )
    build_cmd.add_argument(
        "-w",
        "--max-workers",
        "--workers",
        type=int,
        default=int((cpu_count() or 16) / 2),
    )

    build_cmd.set_defaults(func=build_cli)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
