from argparse import ArgumentParser

from .train import train_cli


def main():
    parser = ArgumentParser()
    parser.set_defaults(func=parser.print_help)

    commands = parser.add_subparsers()

    train_cmd = commands.add_parser(name="train", help="Train the model")
    group = train_cmd.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "-i",
        "--input-dir",
        "--input",
        help="The directory containing the manifest.json file",
    )
    group.add_argument(
        "-r",
        "--ref-str",
        "--ref",
        help="a string containing the LakeFS repo name and ref, formatted as `<repo>@<ref>`",
    )

    train_cmd.set_defaults(func=train_cli)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
