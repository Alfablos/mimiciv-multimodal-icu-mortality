from argparse import ArgumentParser

from train import train_cli


def main():
    parser = ArgumentParser()
    parser.set_defaults(func=parser.print_help)

    commands = parser.add_subparsers()

    train_cmd = commands.add_parser(name="train", help="Train the model")
    train_cmd.add_argument(
        "-m",
        "--manifest-uri",
        "--manifest",
        help="The path to the manifest.json file in URI form. Supported schemes are `file://` and `lakefs://`",
        required=True,
    )
    train_cmd.add_argument(
        "-w",
        "--working-directory",
        "--workdir",
        default="./",
        help="Where training data will be stored. It'll be ignored for filesystem-based stores, that is, for manifest URIs starting with `file://`, to avoid data duplication.",
    )

    train_cmd.add_argument("-v", "--verbose", action="store_true", default=False)

    train_cmd.set_defaults(func=train_cli)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
