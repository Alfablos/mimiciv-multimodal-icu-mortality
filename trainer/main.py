from argparse import ArgumentParser

from .train import train_start


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.set_defaults(func=parser.print_help)

    commands = parser.add_subparsers()

    train_cmd = commands.add_parser(name="train", help="Train the model")
    train_cmd.set_defaults(func=train_start)

    args = parser.parse_args()
    args.func(args)
