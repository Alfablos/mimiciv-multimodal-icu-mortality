default:
  @ just --list

generator *args:
  @uv --project generator run python -m generator.main {{args}}

trainer *args:
  @uv --project trainer run python -m trainer.main {{args}}

test:
  @uv run --group dev pytest -q

generator-test:
  @uv run --project generator --group dev pytest -q generator/tests

trainer-test:
  @uv run --project trainer --group dev pytest -q trainer/tests
