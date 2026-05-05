from lakefs.client import Client as LakeFSClient
from lakefs import Repository

import dagster as dg


class LakeFSResource(dg.ConfigurableResource):
    host: str
    username: str
    password: str

    def get_client(self) -> LakeFSClient:
        return LakeFSClient(
            host=self.host, username=self.username, password=self.password
        )

    def get_text_files_from_ref(
        self, repo: str, ref: str, requested_files: list[str]
    ) -> dict[str, str]:
        _ref = Repository(repository_id=repo, client=self.get_client()).ref(ref)
        found = {}

        for file in requested_files:
            f = _ref.object(file)

            if not f.exists():
                raise FileNotFoundError(
                    f"File {file} is not in repo `{repo}` at ref `{ref}`"
                )

            with f.reader(mode="r") as f_reader:
                found[file] = f_reader.read()

        return found


@dg.definitions
def resources():
    return dg.Definitions(
        resources={
            "lakefs": LakeFSResource(
                host=dg.EnvVar("LAKEFS_HOST"),
                username=dg.EnvVar("LAKEFS_USERNAME"),
                password=dg.EnvVar("LAKEFS_PASSWORD"),
            )
        }
    )
