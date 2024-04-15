from prefect import flow
from prefect.deployments import DeploymentImage
from prefect.runner.storage import GitRepository

if __name__ == "__main__":
    flow.from_source(
        source=GitRepository(
        url="https://github.com/dzarbiv95/prefect-pipeline.git",
        branch="master",
        # credentials={
        #     "access_token": Secret.load("github-access-token")
        # }
    ),
    entrypoint="main.py:father_flow",
    ).deploy(
        name="first-pipeline",
        work_pool_name="base-pipeline",
        image=DeploymentImage({

        })
    )