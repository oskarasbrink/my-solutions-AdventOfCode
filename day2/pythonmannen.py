import requests
import json
import os
from databricks_cli.sdk.api_client import ApiClient
from databricks_cli.workspace.api import WorkspaceApi, WorkspaceFileInfo
from databricks_cli.workspace.cli import *
from databricks_cli.configure.provider import *
from databricks_cli.clusters.api import ClusterApi
from databricks_cli.jobs.api import JobsApi
from databricks_cli.jobs.cli import *
from databricks_cli.runs.api import RunsApi

api_client = ApiClient(
                host  = get_config().host,
                token = get_config().token
                )


workspace = WorkspaceApi(api_client)

workspace.export_workspace("/scalable-data-science/000_0-sds-3-x-projects-2022/student-project-09_group-DistEnsembles/01_Human_Pose_Data","/Users/oskarasbrink/asd","DBC",True)
