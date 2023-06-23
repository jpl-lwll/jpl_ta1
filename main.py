# Copyright (c) 2023 California Institute of Technology (“Caltech”). U.S.
# Government sponsorship acknowledged.
# All rights reserved.
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# * Neither the name of Caltech nor its operating division, the Jet Propulsion
#   Laboratory, nor the names of its contributors may be used to endorse or
#   promote products derived from this software without specific prior written
#   permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

from pathlib import Path
import fire
from jpl_ta1.workflow import Workflow
from jpl_ta1.logger import get_module_logger

class CLI:
    """
    Main CLI class to launch off different configurations of the automated TA1 system

    """

    def __init__(self) -> None:
        pass

    def launch_system(self, dataset_type: str, problem_type: str, dataset_dir: str, environment: str, team_secret: str, skip_dataset: list = [], log_level: str = 'INFO', task_id: str = None) -> None:
        """
        Main launch method that takes our parameters and runs our system
        """
        valid_dataset_types = ['sample', 'full', 'all']
        if dataset_type not in valid_dataset_types:
            raise Exception(f'Invalid `dataset_type`, expected one of {valid_dataset_types}')

        valid_problem_types = ['image_classification', 'object_detection', 'machine_translation', 'video_classification',
                               'all']
        if problem_type not in valid_problem_types:
            raise Exception(f'Invalid `problem_type`, expected one of {valid_problem_types}')

        valid_environments = ['local', 'dev', 'staging', 'prod']
        if environment not in valid_environments:
            raise Exception(f'Invalid `environment`, expected one of {valid_environments}')

        if not Path(dataset_dir).exists():
            raise Exception('`dataset_dir` does not exist..')
        if environment in ['local', 'dev', 'staging'] and not Path(dataset_dir).joinpath('development').exists():
            raise Exception(f"Can't find `development` dataset directory in path {dataset_dir} while running in one of {['local', 'dev', 'staging']} mode")
        if environment == 'prod' and not Path(dataset_dir).joinpath('evaluation').exists():
            raise Exception(f"Can't find `evaluation` dataset directory in path {dataset_dir} while running in `prod` mode")
        if not Path(dataset_dir).joinpath('external').exists():
            raise Exception(f"Can't find `external` dataset directory in path {dataset_dir}")

        valid_log_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR']
        if log_level not in valid_log_levels:
            raise Exception(f'Invalid `log_level`, expected one of {valid_log_levels}, but got {log_level}')
        # log = Logger(__name__, log_level)

        skip_dataset = [skip_dataset] if isinstance(skip_dataset, str) else [dataset for dataset in skip_dataset]

        # Now launch the system
        workflow = Workflow(dataset_type, problem_type, dataset_dir, environment, team_secret, task_id, skip_dataset=skip_dataset, log=log_level)
        workflow.run()


def main() -> None:
    fire.Fire(CLI)


if __name__ == '__main__':
    fire.Fire(CLI)
