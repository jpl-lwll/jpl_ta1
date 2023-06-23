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

from typing import List, Any
from jpl_ta1.api_handler import LwllApiHandler
from jpl_ta1.session import Session
import time
from jpl_ta1.logger import get_module_logger


class Workflow:

    def __init__(self, dataset_type: str, problem_type: str, dataset_dir: str, environment: str, team_secret: str, task_id: str = None, skip_dataset: list = [], **kwargs: Any) -> None:
        self.environment = environment
        self.log_level = kwargs['log']
        self.log = get_module_logger(__name__, self.log_level)
        self.base_url = self._get_endpoint(environment)
        self.api_handler = LwllApiHandler(self.base_url, team_secret, log=kwargs['log'])
        self.skip_dataset = skip_dataset
        self.task_list = self.api_handler.get_all_tasks()
        self.tasks = {
            'image_classification': self._get_task_subset_by_type('image_classification'),
            'object_detection': self._get_task_subset_by_type('object_detection'),
            'machine_translation': self._get_task_subset_by_type('machine_translation'),
            'video_classification': self._get_task_subset_by_type('video_classification')
        }
        '''self.tasks = {
            'image_classification': ['6d5e1f85-5d8f-4cc9-8184-299db03713f4'],
            'object_detection': ['dc75cc32-5db1-4767-b41f-b3dfa6b086a9'],
            'machine_translation': ['problem_test_machine_translation'],
            'video_classification': ['problem_test_video_classification']
        }'''

        self.dataset_type = dataset_type
        self.problem_type = problem_type
        self.dataset_dir = dataset_dir
        self.task_id = task_id

        self._tasks_completed: List[str] = []  # private variable to allow us to keep track of the tasks we completed
        self._task_types_completed = {
            'image_classification--full': 0,
            'image_classification--sample': 0,
            'object_detection--full': 0,
            'object_detection--sample': 0,
            'video_classification--full': 0,
            'video_classification--sample': 0,
            'machine_translation--full': 0,
            'machine_translation--sample': 0,
        }

    def run(self) -> None:
        self.log.info(f"Starting Workflow Loop...")
        t_start = int(time.time())
        self._launch_workflow_loop()
        t_end = int(time.time())
        m, s = divmod(int(t_end - t_start), 60)
        h, m = divmod(m, 60)
        time_str = f'{h:d} Hours {m:02d} Minutes and {s:02d} seconds'
        self.log.info('\n\n')
        self.log.info(f'Tasks Completed: {self._tasks_completed}')
        self.log.info(f'Tasks Completed Breakdown: {self._task_types_completed}')
        self.log.info(f"Finished Complete Workflow run in {time_str}")

    def _launch_workflow_loop(self) -> None:
        if self.dataset_type in ['sample', 'all']:
            if self.task_id:
                self.log.info(f"launching task_id_loop for task: {self.task_id}")
                self._task_id_loop('sample')
            elif self.problem_type:
                self.log.info(f"launching problem_type_loop for type: {self.problem_type}")
                self._problem_type_loop('sample')
        if self.dataset_type in ['full', 'all']:
            if self.task_id:
                self._task_id_loop('full')
            elif self.problem_type:
                self._problem_type_loop('full')
        return

    def _problem_type_loop(self, dataset_config: str) -> None:
        if self.problem_type in ['image_classification', 'all']:
            self._session_loop(dataset_config, 'image_classification')
        if self.problem_type in ['object_detection', 'all']:
            self._session_loop(dataset_config, 'object_detection')
        if self.problem_type in ['machine_translation', 'all']:
            self._session_loop(dataset_config, 'machine_translation')
        if self.problem_type in ['video_classification', 'all']:
            self._session_loop(dataset_config, 'video_classification')
        return

    def _task_id_loop(self, dataset_config: str) -> None:
        self.log.info(f'Creating session for task: {self.task_id}')
        session = Session(self.environment, dataset_config, self.task_id,
                          self.api_handler, self.dataset_dir, log=self.log_level)
        session.run()
        self._tasks_completed.append(f"{self.task_id}--{dataset_config}")
        self._task_types_completed[f"{self.problem_type}--{dataset_config}"] += 1

    def _session_loop(self, dataset_config: str, problem_type_config: str) -> None:
        for _task in self.tasks[problem_type_config]:
            self.log.info(f'Creating session for task: {_task}')
            session = Session(self.environment, dataset_config, _task, self.api_handler,
                              self.dataset_dir, problem_type_config, log=self.log_level)
            session.run()
            self._tasks_completed.append(f"{_task}--{dataset_config}")
            self._task_types_completed[f"{problem_type_config}--{dataset_config}"] += 1
        return

    @staticmethod
    def _get_endpoint(environment: str) -> str:
        """
        Lookup to get our valid url based on the environment
        """
        url_lookup = {
            'local': 'http://localhost:5000',
            'dev': 'https://api-dev.lollllz.com',
            'staging': 'https://api-staging.lollllz.com',
            'prod': 'https://api-prod.lollllz.com',
        }
        url = url_lookup[environment]
        return url

    def _get_task_subset_by_type(self, subset_type: str) -> List[str]:
        """
        Helper function that returns the task ids in a list that match a specified
        problem type

        Params
        ------

        subset_type : str
            The task_type subset you want to get back
        """
        subset_tasks = []
        for _task in self.task_list:
            try:
                task_meta = self.api_handler.get_task_metadata(_task)
                if task_meta['task_metadata']['adaptation_dataset'] in self.skip_dataset or task_meta['task_metadata']['base_dataset'] in self.skip_dataset:
                    self.log.info(f"Skipping task: {_task} - in dataset skip list ({self.skip_dataset})")
                    continue
                if task_meta['task_metadata']['problem_type'] == subset_type:
                    subset_tasks.append(_task)
            except Exception as err:
                self.log.error(f"Error getting task metadata for task: {_task}, error: {err}")
        return subset_tasks
