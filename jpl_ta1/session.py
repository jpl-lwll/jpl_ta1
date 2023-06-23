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

import json
import time
from jpl_ta1.api_handler import LwllApiHandler
from jpl_ta1.model_wrapper import ModelWrapper
from typing import List, Any
from datetime import datetime
from jpl_ta1.logger import get_module_logger


class Session:

    def __init__(self, environment: str, dataset_config: str, task_id: str, api_handler: LwllApiHandler, dataset_dir: str,
                 problem_type: str = None, name_prefix: str = "", name_postfix: str = "", **kwargs: Any) -> None:
        # Configuration variables
        self.environment = environment
        self.dataset_config = dataset_config
        self.task_id = task_id
        self.api_handler = api_handler
        self.dataset_dir = dataset_dir
        self.log_level = kwargs['log']
        self.log = get_module_logger(__name__, self.log_level)
        self.problem_type = problem_type

        # Session state variables
        session_name = f"{name_prefix} - Run starting at {datetime.now().strftime('%m/%d/%Y, %H:%M:%S')} - {name_postfix}"
        self.log.info(f"Starting session with name: {session_name}")
        self.session_token = self.api_handler.start_session(self.task_id, session_name, self.dataset_config)
        self.metadata = self._refresh_metadata()
        self.log.debug(f'Initial Session Metadata: {self.metadata}')

        # Session state evolution variables
        self.label_cache: dict = {}  # this will become more advance as we start implementing actual TA1 logic and not just a shell

    def run(self) -> None:
        # instantiate our ModelWrapper that will have the ability to be parameterized to do everything from ensembles to
        # more complex model constructions
        t_start = int(time.time())
        model = ModelWrapper(self.environment, self.dataset_dir, self.dataset_config, log=self.log_level)

        if self.problem_type == 'machine_translation':
            range_lim = 8
        else:
            range_lim = 4

        for stage in ['base', 'adaptation']:
            self.log.info(f"Starting Stage: {stage}")
            # We make sure our ModelWrapper knows which stage it is in for downstream logic controlling domain adaption methodologies
            model.set_stage(stage, self.metadata['current_dataset'])

            # Stage 1 - 4 Get Seed Labels
            if self.problem_type != 'machine_translation':
                for i in range(4):
                    self.log.info(f"Getting Round {i + 1} Seed Labels")
                    seed_labels = self.api_handler.get_seed_labels(self.session_token)
                    self.metadata = self._refresh_metadata()  # refreshes how many labels we have until checkpoint
                    self.log.info(f"Budget used: {self.metadata['budget_used']}, " +
                                  f"Budget left: {self.metadata['budget_left_until_checkpoint']}")
                    self._add_to_label_cache(seed_labels, seed_round=True)
                    model.fit(self.label_cache)
                    self.api_handler.submit_predictions(self.session_token, model.predict())
                    self.metadata = self._refresh_metadata()
                    # self.log.info(f"Submitted predictions: {self.metadata}")
                    self.log.info(f"Submitted predictions. Budget used: {self.metadata['budget_used']}, "
                                  + f"Budget left: {self.metadata['budget_left_until_checkpoint']}")

            # Stage 5 - 8 -- Active Learning Rounds
            for i in range(range_lim):
                if self.problem_type == 'machine_translation':
                    self.log.info(f"Starting checkpoint `{i}` loop for current stage")
                else:
                    self.log.info(f"Starting checkpoint `{i+5}` loop for current stage")

                while self.metadata['budget_left_until_checkpoint'] > 0:
                    self._request_label_loop(model)
                    model.fit(self.label_cache)
                    self.metadata = self._refresh_metadata()  # refreshes how many labels we have until checkpoint
                    self.log.info(f"Submitted predictions. Budget used: {self.metadata['budget_used']}, "
                                  + f"Budget left: {self.metadata['budget_left_until_checkpoint']}")
                    # While this JPL TA1 is just a shell, we artificially change the metadata `budget_left_until_checkpoint` to 0
                    # When we start actually requesting labels this will go down as we request more labels in our `_request_label_loop` loop
                    self.metadata['budget_left_until_checkpoint'] = 0
                self.api_handler.submit_predictions(self.session_token, model.predict())
                self.metadata = self._refresh_metadata()

                # assert to verify the session metadata is doing what we want it to and we are going on to
                # the adaptation phase
                if i == range_lim and model.pair_stage == 'base':
                    assert self.metadata['pair_stage'] == 'adaptation'

        # Assert that we have finished this particular Session
        assert self.metadata['active'] == 'Complete', f"Session is not complete. Metadata: {self.metadata}"
        t_end = int(time.time())
        m, s = divmod(int(t_end - t_start), 60)
        h, m = divmod(m, 60)
        time_str = f'{h:d} Hours {m:02d} Minutes and {s:02d} seconds'
        self.log.info(f"Successfully completed Session. Finished Session run in {time_str}")

        return

    def _request_label_loop(self, model: ModelWrapper) -> None:
        """
        Active Learning Loop

        In this loop we want to exploit what our model is most uncertain about or other methods to determine what we should be requesting for
        """
        samples_to_request = model.most_uncertain_unlabeled_items()
        labels = self.api_handler.request_labels(self.session_token, samples_to_request)
        self._add_to_label_cache(labels, seed_round=False)

    def _add_to_label_cache(self, labels: List[dict], seed_round: bool = False) -> None:
        """
        Helper method to add to our label cache state throughout a session

        *This doesn't do anything for the time being until we start implementing our own TA1 logic and
        move beyond just an example shell
        """
        pass

    def _refresh_metadata(self) -> dict:
        """
        Helper to refresh metadata on current session
        """
        old_metadata = {}
        # Make sure we've already fetched the metadata before trying to load the old version
        if hasattr(self, "metadata"):
            old_metadata = json.loads(json.dumps(self.metadata))
            # Delete date_last_interacted for comparison as this can give a false positive that things were changed
            del old_metadata["date_last_interacted"]

        metadata = self.api_handler.get_session_metadata(self.session_token)

        # Check if metadata changed and log if it didn't (json conversion hack to do a deep copy)
        new_metadata = json.loads(json.dumps(metadata))
        del new_metadata["date_last_interacted"]
        if json.dumps(old_metadata) == json.dumps(new_metadata):
            self.log.info(f"Session metadata did not change after refresh...")

        return metadata
