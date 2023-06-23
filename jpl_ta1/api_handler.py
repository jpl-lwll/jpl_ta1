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

import requests
import pandas as pd
import time
from typing import List, Any
from requests.exceptions import ConnectionError, Timeout
from jpl_ta1.utils.decorators import retry
from jpl_ta1.logger import get_module_logger

class LwllApiHandler:
    """
    API Handler helper class to abstract away HTTP requests
    """

    def __init__(self, base_url: str, team_secret: str, **kwargs: Any) -> None:
        self.base_url = base_url
        self.team_secret = team_secret
        self.non_session_headers = {'user_secret': self.team_secret}
        self.session_token = None
        # self.log = kwargs['log']
        self.log = get_module_logger(__name__, kwargs['log'])

    @retry((ConnectionError, Timeout))
    def get_all_tasks(self) -> dict:
        """
        Helper method to get all available tasks
        """
        self.log.debug(f"API handler calling `/list_tasks`")
        r = requests.get(f"{self.base_url}/list_tasks", headers=self.non_session_headers)
        response = r.json()
        if response:
            self.log.debug(f"api list_tasks response is {response}")
            tasks: dict = response['tasks']
            return tasks
        else:
            raise Exception("received an empty response from api")

    @retry((ConnectionError, Timeout))
    def get_task_metadata(self, task_id: str) -> dict:
        """
        Helper method to get a particular task metadata
        """
        self.log.debug(f"API handler calling `/task_metadata/{task_id}`")
        r = requests.get(f"{self.base_url}/task_metadata/{task_id}", headers=self.non_session_headers)
        response: dict = r.json()
        self.log.debug(f"{response}")
        return response

    @retry((ConnectionError, Timeout))
    def start_session(self, task_id: str, session_name: str, data_type: str) -> str:
        """
        Helper method to start a session with parameters and get the session token back
        """
        self.log.debug(f"API handler calling `/create_session`")
        r = requests.post(f"{self.base_url}/auth/create_session", json={'session_name': session_name,
                                                                        'data_type': data_type, 'task_id': task_id}, headers=self.non_session_headers)
        response = r.json()
        self.log.debug(f"{response}")
        session_token: str = response['session_token']
        return session_token

    @retry((ConnectionError, Timeout))
    def get_session_metadata(self, session_token: str) -> dict:
        """
        Helper method to get session metadata
        """
        self.log.debug(f"API handler calling `/session_status`")
        r = requests.get(f"{self.base_url}/session_status", headers=self._get_session_headers(session_token))
        response = r.json()
        self.log.debug(f"{response}")
        metadata: dict = response['Session_Status']
        return metadata

    # @retry((ConnectionError, Timeout))
    def get_seed_labels(self, session_token: str) -> List[dict]:
        """
        Helper method to get the first round of seed labels
        """
        self.log.debug(f"API handler calling `/seed_labels`")
        r = requests.get(f"{self.base_url}/seed_labels", headers=self._get_session_headers(session_token))
        response = r.json()
        if r.status_code == 200:
            if 'Labels' in response.keys():
                labels: list = response['Labels']
                return labels
            else:
                print(f"Labels were not in the response, got keys: {response.keys()}")
                raise Exception(response.get('trace', response.get('Error', 'unknown error')))
        else:
            self.log.exception(response.get('trace', response.get('Error', 'unknown error')))

    @retry((ConnectionError, Timeout))
    def submit_predictions(self, session_token: str, predictions: pd.DataFrame) -> None:
        """
        Helper method to submit predictions
        """
        self.log.debug(f"API handler calling `/submit_predictions`")
        r = requests.post(f"{self.base_url}/submit_predictions", json={'predictions': predictions.to_dict()}, headers=self._get_session_headers(session_token))
        response = r.json()
        if r.status_code != 200:
            self.log.error(response.get('trace', response.get('Error', 'unknown error')))

        self.log.debug(f"{response}")
        return

    @retry((ConnectionError, Timeout))
    def request_labels(self, session_token: str, items: List[str]) -> list:
        """
        Helper method to request labels
        """
        self.log.debug(f"API handler calling `/query_labels`")
        r = requests.post(f"{self.base_url}/query_labels", json={'example_ids': items}, headers=self._get_session_headers(session_token))
        response = r.json()
        self.log.debug(f"{response}")
        labels: list = response['Labels']
        return labels

    @retry((ConnectionError, Timeout))
    def _get_session_headers(self, session_token: str) -> dict:
        _headers = self.non_session_headers
        _headers['session_token'] = session_token
        return _headers
