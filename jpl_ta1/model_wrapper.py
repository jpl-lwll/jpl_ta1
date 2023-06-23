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

import pandas as pd
from typing import List, Tuple, Any
from pathlib import Path
import random
from jpl_ta1.logger import get_module_logger

class ModelWrapper:
    """
    This class will hold the overall training and prediction loop that will be called from a Workflow

    In order to allow flexibility of testing, but also ensure automated working of the system, we will parameterize this class
    while keeping an interface of `fit` and `predict` methods. This allows us to create anything from ensembles of models to
    more advanced structures while keeping a single interface the Workflow object knows how to interact with

    dataset_metadata schema example:

        'classes': [
                    '0',
                    '1',
                    '2',
                    '3',
                    '4',
                    '5',
                    '6',
                    '7',
                    '8',
                    '9'
                    ],
        'dataset_type': 'image_classification',
        'name': 'mnist',
        'number_of_channels': 1,
        'number_of_classes': 10,
        'number_of_samples_test': 1000,
        'number_of_samples_train': 5000,
        'uid': 'mnist',
    }

    """

    def __init__(self, environment: str, dataset_dir: str, dataset_config: str, **kwargs: Any) -> None:
        self.environment = environment
        self.working_path = 'development' if self.environment in ['local', 'dev', 'staging'] else 'evaluation'
        self.dataset_dir = Path(dataset_dir)  # how we know where to look for the current dataset ex. 'mnist'
        self.dataset_config = dataset_config  # how we know if we are doing 'sample' or 'full'
        self.dataset_metadata: dict = {}  # gives us information about what type of dataset we are dealing with
        self.pair_stage = ''  # Either `base` or `adaptation` -- Set after instantiation
        self.log = get_module_logger(__name__, kwargs['log'])

    def set_stage(self, stage: str, dataset_metadata: dict) -> None:
        """
        Helper method called at the beginning of each of `base` and `adaptation` phase in order to make sure we are informed on the the stage
        and have the most up to date dataset metadata
        """
        self.pair_stage = stage
        self.dataset_metadata = dataset_metadata
        return

    def fit(self, data: dict) -> None:
        if self.pair_stage == '':
            raise Exception('Forgot to set pair_stage before fitting')
        if self.dataset_metadata == {}:
            raise Exception("Didn't set current dataset metadata")
        pass

    def predict(self) -> pd.DataFrame:
        if self.dataset_metadata['dataset_type'] == 'machine_translation':
            test_df = self._get_test_data_mt()
            df = self._generate_random_predictions_on_test_set(model_type=self.dataset_metadata['dataset_type'], test_df=test_df)
        else:
            test_imgs, dataset_classes = self._get_test_images_and_classes()
            df = self._generate_random_predictions_on_test_set(model_type=self.dataset_metadata['dataset_type'],
                                                               test_imgs=test_imgs, current_dataset_classes=dataset_classes)
        return df

    def most_uncertain_unlabeled_items(self) -> List[str]:
        """
        Main method to trigger active learning loops of our ModelWrapper abstraction

        This will produce the most 'N' uncertain labels that we want to request for and will get called in
        a loop until we satisfy the Session condition of we are up to our budget checkpoint
        """
        return []

    def _get_test_images_and_classes(self) -> Tuple[List[str], List[str]]:
        """
        Helper method to dynamically get the test labels and give us the possible classes that can be submitted
        for the current dataset

        Returns
        -------

        Tuple[List[str], List[str]]
            The list of test image ids needed to submit a prediction and the list of class names that you can predict against
        """
        current_dataset_name = self.dataset_metadata['name']
        current_dataset_classes = self.dataset_metadata['classes']

        test_imgs_dir = self.dataset_dir.joinpath(f"{self.working_path}/{current_dataset_name}/{current_dataset_name}_{self.dataset_config}/test")
        if (self.dataset_metadata['dataset_type'] == 'video_classification'):
            test_imgs = [test_imgs_dir]
        else:
            test_imgs = [f.name for f in test_imgs_dir.iterdir() if f.is_file()]
        return test_imgs, current_dataset_classes

    def _get_test_data_mt(self) -> List[str]:
        """
        Helper method to dynamically get the test labels and give us the possible classes that can be submitted
        for the current dataset

        Params
        ------

        dataset_path : Path
            The path to the `development` dataset downloads

        session_token : str
            Your current session token so that we can look up the current session metadata

        Returns
        -------

        pd.DataFrame
            The DataFrame on which you must make predictions from a 'source' column
        """
        # Then we can just reference our current metadata to get our dataset name and use that in the path
        current_dataset_name = self.dataset_metadata['name']

        _path = str(
            self.dataset_dir.joinpath(f"{self.working_path}/{current_dataset_name}/{current_dataset_name}_{self.dataset_config}/test_data.feather"))
        test_df = pd.read_feather(_path)
        return test_df

    @staticmethod
    def _generate_random_predictions_on_test_set(model_type: str, test_imgs: List[str] = None, current_dataset_classes: List[str] = None, test_df = None) -> pd.DataFrame:
        """
        Generates a prediction dataframe for image classification based on random sampling from our available classes
        """
        if model_type == 'image_classification':
            rand_lbls = [str(random.choice(current_dataset_classes)) for _ in range(len(test_imgs))]
            df = pd.DataFrame({'id': test_imgs, 'class': rand_lbls})
        elif model_type == 'object_detection':
            # We just use random labels for example. Our labels have to have a bounding box, confidence and class for object detection
            # bounding boxes are defined as '<xmin>, <ymin>, <xmax>, <ymax>''
            # This would be your inferences filling this DataFrame though.
            rand_lbls = ['20, 20, 80, 80' for _ in range(len(test_imgs))]
            conf = [0.95 for _ in range(len(test_imgs))]
            classes = [current_dataset_classes[0] for _ in range(len(test_imgs))]
            df = pd.DataFrame({'id': test_imgs, 'bbox': rand_lbls, 'confidence': conf, 'class': classes})
        elif model_type == 'machine_translation':
            # We make fake predictions and want a DataFrame with the columns
            # 'id' and 'text'
            pred = 'The quick brown fox jumps over the lazy dog'
            pred_list = [pred for _ in range(len(test_df))]
            df = pd.DataFrame({'id': test_df['id'].tolist(), 'text': pred_list})
        elif model_type == 'video_classification':
            test_vid_dir = test_imgs[0]
            test_ids =  [vid.name for vid in test_vid_dir.iterdir()]
            # if '.DS_Store' in test_ids:
            #    test_ids.remove('.DS_Store')

            data = []
            for tid in test_ids:
                fr_list = [f.name for f in test_vid_dir.joinpath(tid).iterdir() if f.is_file()]
                # if '.DS_Store' in fr_list:
                #     fr_list.remove('.DS_Store')
                fr_list = sorted(fr_list)
                data.append({'id': tid,
                             'class': str(random.choice(current_dataset_classes))})
            df = pd.DataFrame(columns=['id', 'class']).from_dict(data)
        return df
