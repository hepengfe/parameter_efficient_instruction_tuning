# coding=utf-8
# Copyright 2020 The TensorFlow Datasets Authors and the HuggingFace Datasets Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Natural Instruction V2 Dataset."""


import json
import os
import random
import datasets

logger = datasets.logging.get_logger(__name__)

_CITATION = """
@article{wang2022benchmarking,
  title={Benchmarking Generalization via In-Context Instructions on 1,600+ Language Tasks},
  author={Wang, Yizhong and Mishra, Swaroop and Alipoormolabashi, Pegah and Kordi, Yeganeh and others},
  journal={arXiv preprint arXiv:2204.07705},
  year={2022}
}
"""

_DESCRIPTION = """
Natural-Instructions v2 is a benchmark of 1,600+ diverse language tasks and their expert-written instructions. 
It covers 70+ distinct task types, such as tagging, in-filling, and rewriting. 
These tasks are collected with contributions of NLP practitioners in the community and 
through an iterative peer review process to ensure their quality. 
"""

_URL = "https://instructions.apps.allenai.org/"

class NIConfig(datasets.BuilderConfig):
    def __init__(self, *args, task_dir=None, max_num_instances_per_task=None, max_num_instances_per_eval_task=None, random_seed = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.task_dir: str = task_dir
        self.max_num_instances_per_task: int = max_num_instances_per_task
        self.max_num_instances_per_eval_task: int = max_num_instances_per_eval_task
        self.random_seed = random_seed if random_seed is not None else 42


class NaturalInstructions(datasets.GeneratorBasedBuilder):
    """NaturalInstructions Dataset."""

    VERSION = datasets.Version("2.0.0")
    BUILDER_CONFIG_CLASS = NIConfig
    BUILDER_CONFIGS = [
        NIConfig(name="default", description="Default config for NaturalInstructions")
    ]
    DEFAULT_CONFIG_NAME = "default"

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "Task": datasets.Value("string"),
                    "Contributors": datasets.Value("string"),
                    "Source": [datasets.Value("string")],
                    "URL": [datasets.Value("string")],
                    "Categories": [datasets.Value("string")],
                    "Categories_split": [datasets.Value("string")],
                    "Reasoning": [datasets.Value("string")],
                    "Definition": [datasets.Value("string")],
                    "Positive Examples": [{
                        "input": datasets.Value("string"),
                        "output": datasets.Value("string"),
                        "explanation": datasets.Value("string")
                    }],
                    "Negative Examples": [{
                        "input": datasets.Value("string"),
                        "output": datasets.Value("string"),
                        "explanation": datasets.Value("string")
                    }],
                    "Input_language": [datasets.Value("string")],
                    "Output_language": [datasets.Value("string")],
                    "Instruction_language": [datasets.Value("string")],
                    "Domains": [datasets.Value("string")],
                    "Instance": {
                        "id": datasets.Value("string"),
                        "input": datasets.Value("string"),
                        "output": [datasets.Value("string")]
                    },
                }
            ),
            supervised_keys=None,
            homepage="https://github.com/allenai/natural-instructions",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        if self.config.data_dir is None or self.config.task_dir is None:
            dl_path = dl_manager.download_and_extract(_URL)
            self.config.data_dir = self.config.data_dir or os.path.join(dl_path, "splits")
            self.config.task_dir = self.config.task_dir or os.path.join(dl_path, "tasks")

        split_dir = self.config.data_dir
        task_dir = self.config.task_dir
        length=self.config.max_num_instances_per_task
        random_seed = self.config.random_seed
        random.seed(42)
        numbers = list(range(2*length))
        # draw 200 numbers from 0 to len(all_instances)
        random.shuffle(numbers)
        self.indices = numbers
        self.train_indices = self.indices[:length]
        self.test_indices = self.indices[length:]
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "path": os.path.join(split_dir, "train_tasks.txt"), 
                    "task_dir": task_dir, 
                    "max_num_instances_per_task": self.config.max_num_instances_per_task,
                    "subset": "train"
                }),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "path": os.path.join(split_dir, "dev_tasks.txt"), 
                    "task_dir": task_dir,
                    "max_num_instances_per_task": self.config.max_num_instances_per_eval_task,
                    "subset": "dev"
                }),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "path": os.path.join(split_dir, "test_tasks.txt"), 
                    "task_dir": task_dir, 
                    "max_num_instances_per_task": self.config.max_num_instances_per_eval_task,
                    "subset": "test"
                }),
            datasets.SplitGenerator(
                name="traditional_test",
                gen_kwargs={
                    "path": os.path.join(split_dir, "train_tasks.txt"), 
                    "task_dir": task_dir, 
                    "max_num_instances_per_task": self.config.max_num_instances_per_eval_task,
                    "subset": "traditional_test"
                }),
        ]

    def _generate_examples(self, path=None, task_dir=None, max_num_instances_per_task=None, subset=None):
        """Yields examples."""
        logger.info(f"Generating tasks from = {path}")
        with open(path, encoding="utf-8") as split_f:
            if "train_tasks.txt" in path:
                cur_split = "train"
            elif "dev_tasks.txt" in path:
                cur_split = "dev_"
            elif "test_tasks.txt" in path:
                cur_split = "test"
            else:
                raise ValueError(f"Unknown split: {path}")
            i = 0
            for line in split_f: # each line is a task
                
                task_name = line.strip()
                # in dev file, test data is split into dev and test
                if task_name == "":
                    cur_split = "test_in_dev"
                    continue
                task_path = os.path.join(task_dir, task_name + ".json")
                i += 1
                # test
                # if i > 5 and (cur_split == "train" or cur_split == "test"):
                #     continue
                # print(line, ":", cur_split)
                with open(task_path, encoding="utf-8") as task_f:
                    s = task_f.read()
                    task_data = json.loads(s)
                    task_data["Task"] = task_name
                    # task_data["Task_by_split"] = 
                    if "Instruction Source" in task_data:
                        task_data.pop("Instruction Source")
                    all_instances = task_data.pop("Instances")
                    if subset == "test":
                        # for testing tasks, 100 instances are selected for efficient evaluation and they are label-balanced.
                        # we put them in the first for reproducibility.
                        # so, we use them here
                        instances = all_instances[:100]
                    else:
                        instances = all_instances
                    
                    
                    if max_num_instances_per_task is not None and max_num_instances_per_task >= 0:
                        if len(all_instances) < 200:
                            train_indices = list(range(100))
                            test_indices = list(range(100, len(all_instances)))
                        else:
                            train_indices = self.train_indices
                            test_indices = self.test_indices
                        # random.shuffle(instances)
                        # instances = instances[:max_num_instances_per_task]
                        if subset == "train":
                            instances = [all_instances[i] for i in train_indices if i < len(all_instances)]
                        elif subset == "traditional_test":
                            instances = [all_instances[i] for i in test_indices if i < len(all_instances)]
                        else:
                            instances = instances[:max_num_instances_per_task]
                    for idx, instance in enumerate(instances):
                        example = task_data.copy()
                        example["id"] = instance["id"]
                        example["Instance"] = instance
                        example["Categories"] = [f"{c}_{cur_split}" for c in example["Categories"]]
                        example["Categories_split"] = [cur_split] 
                        yield f"{task_name}_{idx}", example

