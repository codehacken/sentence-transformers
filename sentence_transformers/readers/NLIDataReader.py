from . import InputExample
import csv
import gzip
import os


class NLIDataReader(object):
    """
    Reads in the Stanford NLI dataset and the MultiGenre NLI dataset
    """
    def __init__(self, dataset_folder):
        self.dataset_folder = dataset_folder

    def get_examples(self, filename, max_examples=0, multiplier=False):
        """
        data_splits specified which data split to use (train, dev, test).
        Expects that self.dataset_folder contains the files s1.$data_split.gz,  s2.$data_split.gz,
        labels.$data_split.gz, e.g., for the train split, s1.train.gz, s2.train.gz, labels.train.gz
        """
        s1 = gzip.open(os.path.join(self.dataset_folder, 's1.' + filename),
                       mode="rt", encoding="utf-8").readlines()
        s2 = gzip.open(os.path.join(self.dataset_folder, 's2.' + filename),
                       mode="rt", encoding="utf-8").readlines()
        labels = gzip.open(os.path.join(self.dataset_folder, 'labels.' + filename),
                           mode="rt", encoding="utf-8").readlines()

        examples = []
        id = 0
        for sentence_a, sentence_b, label in zip(s1, s2, labels):
            guid = "%s-%d" % (filename, id)
            id += 1
            if multiplier is True:
                examples.append(InputExample(guid=guid, texts=[sentence_a, sentence_b],
                                             label=self.map_label(label),
                                             multiplier=self.map_multipliers(label)))
            else:
                examples.append(InputExample(guid=guid, texts=[sentence_a, sentence_b],
                                             label=self.map_label(label)))

            if 0 < max_examples <= len(examples):
                break

        return examples

    @staticmethod
    def get_labels():
        return {"contradiction": 0, "entailment": 1, "neutral": 2}

    @staticmethod
    def get_multiplier():
        return {"entailment": 1.0, "contradiction": 0.0, "neutral": -1.0}

    def get_num_labels(self):
        return len(self.get_labels())

    def map_multipliers(self, label):
        return self.get_multiplier()[label.strip().lower()]

    def map_label(self, label):
        return self.get_labels()[label.strip().lower()]
