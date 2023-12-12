import pandas as pd
import math
import random
from classifier.classifier_models import *

__author__ = "Eli Tiao, David Ponce De Leon"
__copyright__ = "Copyright 2023, Westmont College"
__credits__ = ["Eli Tiao", "David Ponce De Leon"]
__email__ = "jtiao@westmont.edu, dponcedeleon@westmont.edu"


# makes a dictionary
class ValorantFeature(Feature):
    def __init__(self, name, value):
        super().__init__(name, value)


class ValorantFeatureSet(FeatureSet):
    """A set of features that represent a single object. Optionally includes the known class of the object.
        Our feature set is going to consist of the individual words within a chunk of each inaugural speech.

        Attributes:
            _feat (set[Feature]): a set of features that define this object for the purposes of a classifier
            _clas (str | None): optional attribute set as the pre-defined classification of this object
        """
    def __init__(self, features: set[Feature], known_clas=None):
        super().__init__(features, known_clas)

    @classmethod
    def build(cls, source_object: pd.Series, known_clas=None, **kwargs) -> FeatureSet:
        """Method that builds and returns an instance of FeatureSet given a source object that requires preprocessing.

        :param source_object: object to build the feature set from
        :param known_clas: pre-defined classification of the source object
        :param kwargs: any additional data needed to preprocess the `source_object` into a feature set
        :return: an instance of `FeatureSet` built based on the `source_object` passed in
        """
        player = source_object['player']
        map = source_object['map']
        numerical_stats = source_object[['kill', 'death', 'assist', 'adr', 'fk', 'fd']]
        features = {ValorantFeature('player' + name, True) for name in player}
        features |= {ValorantFeature('map' + map_name, True) for map_name in map}
        features |= {ValorantFeature(stat, numerical_stats[stat]) for stat in numerical_stats.index}

        return ValorantFeatureSet(features, known_clas)


class ValorantAbstractClassifier(AbstractClassifier):
    """After classifying our train set by hand, the abstract classifier will allow us to see which words can most
        accurately identify which party the speech is from. """
    def __init__(self, classifier: dict):
        self.dict = classifier

    def gamma(self, a_feature_set: ValorantFeatureSet) -> str:
        """Given a single feature set representing an object to be classified, returns the most probable class
        for the object based on the training this classifier received (via a call to `train` class method).

        :param a_feature_set: a single feature set representing an object to be classified
        :return: name of the class with the highest probability for the object
        """
        # TODO: return probability for the sentence and the political party

        role_probabilities = {role: 0.0 for role in ['duelist', 'sentinel', 'initiator', 'controller']}

        for feature in a_feature_set.feat:
            if feature.name in self.dict:
                for role in role_probabilities:
                    role_probabilities[role] += math.log(self.dict[feature.name][role])

        predicted_role = max(role_probabilities, key=role_probabilities.get)
        return predicted_role

    def present_features(self, top_n: int = 1) -> None:
        """Prints `top_n` feature(s) used by this classifier in the descending order of informativeness of the
        feature in determining a class for any object. Informativeness of a feature is a quantity that represents
        how "good" a feature is in determining the class for an object.

        :param top_n: how many of the top features to print; must be 1 or greater
        """
        # TODO: present the features that were most helpful in determining the political party of the sentence
        sorted_features = sorted(self.dict.items(), key=lambda item: abs(item[1][0] - item[1][1]), reverse=True)

        # Print top_n features
        print(f"Top {top_n} features:")
        for feature, (rep_prob, dem_prob) in sorted_features[:top_n]:
            if rep_prob > dem_prob:
                rep = rep_prob / dem_prob
                print(f"{feature} Republican:Democrat, {rep}:1")
            elif dem_prob > rep_prob:
                dem = dem_prob / rep_prob
                print(f"{feature} Democrat:Republican, {dem}:1")

    @classmethod
    def train(cls, training_set: Iterable[FeatureSet]) -> AbstractClassifier:
        """Method that builds a Classifier instance with its training (supervised learning) already completed. That is,
        the `AbstractClassifier` instance returned as the result of invoking this method must support `gamma` and
        present_features` method calls immediately without needing any other method invocations prior to them.

        :param training_set: An iterable collection of `FeatureSet` to use for training the classifier
        :return: an instance of `AbstractClassifier` with its training already completed
        """
        # TODO: Implement it such that it takes in a feature set of sentences of either political party to train
        classifier = {}

        role_tallies = {role: 0 for role in ['duelist', 'sentinel', 'initiator', 'controller']}

        for fset in training_set:
            role_tallies[fset.clas] += 1

        for feature_set in training_set:
            for feature in feature_set.feat:
                if classifier.get(feature.name, 0) == 0:
                    classifier[feature.name] = {role: 0 for role in ['duelist', 'sentinel', 'initiator', 'controller']}
                    classifier[feature.name][feature_set.clas] += 1

        for feature in classifier.keys():
            for role in ['duelist', 'sentinel', 'initiator', 'controller']:
                classifier[feature][role] = (classifier[feature][role] + 1) / (role_tallies[role])
        return ValorantAbstractClassifier(classifier)
