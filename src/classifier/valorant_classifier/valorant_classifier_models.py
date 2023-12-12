import pandas as pd
import math
from classifier.classifier_models import *

__author__ = "Eli Tiao, David Ponce De Leon"
__copyright__ = "Copyright 2023, Westmont College"
__credits__ = ["Eli Tiao", "David Ponce De Leon"]
__email__ = "jtiao@westmont.edu, dponcedeleon@westmont.edu"


# Represents a feature in the Valorant context
class ValorantFeature(Feature):
    def __init__(self, name, value):
        super().__init__(name, value)


# Represents a set of features for a single object in the Valorant context
class ValorantFeatureSet(FeatureSet):
    def __init__(self, features: set[Feature], known_clas=None):
        super().__init__(features, known_clas)

    @classmethod
    def build(cls, source_object: pd.Series, known_clas=None, **kwargs) -> FeatureSet:
        # Extracts relevant features from the source object (CSV)
        player = source_object['player']
        mapname = source_object['map']
        kill = source_object['kill']
        firstblood = source_object['fk']
        assists = source_object['assist']

        # Creates a set of Valorant features based on extracted data
        features: set[ValorantFeature] = set()
        features.add(ValorantFeature('player' + player, True))
        features.add(ValorantFeature('map' + mapname, True))
        if kill > 14:
            features |= {ValorantFeature('15+ kills', True)}
        if firstblood > 2:
            features |= {ValorantFeature('3+ firstbloods', True)}
        if assists > 6:
            features |= {ValorantFeature('7+ assists', True)}

        return ValorantFeatureSet(features, known_clas)


# Represents an abstract classifier for Valorant roles
class ValorantAbstractClassifier(AbstractClassifier):
    def __init__(self, classifier: dict):
        self.dict = classifier

    def gamma(self, a_feature_set: ValorantFeatureSet) -> str:
        # Given a feature set, calculate the most probable class (role) using the trained classifier
        role_probabilities = {role: 0.0 for role in ['duelist', 'sentinel', 'initiator', 'controller']}
        for feature in a_feature_set.feat:
            if feature.name in self.dict:
                for role in role_probabilities:
                    role_probabilities[role] += math.log(self.dict[feature.name][role])

        total_probability = sum(role_probabilities.values())
        normalized_probabilities = {role: prob / total_probability for role, prob in role_probabilities.items()}
        predicted_role = max(normalized_probabilities, key=role_probabilities.get)
        return predicted_role

    def present_features(self, top_n: int = 1) -> None:
        # Prints the top N features
        sorted_features = sorted(self.dict.items(), key=lambda item: max(item[1].values()), reverse=True)
        print(f"Top {top_n} features:")
        for feature, role_probs in sorted_features[:top_n]:
            max_role = max(role_probs, key=role_probs.get)
            max_prob = role_probs[max_role]
            print(f"{feature} predicts the role: {max_role}, Probability: {max_prob:.2%}")

    @classmethod
    def train(cls, training_set: Iterable[FeatureSet]) -> AbstractClassifier:
        # Trains the classifier
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
