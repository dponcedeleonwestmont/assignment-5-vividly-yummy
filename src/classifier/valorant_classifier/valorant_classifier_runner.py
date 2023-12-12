from valorant_classifier_models import *
from nltk.corpus import stopwords
import pandas as pd
import random

__author__ = "Eli Tiao, David Ponce De Leon"
__copyright__ = "Copyright 2023, Westmont College"
__credits__ = ["Eli Tiao", "David Ponce De Leon"]
__email__ = "jtiao@westmont.edu, dponcedeleon@westmont.edu"

data = pd.read_csv('player_stats.csv')
features = data[['player', 'map', 'kill', 'death', 'assist', 'adr', 'fk', 'fd']]

# Training data

training_data = data.sample(n=300, random_state=42)

test_data = data.drop(training_data.index).sample(n=300, random_state=42)

# Define stop words
stop_words = set(stopwords.words('english'))

# Create a list of training feature sets
training_feature_sets = []

for _, row in training_data.iterrows():
    known_clas = row['role']

    training_feature_set = ValorantFeatureSet.build(row, known_clas=known_clas)

    training_feature_sets.append(training_feature_set)
# Train the classifier
classifier = ValorantAbstractClassifier.train(training_feature_sets)


# top = classifier.present_features(50)
# print(top)
i = 0
for index, row in test_data.iterrows():
    test_feature_set = ValorantFeatureSet.build(row)
    predicted_role = classifier.gamma(test_feature_set)
    print(f"The predicted role for the player in {index} is: {predicted_role}")
    i += 1
