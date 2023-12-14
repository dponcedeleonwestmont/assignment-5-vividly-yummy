from valorant_classifier_models import ValorantFeatureSet, ValorantAbstractClassifier
import pandas as pd
import random

__author__ = "Eli Tiao, David Ponce De Leon"
__copyright__ = "Copyright 2023, Westmont College"
__credits__ = ["Eli Tiao", "David Ponce De Leon"]
__email__ = "jtiao@westmont.edu, dponcedeleon@westmont.edu"

data = pd.read_csv('../../player_stats.csv')

# Determines the size of the training and test sets
training_size = int(len(data) * 0.5)  # 70% for training
test_size = len(data) - training_size

# Creates a list of training feature sets
training_feature_sets = []

# Shuffles the data
shuffled_indices = random.sample(range(len(data)), len(data))

# Splits the shuffled indices into training and test sets
training_indices = shuffled_indices[:training_size]
test_indices = shuffled_indices[training_size:]

for idx in training_indices:
    row = data.iloc[idx]
    known_clas = row['role']
    training_feature_set = ValorantFeatureSet.build(row, known_clas=known_clas)
    training_feature_sets.append(training_feature_set)

# Trains the classifier
classifier = ValorantAbstractClassifier.train(training_feature_sets)

# Outputs the predicted role in for loop
i = 0
correct_predictions = 0
for idx in test_indices:
    row = data.iloc[idx]
    test_feature_set = ValorantFeatureSet.build(row)
    predicted_role = classifier.gamma(test_feature_set)
    actual_role = row['role']
    agent = row['agent']
    print(f"[{idx}] The classified role of the agent {agent} is: {predicted_role}; actual role: {actual_role}")
    if predicted_role == actual_role:
        correct_predictions += 1
    i += 1

# Prints accuracy
accuracy = round((correct_predictions / len(test_indices)) * 100, 2)
print(f"Accuracy: {accuracy}%")

# Prints present features
top = classifier.present_features(3)
print(top)
