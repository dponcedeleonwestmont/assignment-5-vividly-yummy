# Valorant Role Prediction Classifier

### Demo Code: 

### Overview

This Python program utilizes match history from the 2023 VCT Season (via Kaggle CSV) 
to predict the in-game role (duelist, sentinel, initiator, controller) of Valorant players 
based on their performance statistics as well as other factors such as player and map. 

# Project Structure

The project is organized following a typical structure for a Python application. Below is an overview of the key directories and files:

## Directory Structure

```plaintext
valorant-role-prediction/
├── src/                         # Source code files
│   ├── __init__.py              # Initialization file for the source code package
│   ├── classifier_models.py     # Definitions of the superclass classifier models
│   └── valorant_classifier/     # Package for Valorant-specific classifier implementation
│       ├── __init__.py          # Initialization file for the Valorant classifier package
│       ├── player_stats.csv     # Dataset for both training and testing [half allocated towards each]
│       ├── valorant_classifier_models.py  # Valorant-specific classifier models
│       └── valorant_classifier_runner.py  # Valorant-specific classifier runner
└── README.md                    # Project documentation
```

### Program Details

* The program uses statistical analysis processing techniques to predict the in-game role of Valorant players.
* It leverages a custom-built Valorant FeatureSet and AbstractClassifier for role prediction.

## How the Program Works

### ValorantFeature and ValorantFeatureSet

- **`ValorantFeature`**: Represents a feature in the Valorant context. It extends the base `Feature` class.

- **`ValorantFeatureSet`**: Represents a set of features for a single object in the Valorant context. It extends the base `FeatureSet` class.

### ValorantFeatureSet.build

- The `build` method is a class method that constructs a `ValorantFeatureSet` from a source object (in this case, a row from a CSV file).

- It extracts relevant features such as player name, map, kills, first bloods, and assists, and creates a set of `ValorantFeature` instances.

### ValorantAbstractClassifier

- Represents an abstract classifier for Valorant roles.

#### `gamma` Method

- Given a single feature set representing an object, it returns the most probable class (role) for the object based on the training the classifier received.

#### `present_features` Method

- Prints the top features used by the classifier in the descending order of percentage for determining a class.

#### `train` Class Method

- Builds a Valorant classifier instance with its training (supervised learning) already completed.

- The `train` method takes an iterable collection of `ValorantFeatureSet` instances as its parameter.

### Valorant Player Stats
The "VCT All Tournament 2023 Stats" dataset by Ediashta Revin was selected, containing performance statistics of Valorant players. The goal is to predict the in-game role of players (duelist, sentinel, initiator, controller) based on their individual and team statistics, contributing to strategic insights for Valorant teams.