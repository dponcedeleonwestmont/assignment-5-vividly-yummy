from io import StringIO
from unittest.mock import patch

import pytest
import unittest
from classifier.political_party_classifier.political_party_classifier_models import *
from classifier.classifier_models import *
from nltk.corpus import stopwords

__author__ = "David Ponce De Leon, Eli Tiao"
__credits__ = ["David Ponce De Leon, Eli Tiao"]
__email__ = "dponcedeleon@westmont.edu, jtiao@westmont.edu"


import pytest
import unittest
from classifier.political_party_classifier.political_party_classifier_models import OurAbstractClassifier, OurFeatureSet, OurFeature
from classifier.classifier_models import Feature


class Test(unittest.TestCase):
    def test_build(self):
        source_object = "Some test data"
        known_class = "Positive"
        stop_words = set(stopwords.words('english'))
        feature_set = OurFeatureSet.build(source_object, known_class, stop_words=stop_words)
        self.assertIsInstance(feature_set, OurFeatureSet)
        self.assertEqual(feature_set.clas, known_class)

        self.assertIsNotNone(feature_set.feat)
        self.assertGreater(len(feature_set.feat), 0)
        self.assertIn("test", [feature.name for feature in feature_set.feat])
        self.assertIn("data", [feature.name for feature in feature_set.feat])

    def test_gamma(self):
        training_data = [
            OurFeatureSet({OurFeature("feature1", True), OurFeature("feature2", True)}, known_clas="Republican"),
            OurFeatureSet({OurFeature("feature1", True), OurFeature("feature2", True)}, known_clas="Democrat"),
        ]
        classifier = OurAbstractClassifier.train(training_data)
        test_feature_set_1 = OurFeatureSet({OurFeature("feature1", True), OurFeature("feature2", True)})

        predicted_class_1 = classifier.gamma(test_feature_set_1)
        self.assertEqual(predicted_class_1, "Democrat")

        test_feature_set_2 = OurFeatureSet({OurFeature("feature1", True), OurFeature("feature2", True)})
        predicted_class_2 = classifier.gamma(test_feature_set_2)
        self.assertEqual(predicted_class_2, "Democrat")

    def test_present_features(self):
        classifier_data = {
            "word1": [0.2, 0.8],
            "word2": [0.7, 0.3],
            "word3": [0.4, 0.6],
        }

        classifier = OurAbstractClassifier(classifier_data)

        with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
            classifier.present_features(top_n=2)

            printed_output = mock_stdout.getvalue()

        output_lines = printed_output.strip().split("\n")

        self.assertGreaterEqual(len(output_lines), 2)
        self.assertIn("Top 2 features:", output_lines[0])
        self.assertIn("word1 Democrat:Republican, 4.0:1", output_lines[1])
        self.assertIn("word2 Republican:Democrat, 2.3333333333333335:1", output_lines[2])

    def test_train(self):
        training_data = [
            OurFeatureSet({OurFeature("feature1", True), OurFeature("feature2", True)}, known_clas="Republican"),
            OurFeatureSet({OurFeature("feature3", True), OurFeature("feature4", True)}, known_clas="Democrat"),
        ]

        classifier = OurAbstractClassifier.train(training_data)

        self.assertIsInstance(classifier, OurAbstractClassifier)
        self.assertIsNotNone(classifier)

        for feature, (rep_prob, dem_prob) in classifier.dict.items():
            self.assertGreaterEqual(rep_prob, 0)
            self.assertLessEqual(rep_prob, 2)
            self.assertGreaterEqual(dem_prob, 0)
            self.assertLessEqual(dem_prob, 2)

        for feature, (rep_prob, dem_prob) in classifier.dict.items():
            self.assertAlmostEqual(rep_prob + dem_prob, 3, delta=0.0001)


if __name__ == '__main__':
    unittest.main()