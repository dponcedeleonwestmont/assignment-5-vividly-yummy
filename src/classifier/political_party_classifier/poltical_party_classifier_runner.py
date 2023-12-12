from political_party_classifier_models import *
from nltk.corpus import stopwords

__author__ = "Eli Tiao, David Ponce De Leon"
__copyright__ = "Copyright 2023, Westmont College"
__credits__ = ["Eli Tiao", "David Ponce De Leon"]
__email__ = "jtiao@westmont.edu, dponcedeleon@westmont.edu"

speeches = {file_id: inaugural.raw(file_id) for file_id in inaugural.fileids()}
# Training data
data = [
    (speeches['1853-Pierce.txt'], 'Democratic'),        # 0
    (speeches['1857-Buchanan.txt'], 'Democratic'),      # 1
    (speeches['1861-Lincoln.txt'], 'Republican'),       # 2
    (speeches['1865-Lincoln.txt'], 'Republican'),       # 3
    (speeches['1869-Grant.txt'], 'Republican'),         # 4
    (speeches['1873-Grant.txt'], 'Republican'),         # 5
    (speeches['1877-Hayes.txt'], 'Republican'),         # 6
    (speeches['1881-Garfield.txt'], 'Republican'),      # 7
    (speeches['1885-Cleveland.txt'], 'Democratic'),     # 8
    (speeches['1889-Harrison.txt'], 'Republican'),      # 9
    (speeches['1893-Cleveland.txt'], 'Democratic'),     # 10
    (speeches['1897-McKinley.txt'], 'Republican'),      # 11
    (speeches['1901-McKinley.txt'], 'Republican'),      # 12
    (speeches['1905-Roosevelt.txt'], 'Republican'),     # 13
    (speeches['1909-Taft.txt'], 'Republican'),          # 14
    (speeches['1913-Wilson.txt'], 'Democratic'),        # 15
    (speeches['1917-Wilson.txt'], 'Democratic'),        # 16
    (speeches['1921-Harding.txt'], 'Republican'),       # 17
    (speeches['1925-Coolidge.txt'], 'Republican'),      # 18
    (speeches['1929-Hoover.txt'], 'Republican'),        # 19
    (speeches['1933-Roosevelt.txt'], 'Democratic'),     # 20
    (speeches['1937-Roosevelt.txt'], 'Democratic'),     # 21
    (speeches['1941-Roosevelt.txt'], 'Democratic'),     # 22
    (speeches['1945-Roosevelt.txt'], 'Democratic'),     # 23
    (speeches['1949-Truman.txt'], 'Democratic'),        # 24
    (speeches['1953-Eisenhower.txt'], 'Republican'),    # 25
    (speeches['1957-Eisenhower.txt'], 'Republican'),    # 26
    (speeches['1961-Kennedy.txt'], 'Democratic'),       # 27
    (speeches['1965-Johnson.txt'], 'Democratic'),       # 28
    (speeches['1969-Nixon.txt'], 'Republican'),         # 29
    (speeches['1973-Nixon.txt'], 'Republican'),         # 30
    (speeches['1977-Carter.txt'], 'Democratic'),        # 31
    (speeches['1981-Reagan.txt'], 'Republican'),        # 32
    (speeches['1985-Reagan.txt'], 'Republican'),        # 33
    (speeches['1989-Bush.txt'], 'Republican'),          # 34
    (speeches['1993-Clinton.txt'], 'Democratic'),       # 36
    (speeches['1997-Clinton.txt'], 'Democratic'),       # 37
    (speeches['2001-Bush.txt'], 'Republican'),          # 38
    (speeches['2005-Bush.txt'], 'Republican'),          # 39
    (speeches['2009-Obama.txt'], 'Democratic'),         # 40
    (speeches['2013-Obama.txt'], 'Democratic'),         # 41
    (speeches['2017-Trump.txt'], 'Republican'),         # 42
    (speeches['2021-Biden.txt'], 'Democratic'),         # 43
]
training_data = [
    (speeches['1933-Roosevelt.txt'], 'Democrat'),
    (speeches['1929-Hoover.txt'], 'Republican'),
    (speeches['1953-Eisenhower.txt'], 'Republican'),
    (speeches['1921-Harding.txt'], 'Republican'),
    (speeches['1949-Truman.txt'], 'Democrat'),
    (speeches['1961-Kennedy.txt'], 'Democrat'),
    (speeches['1841-Harrison.txt'], 'Republican'),
    (speeches['2001-Bush.txt'], 'Republican'),
    (speeches['2021-Biden.txt'], 'Democrat')
]

test_data = [
    speeches['1877-Hayes.txt'],         # 0 Expected: 'Republican' Actual: 'Republican'
    speeches['1881-Garfield.txt'],      # 1 Expected: 'Republican' Actual: 'Republican'
    speeches['1885-Cleveland.txt'],     # 2 Expected: Actual: 'Republican'
    speeches['1889-Harrison.txt'],      # 3 Expected: 'Republican' Actual: 'Republican'
    speeches['1893-Cleveland.txt'],     # 4 Expected: Actual: 'Republican'
    speeches['1897-McKinley.txt'],      # 5 Expected: 'Republican' Actual:
    speeches['1901-McKinley.txt'],      # 6 Expected: 'Republican' Actual: 'Republican'
    speeches['1905-Roosevelt.txt'],     # 7 Expected: 'Republican' Actual: 'Republican'
    speeches['1909-Taft.txt'],          # 8 Expected: 'Republican' Actual:
    speeches['1913-Wilson.txt'],        # 9 Expected: Actual: 'Republican'
    speeches['1917-Wilson.txt'],        # 10 Expected: Actual: 'Republican'
    speeches['1921-Harding.txt'],       # 11 Expected: 'Republican' Actual: 'Republican'
    speeches['1925-Coolidge.txt'],      # 12 Expected: 'Republican' Actual:
    speeches['1929-Hoover.txt'],        # 13 Expected: 'Republican' Actual: 'Republican'
    speeches['1933-Roosevelt.txt'],     # 14 Expected: Actual:
    speeches['1937-Roosevelt.txt'],     # 15 Expected: Actual: 'Republican'
    speeches['1941-Roosevelt.txt'],     # 16 Expected: Actual: 'Republican'
    speeches['1945-Roosevelt.txt'],     # 17 Expected: Actual: 'Republican'
    speeches['1949-Truman.txt'],        # 18 Expected: Actual:
    speeches['1953-Eisenhower.txt'],    # 19 Expected: 'Republican' Actual: 'Republican'
    speeches['1957-Eisenhower.txt'],    # 20 Expected: 'Republican' Actual: 'Republican'
    speeches['1961-Kennedy.txt'],       # 21 Expected: Actual:
    speeches['1965-Johnson.txt'],       # 22 Expected: Actual: 'Republican'
    speeches['1969-Nixon.txt'],         # 23 Expected: 'Republican' Actual: 'Republican'
    speeches['1973-Nixon.txt'],         # 24 Expected: 'Republican' Actual: 'Republican'
    speeches['1977-Carter.txt'],        # 25 Expected: Actual: 'Republican'
    speeches['1981-Reagan.txt'],        # 26 Expected: 'Republican' Actual: 'Republican'
    speeches['1985-Reagan.txt'],        # 27 Expected: 'Republican' Actual:
    speeches['1989-Bush.txt'],          # 28 Expected: 'Republican' Actual: 'Republican'
    speeches['1993-Clinton.txt'],       # 29 Expected: Actual: 'Republican'
    speeches['1997-Clinton.txt'],       # 30 Expected: Actual: 'Republican'
    speeches['2001-Bush.txt'],          # 31 Expected: 'Republican' Actual: 'Republican'
    speeches['2005-Bush.txt'],          # 32 Expected: 'Republican' Actual: 'Republican'
    speeches['2009-Obama.txt'],         # 33 Expected: Actual:
    speeches['2013-Obama.txt'],         # 34 Expected: Actual: 'Republican'
    speeches['2017-Trump.txt'],         # 35 Expected: 'Republican' Actual: 'Republican'
]
# Define stop words
stop_words = set(stopwords.words('english'))

# Create a list of training feature sets
training_feature_sets = [OurFeatureSet.build(speech, known_clas=party, stop_words=stop_words) for speech, party in
                         training_data]
i = 0
for index in training_feature_sets:
    print(training_feature_sets[i].clas)
    i += 1

# Train the classifier
classifier = OurAbstractClassifier.train(training_feature_sets)



top = classifier.present_features(50)
print(top)
i = 0
for speech in test_data:
    test_feature_set = OurFeatureSet.build(speech, stop_words=stop_words)
    predicted_party = classifier.gamma(test_feature_set)
    print(f"The predicted party for the speech in {i} is: {predicted_party}")
    i += 1
