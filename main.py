# Ravindra Bisram
# ECE 467 - Natural Language Processing
# Project 1: Text categorization

import nltk
# nltk.download('punkt')

from naive_bayes import NaiveBayesClassifier

print('For this program, each row of the training file should consist of the relative path and filename of one training document followed by a single space and then the category of the document. Regarding the test file, each row should consist of a single string representing the relative path and filename of one test document (document that needs to be assigned a label by the model).')

# Obtain file paths
train_labels_filename = input('Enter training file: ')
test_filename = input('Enter testing file: ')
output_filename = input('Enter output file: ')

# Relative path names to find the corpus articles
path_to_train_labels_file = '/'.join(train_labels_filename.split('/')[0:-1])
path_to_test_file = '/'.join(test_filename.split('/')[0:-1])

# Identify the unique classes for any generalized dataset by analyzing the training set and inserting each class to a python set (guaranteed unique)
# Document - classification pairs stored in dictionary
document_classification_map = {}
class_set = set()

# Parse training file into set and dictionary
with open(train_labels_filename) as training_file_handle:
    for line in training_file_handle:
        file, class_name = line.rstrip('\n').split(' ')
        class_set.add(class_name)
        # Insert pair into dictionary
        document_classification_map[file] = class_name

# Use the training documents to build a vocabulary (in this instance not time intensive)
vocab_set = set()

for document_filename in document_classification_map.keys():
    # https://www.nltk.org/api/nltk.tokenize.html
    # This tokenization method return a tokenized copy of text, using NLTK’s recommended word tokenizer (currently an improved TreebankWordTokenizer along with PunktSentenceTokenizer for the specified language). Proven to work well for European languages.
    with open(f'{path_to_train_labels_file}/{document_filename}', 'r') as document_handle:
        vocab_set.update(nltk.tokenize.word_tokenize(''.join(document_handle.readlines())))


# Instance of Naive Bayes Classifier created locally in naive_bayes.py
naive_bayes_classifier = NaiveBayesClassifier(vocab_set, class_set)

# main training loop: loop through corpora, tokenize and get necessary counts
for document_filename, document_class in document_classification_map.items():
    with open(f'{path_to_train_labels_file}/{document_filename}', 'r') as document_handle:
        tokens = nltk.tokenize.word_tokenize(''.join(document_handle.readlines()))
    naive_bayes_classifier.train_document(tokens, document_class)

# convert counts to log probabilities to prepare for inference
naive_bayes_classifier.build_model()

# Read in the test file paths
with open(test_filename, 'r') as test_file_handle:
    test_filenames = test_file_handle.read().splitlines()

predictions = []

# Execute model and populate predictions list
for i, document_filename in enumerate(test_filenames):
    with open(f'{path_to_test_file}/{document_filename}', 'r') as document_handle:
        tokens = nltk.tokenize.word_tokenize(''.join(document_handle.readlines()))
    predictions.append((document_filename, naive_bayes_classifier.predict(tokens)))

# print(predictions)

# Print predictions to output file
with open(output_filename, 'w+') as out_file_handle:
    for filename, likely_class in predictions:
        out_file_handle.write(f'{filename} {likely_class}\n')
