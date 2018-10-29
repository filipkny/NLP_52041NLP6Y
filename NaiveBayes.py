from collections import defaultdict
import numpy as np

class NaiveBayesClassifier(object):
    def __init__(self):
        self.prior = defaultdict(int)
        self.logprior = {}
        self.bigdoc = defaultdict(list)
        self.loglikelihoods = defaultdict(defaultdict)
        self.V = []

    def compute_prior_and_bigdoc(self, training_set, training_labels):
        '''
        Computes the prior and the bigdoc (from the book's algorithm)
        :param training_set:
            a list of all documents of the training set
        :param training_labels:
            a list of labels corresponding to the documents in the training set
        :return:
            None
        '''
        for set,label in zip(training_set,training_labels):
            self.prior[label] += len(set.split(" "))
            self.bigdoc[label].append(set)

    def compute_vocabulary(self,documents):
        '''
        Computes the vocabulary of a certain document
        :param documents:
            list of strings corresponding to the documents
        :return:
            returns the vocabulary corresponding to the given documents
        '''
        vocabulary = set()
        for doc in documents:
            for word in doc.split(" "):
                vocabulary.add(word)

        return vocabulary

    def count_word_in_classes(self, classes, word = None ):
        '''
        Counts a certain word within the documents of a certain class
        If no word is provided counts all words within that class.
        :param classes:
            a list of the classes in which the word will be counted
        :param word:
            the word to be counted
        :return:
            the count
        '''
        count = 0
        for c in classes:
            docs = self.bigdoc[c]
            for doc in docs:
                if word:
                    count += doc.count(word)
                else:
                    count += len(doc.split(" "))

        return count

    def train(self, training_set, training_labels, alfa = 1):
        '''
        Trains the classifier according to the Naive Bayes training algorithm given in the book
        :param training_set:
            a list of all documents of the training set
        :param training_labels:
            a list of labels corresponding to the documents in the training set:param alfa:
        :alfa:
            the smoothing factor
        :return:
            None
        '''
        # Get number of documents
        N_doc = float(len(training_set))

        # Get vocabulary used in training set
        self.V = self.compute_vocabulary(training_set)

        # For each class
        for c in self.prior.keys():

            # Get number of documents for that class
            N_c = float(sum(training_labels == c))

            # Compute logprior for class
            self.logprior[c] = np.log(N_c/N_doc)

            # Calculate the sum of counts of words in current class
            total_count = 0
            for word in self.V:
                total_count += self.count_word_in_classes([c],word=word)

            # For every word, get the count and compute the loglikelihood for this class
            for word in self.V:
                count = self.count_word_in_classes([c], word=word)
                self.loglikelihoods[c][word] = np.log((count + alfa)/(total_count + len(self.V)))



    def predict(self,testdoc):
        """
        Predicts the sentiment for a certain document
        :param testdoc:
            the document (a string)
        :return:
            a dictionary with the probabilites of belonging to either class
        """
        sums = {}
        for c in self.bigdoc.keys():
            sums[c] = self.logprior[c]
            for word in testdoc.split(" "):
                if word in self.V:
                    sums[c] += self.loglikelihoods[c][word]

        return sums



doc1 = "just plain boring"                      # -
doc2 = "entirely predictable and lacks energy" # -
doc3 = "no surprises and very few laughs"       # -
doc4 = "very powerful"                          # +
doc5 = "the most fun film of the summer"        # +

training_set = [doc1, doc2, doc3, doc4, doc5]
training_labels = np.array([0, 0, 0, 1 ,1])

doc6 = "predictable with no fun" # ?

NBclassifier = NaiveBayesClassifier()
NBclassifier.compute_prior_and_bigdoc(training_set, training_labels)
NBclassifier.train(training_set,training_labels)

result = NBclassifier.predict(doc6)
print(result)

