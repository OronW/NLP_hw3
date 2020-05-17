import os
import random
import sys

import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest

from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn.model_selection import KFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression

summaryToFile = []
bestWords = []


def main(userDir=sys.argv[1], countryEqualizedInput=sys.argv[2], summaryOutputPath=sys.argv[3], bestWords1=sys.argv[4], bestWords2=sys.argv[5]):

    # prep for country files
    # createShuffledFiles(countryDir)    # only needed if no shuffled files exists
    # equalizeLength(countryOut)  # only needed if files do not have same amount of sentences
    # combineSentences(countryEqualizedInput)     # combines every 20 sentences into one

    # \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/
    print('Phase1 (Bag of Words):')
    print('Author Identification:')
    summaryToFile.append('Phase1 (Bag of Words):')
    summaryToFile.append('Author Identification:')

    # BOW for 2 users files
    totalCorpus = readAndLabel(userDir)
    createFeatureVectors(totalCorpus, 'NB')
    createFeatureVectors(totalCorpus, 'LR')

    print('Native Language Identification:')
    summaryToFile.append('Native Language Identification:')

    # BOW for country files
    totalCorpus = readAndLabel(countryEqualizedInput)
    createFeatureVectors(totalCorpus, 'NB')
    createFeatureVectors(totalCorpus, 'LR')

    print('-------------------------------------------------------------------------------------------------------------------')
    summaryToFile.append('-------------------------------------------------------------------------------------------------------------------')
    # /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\


    # \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/
    print('Phase2 (My features):')
    print('Author Identification:')
    summaryToFile.append('Phase2 (My features):')
    summaryToFile.append('Author Identification:')

    # Manual vector for 2 users files
    totalCorpus = readAndLabel(userDir)
    createFeatureVectors(totalCorpus, 'NB', vectorType='manual')
    createFeatureVectors(totalCorpus, 'LR', vectorType='manual')

    print('Native Language Identification:')
    summaryToFile.append('Native Language Identification:')

    # Manual vector for country files
    totalCorpus = readAndLabel(countryEqualizedInput)
    createFeatureVectors(totalCorpus, 'NB', vectorType='manual')
    createFeatureVectors(totalCorpus, 'LR', vectorType='manual')

    print('-------------------------------------------------------------------------------------------------------------------')
    summaryToFile.append('-------------------------------------------------------------------------------------------------------------------')
    # /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\


    # \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/
    print('Phase3 (Best features):')
    print('Author Identification:')
    summaryToFile.append('Phase3 (Best features):')
    summaryToFile.append('Author Identification:')

    # Get K best for 2 users files
    totalCorpus = readAndLabel(userDir)
    featureList = getKbest(totalCorpus)
    # featureMatrix = np.array(featureList)
    # print(featureMatrix.reshape((20, 5)))

    createWordsFile(featureList, bestWords1)
    createFeatureVectors(totalCorpus, 'NB', featureList, vectorType='kbest')
    createFeatureVectors(totalCorpus, 'LR', featureList, vectorType='kbest')

    print('Native Language Identification:')
    summaryToFile.append('Native Language Identification:')

    # Get K best for country files
    totalCorpus = readAndLabel(countryEqualizedInput)
    featureList = getKbest(totalCorpus)
    # featureMatrix = np.array(featureList)
    # print(featureMatrix.reshape((20, 5)))

    createWordsFile(featureList, bestWords2)
    createFeatureVectors(totalCorpus, 'NB', featureList, vectorType='kbest')
    createFeatureVectors(totalCorpus, 'LR', featureList, vectorType='kbest')
    # /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\

    createSummaryFile(summaryOutputPath)


def createWordsFile(featureList, path):

    f = open(path, 'w+', encoding='utf-8')

    for feature in featureList:
        f.write(feature + '\n')


def createSummaryFile(path):

    f = open(path, 'w+', encoding='utf-8')

    for line in summaryToFile:
        f.write(line + '\n')


def getKbest(totalCorpus):

    totalDf = pd.DataFrame.from_dict(totalCorpus)  # create a data frame for the labeled sentences
    y = totalDf['class']  # create a column for of the labels
    X = totalDf['text']

    vect = CountVectorizer()
    transformer = TfidfTransformer()
    cv = vect.fit_transform(X)
    totalFeatures = vect.get_feature_names()

    tr = transformer.fit_transform(cv)

    ch2 = SelectKBest(k=100)

    count_new = ch2.fit_transform(tr, y)
    # print(count_new.shape)

    features = []
    featureList = []

    indices = ch2.get_support(indices="true")
    for ind in indices:
        features.append(totalFeatures[ind])

    for feature in features:
        featureList.append(feature)

    return featureList


def combineSentences(directory):

    largeSentence = ''

    for currentFile in os.listdir(directory):
        if currentFile.endswith(".txt"):
            path = directory + '\\' + currentFile
            # print()
            # print('Reading the file: ')
            # print(path)

            f = open(path, 'r', encoding='utf-8')
            sentences = f.read().splitlines()

            newPath = directory + '\\' + 'combined' + currentFile

            f = open(newPath, 'w', encoding='utf-8')
            counter = 0

            for sentence in sentences:
                counter += 1
                largeSentence += sentence + ' '

                if counter == 20:
                    if sentence == sentences[len(sentences)-1]:
                        f.write(largeSentence)
                    else:
                        f.write(largeSentence + '\n')
                        largeSentence = ''
                        counter = 0


def equalizeLength(directory):

    lengthsOfFiles = []

    for currentFile in os.listdir(directory):
        if currentFile.endswith(".txt"):
            path = directory + '\\' + currentFile
            # print()
            # print('Reading the file: ')
            # print(path)

            f = open(path, 'r', encoding='utf-8')
            sentences = f.read().splitlines()
            lengthsOfFiles.append(len(sentences))

    minLength = min(lengthsOfFiles) + 1
    neededLength = minLength - (minLength % 20)

    for currentFile in os.listdir(directory):
        if currentFile.endswith(".txt"):
            path = directory + '\\' + currentFile
            # print()
            # print('Equalizing the file: ')
            # print(path)

            f = open(path, 'r', encoding='utf-8')
            sentences = f.read().splitlines()

            newPath = directory + '\\' + 'equalized' + currentFile

            f = open(newPath, 'w', encoding='utf-8')
            for sentence in sentences[:neededLength]:
                f.write(sentence + '\n')


def createShuffledFiles(directory):

    for currentFile in os.listdir(directory):
        if currentFile.endswith(".txt"):
            path = directory + '\\' + currentFile
            # print()
            # print('Reading the file: ')
            # print(path)

            f = open(path, 'r', encoding='utf-8')
            sentences = f.read().splitlines()

            random.shuffle(sentences)

            # f = open(countryOut + '\\' + 'Shuffled' + currentFile, 'w+', encoding='utf-8')
            for sentence in sentences:
                f.write(sentence + '\n')


@ignore_warnings(category=ConvergenceWarning)
def createFeatureVectors(totalCorpus, classifier, featureList=None, vectorType='normal'):

    totalDf = pd.DataFrame.from_dict(totalCorpus)     # create a data frame for the labeled sentences
    y = totalDf['class']     # create a column for of the labels
    X = totalDf['text']

    cv = CountVectorizer()
    words = cv.fit_transform(X)

    sum = 0

    kf = KFold(n_splits=10, random_state=1, shuffle=True)

    for train_index, test_index in kf.split(X):
        # print("\nTRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        if vectorType == 'normal':

            vect = CountVectorizer()
            X_train_dtm = vect.fit_transform(X_train)  # create document - term matrix for the words

            transformer = TfidfTransformer()
            X_train_dtm = transformer.fit_transform(X_train_dtm)

            X_test_dtm = vect.transform(X_test)
            X_test_dtm = transformer.fit_transform(X_test_dtm)


        elif vectorType == 'manual':
            myVectorXtrain = []
            myVectorXtest = []

            for sentence in X_train:
                st = sentence.split()
                myVectorXtrain.append([len(sentence), len(st), sentence.count('!'), sentence.count('?'), sentence.count('.'), sentence.count('\''), sentence.count('I am'), sentence.count('you \' re'), sentence.count('. . .'), sentence.count('I \' m')])

            for sentence in X_test:
                st = sentence.split()
                myVectorXtest.append([len(sentence), len(st), sentence.count('!'), sentence.count('?'), sentence.count('.'), sentence.count('\''), sentence.count('I am'), sentence.count('you \' re'), sentence.count('. . .'), sentence.count('I \' m')])


            X_train_dtm = np.array(myVectorXtrain)  # create document - term matrix for the words
            X_test_dtm = np.array(myVectorXtest)


        elif vectorType == 'kbest':

            myVectorXtrain = []
            myVectorXtest = []

            sentenceFeatures = []

            for sentence in X_train:
                for feature in featureList:
                    sentenceFeatures.append(sentence.count(feature))

                myVectorXtrain.append(sentenceFeatures)
                sentenceFeatures = []

            for sentence in X_test:
                for feature in featureList:
                    sentenceFeatures.append(sentence.count(feature))

                myVectorXtest.append(sentenceFeatures)
                sentenceFeatures = []

            X_train_dtm = np.array(myVectorXtrain)  # create document - term matrix for the words
            X_test_dtm = np.array(myVectorXtest)


        # -- this part for NB classifier --
        if classifier == 'NB':
            nb = MultinomialNB()
            nb.fit(X_train_dtm, y_train)
            y_pred_class = nb.predict(X_test_dtm)
            sum += metrics.accuracy_score(y_test, y_pred_class)


        # -- this part for LR classifier --
        elif classifier == 'LR':
            lr = LogisticRegression(max_iter=500)
            lr.fit(X_train_dtm, y_train)
            y_pred_class = lr.predict(X_test_dtm)
            sum += metrics.accuracy_score(y_test, y_pred_class)

        else:
            print('NO CLASSIFIER SELECTED FOR \'createFeatureVectors\' FUNCTION. ENDING RUN! ')
            exit()


        # print('\nAccuracy score: ', metrics.accuracy_score(y_test, y_pred_class))
        # print('Test sentences by classes:')
        # print(y_test.value_counts())
        # print('Confusion matrix:\n', metrics.confusion_matrix(y_test, y_pred_class))

    # print('\n**********************************')

    acc = sum/10
    total = int(sum*1000)/100

    if classifier == 'NB':
        print('Naïve Bayes:', total)
        summary = str('Naïve Bayes: ' + str(total))
        summaryToFile.append(summary)

    if classifier == 'LR':
        print('Logistic Regression: ', total)
        summary = str('Logistic Regression: ' + str(total))
        summaryToFile.append(summary)


def readAndLabel(directory):

    label = 0
    totalCorpus = []
    fileCorpus = []

    # TODO: remove list limitations
    for currentFile in os.listdir(directory)[:]:
        if currentFile.endswith(".txt"):
            path = directory + '\\' + currentFile
            # print()
            # print('Reading the file: ')
            # print(path)

            f = open(path, 'r', encoding='utf-8')
            sentences = f.read().splitlines()

            for sentence in sentences:
                totalCorpus.append({'text': sentence, 'class': label})

        label += 1

    return totalCorpus


main()
