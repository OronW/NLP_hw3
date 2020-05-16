import os
import random
import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, chi2, f_classif

from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

from sklearn import metrics
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer

from sklearn.model_selection import KFold, RepeatedKFold, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, FeatureUnion

import gc

userDir = r'C:\Users\oron.werner\PycharmProjects\NLP\hw3Input\byUser'
countryDir = r'C:\Users\oron.werner\PycharmProjects\NLP\hw2Input'
countryOut = r'C:\Users\oron.werner\PycharmProjects\NLP\hw3Input\byCountry'
countryEqualizedInput = r'C:\Users\oron.werner\PycharmProjects\NLP\hw3Input\byCountry\equalized'
summaryOutputPath = r'C:\Users\oron.werner\PycharmProjects\NLP\hw3Output\summary\summary.txt'
bestWords1 = r'C:\Users\oron.werner\PycharmProjects\NLP\hw3Output\summary\words1.txt'
bestWords2 = r'C:\Users\oron.werner\PycharmProjects\NLP\hw3Output\summary\words2.txt'

summaryToFile = []
bestWords = []

def main():


    # \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/
    print('Phase1 (Bag of Words):')
    print('Author Identification:')
    summaryToFile.append('Phase1 (Bag of Words):')
    summaryToFile.append('Author Identification:')
    # BOW for 2 users files
    # totalCorpus = readAndLabel(userDir)
    # createFeatureVectors(totalCorpus, 'NB')
    # createFeatureVectors(totalCorpus, 'LR')


    # prep for country files
    # createShuffledFiles(countryDir)    # only needed if no shuffled files exists
    # equalizeLength(countryOut)  # only needed if files do not have same amount of sentences
    # combineSentences(countryEqualizedInput)     # combines every 20 sentences into one

    # BOW for country files
    print('Native Language Identification:')
    summaryToFile.append('Native Language Identification:')

    # totalCorpus = readAndLabel(countryEqualizedInput)
    # createFeatureVectors(totalCorpus, 'NB')
    # createFeatureVectors(totalCorpus, 'LR')
    print('-------------------------------------------------------------------------------------------------------------------')
    summaryToFile.append('-------------------------------------------------------------------------------------------------------------------')

    # /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\


    # \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/
    print('Phase2 (My features):')
    print('Author Identification:')
    summaryToFile.append('Phase2 (My features):')
    summaryToFile.append('Author Identification:')


    # Manual vector for 2 users files
    # totalCorpus = readAndLabel(userDir)
    # createFeatureVectors(totalCorpus, 'NB', vectorType='manual')
    # createFeatureVectors(totalCorpus, 'LR', vectorType='manual')

    # Manual vector for country files
    print('Native Language Identification:')
    summaryToFile.append('Native Language Identification:')

    # totalCorpus = readAndLabel(countryEqualizedInput)
    # createFeatureVectors(totalCorpus, 'NB', vectorType='manual')
    # createFeatureVectors(totalCorpus, 'LR', vectorType='manual')
    print('-------------------------------------------------------------------------------------------------------------------')
    summaryToFile.append('-------------------------------------------------------------------------------------------------------------------')

    # /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\


    # \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/
    print('Phase3 (Best features):')
    print('Author Identification:')
    summaryToFile.append('Phase3 (Best features):')
    summaryToFile.append('Author Identification:')

    totalCorpus = readAndLabel(userDir)
    featureList = getKbest(totalCorpus)
    # featureMatrix = np.array(featureList)
    # print(featureMatrix.reshape((20, 5)))

    createWordsFile(featureList, bestWords1)


    # createFeatureVectors(totalCorpus, 'NB', featureList, vectorType='kbest')
    # createFeatureVectors(totalCorpus, 'LR', featureList, vectorType='kbest')

    print('Native Language Identification:')
    summaryToFile.append('Native Language Identification:')

    totalCorpus = readAndLabel(countryEqualizedInput)
    featureList = getKbest(totalCorpus)
    # featureMatrix = np.array(featureList)
    # print(featureMatrix.reshape((20, 5)))

    createWordsFile(featureList, bestWords2)

    # createFeatureVectors(totalCorpus, 'NB', featureList, vectorType='kbest')
    # createFeatureVectors(totalCorpus, 'LR', featureList, vectorType='kbest')
    # /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\

    createSummaryFile()


def createWordsFile(featureList, path):

    f = open(path, 'w+', encoding='utf-8')

    for feature in featureList:
        f.write(feature + '\n')

def createSummaryFile():

    f = open(summaryOutputPath, 'w+', encoding='utf-8')

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

    # print(cv.shape)
    tr = transformer.fit_transform(cv)
    # print(tr.shape)

    ch2 = SelectKBest(k=100)

    count_new = ch2.fit_transform(tr, y)
    # print(count_new.shape)

    features = []
    featureList = []

    indices = ch2.get_support(indices="true")
    for ind in indices:
        features.append(totalFeatures[ind])
    # print(features)
    for feature in features:
        # print(str(feature))
        featureList.append(feature)




    # return best_k_words
    # print(featureList)

    return featureList

    # print(count_new.shape)
    # print(ch2.get_support(indices=True))

    # vector_names = list(count_new.columns[ch2.get_support(indices=True)])
    # print(vector_names)

    # for i in range(len(X)):
    #     if i == 1498:
    #         print(X[i])

    # dic = np.asarray(count_new.get_feature_names())[ch2.get_support()]
    # count_vectorizer = CountVectorizer(strip_accents='unicode', ngram_range=(1, 1), binary=True, vocabulary=dict)

    # print(count_vectorizer)

    # # selector = SelectKBest(chi2, k=100)
    # # kbest = selector.fit(tr, y)
    #
    # selector = SelectKBest(k=100)
    # sel = selector.fit_transform(tr, y)
    # # print(sel)
    # # print(sel.get_params())



def manualFeatureVector():
    manualVector = []



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
                        print('HERE')
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

    # print(minLength)
    # print(neededLength)

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

            f = open(countryOut + '\\' + 'Shuffled' + currentFile, 'w+', encoding='utf-8')
            for sentence in sentences:
                f.write(sentence + '\n')


@ignore_warnings(category=ConvergenceWarning)
def createFeatureVectors(totalCorpus, classifier, featureList=None, vectorType='normal'):
    # X = numpy.array(totalCorpus)
    totalDf = pd.DataFrame.from_dict(totalCorpus)     # create a data frame for the labeled sentences
    y = totalDf['class']     # create a column for of the labels

    X = totalDf['text']
    # print('X shape:', X.shape)

    cv = CountVectorizer()
    words = cv.fit_transform(X)
    # print(len(cv.get_feature_names()))

    sum = 0

    kf = KFold(n_splits=10, random_state=1, shuffle=True)
    # print('Number of splits for run:', kf.get_n_splits(totalDf))

    for train_index, test_index in kf.split(X):
        # print("\nTRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        if vectorType == 'normal':

            vect = CountVectorizer()
            X_train_dtm = vect.fit_transform(X_train)  # create document - term matrix for the words TODO: verify this part
            # print(X_train_dtm.shape)

            transformer = TfidfTransformer()
            X_train_dtm = transformer.fit_transform(X_train_dtm)

            X_test_dtm = vect.transform(X_test)
            X_test_dtm = transformer.fit_transform(X_test_dtm)


        elif vectorType == 'manual':
            myVectorXtrain = []
            myVectorXtest = []

            for sentence in X_train:
                st = sentence.split()
                # print('sentenceLen ' + str(len(sentence)) + ' wordLen ' + str(len(st)))
                myVectorXtrain.append([len(sentence), len(st), sentence.count('!'), sentence.count('?'), sentence.count('.'), sentence.count('\''), sentence.count('I am'), sentence.count('you \' re'), sentence.count('. . .'), sentence.count('I \' m')])
                # print(myVectorXtrain)

            for sentence in X_test:
                st = sentence.split()
                # print('sentenceLen ' + str(len(sentence)) + ' wordLen ' + str(len(st)))
                myVectorXtest.append([len(sentence), len(st), sentence.count('!'), sentence.count('?'), sentence.count('.'), sentence.count('\''), sentence.count('I am'), sentence.count('you \' re'), sentence.count('. . .'), sentence.count('I \' m')])

            # print(np.array(myVectorXtrain))

            X_train_dtm = np.array(myVectorXtrain)  # create document - term matrix for the words TODO: verify this part
            X_test_dtm = np.array(myVectorXtest)

            # print('X train dtm ', X_train_dtm.shape)
            # print('X test dtm ', X_test_dtm.shape)

        elif vectorType == 'kbest':

            myVectorXtrain = []
            myVectorXtest = []

            sentenceFeatures = []

            for sentence in X_train:
                # words = sentence.split()
                # print(sentence)
                for feature in featureList:
                    sentenceFeatures.append(sentence.count(feature))
                # print(sentenceFeatures)
                # print('sentenceLen ' + str(len(sentence)) + ' wordLen ' + str(len(st)))

                myVectorXtrain.append(sentenceFeatures)
                sentenceFeatures = []
                # print(myVectorXtrain)
            # for lists in myVectorXtrain:
            #     print(lists)

            for sentence in X_test:
                st = sentence.split()
                # print('sentenceLen ' + str(len(sentence)) + ' wordLen ' + str(len(st)))
                for feature in featureList:
                    sentenceFeatures.append(sentence.count(feature))

                myVectorXtest.append(sentenceFeatures)
                sentenceFeatures = []
            # print(np.array(myVectorXtrain))

            X_train_dtm = np.array(myVectorXtrain)  # create document - term matrix for the words TODO: verify this part
            X_test_dtm = np.array(myVectorXtest)



        # -- uncomment this part for NB classifier --
        if classifier == 'NB':
            nb = MultinomialNB()
            nb.fit(X_train_dtm, y_train)
            y_pred_class = nb.predict(X_test_dtm)
            sum += metrics.accuracy_score(y_test, y_pred_class)


        # -- uncomment this part for LR classifier --
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

    # print(totalCorpus[47356:47360])
    return totalCorpus


def CFV(fileCorpus):

    vectorizer = CountVectorizer()
    vectorizer.fit(fileCorpus)

    # print(len(vectorizer.vocabulary_))

    # create the transform
    vectorizer = TfidfVectorizer()
    # tokenize and build vocab
    vectorizer.fit(fileCorpus)
    # summarize
    # print(vectorizer.vocabulary_)
    # print(vectorizer.idf_)
    # encode document
    vector = vectorizer.transform([fileCorpus[0]])
    # summarize encoded vector
    # print(vector.shape)
    # print(vector.toarray())

    return vectorizer

# def fixNumRows(currentFile, sentences, numRows):
#
#     # this run at main for the initial creation of 2 users from Argentina files
#     for currentFile in os.listdir(directory):
#         if currentFile.startswith("Wild"):
#             path = directory + '\\' + currentFile
#             print()
#             print('Reading the file: ')
#             print(path)
#
#             sentences = []
#
#             f = open(path, 'r', encoding='utf-8')
#             sentences = f.read().splitlines()
#
#             g = open(directory + "\\" + str(currentFile)[:-4] + '_fixed' + '.txt', 'w+',
#                      encoding='utf-8')  # create a file with name of "file" .txt.  w+ is write privileges
#             for sentence in sentences[:47357]:
#                 g.write(sentence.lstrip() + '\n')
#
#             # fixNumRows(currentFile, sentences, 47358)
#
#         print('*********************************')


main()
