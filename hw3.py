import os
import random
import pandas as pd
import numpy as np

from sklearn import metrics
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer

from sklearn.model_selection import KFold, RepeatedKFold, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, FeatureUnion

userDir = r'C:\Users\oron.werner\PycharmProjects\NLP\hw3Input\byUser'
countryDir = r'C:\Users\oron.werner\PycharmProjects\NLP\hw2Input'
countryOut = r'C:\Users\oron.werner\PycharmProjects\NLP\hw3Input\byCountry'
countryEqualizedInput = r'C:\Users\oron.werner\PycharmProjects\NLP\hw3Input\byCountry\equalized'

def main():


    print('Phase1 (Bag of Words):')
    print('Author Identification:')
    # \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/
    # BOW for 2 users from Argentina

    # totalCorpus = readAndLabel(userDir)
    # createFeatureVectors(totalCorpus, 'NB')
    # createFeatureVectors(totalCorpus, 'LR')
    # /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\


    # \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/
    # BOW for country files

    # createShuffledFiles(countryDir)    # only needed if no shuffled files exists
    # equalizeLength(countryOut)  # only needed if files do not have same amount of sentences
    # combineSentences(countryEqualizedInput)     # combines every 20 sentences into one

    print('Native Language Identification:')
    # totalCorpus = readAndLabel(countryEqualizedInput)
    # createFeatureVectors(totalCorpus, 'NB')
    # createFeatureVectors(totalCorpus, 'LR')
    # /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\

    totalCorpus = readAndLabel(userDir)
    createFeatureVectors(totalCorpus, 'NB', vectorType='manual')

    # manualVector = manualFeatureVector()


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



def createFeatureVectors(totalCorpus, classifier, vectorType='normal'):
    # X = numpy.array(totalCorpus)
    totalDf = pd.DataFrame.from_dict(totalCorpus)     # create a data frame for the labeled sentences
    y = totalDf['class']     # create a column for of the labels

    # print(totalDf)
    # print(y)
    # print('totalDf shape:', totalDf.shape)
    # print('y shape:', y.shape)

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
            print(X_train_dtm.shape)

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
                myVectorXtrain.append([len(sentence), len(st)])

            for sentence in X_test:
                st = sentence.split()
                # print('sentenceLen ' + str(len(sentence)) + ' wordLen ' + str(len(st)))
                myVectorXtest.append([len(sentence), len(st)])

            print(np.array(myVectorXtrain))

            X_train_dtm = np.array(myVectorXtrain)  # create document - term matrix for the words TODO: verify this part
            X_test_dtm = np.array(myVectorXtest)

            print('X train dtm ', X_train_dtm.shape)
            print('X test dtm ', X_test_dtm.shape)




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


        print('\nAccuracy score: ', metrics.accuracy_score(y_test, y_pred_class))
        # print('Test sentences by classes:')
        # print(y_test.value_counts())
        # print('Confusion matrix:\n', metrics.confusion_matrix(y_test, y_pred_class))

    # print('\n**********************************')

    acc = sum/10
    total = int(sum*1000)/100

    if classifier == 'NB':
        print('Naïve Bayes: ', total)
    if classifier == 'LR':
        print('Logistic Regression: ', total)



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
