import os
import numpy
import pandas
from pandas import DataFrame
from pandas.core.common import random_state
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import KFold, RepeatedKFold

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

directory = r'C:\Users\oron.werner\PycharmProjects\NLP\hw3Input'


def main():

    # BOW for 2 users from Argentina
    totalCorpus = readAndLabel(directory)

    tot = numpy.array(totalCorpus)

    # x = numpy.array(i['text'] for i in totalCorpus)
    # y = numpy.array(i['class'] for i in totalCorpus)

    x = [i['text'] for i in totalCorpus]
    y = [i['class'] for i in totalCorpus]
    print('len = ' + str(len(x)) + ' | ' + str(len(y)))
    # print(x[:2])
    # print(y[:2])


    # xTrain, yTrain = x[:int(len(x)*0.9)], y[:int(len(x)*0.9)]
    # xTest, yTest = x[int(len(x)*0.9):], y[int(len(x)*0.9):]

    kf = KFold(n_splits=10, shuffle=True)  # Define the split - into 2 folds
    # kf.get_n_splits(tot)  # returns the number of splitting iterations in the cross-validator

    # print(kf)

    for train_index, test_index in kf.split(x):
        print('TRAIN:', train_index, 'TEST:', test_index)

        xTrain = []
        xTest = []
        yTest = []
        yTrain = []
        for index in train_index:
            xTrain.append(x[index])
            yTrain.append(y[index])

        for index in test_index:
            xTest.append(x[index])
            yTest.append(y[index])


        # y_train, y_test = y[train_index], y[test_index]

    # rkf = RepeatedKFold(n_splits=10, n_repeats=10)
    # # kf = KFold(n_splits=10)
    # for train_index, test_index in rkf.split(x):
    #     # print("%s %s" % (train_index, test_index))
    #
    #     # print("TRAIN:", train_index, "TEST:", test_index)
    #     xTrain, xTest = x[train_index], x[test_index]
    #     yTrain, yTest = y[train_index], y[test_index]
    #

        cv = CountVectorizer()
        features = cv.fit_transform(xTrain)
        transformer = TfidfTransformer()

        model = MultinomialNB()
        model.fit(features, yTrain)

        featureTest = cv.transform(xTest)

        print(model.score(featureTest, yTest))

    # return result_vectors
    # print(result_vectors)


    # pipeline = Pipeline([
    #     ('count_vectorizer', CountVectorizer()),
    #     ('classifier', MultinomialNB())
    # ])
    #
    # k_fold = KFold(n_splits=2)
    # scores = []
    # confusion = numpy.array([[0, 0], [0, 0]])
    # for train_indices, test_indices in k_fold.split(result_vectors):
    #     train_text = result_vectors.iloc[train_indices]['text'].values
    #     train_y = result_vectors.iloc[train_indices]['class'].values.astype(str)
    #
    #     test_text = result_vectors.iloc[test_indices]['text'].values
    #     test_y = result_vectors.iloc[test_indices]['class'].values.astype(str)
    #
    #     pipeline.fit(train_text, train_y)
    #     predictions = pipeline.predict(test_text)
    #
    #     confusion += confusion_matrix(test_y, predictions)
    #     score = f1_score(test_y, predictions, pos_label=1)
    #     scores.append(score)
    #
    # print('Total emails classified:', len(result_vectors))
    # print('Score:', sum(scores) / len(scores))
    # print('Confusion matrix:')
    # print(confusion)





def readAndLabel(directory):

    label = 0
    totalCorpus = []
    fileCorpus = []

    # TODO: remove list limitations
    for currentFile in os.listdir(directory)[:]:
        if currentFile.endswith(".txt"):
            path = directory + '\\' + currentFile
            print()
            print('Reading the file: ')
            print(path)

            f = open(path, 'r', encoding='utf-8')
            sentences = f.read().splitlines()

            for sentence in sentences:
                totalCorpus.append({'text': sentence, 'class': label})

        label += 1

    # print(totalCorpus[47356:47360])
    return totalCorpus


def createFeatureVectors(fileCorpus):

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
