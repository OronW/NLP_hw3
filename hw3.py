import os

directory = r'C:\Users\oron.werner\PycharmProjects\NLP\hw3Output'


def main():




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
