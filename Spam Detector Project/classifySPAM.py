'''
  This program shell reads email data for the spam classification problem.
  The input to the program is the path to the Email directory "corpus" and a limit number.
  The program reads the first limit number of ham emails and the first limit number of spam.
  It creates an "emaildocs" variable with a list of emails consisting of a pair
    with the list of tokenized words from the email and the label either spam or ham.
  It prints a few example emails.
  Your task is to generate features sets and train and test a classifier.

  Usage:  python classifySPAM.py  <corpus directory path> <limit number>
'''
# open python and nltk packages needed for processing
import os
import sys
import random
import nltk
from nltk.corpus import stopwords
import os
import sys
import random
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.collocations import *
from nltk.collocations import TrigramAssocMeasures, TrigramCollocationFinder
from nltk.metrics import ConfusionMatrix
import re
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
random.seed(10)

# define a feature definition function here
def word_freq(email,word_features):
    email_words=set(email)
    features={}
    for word in word_features:
        features['contains({})'.format(word)]=(word in email_words)
    return features
    
def bigram_features(email, word_features, bigram_features):
  email_words = set(email)
  email_bigrams = nltk.bigrams(email)
  features = {}
  for word in word_features:
    features['contains({})'.format(word)] = (word in email_words)
  for bigram in bigram_features:
    features['bigram({} {})'.format(bigram[0], bigram[1])] = (bigram in email_bigrams)
  return features
def trigram_features(email, word_features, trigrams):
  email_words = set(email)
  trigram_finder_email = TrigramCollocationFinder.from_words(email)
  trigram_scores_email = trigram_finder_email.score_ngrams(TrigramAssocMeasures.chi_sq)
  email_trigrams = [trigram for trigram,score in trigram_scores_email]
  features = {}
  for word in word_features:
    features['contains({})'.format(word)] = (word in email_words)
  for trigram in trigrams:
    features['trigram({} {} {})'.format(trigram[0], trigram[1], trigram[2])] = (trigram in email_trigrams)
  return features
  
def POS_features(email, word_features):
    email_words = set(email)
    tagged_words = nltk.pos_tag(email)
    features = {}
    for word in word_features:
        features['contains({})'.format(word)] = (word in email_words)
    numNoun = 0
    numVerb = 0
    numAdj = 0
    numAdverb = 0
    for (word, tag) in tagged_words:
        if tag.startswith('N'): numNoun += 1
        if tag.startswith('V'): numVerb += 1
        if tag.startswith('J'): numAdj += 1
        if tag.startswith('R'): numAdverb += 1
    features['nouns'] = numNoun
    features['verbs'] = numVerb
    features['adjectives'] = numAdj
    features['adverbs'] = numAdverb
    return features


def SL_features(email, word_features):
  
  SL = readSubjectivity()
  email_words = set(email)
  features = {}
  for word in word_features:
    features['contains({})'.format(word)] = (word in email_words)
  # count variables for the 4 classes of subjectivity
  weakPos = 0
  strongPos = 0
  weakNeg = 0
  strongNeg = 0
  for word in email_words:
    if word in SL:
      strength, posTag, isStemmed, polarity = SL[word]
      if strength == 'weaksubj' and polarity == 'positive':
        weakPos += 1
      if strength == 'strongsubj' and polarity == 'positive':
        strongPos += 1
      if strength == 'weaksubj' and polarity == 'negative':
        weakNeg += 1
      if strength == 'strongsubj' and polarity == 'negative':
        strongNeg += 1
      features['positivecount'] = weakPos + (2 * strongPos)
      features['negativecount'] = weakNeg + (2 * strongNeg)
  
  if 'positivecount' not in features:
    features['positivecount']=0
  if 'negativecount' not in features:
    features['negativecount']=0
  return features

def combined_features(email, word_features, bigram_features_all, trigrams):
    email_words = set(email)
    SL = readSubjectivity()
    email_bigrams=nltk.bigrams(email)
    trigram_finder_email = TrigramCollocationFinder.from_words(email)
    trigram_scores_email = trigram_finder_email.score_ngrams(TrigramAssocMeasures.chi_sq)
    email_trigrams = [trigram for trigram,score in trigram_scores_email]
    features={}
    weakPos = 0
    strongPos = 0
    weakNeg = 0
    strongNeg = 0
    for word in email_words:
      if word in SL:
        strength, posTag, isStemmed, polarity = SL[word]
        if strength == 'weaksubj' and polarity == 'positive':
          weakPos += 1
        if strength == 'strongsubj' and polarity == 'positive':
          strongPos += 1
        if strength == 'weaksubj' and polarity == 'negative':
          weakNeg += 1
        if strength == 'strongsubj' and polarity == 'negative':
          strongNeg += 1
        features['positivecount'] = weakPos + (2 * strongPos)
        features['negativecount'] = weakNeg + (2 * strongNeg)
    if 'positivecount' not in features:
      features['positivecount']=0
    if 'negativecount' not in features:
      features['negativecount']=0
    for word in word_features:
      features['contains({})'.format(word)] = (word in email_words)
    for bigram in bigram_features_all:
      features['bigram({} {})'.format(bigram[0], bigram[1])] = (bigram in email_bigrams)
    for trigram in trigrams:
      features['trigram({} {} {})'.format(trigram[0], trigram[1], trigram[2])] = (trigram in email_trigrams)
      return features


#def svm_classifier(emails,classification):
#  x_train, x_test, y_train, y_test = train_test_split(emails,classification,test_size=0.3,random_state=100)
#  classifier = svm.SVC(kernel='linear')
#  classifier.fit(x_train,y_train)
#  y_pred = classifier.predict(x_test)
#  print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


def model_training(featuresets):
  training_size = int(0.7*len(featuresets))
  test_set = featuresets[:training_size]
  training_set = featuresets[training_size:]
  classifier = nltk.NaiveBayesClassifier.train(training_set)
  goldlist = []
  predictedlist = []
  for (features, label) in test_set:
      goldlist.append(label)
      predictedlist.append(classifier.classify(features))
  cm = nltk.ConfusionMatrix(goldlist, predictedlist)
  print ("Confusion Matrix : ")
  print(cm)
  print ("Model Accuracy : ")
  print (nltk.classify.accuracy(classifier, test_set))
  print ("k-fold cross validation mean accuracy : ")
  print (cross_validation(10,featuresets))
  print ("Other Evaluation Measures")
  eval_measures(goldlist,predictedlist)
#  print ("Showing most informative features")
#  classifier.show_most_informative_features(30)
#  print (nltk.classify.accuracy(classifier,test_set))


def eval_measures(gold, predicted):
    # get a list of labels
    labels = list(set(gold))
    # these lists have values for each label
    recall_list = []
    precision_list = []
    F1_list = []
    for lab in labels:
        # for each label, compare gold and predicted lists and compute values
        TP = FP = FN = TN = 0
        for i, val in enumerate(gold):
            if val == lab and predicted[i] == lab:  TP += 1
            if val == lab and predicted[i] != lab:  FN += 1
            if val != lab and predicted[i] == lab:  FP += 1
            if val != lab and predicted[i] != lab:  TN += 1
        # use these to compute recall, precision, F1
        recall = TP / (TP + FP)
        precision = TP / (TP + FN)
        recall_list.append(recall)
        precision_list.append(precision)
        F1_list.append( 2 * (recall * precision) / (recall + precision))

    # the evaluation measures in a table with one row per label
    print('\tPrecision\tRecall\t\tF1')
    # print measures for each label
    for i, lab in enumerate(labels):
        print(lab, '\t', "{:10.3f}".format(precision_list[i]), \
          "{:10.3f}".format(recall_list[i]), "{:10.3f}".format(F1_list[i]))


def cross_validation(num_folds, featuresets):
      subset_size = int(len(featuresets)/num_folds)
      accuracy_list = []
      # iterate over the folds
      for i in range(num_folds):
          test_this_round = featuresets[i*subset_size:][:subset_size]
          train_this_round = featuresets[:i*subset_size]+featuresets[(i+1)*subset_size:]
          # train using train_this_round
          classifier = nltk.NaiveBayesClassifier.train(train_this_round)
          # evaluate against test_this_round and save accuracy
          accuracy_this_round = nltk.classify.accuracy(classifier, test_this_round)
          #print (i, accuracy_this_round)
          accuracy_list.append(accuracy_this_round)
      # find mean accuracy over all rounds
      mean_accuracy = sum(accuracy_list) / num_folds
      return(mean_accuracy)
      
def writeFeatureSets(featuresets, outpath):
    # open outpath for writing
    f = open(outpath, 'w')
    # get the feature names from the feature dictionary in the first featureset
    featurenames = featuresets[0][0].keys()
    # create the first line of the file as comma separated feature names
    #    with the word class as the last feature name
    featurenameline = ''
    for featurename in featurenames:
        # replace forbidden characters with text abbreviations
        featurename = featurename.replace(',','CM')
        featurename = featurename.replace("'","DQ")
        featurename = featurename.replace('"','QU')
        featurenameline += featurename + ','
    featurenameline += 'class'
    # write this as the first line in the csv file
    f.write(featurenameline)
    f.write('\n')
    # convert each feature set to a line in the file with comma separated feature values,
    # each feature value is converted to a string
    #   for booleans this is the words true and false
    #   for numbers, this is the string with the number
    for featureset in featuresets:
        featureline = ''
        for key in featurenames:
            featureline += str(featureset[0][key]) + ','
        featureline += featureset[1]
        # write each feature set values to the file
        f.write(featureline)
        f.write('\n')
    f.close()
    
def readSubjectivity():
    path = "sub.tff"
    flexicon = open(path, 'r')
    # initialize an empty dictionary
    sldict = { }
    for line in flexicon:
        fields = line.split()   # default is to split on whitespace
        # split each field on the '=' and keep the second part as the value
        strength = fields[0].split("=")[1]
        word = fields[2].split("=")[1]
        posTag = fields[3].split("=")[1]
        stemmed = fields[4].split("=")[1]
        polarity = fields[5].split("=")[1]
        if (stemmed == 'y'):
            isStemmed = True
        else:
            isStemmed = False
        # put a dictionary entry with the word as the keyword
        #     and a list of the other values
        sldict[word] = [strength, posTag, isStemmed, polarity]
    return sldict

# function to read spam and ham files, train and test a classifier 
def processspamham(dirPath,limitStr):
  # convert the limit argument from a string to an int
  limit = int(limitStr)
  
  # start lists for spam and ham email texts
  hamtexts = []
  spamtexts = []
  os.chdir(dirPath)
  # process all files in directory that end in .txt up to the limit
  #    assuming that the emails are sufficiently randomized
  for file in os.listdir("./spam"):
    if (file.endswith(".txt")) and (len(spamtexts) < limit):
      # open file for reading and read entire file into a string
      f = open("./spam/"+file, 'r', encoding="latin-1")
      spamtexts.append (f.read())
      f.close()
  for file in os.listdir("./ham"):
    if (file.endswith(".txt")) and (len(hamtexts) < limit):
      # open file for reading and read entire file into a string
      f = open("./ham/"+file, 'r', encoding="latin-1")
      hamtexts.append (f.read())
      f.close()
  
  # print number emails read
  print ("Number of spam files:",len(spamtexts))
  print ("Number of ham files:",len(hamtexts))
 
  # create list of mixed spam and ham email documents as (list of words, label)
  emaildocs = []
  # add all the spam
  for spam in spamtexts:
    tokens = nltk.word_tokenize(spam)
    emaildocs.append((tokens, 'spam'))
  # add all the regular emails
  for ham in hamtexts:
    tokens = nltk.word_tokenize(ham)
    emaildocs.append((tokens, 'ham'))
  
  # randomize the list
  random.shuffle(emaildocs)
#
#  # print a few token lists
#  for email in emaildocs[:4]:
#    print (email)
#    print
#

  # continue as usual to get all words and create word features

  all_words=[]
  for email in emaildocs:
    for words in email:
      for word in words:
        all_words.append(word)
  all_original = all_words
  
  # possibly filter tokens
  
  all_words = [w for w in all_words if not w.lower() in stopwords.words('english')]
  #Unigram processed
  distribution = nltk.FreqDist(all_words)
  word_items = distribution.most_common(2000)
  word_features = [word for (word, freq) in word_items]
  #Unigram unprocessed
  distribution_o = nltk.FreqDist(all_original)
  word_items_o = distribution.most_common(2000)
  word_features_o = [word for (word, freq) in word_items_o]
  #Bigram processed
  bigram_measures = nltk.collocations.BigramAssocMeasures()
  finder = BigramCollocationFinder.from_words(tokens,window_size=3)
  bigram_features_all = finder.nbest(bigram_measures.chi_sq, 3000)
  #Trigram processed
  trigram_measures = nltk.collocations.TrigramAssocMeasures()
  trigram_finder = TrigramCollocationFinder.from_words(tokens)
  trigram_scores = trigram_finder.nbest(trigram_measures.chi_sq,3000)
  
  # feature sets from a feature definition function
  featureset_freq_original = [(word_freq(email, word_features),classification) for (email,classification) in emaildocs]
  featureset_freq = [(word_freq(email, word_features),classification) for (email,classification) in emaildocs]
  featureset_bigram = [(bigram_features(email,word_features,bigram_features_all),classification) for (email,classification) in emaildocs]
  featureset_POS = [(POS_features(email, word_features), classification) for (email, classification) in emaildocs]
  featureset_SL = [(SL_features(email, word_features), classification) for (email, classification) in emaildocs]
  featureset_trigram = [(trigram_features(email,word_features,trigram_scores),classification) for (email, classification) in emaildocs]
  featureset_combined = [(combined_features(email, word_features, bigram_features_all, trigram_scores),classification) for email, classification in emaildocs]
  
  # Features for SVM
  writeFeatureSets(featureset_freq,"BOW")
  #svm_model=svm_classifier(emails,classification)
  
  # train classifier and show performance in cross-validation
  print ("-------------------------------")
  print ("-------------------------------")
  print ("Feature : Unprocessed BOW")
  writeFeatureSets(featureset_freq,"BOW_unprocessed")
  model_training(featureset_freq_original)
  print ("-------------------------------")
  print ("-------------------------------")
  print ("Feature : BOW")
  model_training(featureset_freq)
  writeFeatureSets(featureset_freq,"BOW")
  print ("-------------------------------")
  print ("-------------------------------")
  print ("Feature : Bigram")
  model_training(featureset_bigram)
  writeFeatureSets(featureset_bigram,"Bigram")
  print ("-------------------------------")
  print ("-------------------------------")
  print ("Feature : POS")
  model_training(featureset_POS)
  writeFeatureSets(featureset_POS,"POS")
  print ("-------------------------------")
  print ("-------------------------------")
  print ("Feature : SL")
  model_training(featureset_SL)
  writeFeatureSets(featureset_SL,"SL")
  print ("-------------------------------")
  print ("-------------------------------")
  print ("Feature : Trigram")
  model_training(featureset_trigram)
  writeFeatureSets(featureset_trigram,"Trigram")
  print ("-------------------------------")
  print ("-------------------------------")
  print ("Feature : Combined")
  model_training(featureset_combined)
  writeFeatureSets(featureset_freq,"Combined")
  

"""
commandline interface takes a directory name with ham and spam subdirectories
   and a limit to the number of emails read each of ham and spam
It then processes the files and trains a spam detection classifier.

"""
if __name__ == '__main__':
    if (len(sys.argv) != 3):
        print ('usage: python classifySPAM.py <corpus-dir> <limit>')
        sys.exit(0)
    processspamham(sys.argv[1], sys.argv[2])
        
print(sys.argv)
