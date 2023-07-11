import re
import DataReader as DR
import CountVect as CV
import TFIDF as TF
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.linear_model import LogisticRegression as LR
from sklearn.neighbors import KNeighborsClassifier

from sklearn import svm
from nltk.tokenize import word_tokenize,RegexpTokenizer
from nltk.stem import *
import re
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.linear_model import LogisticRegression as LR
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier as NN
#from sklearn.metrics import confusion_matrix
#from pandas_ml import ConfusionMatrix
from sklearn.decomposition import  LatentDirichletAllocation

filename="lyrics_got.txt"
def ReadNew():
    Data=[]
    lyrics=[]   
    for i in range(1,2):
        #filename="lyrics_got.txt"
        try:
            with open(filename,'r',encoding='latin-1') as file:
                lyric=file.read()
                (lyric,count)=re.subn(r"(\n|\t)"," ",lyric)
                (lyric,count)=re.subn(r"(\[\d\d:\d\d\.\d\d\])","",lyric)
                lyrics.append(lyric)
                # print("\n\nThese are the lyrics from datareadrer:\n")
                # print(lyrics)
        except:
            break
    # except:
    #     break

    Data.append(lyrics)
    return Data   

# NewData=ReadNew()

# ##print(NewData)

# SongsWordsNew=[[],[]]

# for song in NewData:
# 	s=song[0]
# 	SongsWordsNew[0].append(s)


# stemmer = SnowballStemmer("english")
# tokenizer = RegexpTokenizer("[\w’]+", flags=re.UNICODE)


stemmer = SnowballStemmer("english")
tokenizer = RegexpTokenizer("[\w’]+", flags=re.UNICODE)

def tokenize(text):
    #tokens = word_tokenize(text)
    tokens = tokenizer.tokenize(text)
    stems = []
    for item in tokens:
        stems.append(stemmer.stem(item))
    return stems


# vectorizer1 = CountVectorizer(min_df=2,ngram_range = (1,3))
# vectorizer2 = TfidfVectorizer(tokenizer=tokenize,min_df=2, ngram_range = (1,3), sublinear_tf = True, stop_words = "english")

# print ("Vectorizing training...")

# train_x1 = vectorizer1.fit_transform(CV.SongsWordsTrain[0])
# train_x2 = vectorizer2.fit_transform(TF.SongsWordsTrain[0])
# #print(train_x)
# print("\nPrinting count vect values\n")
# print(train_x1.shape)
# print(train_x1.getnnz())

# print("\nPrinting tfid vect values\n")
# print(train_x2.shape)
# print(train_x2.getnnz())
# print ("Vectorizing test...")
# # print("\nThese are the feature names\n")
# # print(vectorizer.get_feature_names())

# test_new1 = vectorizer1.transform(SongsWordsNew[0])
# test_new2 = vectorizer2.transform(SongsWordsNew[0])

# #print(test_x)
# print ("Training...")
# print ("\nNB...")
# modelA1 = MultinomialNB()
# modelA2 = MultinomialNB()
# modelA1.fit(train_x1, CV.SongsWordsTrain[1])
# modelA2.fit(train_x2, TF.SongsWordsTrain[1])
# #score on training set
# # print(modelA.score(train_x, CV.SongsWordsTrain[1])) 

# # print(modelA.score(test_new,SongsWordsTTrain[1]))
# print("NB Prediction:")
# print("\nCount Vect:")
# print(modelA1.predict(test_new1))
# print("TFIDF:")
# print(modelA2.predict(test_new2))




# print ("\nSVM...")
# modelB1 = svm.SVC(kernel='linear', C=1, gamma=1) 
# modelB2 = svm.SVC(kernel='linear', C=1, gamma=1)
# modelB1.fit( train_x1, CV.SongsWordsTrain[1])
# modelB2.fit( train_x2, TF.SongsWordsTrain[1])

# print("SVM Prediction:")
# print("\nCount Vect:")
# print(modelB1.predict(test_new1))
# print("TFIDF:")
# print(modelB2.predict(test_new2))

# print ("\nLR...")
# modelC1 = LR(multi_class='multinomial',solver='newton-cg')
# modelC2 = LR(multi_class='multinomial',solver='newton-cg')

# modelC1.fit( train_x1, CV.SongsWordsTrain[1])
# modelC2.fit( train_x2, TF.SongsWordsTrain[1])

# print("LR Prediction:")
# print("\nCount Vect:")
# print(modelC1.predict(test_new1))
# print("TFIDF:")
# print(modelC2.predict(test_new2))

# print ("\nKNN...")

# knn1 = KNeighborsClassifier(n_neighbors=5)
# knn2 = KNeighborsClassifier(n_neighbors=5)

# knn1.fit(train_x1,CV.SongsWordsTrain[1])
# knn2.fit(train_x2,TF.SongsWordsTrain[1])

# print("KNN Prediction:")
# print("\nCount Vect:")
# print(knn1.predict(test_new1))
# print("TFIDF:")
# print(knn2.predict(test_new2))


def mainPredict():


    NewData=ReadNew()
    resu=[]
    ##print(NewData)

    SongsWordsNew=[[],[]]

    for song in NewData:
        s=song[0]
        SongsWordsNew[0].append(s)



    vectorizer1 = CountVectorizer(min_df=2,ngram_range = (1,3))
    vectorizer2 = TfidfVectorizer(tokenizer=tokenize,min_df=2, ngram_range = (1,3), sublinear_tf = True, stop_words = "english")

    print ("Vectorizing training...")

    train_x1 = vectorizer1.fit_transform(CV.SongsWordsTrain[0])
    train_x2 = vectorizer2.fit_transform(TF.SongsWordsTrain[0])
    #print(train_x)
    print("\nPrinting count vect values\n")
    print(train_x1.shape)
    print(train_x1.getnnz())

    print("\nPrinting tfid vect values\n")
    print(train_x2.shape)
    print(train_x2.getnnz())
    print ("Vectorizing test...")
    # print("\nThese are the feature names\n")
    # print(vectorizer.get_feature_names())

    test_new1 = vectorizer1.transform(SongsWordsNew[0])
    test_new2 = vectorizer2.transform(SongsWordsNew[0])

    #print(test_x)
    print ("Training...")
    print ("\nNB...")
    modelA1 = MultinomialNB()
    modelA2 = MultinomialNB()
    modelA1.fit(train_x1, CV.SongsWordsTrain[1])
    modelA2.fit(train_x2, TF.SongsWordsTrain[1])

    #score on training set
    # print(modelA.score(train_x, CV.SongsWordsTrain[1])) 

    # print(modelA.score(test_new,SongsWordsTTrain[1]))
    print("NB Prediction:")
    print("\nCount Vect:")
    ress=modelA1.predict(test_new1)
    resu.append(str(ress[0]))
    print("TFIDF:")
    ress=modelA2.predict(test_new2)
    resu.append(str(ress[0]))




    print ("\nSVM...")
    modelB1 = svm.SVC(kernel='linear', C=1, gamma=1) 
    modelB2 = svm.SVC(kernel='linear', C=1, gamma=1)
    modelB1.fit( train_x1, CV.SongsWordsTrain[1])
    modelB2.fit( train_x2, TF.SongsWordsTrain[1])

    print("SVM Prediction:")
    print("\nCount Vect:")
    ress=modelB1.predict(test_new1)
    resu.append(str(ress[0]))
    print("TFIDF:")
    ress=modelB2.predict(test_new2)
    resu.append(str(ress[0]))

    print ("\nLR...")
    modelC1 = LR(multi_class='multinomial',solver='newton-cg')
    modelC2 = LR(multi_class='multinomial',solver='newton-cg')

    modelC1.fit( train_x1, CV.SongsWordsTrain[1])
    modelC2.fit( train_x2, TF.SongsWordsTrain[1])

    print("LR Prediction:")
    print("\nCount Vect:")
    ress=modelC1.predict(test_new1)
    resu.append(str(ress[0]))
    print("TFIDF:")
    ress=modelC2.predict(test_new2)
    resu.append(str(ress[0]))

    print ("\nKNN...")

    knn1 = KNeighborsClassifier(n_neighbors=5)
    knn2 = KNeighborsClassifier(n_neighbors=5)

    knn1.fit(train_x1,CV.SongsWordsTrain[1])
    knn2.fit(train_x2,TF.SongsWordsTrain[1])

    print("KNN Prediction:")
    print("\nCount Vect:")
    ress=knn1.predict(test_new1)
    resu.append(str(ress[0]))
    print("TFIDF:")
    ress=knn2.predict(test_new2)
    resu.append(str(ress[0]))

    #results.append(str(res[0]))
    #print(res[0])

    # plot_cnf(neural,test_x,test_y,GENRES)
    # print("\n\nPRINTING SET APPENDED\n\n")
    # print(resu)
    word_counter = {}
    for word in resu:
        if word in word_counter:
            word_counter[word] += 1
        else:
            word_counter[word] = 1  
    popular_words = sorted(word_counter, key = word_counter.get, reverse = True)
    resu = popular_words[:1]
    print("\n\nPRINTING APPENDED RESULTS")
    print(resu)
    n=str(resu[0])
    if (n=="0"):
        return "Angry"
    elif (n=="1"):
        return "Happy"
    elif (n=="2"):
        return "Sad"

s=mainPredict()
print(s)
# genre = pred()[:-4]
# print(genre)


