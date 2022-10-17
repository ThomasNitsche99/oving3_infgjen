import pprint
import random
from unittest import skip; random.seed(123)
import codecs
import string
from nltk.stem.porter import PorterStemmer;
from nltk.probability import FreqDist

f = codecs.open("pg3300.txt", "r", "utf-8")


#--------------------PART 1: Data loading and processing --------------------------------#
#1.1
read = f.read()

#1.2 split into paragraphs
Split = read.split("\r\n\r\n")

#1.3 filter out "gutenberg"
for par in Split.copy():
    if "Gutenberg" in par:
        Split.remove(par)


#1.4 list of list containing each word in paragraph
newParagraphList = []
for par in Split:
    list_of_words = par.split(" ")
    newParagraphList.append(list_of_words)
    
#1.5 remove punctionation
#1.6 stemming
stemmer = PorterStemmer()
list_without_punc_and_stemmed = []                      
for listOWords in newParagraphList:
    newList = []
    
    for word in listOWords:
        new_string = word.translate(str.maketrans('', '', (string.punctuation+"\n\r\t")))
        new_string2 = stemmer.stem(new_string.lower())
        
        if new_string2!="":
            newList.append(new_string2)
            
    list_without_punc_and_stemmed.append(newList)
    
    
#1.7 freDist over a list of words
listen = list_without_punc_and_stemmed[2]
fdist = FreqDist(listen)
 
# ---------------------------------PART 2: Dictionary building ------------------------------#

#2.0
import gensim
##dictionary of the document
dictionary_doccument = gensim.corpora.Dictionary(list_without_punc_and_stemmed)

#2.1
#getting stopwords from document
Stopwords = codecs.open("common-english-words.txt", "r", "utf-8")
read_stopwords = Stopwords.read()
list_of_stopwords = read_stopwords.split(",")

stopword_ids = []
for stopword in list_of_stopwords:
    try :
        stopword_ids.append(dictionary_doccument.token2id[stopword])
    except:
        continue

##filter on stopword-tokens
dictionary_doccument.filter_tokens(stopword_ids)

#2.2
corpus = []
for l in list_without_punc_and_stemmed:
    corpus.append(dictionary_doccument.doc2bow(l))

##Translate from number to text
id2token = dict((i,j) for j, i in dictionary_doccument.token2id.items())
#corpus is list of tuple pairs

# ----------------------PART 3: Retrieval models -------------------------

#3.1
#tf-idf model from corpus
tfidf_model = gensim.models.TfidfModel(corpus)

#3.2
#create tdidf corpus from model
tfidf_corpus = tfidf_model[corpus]

#3.3 matrixSimilarity
tfidf_MatrixSim = gensim.similarities.MatrixSimilarity(tfidf_corpus)
#189 docs, 7194 features

#3.4
#creating LSI model
LSI_model = gensim.models.LsiModel(tfidf_corpus, id2word=dictionary_doccument,
num_topics=100)

##creating LSI corpus
LSI_corpus = LSI_model[tfidf_corpus] 

##creating LSI matrix Sim from LSI corpus
LSI_matrixSim = gensim.similarities.MatrixSimilarity(LSI_corpus)

#3.5
##this works, print the 3 first topics :D
print(LSI_model.show_topics()[0:3])


# --------------------------PART 4: Query  ------------------------------------------#

#4.1
query = "What is the function of money?"
#tokenize
tokenize_query  = query.split(" ")

query_stemmed_and_puncted = []  
#stem and remove punctuation             
for qword in tokenize_query:
    word1 = qword.translate(str.maketrans('', '', (string.punctuation+"\n\r\t")))
    word2 = stemmer.stem(word1.lower())
        
    if word2!="":
        query_stemmed_and_puncted.append(word2)

#doc2bow
fixedQuery = dictionary_doccument.doc2bow(query_stemmed_and_puncted)


#4.2
tfidfQ = tfidf_model[fixedQuery]

#4.3

#3 most relevant
doc2similarity = enumerate(tfidf_MatrixSim[tfidfQ])
sim = sorted(doc2similarity, key=lambda kv: -kv[1])[:3]
print(sim)
##fikse representasjon her


#4.4
lsi_query = LSI_model[tfidfQ]
print( "LSI_query" , sorted(lsi_query, key=lambda kv: -abs(kv[1]))[:3] ) 

topics = LSI_model.show_topics()[0:3]

for topic in topics:
    print("Topic ", topic, "\n")
 

doc2similarity = enumerate(LSI_matrixSim[lsi_query])
print( "doc2similarity", sorted(doc2similarity, key=lambda kv: -kv[1])[:3] ) 

    


 

  









    







        


