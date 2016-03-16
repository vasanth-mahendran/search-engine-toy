import os
import math
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords

corpus_root = '/Users/vasanthmahendran/Documents/study/2016-spring/5334-min/project-1/presidential_debates/'
stemmer = PorterStemmer()
tokenizer = RegexpTokenizer(r'[a-zA-Z]+')

class search_engine_toy:

    def __init__(self):
        self.parsetokenfrequency()
        self.calculatedocumentvector()
        self.normalizedocumentvector()
        #print("document_vectors-->",self.document_vectors)
        #print("normalized_document_vectors-->",self.normalized_document_vectors)

    def query(self,qstring):
        normalized_query_vector = self.calculatequeryvector(qstring)
        high_cos_sim_doc = ""
        high_cos_sim_val = 0
        for filename,normalized_document_vector in self.normalized_document_vectors.items():
            cos_sim = self.calculatedotproduct(normalized_query_vector,normalized_document_vector)
            #print("cos_sim ->",cos_sim," for file name->",filename)
            if not high_cos_sim_doc:
                high_cos_sim_doc = filename
                high_cos_sim_val = cos_sim
            else:
                if cos_sim > high_cos_sim_val:
                    high_cos_sim_val = cos_sim
                    high_cos_sim_doc = filename
        return high_cos_sim_doc

    def getcount(self,token):
        count_token = 0
        for filename,word_dic in self.doc_dictionary.items():
            if token in word_dic.keys():
                count_token += word_dic[token]
        return count_token

    def getidf(self,token):
        doc_token_count = 0
        for filename,word_dic in self.doc_dictionary.items():
            if token in word_dic.keys():
                doc_token_count += 1
        if doc_token_count == 0:
            return 0
        else:
            return math.log10(self.doc_count/doc_token_count)

    def docdocsim(self,filename1,filename2):
        #print(self.document_vectors[filename1])
        #print(len(self.document_vectors[filename1]))
        normalized_document_vector1 = self.normalized_document_vectors[filename1]
        normalized_document_vector2 = self.normalized_document_vectors[filename2]
        return self.calculatedotproduct(normalized_document_vector1,normalized_document_vector2)

    def querydocsim(self,qstring,filename):
        normalized_query_vector = self.calculatequeryvector(qstring)
        normalized_document_vector = self.normalized_document_vectors[filename]
        return self.calculatedotproduct(normalized_query_vector,normalized_document_vector)

    def parsetokenfrequency(self):
        doc_dictionary = dict()
        doc_count = 0
        for filename in os.listdir(corpus_root):
            count = 0
            doc_count += 1
            file = open(os.path.join(corpus_root, filename), "r", encoding='UTF-8')
            doc = file.read()
            file.close()
            doc = doc.lower()
            word_frequency_tokens = dict()
            tokens = tokenizer.tokenize(doc)
            stop_words = stopwords.words('english')
            for word in tokens: # iterate over word_list
                if word not in stop_words:
                    count += 1
                    word = stemmer.stem(word)
                    if word in word_frequency_tokens.keys():
                        word_frequency_tokens[word] = word_frequency_tokens[word]+1
                    else:
                        word_frequency_tokens[word] = 1
            doc_dictionary[filename] = word_frequency_tokens

        '''
        self.doc_dictionary  is the dictionary to collect document name as key and a word dictionary as
        value which in return collect all the words in the document along with its frequency in the document
        '''
        self.doc_dictionary = doc_dictionary

        '''
        self.doc_count holds the total count of documents in the corpus.
        '''
        self.doc_count = doc_count

    def calculatetermweight(self,term,frequency):
        return (1+math.log10(frequency))*self.getidf(term)

    def calculatedocumentvector(self):
        document_vectors = dict()
        for filename,word_dic in self.doc_dictionary.items():
            document_vector = dict()
            for token,frequency in word_dic.items():
                tf_idf_weight = self.calculatetermweight(token,frequency)
                document_vector[token] = tf_idf_weight
            document_vectors[filename] = document_vector
        self.document_vectors = document_vectors

    def normalizedocumentvector(self):
        normalized_document_vectors = dict()
        for filename,document_vector in self.document_vectors.items():
            normalized_document_vector = dict()
            euclidian_distance_sum = 0
            for token,tf_idf_weight in document_vector.items():
                euclidian_distance_sum += math.pow(tf_idf_weight,2)
            euclidian_distance = math.sqrt(euclidian_distance_sum)
            for token,tf_idf_weight in document_vector.items():
                normalized_tf_idf_weight = tf_idf_weight/euclidian_distance
                normalized_document_vector[token] = normalized_tf_idf_weight
            normalized_document_vectors[filename] = normalized_document_vector
        self.normalized_document_vectors = normalized_document_vectors

    def calculatequeryvector(self,qstring):
        query_frequency_vector = dict()
        for word in tokenizer.tokenize(qstring): # iterate over query terms
            if word not in stopwords.words('english'):
                word = stemmer.stem(word)
                if word in query_frequency_vector.keys():
                    query_frequency_vector[word] = query_frequency_vector[word]+1
                else:
                    query_frequency_vector[word] = 1
        query_vector = dict()
        for term,frequency in query_frequency_vector.items():
            tf_weight = 1 + math.log10(frequency)
            query_vector[term] = tf_weight
        euclidian_distance_sum = 0
        for token,tf_weight in query_vector.items():
            euclidian_distance_sum += math.pow(tf_weight,2)
        euclidian_distance = math.sqrt(euclidian_distance_sum)
        normalized_query_vector = dict()
        for token,tf_weight in query_vector.items():
            normalized_tf_weight = tf_weight/euclidian_distance
            normalized_query_vector[token] = normalized_tf_weight
        return normalized_query_vector

    def calculatedotproduct(self,vector1,vector2):
        dotproduct = 0
        #print("vector1-->",vector1)
        #print("vector2-->",vector2)
        for key,value in vector1.items():
            if key in vector2.keys():
                dotproduct+=value*vector2[key]
        return dotproduct

se_toy = search_engine_toy()
print("----------output 1--------")
print(se_toy.query("health insurance wall street"))
print(se_toy.getcount("health"))
print("%.12f" % se_toy.getidf("health"))
print("%.12f" % se_toy.docdocsim("1960-09-26.txt", "1980-09-21.txt"))
print("%.12f" % se_toy.querydocsim("health insurance wall street", "1996-10-06.txt"))
print("----------output 2--------")
print(se_toy.query("security conference ambassador"))
print(se_toy.getcount("attack"))
print("%.12f" % se_toy.getidf("agenda"))
print("%.12f" % se_toy.docdocsim("1960-10-21.txt", "1980-09-21.txt"))
print("%.12f" % se_toy.querydocsim("particular constitutional amendment", "2000-10-03.txt"))
print("----------output 3--------")
print(se_toy.query("particular constitutional amendment"))
print(se_toy.getcount("amend"))
print("%.12f" % se_toy.getidf("particular"))
print("%.12f" % se_toy.docdocsim("1960-09-26.txt", "1960-10-21.txt"))
print("%.12f" % se_toy.querydocsim("health insurance wall street", "2000-10-03.txt"))