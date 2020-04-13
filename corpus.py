import convokit
import nltk
import string
from nltk.corpus import stopwords
from convokit import Corpus, download
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from gensim.models import Word2Vec

nltk.download('punkt')
corpus = Corpus(filename=download("subreddit-creepypasta"))

corpusTXT = open("corpus.txt", "w")

utter_ids = corpus.get_utterance_ids()
length = len(utter_ids)

# Print the posts from subreddit to a file
i = 0
while i < 2:
    corpusTXT.write(corpus.get_utterance(utter_ids[i]).text)
    i += 1

corpusTXT.close()

corpusTXT_2 = open("corpus.txt", "r")

text = corpusTXT_2.read()

# Tokenize all of corpus.txt
nltk_sentences = sent_tokenize(text)
tokenized_sents = [word_tokenize(i) for i in nltk_sentences]
new_sents = []
stop_words = set(stopwords.words('english'))
whitespace = ' '
punctuation = string.punctuation

i = 0
j = 0
while i < len(tokenized_sents):
    while j < len(tokenized_sents[i]):
        if tokenized_sents[i][j] in stop_words or tokenized_sents[i][j] == whitespace or tokenized_sents[i][j] in punctuation: 
            del tokenized_sents[i][j]
        j = j+1
    new_sents.append(tokenized_sents[i])
    i = i+1
    j = 0
    
#print(new_sents)

corpusTXT_2.close()

#train model
model = Word2Vec(new_sents, min_count=1)
#summarize the loaded model
print("\nSummary of loaded model:")
print(model)
#summarize vocabulary
words = list(model.wv.vocab)
print("\nSummary of vocabulary:")
print(words)
#access vector for one word
print("\nVector for pokemon:")
print(model['eyes'])
#save model
model.save('model.txt')
#load model
new_model = Word2Vec.load('model.txt')
print("\nNew Model:")
print(new_model)

print("CODE COMPLETED")
