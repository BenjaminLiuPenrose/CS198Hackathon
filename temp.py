import nltk
import numpy as np
import random
import string # to process standard python strings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ### Reading in the data
# f=open('exampledir/basic.txt','r',errors = 'ignore')
# raw=f.read()
# raw=raw.lower()# converts to lowercase
# nltk.download('punkt') # first-time use only
# nltk.download('wordnet') # first-time use only
# nltk.download('maxent_ne_chunker')
# nltk.download('words')
# nltk.download("treebank")
# nltk.download("averaged_perceptron_tagger")
# dir(nltk)

# sent_tokens = nltk.sent_tokenize(raw)# converts to list of sentences
# word_tokens = nltk.word_tokenize(raw)# converts to list of words

# sent_tokens[:2]
# word_tokens[:2]

# ### Pre-processing the raw text
# lemmer = nltk.stem.WordNetLemmatizer()
# # WordNet is a semantically-oriented dictionary of English included in NLTK.
# def LemTokens(tokens):
# 	return [lemmer.lemmatize(token) for token in tokens]
# remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)
# def LemNormalize(text):
# 	return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

# ### Keyword matching
# GREETING_INPUTS = ("hello", "hi", "greetings", "sup", "what's up","hey",)
# GREETING_RESPONSES = ["hi", "hey", "*nods*", "hi there", "hello", "I am glad! You are talking to me"]
# def greeting(sentence):
# 	for word in sentence.split():
# 		if word.lower() in GREETING_INPUTS:
# 			return random.choice(GREETING_RESPONSES)

# ### Generating Response
# def response(user_response):
# 	robo_response=''
# 	TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
# 	tfidf = TfidfVec.fit_transform(sent_tokens)
# 	vals = cosine_similarity(tfidf[-1], tfidf)
# 	idx=vals.argsort()[0][-2]
# 	flat = vals.flatten()
# 	flat.sort()
# 	req_tfidf = flat[-2]
# 	if(req_tfidf==0):
# 		robo_response=robo_response+"I am sorry! I don't understand you"
# 		return robo_response
# 	else:
# 		robo_response = robo_response+sent_tokens[idx]
# 		return robo_response

# ### Main
# flag=True
# print("ROBO: My name is Robo. I will answer your queries about Chatbots. If you want to exit, type Bye!")
# while(flag==True):
# 	user_response = input()
# 	user_response=user_response.lower()
# 	if(user_response!='bye'):
# 		if(user_response=='thanks' or user_response=='thank you' ):
# 			flag=False
# 			print("ROBO: You are welcome..")
# 		else:
# 			if(greeting(user_response)!=None):
# 				print("ROBO: "+greeting(user_response))
# 			else:
# 				sent_tokens.append(user_response)
# 				word_tokens=word_tokens+nltk.word_tokenize(user_response)
# 				final_words=list(set(word_tokens))
# 				print("ROBO: ",end="")
# 				print(response(user_response))
# 				sent_tokens.remove(user_response)
# 	else:
# 		flag=False
# 		print("ROBO: Bye! take care..")


# ===============================================IDENTIFY ORDINAL NUMBER======================================================
import spacy
from spacy import displacy
import en_core_web_sm
nlp = en_core_web_sm.load()

doc = nlp('European authorities fined Google a record $5.1 billion on Wednesday for abusing its power in the mobile phone market and ordered the company to alter its practices')
print([(X, X.ent_iob_, X.ent_type_) for X in doc])
print([(X.text, X.label_) for X in doc.ents])

[(x.orth_,x.pos_, x.lemma_) for x in [y
                                      for y
                                      in doc
                                      if not y.is_stop and y.pos_ != 'PUNCT']]

from pattern.en import number
number('two thousand fifty and a half')

