import nltk
import numpy as np
import random
import string # to process standard python strings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
import spacy
import en_core_web_sm
from num2words import num2words
from rake_nltk import Rake
from utils import *
nlp = en_core_web_sm.load()

class Conversation:

    def __init__(self, chatbot_txt = "datadir/chatbot.txt"):
        """
        The init function: Here, you should load any
        PyTorch models / word vectors / etc. that you
        have previously trained.
        class attributes:
        notes
        _facet_dict
        _reponse
        _ready_take_note    -- bool
        _non_facet_noun
        _sent_tokens        -- tokenized sentence
        _word_tokens        -- tokenized word


        class methods:
        calc_num_notes
        prepro_sentence
        prepare_take_one_note
        take_one_note
        update_facet_dict
        retrieve_one_note
        delete_one_note
        take_note_with_content
        respond
        """
        self.original = ""
        self.notes = []
        self.intent = {"take note": "take a note",
                       "add note": "take a note",
                       "remind me": "take a note with content",
                       "delete note": "delete a note",
                       "remove note": "delete a note",
                       "read note": "retrieve a note",
                       "what last note": "retrieve a note",
                       "which note": "retrieve a note",
                       "total number" : "total notes",
                       "many number" : "total notes"
                       }

        self._response = {
                        "calc_num_notes": [("You have ", " notes in total"),
                                           ("You have ", " notes in total")],
                        "prepare_take_one_note": ["OK, what would you like me to take?",
                                                  "Oh, sure. I am ready.",
                                                  "Say it.",
                                                  "I will record what you say next."],
                        "take_one_note": ["Alright, I have noted that.",
                                          "Sir, your note is recorded.",
                                          "Cool, I get it.",
                                          "Note recieved. You can retrieve it or delete it any time",
                                          "Okay, I have taken the note."],
                        "retrieve_one_note": ["That note was: ",
                                              "The note you are referring to was: "],
                        "retrieve_one_note_sorry": ["Sorry, the note you want to retrive doesn't exist.",
                                                    "Note to be retrieve not found",
                                                    "Sorry, note cannot be retrieved",
                                                    "Sorry, retrieve failure."],
                        "delete_one_note": [("That note '", "' is deleted"),
                                            ("That note '", "' is deleted")],
                        "delete_one_note_sorry": ["Sorry, the note you want to delete doesn't exist."
                                                    "Note to be delete not found",
                                                    "Sorry, note cannot be deleted",
                                                    "Sorry, deletion failure."],
                        "intent not found": "I am sorry! I don't understand you. Please try again!"
                        } # each value can be a list and then randomized choice

        self._ready_take_note = False
        self._facet_dict = {}
        self._non_facet_noun = ["note", "tomorrow"] # words to be removed from facet dict

        self._raw = (open(chatbot_txt,'r',errors = 'ignore').read()
                                                            .lower())
        self._sent_tokens = nltk.sent_tokenize(". ".join(["take note", "add note", "remind me", "delete note", "remove note", "read note"]))
        self._sent_tokens_what = nltk.sent_tokenize(". ".join(["what last note", "which note", "total number", "many number"]))

        self._word_toekns = nltk.word_tokenize(self._raw)

    def calc_num_notes(self):
        """
        No Arguments
        Rets:
        int, number of notes
        """
        return len(self.notes)

    def prepro_sentence(self, sentence):
        """
        Args:
        sentence

        Rets:
        str, sentence after preprocess
        """
        sentence = sentence.replace(r"[^\w\s]", "").strip().lower()
        s = set(stopwords.words('english'))
        s.remove("me")
        s.remove("what")
        s.remove("about")
        s.remove("have")

        split_sentence = list (filter (lambda w: not w in s, sentence.split()))
        full_sentence = ""
        for token in split_sentence:
            full_sentence += " " + token

        return full_sentence
    def get_input_vec(self, preprocessed_input):
        return [nltk.word_tokenize(elem) for elem in preprocessed_input]

    def get_cloest_intent(self, intput_vec):
        return

    def prepare_take_one_note(self, sentence):
        """
        Args:
        sentence

        Rets:
        str, proper response
        """
        self._ready_take_note = True # set ready_take_note to be true, then the bot will only record whatever the next sentence is.
        return random.choice(self._response["prepare_take_one_note"])

    def take_one_note(self, sentence):
        """
        Args:
        sentence

        Rets:
        str, proper response
        """
        self.notes.append(sentence)
        self.update_facet_dict(sentence)
        self._ready_take_note = False
        return random.choice(self._response["take_one_note"])


    def update_facet_dict(self, sentence):
        """
        Args:
        sentence

        Rets:
        Nothing
        """
        non_facet_noun = self._non_facet_noun
        r = Rake()
        r.extract_keywords_from_text(sentence)
        dat = r.get_ranked_phrases_with_scores() # extract the (key phrases, score) pair
        idx = self.calc_num_notes() - 1 # index of the added sentence, aka current sentence
        tmp = []
        for pair in dat:
            score = pair[0]
            phrase = nlp(pair[1])
            tmp.extend([(token.lemma_, (idx, score)) for token in phrase if token.pos_=="NOUN" and token.lemma_ not in non_facet_noun])

        self._facet_dict.update(dict(tmp)) # update the dict with {lemma_word: (idx, score)} # can use score in an advanced version

    def retrieve_one_note(self, idx=-1):
        """
        Args:
        idx     -- index of the relevent note

        Rets:
        str, proper response
        """
        if self.calc_num_notes() == 0: # there is no note
            return random.choice(self._response["retrieve_one_note_sorry"])
        elif idx == np.inf or idx == -np.inf: # the quested note idx is outside of notes length
            return random.choice(self._response["retrieve_one_note_sorry"])
        elif (idx > 0 and idx >= self.calc_num_notes()) or (idx < 0 and -idx > self.calc_num_notes()):
            # the quested note idx is outside of notes length
            return random.choice(self._response["retrieve_one_note_sorry"])
        else:
            return "{}{}".format(random.choice(self._response["retrieve_one_note"]), self.notes[idx])

    def delete_one_note(self, idx=-1):
        """
        Args:
        idx     -- index of the relevent note

        Rets:
        str, proper response
        """
        if self.calc_num_notes() == 0:
            return random.choice(self._response["delete_one_note_sorry"])
        elif idx == np.inf or idx == -np.inf:
            return random.choice(self._response["delete_one_note_sorry"])
        elif (idx > 0 and idx >= self.calc_num_notes()) or (idx < 0 and -idx > self.calc_num_notes()):
            return random.choice(self._response["delete_one_note_sorry"])
        else:
            note_del = self.notes.pop(idx)
            res = random.choice(self._response["delete_one_note"])
            return "{}{}{}' is deleted".format(res[0], note_del, res[1])

    def take_note_with_content(self, sentence):
        """
        call method take_one_note
        """
        return self.take_one_note(sentence)


    def delete_one_note(self, idx=-1):
        """
        Args:
        idx     -- index of the relevent note

        Rets:
        str, proper response
        """
        if self.calc_num_notes() == 0:
            return "Sorry, the note you want to delete doesn't exist."
        elif idx == np.inf or idx == -np.inf:
            return "Sorry, the note you want to delete doesn't exist."
        elif (idx > 0 and idx >= self.calc_num_notes()) or (idx < 0 and -idx > self.calc_num_notes()):
            return "Sorry, the note you want to delete doesn't exist."
        else:
            note_del = self.notes.pop(idx)
            return "That note '{}' is deleted".format(note_del)

    def take_note_with_content(self, sentence):
        """
        call method take_one_note
        """
        return self.take_one_note(sentence)

    def respond(self, sentence):
        """
        This is the only method you are required to support
        for the Conversation class. This method should accept
        some input from the user, and return the output
        from the chatbot.

        Args:
        sentence    -- str, input sentence from user

        Rets:
        response    -- str, proper response
        """
        ############################
        ### Preparation Phrase #####
        ############################
        self.original = sentence
        response = "" # response string, to be returned, capable be appended with multiple responses in advanced version
        sentence = self.prepro_sentence(sentence); # print(sentence)
        idx = None
        non_facet_noun = self._non_facet_noun
        TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')

        if self._ready_take_note: # if ready_take_note is true, the agent will take whatever the sentence is
            return self.take_one_note(self.original)



        ############################
        ### Handling Phrase ########
        ############################
        ### identify questions or not
        question = False
        sentence_lst = self.original.lower().split()
        for word in ["What","what", "what's", "how", "which", "?"]:
            if word in sentence_lst:
                question = True
            if self.original and'?' == self.original[-1]:
                question = True

        if not question:
            sent_tokens = self._sent_tokens.copy() # make a copy of sent_tokens, so that the new sentence won't be stored after calling method
            sent_tokens.append(sentence)
            tfidf = TfidfVec.fit_transform(sent_tokens)
            # Identify intention
            vals = cosine_similarity(tfidf[-1], tfidf) # use tfidf model to identify intention
            idxx = vals.argsort()[0][-2]
            flat = vals.flatten()
            flat.sort()
            req_tfidf = flat[-2]
            intent = self._sent_tokens[idxx]
        else:
            sent_tokens = self._sent_tokens_what.copy() # make a copy of sent_tokens, so that the new sentence won't be stored after calling method
            sent_tokens.append(sentence)
            tfidf = TfidfVec.fit_transform(sent_tokens)
            # Identify intention
            vals = cosine_similarity(tfidf[-1], tfidf) # use tfidf model to identify intention
            idxx = vals.argsort()[0][-2]
            flat = vals.flatten()
            flat.sort()
            req_tfidf = flat[-2]
            intent = self._sent_tokens_what[idxx]
            if "how many" in self.original.lower():
                intent = "total number"



        # Identify ordinal number (with "notes") => to number + "last"/"previous" => retrive/delete notes or just change idx
        sentence_nlp = nlp(sentence)
        if any([True for token in sentence_nlp.ents if token.label_ == 'ORDINAL']): # identify ordinal number
            ordinal = [token for token in sentence_nlp.ents if token.label_ == 'ORDINAL'][-1] # only work on last ordinal number, can do more in advanced version
            ordinal_to_num = {} # map word ordinal number to digit
            for i in range(1, self.calc_num_notes()+1):
                ordinal_key = num2words(i, to="ordinal")
                ordinal_to_num[ordinal_key] = i
                ordinal_to_num[ordinal_key.replace("-", " ")] = i # pay attention to the case whether "-" exists
            idx = ordinal_to_num.get(str(ordinal), np.inf) - 1 # index matching
            if "last" in sentence.split() or "previous" in sentence.split():
                idx = -idx - 1 # index matching

        # Identify facet of a sentence (NOUN, keyword extraction/high tfidf, without "note") => idx
        if any([True for token in sentence_nlp if token.pos_=="NOUN" and token.lemma_ not in non_facet_noun]): # identify facet
            facets = [token.lemma_ for token in sentence_nlp if token.pos_=="NOUN" and token.lemma_ not in non_facet_noun]
            for facet in facets:
                qry = self._facet_dict.get(facet, "Not Found")
                if qry == "Not Found":
                    continue
                else:
                    idx = qry[0]




        ############################
        ### Responding Phrase ######
        ############################
        if(req_tfidf==0): # sentence doesn't fit into any provided intend in chatbot.txt
            response = response + self._response["intent not found"]
            return response
        intent = intent.replace(".", "")
        #print("DEBUG: intent is - " + intent)
        if self.intent[intent] == "take a note": # intent is prepare_take_one_note
            response = response + self.prepare_take_one_note(self.original)
            return response
        elif self.intent[intent] == "total notes": # intent is calc_num_notes
            response = response + "You have {} notes in total".format(self.calc_num_notes())
            return response
        elif self.intent[intent] == "retrieve a note": # intent is retrive_one_note
            if idx == None:
                response = response + self.retrieve_one_note(idx = -1)
            elif idx != None:
                response = response + self.retrieve_one_note(idx = idx)
            else :
                raise NotImplementedError("Invalid Case!")
            return response
        elif self.intent[intent] == "delete a note": # intent is delete_one_note
            if idx == None:
                response = response + self.delete_one_note(idx = -1)
            elif idx != None:
                response = response + self.delete_one_note(idx = idx)
            else :
                raise NotImplementedError("Invalid Case!")
            return response
        elif self.intent[intent] == "take a note with content": # intent is take_one_note_with_content
            return self.take_note_with_content(self.original)
        else :
            response = response + self._response["intent not found"]
            return response
