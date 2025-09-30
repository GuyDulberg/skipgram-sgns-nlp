# you can use these packages (uncomment as needed)
import pickle
import pandas as pd
import numpy as np
import os,time, re, sys, random, math, collections, nltk
from collections import defaultdict
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
#nltk.download('punkt')
#nltk.download('stopwords')

#static functions
def who_am_i():  # this is not a class method
    """Returns a dictionary with your name, id number and email. keys=['name', 'id','email']
        Make sure you return your own info!
    """
    return {'name': 'Guy Dulberg', 'id': '206562977', 'email': 'dulbergg@post.bgu.ac.il'}

def helper_normalize_sentence(sentence):
    clean_sentence = re.sub(r"[^0-9A-Za-z\s]", "", sentence)
    clean_sentence = re.sub(r'([a-z])([A-Z])', r'\1 \2', clean_sentence)
    clean_sentence = clean_sentence.lower().strip()
    clean_sentence = re.sub(r'\s+', ' ', clean_sentence)
    return clean_sentence


def normalize_text(fn):
    """ Loading a text file and normalizing it, returning a list of sentences.


    Args:
        fn: full path to the text file to process
    """
    sentences = []
    with open(fn, 'r', encoding ='utf-8') as file:
        text = file.read()
    normalize_sentences = nltk.sent_tokenize(text)
    for sentence in normalize_sentences:
        clean_sentence = re.sub(r'[^a-zA-Z0-9\s]', '', sentence)
        lower_clean_sentence = clean_sentence.lower().strip()
        if len(lower_clean_sentence) > 0 :
            sentences.append(helper_normalize_sentence(lower_clean_sentence))
            #print(helper_normalize_sentence(sentence))
    return sentences


    #TODO





def sigmoid(x): return 1.0 / (1 + np.exp(-x))




def load_model(fn):
    """ Loads a model pickle and return it.


    Args:
        fn: the full path to the model to load.
    """
    with open(fn, 'rb') as file:
        sg_model = pickle.load(file)
    return sg_model


    #TODO



class SkipGram:


    def __init__(self, sentences, d=100, neg_samples=4, context=4, word_count_threshold=5):
        self.sentences = sentences
        self.d = d  # embedding dimension
        self.neg_samples = neg_samples  # num of negative samples for one positive sample
        self.context = context #the size of the context window (not counting the target word)
        self.word_count_threshold = word_count_threshold#ignore low frequency words (appearing under the threshold)

        self.word_count = defaultdict(int)
        for sentence in sentences:
            for word in sentence.split():
                self.word_count[word] += 1
        self.word_count = {word: count for word, count in self.word_count.items() if count >= self.word_count_threshold}
        self.word_count = {word: count for word, count in self.word_count.items() if count >= self.word_count_threshold}
        self.word_to_index = {word: idx for idx, word in enumerate(self.word_count)}
        self.index_to_word = {idx: word for word, idx in self.word_to_index.items()}
        vocab_size = len(self.word_to_index)
        self.T = np.random.rand(self.d, vocab_size)
        self.C = np.random.rand(vocab_size, self.d)
        # Tips:
        # 1. It is recommended to create a word:count dictionary
        # 2. It is recommended to create a word-index map


        # TODO

    def normalize_sentence(self, sentence):
        clean_sentence = re.sub(r"[^0-9A-Za-z ]", "", sentence)
        clean_sentence = re.sub(r'([a-z])([A-Z])', r'\1 \2', clean_sentence)
        clean_sentence = clean_sentence.lower().strip()
        clean_sentence = re.sub(r'\s+', ' ', clean_sentence)
        return clean_sentence
    def compute_similarity(self, w1, w2):
        """ Returns the cosine similarity (in [0,1]) between the specified words.
        Args:
            w1: a word
            w2: a word
        Retunrns: a float in [0,1]; defaults to 0.0 if one of specified words is OOV.
    """
        if w1 not in self.word_to_index:
            return 0
        if w2 not in self.word_to_index:
            return 0
        idx1 = self.word_to_index[w1]
        idx2 = self.word_to_index[w2]
        vector_word1 = self.T[:, idx1]
        vector_word2 = self.T[:, idx2]
        dot_product = np.dot(vector_word1, vector_word2)
        norm_vector1 = np.linalg.norm(vector_word1)
        norm_vector2 = np.linalg.norm(vector_word2)
        if norm_vector1 == 0 or norm_vector2 ==0 :
            return 0
        similarity = dot_product/ (norm_vector1 * norm_vector2)
        return similarity
        #sim  = 0.0 # default
        #TODO


        #return sim # default


    def get_closest_words(self, w, n=5):
        """Returns a list containing the n words that are the closest to the specified word.


        Args:
            w: the word to find close words to.
            n: the number of words to return. Defaults to 5.
        """
        similarities_calculations = []
        if w not in self.word_to_index :
            return []
        idx = self.word_to_index[w]
        target_vector = self.T[:, idx]
        for word,index in self.word_to_index.items():
            if word != w:
                helper_vector = self.T[:, index]
                dot_product = np.dot(target_vector, helper_vector)
                norm_target = np.linalg.norm(target_vector)
                norm_other = np.linalg.norm(helper_vector)
                if norm_target == 0:
                    return 0
                if norm_other == 0:
                    return 0
                else:
                    similarity = dot_product/ (norm_other * norm_target)
                similarities_calculations.append((word, similarity))
        similarities_calculations.sort(key=lambda x: x[1], reverse=True)
        closest_words = [word for word, _ in similarities_calculations]
        return closest_words[0:n]

    def create_context_pairs(self):
        context_pairs = []
        for sentence in self.sentences:
            clean_sentence = self.normalize_sentence(sentence)
            words = clean_sentence.split()
            for i, target_word in enumerate(words):
                if target_word not in self.word_to_index:
                    continue
                target_index = self.word_to_index[target_word]
                start = max(0, i-self.context)
                end = min(len(words), i + self.context + 1)
                for j in range(start, end):
                    if i != j :
                        if words[j] in self.word_to_index:
                            context_index = self.word_to_index[words[j]]
                            context_pairs.append((target_index, context_index))
        return context_pairs


    def negative_sampling(self, target_word, num_samples = 5):
        if not target_word :
            return []
        if target_word not in self.word_to_index :
            return []
        negative_samples = []
        all_words = list(self.word_to_index.keys())
        all_words.remove(target_word)
        negative_samples = random.sample(all_words, num_samples)
        negative_indices = [self.word_to_index[word] for word in negative_samples]
        return negative_indices

    def load_and_normalize_text(self, file_path):
        with open(file_path, 'r', encoding = 'utf-8') as file :
            text = file.read()
        sentences = []
        tokenize_between_lines = text.split('\n\n')
        for paragraph in tokenize_between_lines:
            tokenize_sentences = nltk.sent_tokenize(paragraph)
            for sentence in tokenize_sentences:
                clean_sentence = self.normalize_sentence(sentence)
                if len(clean_sentence) > 0:
                    sentences.append(clean_sentence)
        return sentences

    def learn_embeddings(self, step_size=0.0001, epochs=50, early_stopping=3, model_path=None):
        """Returns a trained embedding models and saves it in the specified path


        Args:
            step_size: step size for  the gradient descent. Defaults to 0.0001
            epochs: number or training epochs. Defaults to 50
            early_stopping: stop training if the Loss was not improved for this number of epochs
            model_path: full path (including file name) to save the model pickle at.
        """
        context_pairs = self.create_context_pairs()

        no_improvement = 0
        last_loss = float('inf')

        for epoch in range(epochs):
            total_loss = 0
            random.shuffle(context_pairs)
            for target_index, context_index in context_pairs:
                v_t = self.T[:, target_index]
                v_c = self.C[context_index, :]

                # Positive sample
                score_pos = sigmoid(np.dot(v_t, v_c))
                grad_pos = step_size * (1 - score_pos)
                self.T[:, target_index] += grad_pos * v_c
                self.C[context_index, :] += grad_pos * v_t
                total_loss -= np.log(score_pos + 1e-7)

                # Negative samples
                neg_indices = self.negative_sampling(self.index_to_word[target_index], self.neg_samples)
                for neg_index in neg_indices:
                    v_neg = self.C[neg_index, :]
                    score_neg = sigmoid(-np.dot(v_t, v_neg))
                    grad_neg = step_size * (-score_neg)
                    self.T[:, target_index] -= grad_neg * v_neg
                    self.C[neg_index, :] -= grad_neg * v_t
                    total_loss -= np.log(score_neg + 1e-7)

            if total_loss >= last_loss:
                no_improvement += 1
            else:
                no_improvement = 0
            last_loss = total_loss

            if no_improvement >= early_stopping:
                break

            if model_path:
                with open(model_path, 'wb') as f:
                    pickle.dump(self, f)



        #vocab_size = ... #todo: set to be the number of words in the model (how? how many, indeed?)
        #T = np.random.rand(self.d, vocab_size) # embedding matrix of target words
        #C = np.random.rand(vocab_size, self.d)  # embedding matrix of context words


        #tips:
        # 1. have a flag that allows printing to standard output so you can follow timing, loss change etc.
        # 2. print progress indicators every N (hundreds? thousands? an epoch?) samples
        # 3. save a temp model after every epoch
        # 4.1 before you start - have the training examples ready - both positive and negative samples
        # 4.2. it is recommended to train on word indices and not the strings themselves.


        # TODO


        #return T,C


    def combine_vectors(self, T, C, combo=0, model_path=None):
        """Returns a single embedding matrix and saves it to the specified path


        Args:
            T: The learned targets (T) embeddings (as returned from learn_embeddings())
            C: The learned contexts (C) embeddings (as returned from learn_embeddings())
            combo: indicates how wo combine the T and C embeddings (int)
                   0: use only the T embeddings (default)
                   1: use only the C embeddings
                   2: return a pointwise average of C and T
                   3: return the sum of C and T
                   4: concat C and T vectors (effectively doubling the dimention of the embedding space)
            model_path: full path (including file name) to save the model pickle at.
        """
        # TODO
        if combo == 0:
            V = T
        elif combo == 1:
            V = C
        elif combo == 2:
            V = (T + C.T) / 2
        elif combo == 3:
            V = T + C.T
        elif combo == 4:
            V = np.concatenate((T, C.T), axis = 0)
        else:
            raise ValueError("Invalid combo value")
        if model_path:
            with open(model_path, 'wb') as file:
                pickle.dump(V, file)
        return V


        #return V


    def find_analogy(self, w1,w2,w3):
        """Returns a word (string) that matches the analogy test given the three specified words.
           Required analogy: w1 to w2 is like ____ to w3.


        Args:
             w1: first word in the analogy (string)
             w2: second word in the analogy (string)
             w3: third word in the analogy (string)
        """
        if w1 not in self.word_to_index or w2 not in self.word_to_index or w3 not in self.word_to_index:
            return None

        idx1 = self.word_to_index[w1]
        idx2 = self.word_to_index[w2]
        idx3 = self.word_to_index[w3]

        analogy_vector = self.T[:, idx1] - self.T[:, idx2] + self.T[:, idx3]

        best_word = None
        best_similarity = -1

        for word, idx in self.word_to_index.items():
            if word not in {w1, w2, w3}:
                vector = self.T[:, idx]
                cosine_sim = np.dot(analogy_vector, vector) / (
                            np.linalg.norm(analogy_vector) * np.linalg.norm(vector) + 1e-7)

                if cosine_sim > best_similarity:
                    best_similarity = cosine_sim
                    best_word = word
        return best_word

        #TODO


        #return w


    def test_analogy(self, w1, w2, w3, w4, n=1):
        """Returns True if sim(w1-w2+w3, w4)@n; Otherwise return False.
            That is, returning True if w4 is one of the n closest words to the vector w1-w2+w3.
            Interpretation: 'w1 to w2 is like w4 to w3'


        Args:
             w1: first word in the analogy (string)
             w2: second word in the analogy (string)
             w3: third word in the analogy (string)
             w4: forth word in the analogy (string)
             n: the distance (work rank) to be accepted as similarity
            """
        if w1 not in self.word_to_index or w2 not in self.word_to_index or w3 not in self.word_to_index or w4 not in self.word_to_index:
            #print(f"One of the words '{w1}', '{w2}', '{w3}', or '{w4}' is not in the vocabulary.")
            return False

        idx1 = self.word_to_index[w1]
        idx2 = self.word_to_index[w2]
        idx3 = self.word_to_index[w3]

        analogy_vector = self.T[:, idx1] - self.T[:, idx2] + self.T[:, idx3]

        similarities = []
        for word, idx in self.word_to_index.items():
            if word not in {w1, w2, w3}:
                vector = self.T[:, idx]
                cosine_sim = np.dot(analogy_vector, vector) / (
                        np.linalg.norm(analogy_vector) * np.linalg.norm(vector) + 1e-7)
                similarities.append((word, cosine_sim))
        similarities.sort(key=lambda x: x[1], reverse=True)
        for word, _ in similarities[:n]:
            if word == w4:
                return True
        return False

        # TODO


        #return False

