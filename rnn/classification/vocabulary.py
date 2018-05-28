import tensorlayer as tl
import tensorflow as tf

MAX_DOCUMENT_LENGTH = 10
EMBEDING_SIZE = 300 #词向量的长度

class Vocabulary(object):

    def __init__(self, filename):
        self.vocab = tl.nlp.build_vocab(tl.nlp.read_words(filename))


    def parseLine(self, text, category):
        data = tl.nlp.basic_tokenizer(text)
        text_embeddings = [self.vocab[word] for word in data if word in self.vocab]
        return text_embeddings, category