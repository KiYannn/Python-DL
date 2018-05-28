import tensorlayer as tl
import tensorflow as tf

MAX_DOCUMENT_LENGTH = 10
EMBEDING_SIZE = 300 #词向量的长度

class Vocabulary(object):

    def __init__(self, filename):
        self.vocab = tl.nlp.build_vocab(tl.nlp.read_words(filename))


    def text_to_word_ids(self, text):
        data = tl.nlp.basic_tokenizer(text[0])
        return [self.vocab[word] for word in data if word in self.vocab]

    def parseLine(self, line):
        fields = tf.decode_csv([line], record_defaults=[[0], [''], ['']])
        text_embeddings = self.text_to_word_ids(fields[2])
        return text_embeddings, fields[0]