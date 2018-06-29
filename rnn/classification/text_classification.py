#  Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
"""Example of Estimator for DNN-based text classification with DBpedia data."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import os
import tarfile

from tensorflow.contrib.learn import Experiment
import pandas as pd
import tensorflow as tf
import tensorlayer as tl

from tensorflow.python import debug as tf_debug

hooks = [tf_debug.LocalCLIDebugHook()]

FLAGS = None

MAX_DOCUMENT_LENGTH = 10
EMBEDDING_SIZE = 50
MAX_LABEL=15

DATA_DIR = 'data'
DBPEDIA_URL = 'https://github.com/le-scientifique/torchDatasets/raw/master/dbpedia_csv.tar.gz'
VOCAB_PATH = ''
BATCH_SIZE = 10
CVS_COLUMN_NAME = ['Category','Name','Text']

def load_dataset(file_path, vocab):
    csv = pd.read_csv(file_path, names=CVS_COLUMN_NAME, header=0)
    data = csv.pop('Text').values
    category = csv.pop('Category').values

    text_embedings = []
    for text in data:
        desc = tl.nlp.process_sentence(text, start_word=None, end_word=None,  )
        embedding = tl.nlp.words_to_word_ids(desc, word_to_id= vocab)
        text_embedings.append(embedding)

    text_embedings = tl.prepro.pad_sequences(text_embedings, maxlen=MAX_DOCUMENT_LENGTH)
    category = tf.one_hot(category,depth= MAX_LABEL)
    dataset = tf.data.Dataset.from_tensor_slices(({"Text":text_embedings}, category))
    if FLAGS.small:
        dataset = dataset.shuffle(1000)

    return dataset


def train_input_fn(vocab):
    train_path = os.path.join(sys.path[0], DATA_DIR, 'dbpedia_csv', 'train.csv')

    train_dataset = load_dataset(train_path, vocab)
    train_dataset = train_dataset.repeat(3)
    train_dataset = train_dataset.batch(BATCH_SIZE)

    return train_dataset.make_one_shot_iterator().get_next()


def test_input_fn(vocab):
    test_path = os.path.join(sys.path[0], DATA_DIR, 'dbpedia_csv' ,'test.csv')
    test_dataset = load_dataset(test_path, vocab)
    test_dataset = test_dataset.batch(BATCH_SIZE)
    return test_dataset.make_one_shot_iterator().get_next()


def estimator_spec_for_softmax_classification(logits, labels, mode):
  """Returns EstimatorSpec instance for softmax classification."""
  predicted_classes = tf.argmax(logits, 1)
  predicted_onehot = tf.one_hot(predicted_classes, MAX_LABEL)
  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions={
            'class': predicted_classes,
            'prob': tf.nn.softmax(logits)
        })

  loss = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=logits)
  accuracy = tf.metrics.accuracy(labels=labels, predictions=predicted_onehot)
  tf.summary.scalar('Accuracy', accuracy[1])
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

  eval_metric_ops = {
      'accuracy':accuracy
  }
  return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def bag_of_words_model(features, labels, mode):
  """A bag-of-words model. Note it disregards the word order in the text."""
  bow_column = tf.feature_column.categorical_column_with_identity(
      key='Text', num_buckets=n_words)
  bow_embedding_column = tf.feature_column.embedding_column(
      bow_column, dimension=EMBEDDING_SIZE)
  bow = tf.feature_column.input_layer(
      features, feature_columns=[bow_embedding_column])
  logits = tf.layers.dense(bow, MAX_LABEL, activation=None)

  return estimator_spec_for_softmax_classification(
      logits=logits, labels=labels, mode=mode)


def rnn_model(features, labels, mode):
  """RNN model to predict from sequence of words to a class."""
  # Convert indexes of words into embeddings.
  # This creates embeddings matrix of [n_words, EMBEDDING_SIZE] and then
  # maps word indexes of the sequence into [batch_size, sequence_length,
  # EMBEDDING_SIZE].
  word_vectors = tf.contrib.layers.embed_sequence(
      features['Text'], vocab_size=n_words, embed_dim=EMBEDDING_SIZE)

  # Split into list of embedding per word, while removing doc length dim.
  # word_list results to be a list of tensors [batch_size, EMBEDDING_SIZE].
  word_list = tf.unstack(word_vectors, axis=1)

  # Create a Gated Recurrent Unit cell with hidden size of EMBEDDING_SIZE.
  cell = tf.nn.rnn_cell.GRUCell(EMBEDDING_SIZE)

  # Create an unrolled Recurrent Neural Networks to length of
  # MAX_DOCUMENT_LENGTH and passes word_list as inputs for each unit.
  _, encoding = tf.nn.static_rnn(cell, word_list, dtype=tf.float32)

  # Given encoding of RNN, take encoding of last step (e.g hidden size of the
  # neural network of last step) and pass it as features for softmax
  # classification over output classes.
  logits = tf.layers.dense(encoding, MAX_LABEL, activation=None)
  return estimator_spec_for_softmax_classification(
      logits=logits, labels=labels, mode=mode)


def main(_):
    global n_words
    tf.logging.set_verbosity(tf.logging.INFO)

    if not tf.gfile.Exists(sys.path[0] +'/'+ DATA_DIR):
        tf.gfile.MakeDirs(sys.path[0] +'/'+ DATA_DIR)
        fname = os.path.join(sys.path[0], DATA_DIR, DBPEDIA_URL.split('/')[-1])
        tf.keras.utils.get_file(fname, DBPEDIA_URL, extract = True)
        zipped_file = tarfile.open(fname,'r:*')
        zipped_file.extractall(sys.path[0] +'/'+ DATA_DIR)

        print('Successfully downloaded', fname)


    train_path = os.path.join(sys.path[0] ,DATA_DIR, 'dbpedia_csv', 'train.csv')
    vocab = tl.nlp.build_vocab(tl.nlp.read_words(train_path))
    n_words = len(vocab)
    vocab['UNK'] = n_words
    n_words += 1


    # Build model
    # Switch between rnn_model and bag_of_words_model to test different models.
    model_fn = rnn_model
    if FLAGS.bow_model:
        # Subtract 1 because VocabularyProcessor outputs a word-id matrix where word
        # ids start from 1 and 0 means 'no word'. But
        # categorical_column_with_identity assumes 0-based count and uses -1 for
        # missing word.
        model_fn = bag_of_words_model
    classifier = tf.estimator.Estimator(model_fn=model_fn, model_dir='ckpt/')

    # Train.
    classifier.train(input_fn=lambda:train_input_fn(vocab), steps=1000)

    # Score with tensorflow.
    scores = classifier.evaluate(input_fn=lambda: test_input_fn(vocab))

    #debug
    '''
    
    ex = Experiment(classifier,
                    train_input_fn=lambda:train_input_fn(vocab),
                    eval_input_fn=lambda: test_input_fn(vocab),
                    train_steps=1000,
                    eval_delay_secs=0,
                    eval_steps=1,
                    train_monitors=hooks,
                    eval_hooks=hooks)

    ex.train()
    scores = ex.evaluate()
    '''

    print('Accuracy: {0:f}'.format(scores['accuracy']))

    # Predict.
    #predictions = classifier.predict(input_fn=lambda: test_input_fn(vocab))
    #y_predicted = np.array(list(p['class'] for p in predictions))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
      '--small',
      default=False,
      help='load 1000 record only.')
    parser.add_argument(
      '--bow_model',
      default=True,
      help='Run with BOW model instead of RNN.')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
