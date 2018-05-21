import os
import tensorflow as tf

DATA_DIR = 'data'
DBPEDIA_URL = 'https://github.com/le-scientifique/torchDatasets/raw/master/dbpedia_csv.tar.gz'

def __parseLine(line):
    items = line.split(',')
    return items[2], items[0]

def load_dbpedia(size='small'):
    """Get DBpedia datasets from CSV files."""
    if not tf.gfile.Exists(DATA_DIR):
        tf.gfile.MakeDirs(DATA_DIR)
        fname = os.path.join(DATA_DIR, DBPEDIA_URL.split('/')[-1])
        tf.keras.utils.get_file(fname, DBPEDIA_URL, extract = True)
        print('Successfully downloaded', fname)

    train_path = os.path.join(DATA_DIR, 'dbpedia_csv','train.csv')
    test_path = os.path.join(DATA_DIR, 'dbpedia_csv' ,'test.csv')

    train_dataset = tf.data.TextLineDataset(train_path).map(__parseLine())
    test_dataset = tf.data.TextLineDataset(test_path).map(__parseLine())


    if size == 'small':
        train_dataset = train_dataset.shuffle(1000)
        test_dataset = test_dataset.shuffle(1000)

    return train_dataset, test_dataset