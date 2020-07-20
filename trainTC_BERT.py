import sys
import glob
import os.path
import random
import numpy as np
import logging
import argparse
import tensorflow as tf
import sklearn.metrics
from sklearn.utils import class_weight
# from keras import backend as K
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, TFBertForTokenClassification, TFBertForSequenceClassification
# from transformers import RobertaConfig, RobertaTokenizer, TFRobertaForSequenceClassification

# from keras.preprocessing.sequence import pad_sequences
# input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", value=0, truncating="post", padding="post")

random.seed(2019)

logger = logging.getLogger('propaganda_train_TC')

PROP_CLASS = ['Appeal_to_Authority', 'Appeal_to_fear-prejudice', 'Bandwagon,Reductio_ad_hitlerum',
                'Black-and-White_Fallacy', 'Causal_Oversimplification', 'Doubt', 'Exaggeration,Minimisation',
                'Flag-Waving', 'Loaded_Language', 'Name_Calling,Labeling', 'Repetition', 'Slogans',
                'Thought-terminating_Cliches', 'Whataboutism,Straw_Men,Red_Herring']


PRETRAINED_MODEL = 'bert-large-uncased'
MAX_TOKEN = 128
EPOCHS = 5
BATCH_SIZE = 16

tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL)

logger.info("Bert pretrained model: {0}".format(PRETRAINED_MODEL))

ep = 5


# load article files
def loadArticleFiles(article_dir):
    articles = {}

    article_files = glob.glob(os.path.join(article_dir, "*.txt"))
    for filename in article_files:
        with open(filename, "r", encoding="utf-8") as f:
            content = f.read()
            article_id = os.path.basename(filename).split(".")[0][7:]

            if 'uncased' in PRETRAINED_MODEL:
                content = content.lower()
            
            articles[article_id] = {}
            articles[article_id]['content'] = content

    return articles



# load label file
def loadLabelFile(articles, labelFilename, hasLabel=True) :
    articleIdList = []
    propTokenIdList = []
    propClassList = []
    protTextList = []

    with open(labelFilename, 'r', encoding="utf-8") as f:
        spans = []
        label_id = ''
        for line in f.readlines():
            line = line.strip().split('\t')
            if len(line) > 0:
                label_id = line[0]
                articleIdList.append(label_id)
                span = (int(line[-2]), int(line[-1]))    

                if 'spans' in articles[label_id]:
                    articles[label_id]['spans'].append(span)
                else:
                    articles[label_id]['spans'] = [span]

                if hasLabel:
                    propType = PROP_CLASS.index(line[1])
                    assert propType not in PROP_CLASS, "{0} is not class".format(line[1])

                    # make one-hot
                    # oneHot = np.zeros(len(PROP_CLASS))
                    # oneHot[propType] = 1
                    
                    propClassList.append(propType)
                    # propClassList.append(oneHot)

                propText = articles[label_id]['content'][span[0]:span[1]]
                
                ids = np.zeros(MAX_TOKEN, dtype=int)
                tokens = tokenizer.tokenize(propText)

                tokens.insert(0, '[CLS]')
                if len(tokens) < MAX_TOKEN:
                    tokens.append('[SEP]')
                else:
                    tokens = tokens[0:MAX_TOKEN]
                    tokens[-1] = '[SEP]'

                tokenIds = tokenizer.convert_tokens_to_ids(tokens)
                ids[0:len(tokenIds)] = tokenIds
                propTokenIdList.append(ids)

                # encodedText = tokenizer.encode_plus(propText)
                # propTokenIdList.append(encodedText)
                
                protTextList.append(propText)
                
    return articleIdList, propTokenIdList, propClassList

import imblearn

# build data
def buildDataAll(article_dir, label_file):
    articles = loadArticleFiles(article_dir)
    articleIdList, propTokenIdList, propClassList = loadLabelFile(articles, label_file)

    assert len(propTokenIdList) == len(propClassList), 'x != y'

    data = list(zip(propTokenIdList, propClassList))
    # data = data[0:20]
    random.shuffle(data)

    tr_tokenIds, tr_propClass = zip(*data)

    smote = imblearn.over_sampling.SMOTE('auto')
    x_sm, y_sm = smote.fit_sample(tr_tokenIds, tr_propClass)

    print(len(tr_tokenIds), len(tr_propClass), len(x_sm), len(y_sm))

    np.save('x_sm.npy', np.array(x_sm, dtype=np.int32))
    np.save('y_sm.npy', np.array(y_sm, dtype=np.int32))



    
    tr_masks = np.copy(tr_tokenIds)
    tr_masks[tr_masks>0] = 1

    train_x = dict(
       input_ids = np.array(tr_tokenIds, dtype=np.int32),
       attention_mask = np.array(tr_masks, dtype=np.int32),
       token_type_ids = np.zeros(shape=(len(tr_tokenIds), MAX_TOKEN)))
    train_y = np.array(tr_propClass, dtype=np.int32)

    return train_x, train_y



# build data
def buildData(article_dir, label_file, split=0.2):
    articles = loadArticleFiles(article_dir)
    articleIdList, propTokenIdList, propClassList = loadLabelFile(articles, label_file)

    data = list(zip(articleIdList, propTokenIdList, propClassList))
    # data = data[0:20]
    random.shuffle(data)

    tr_articles, val_articles = train_test_split(data, random_state=2019, test_size=split) # split data into train & validation
    tr_articleIds, tr_tokenIds, tr_propClass = zip(*tr_articles)
    val_articleIds, val_tokenIds, val_propClass = zip(*val_articles)

    # print(val_articleIds)
    # print(val_propClass)

    tr_masks = np.copy(tr_tokenIds)
    tr_masks[tr_masks>0] = 1

    val_masks = np.copy(val_tokenIds)
    val_masks[val_masks>0] = 1

    train_x = dict(
       input_ids = np.array(tr_tokenIds, dtype=np.int32),
       attention_mask = np.array(tr_masks, dtype=np.int32),
       token_type_ids = np.zeros(shape=(len(tr_tokenIds), MAX_TOKEN)))
    train_y = np.array(tr_propClass, dtype=np.int32)

    val_x = dict(
       input_ids = np.array(val_tokenIds, dtype=np.int32),
       attention_mask = np.array(val_masks, dtype=np.int32),
       token_type_ids = np.zeros(shape=(len(val_tokenIds), MAX_TOKEN)))
    val_y = np.array(val_propClass, dtype=np.int32)

    return train_x, train_y, val_x, val_y



def getBertModel():
    # def f1(y_true, y_pred):
    #     def recall(y_true, y_pred):
    #         true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    #         possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    #         recall = true_positives / (possible_positives + K.epsilon())
    #         return recall

    #     def precision(y_true, y_pred):
    #         true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    #         predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    #         precision = true_positives / (predicted_positives + K.epsilon())
    #         return precision

    #     precision = precision(y_true, y_pred)
    #     recall = recall(y_true, y_pred)
    #     return 2*((precision*recall)/(precision+recall+K.epsilon()))

    print(PRETRAINED_MODEL + ' loaded')
    bertModel = TFBertForSequenceClassification.from_pretrained(PRETRAINED_MODEL, num_labels=len(PROP_CLASS))
    # bertModel = TFBertForSequenceClassification.from_pretrained('./model-TC-roBERTa/', num_labels=len(PROP_CLASS))
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5, epsilon=1e-08, clipnorm=1.0)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')

    # loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    # metric = tf.keras.metrics.CategoricalAccuracy('categorical_accuracy')
    # loss = "binary_crossentropy"
    # metric = "accuracy"

    # different cross entropy... X
    # add context sentence

    # bertModel.compile(optimizer=optimizer, loss=loss, metrics=[metric, f1])
    bertModel.compile(optimizer=optimizer, loss=loss, metrics=[metric])

    return bertModel

def trainBert(train_x, train_y):

    # checkpoint_path = "model-TC-BERT/cp-base-sparse-{epoch:04d}.ckpt"
    # cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, verbose=1, save_weights_only=True, save_freq=1)
    
    cw = class_weight.compute_class_weight('balanced', np.unique(train_y), train_y)
    

    bertModel = getBertModel()
    history = bertModel.fit(x=train_x, y=train_y, 
                  epochs=EPOCHS,
                  batch_size=BATCH_SIZE,
                #   callbacks=[cp_callback]
                  class_weight=cw 
                  )

    # print(history.history['val_f1'])

    bertModel.save_pretrained('./model-TC-BERT/{0}/'.format(ep))

    return history



def main():
    logger.setLevel(logging.INFO)
    np.set_printoptions(threshold=sys.maxsize)

    train_article_dir = "datasets/train-articles"
    dev_article_dir = "datasets/dev-articles"

    label_File = "datasets/train-task2-TC.labels"
    dev_label_file = "datasets/dev-task-TC-template.out"

    ## train all data
    tr_x, tr_y = buildDataAll(train_article_dir, label_File)
    print("Train: {0}".format(len(tr_y)))
    
    # trainBert(tr_x, tr_y)

    ## split data    
    # tr_x, tr_y, val_x, val_y = buildData(train_article_dir, label_File)
    # print("Train: {0}, Val: {1}".format(len(tr_y), len(val_y)))
    # trainValBert(tr_x, tr_y, val_x, val_y)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("task", nargs='?', default="TC")
    parser.add_argument("ep", nargs='?', default="5")
    args = parser.parse_args()
    
    ep = int(args.ep)
    
    PRETRAINED_MODEL = 'bert-large-uncased' if ep == 5 else './model-TC-BERT/{0}/'.format(ep - 5)

    main()


