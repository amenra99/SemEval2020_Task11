import sys
import glob
import os.path
import random
import numpy as np
import logging
import argparse
import tensorflow as tf
import sklearn.metrics
# from keras import backend as K
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, TFBertForTokenClassification, TFBertForSequenceClassification
import transformers
# from transformers import RobertaConfig, RobertaTokenizer, TFRobertaForSequenceClassification, TFBertModel


# from keras.preprocessing.sequence import pad_sequences
# input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", value=0, truncating="post", padding="post")

random.seed(2019)

logger = logging.getLogger('propaganda_predict_TC')

PROP_CLASS = ['Appeal_to_Authority', 'Appeal_to_fear-prejudice', 'Bandwagon,Reductio_ad_hitlerum',
                'Black-and-White_Fallacy', 'Causal_Oversimplification', 'Doubt', 'Exaggeration,Minimisation',
                'Flag-Waving', 'Loaded_Language', 'Name_Calling,Labeling', 'Repetition', 'Slogans',
                'Thought-terminating_Cliches', 'Whataboutism,Straw_Men,Red_Herring']


PRETRAINED_MODEL = 'bert-base-uncased'
MAX_TOKEN = 128
EPOCHS = 5
BATCH_SIZE = 64 

tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL)
logger.info("Bert pretrained model: {0}".format(PRETRAINED_MODEL))


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
            articles[article_id]['content'] = content.lower()

    return articles



# load label file
def loadLabelFile(articles, labelFilename, hasLabel=True) :
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
                span = (int(line[-2]), int(line[-1]))    

                if 'spans' in articles[label_id]:
                    articles[label_id]['spans'].append(span)
                else:
                    articles[label_id]['spans'] = [span]

                if hasLabel:
                    propType = PROP_CLASS.index(line[1])
                    assert propType not in PROP_CLASS, "{0} is not class".format(line[1])

                    # make one-hot
                    oneHot = np.zeros(len(PROP_CLASS))
                    oneHot[propType] = 1
                    
                    # propClassList.append(propType)
                    propClassList.append(oneHot)

                propText = articles[label_id]['content'][span[0]:span[1]]
                
                ids = np.zeros(MAX_TOKEN, dtype=int)
                tokens = tokenizer.tokenize(propText)
                tokens.insert(0, '[CLS]')
                if len(tokens) <= MAX_TOKEN:
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
     
    return propTokenIdList, propClassList



# build data
def buildData(article_dir, label_file, split=0.2):
    articles = loadArticleFiles(article_dir)
    propTokenIdList, propClassList = loadLabelFile(articles, label_file)

    data = list(zip(propTokenIdList, propClassList))
    # data = data[0:20]
    random.shuffle(data)

    tr_articles, val_articles = train_test_split(data, random_state=2019, test_size=split) # split data into train & validation
    tr_tokenIds, tr_propClass = zip(*tr_articles)
    val_tokenIds, val_propClass = zip(*val_articles)

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
    
    class BERT(transformers.TFBertModel):
       def __init__(self, config, *inputs, **kwargs):
           super(BERT, self).__init__(config, *inputs, **kwargs)
           self.bert.call = tf.function(self.bert.call)

    token_inputs = tf.keras.Input(shape=(None,), name='input_ids', dtype='int32')
    mask_inputs = tf.keras.Input(shape=(None,), name='attention_mask', dtype='int32')
    segment_inputs = tf.keras.Input(shape=(None,), name='token_type_ids', dtype='int32')

    bert = BERT.from_pretrained(PRETRAINED_MODEL, num_labels=len(PROP_CLASS))
    token_encodings = bert([token_inputs, mask_inputs, segment_inputs])[0]
    sentence_encoding = tf.squeeze(token_encodings[:, 0:1, :], axis=1)
    sentence_encoding = tf.keras.layers.Dropout(0.2)(sentence_encoding)
    outputs = tf.keras.layers.Dense(len(PROP_CLASS), activation='sigmoid', name='outputs')(sentence_encoding)

    model = tf.keras.Model(inputs=[token_inputs, mask_inputs, segment_inputs], outputs=[outputs])
    optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5, epsilon=1e-08, clipnorm=1.0)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    return model



def predictFromModel(articleFile, testFile, modelFile):

    articles = loadArticleFiles(articleFile)
    propTokenIdList, _ = loadLabelFile(articles, testFile, hasLabel=False)

    test_masks = np.copy(propTokenIdList)
    test_masks[test_masks>0] = 1

    test_x = dict(
       input_ids = np.array(propTokenIdList, dtype=np.int32),
       attention_mask = np.array(test_masks, dtype=np.int32),
       token_type_ids = np.zeros(shape=(len(propTokenIdList), MAX_TOKEN)))


    bertModel = getBertModel()
    bertModel.load_weights(modelFile)
    print('model loaded')

    result = bertModel.predict(test_x, batch_size=BATCH_SIZE)
    np.save("npy/TC-roBERTa-binary-result.npy", result)
    pred = np.argmax(result, axis=1)
    print('npy_saved')


    writeSubmission(testFile, pred)


def predictFromMultiModel(articleFile, testFile, modelFile, start=1, end=10):

    articles = loadArticleFiles(articleFile)
    propTokenIdList, _ = loadLabelFile(articles, testFile, hasLabel=False)

    test_masks = np.copy(propTokenIdList)
    test_masks[test_masks>0] = 1

    test_x = dict(
       input_ids = np.array(propTokenIdList, dtype=np.int32),
       attention_mask = np.array(test_masks, dtype=np.int32),
       token_type_ids = np.zeros(shape=(len(propTokenIdList), MAX_TOKEN)))

    bertModel = getBertModel()
    for i in range(start, end + 1):
        bertModel.load_weights(modelFile.format(i))
        print(modelFile.format(i) + ' model loaded')

        result = bertModel.predict(test_x, batch_size=BATCH_SIZE)
        np.save("npy/TC-roBERTa-binary-result-{0:02d}.npy".format(i), result)
        print("npy/TC-roBERTa-binary-result-{0:02d}.npy".format(i) + 'npy_saved')
        pred = np.argmax(result, axis=1)
        
        writeSubmission(testFile, pred, i)


def predictFromNPY(testFile, npyPath):

    results = np.load(npyPath)
    # print(results[0:5])
    # results = weightedResult(results)
    # print(results[0:5])
    pred = np.argmax(results, axis=1)

    writeSubmission(testFile, pred, 0)

def weightedResult(results):
    weightVector = [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2]
    for i, weight in enumerate(weightVector):
        results[:, i] = results[:, i] * weight

    return results
    

def writeSubmission(testFile, pred, num):
    with open(testFile, 'r', encoding='utf-8') as f:
        with open('submission_TC/submission_TC-binaly-{0:02d}.txt'.format(num), 'w', encoding='utf-8') as out:
            lines = f.readlines()
            assert len(lines) == len(pred), 'wrong test file'

            for i, line in enumerate(lines):
                row = line.rstrip().split('\t')
                out.write("{0}\t{1}\t{2}\t{3}\n".format(row[0], PROP_CLASS[pred[i]], row[2], row[3]))

    print('Building submission file completed')



def main(mode='model', task='single'):
    logger.setLevel(logging.INFO)
    np.set_printoptions(threshold=sys.maxsize)

    train_article_dir = "datasets/train-articles"
    dev_article_dir = "datasets/dev-articles"

    label_File = "datasets/train-task2-TC.labels"
    dev_label_file = "datasets/dev-task-TC-template.out"

    if mode == 'model':
        if task == 'multi':
            modelFile = 'model-TC-roBERTa/cp-base-{0:04d}.ckpt'
            predictFromMultiModel(dev_article_dir, dev_label_file, modelFile, 1, EPOCHS)
        else:
            modelFile = 'model-TC/best/cp-0018.ckpt'
            print('predict from model: ' + modelFile)
            predictFromModel(dev_article_dir, dev_label_file, modelFile)
    elif mode == 'npy':
        npyFile = 'npy/TC-roBERTa-binary-result-05.npy'
        print('predict from npy: ' + npyFile)
        predictFromNPY(dev_label_file, npyFile)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", nargs='?', default="npy")
    parser.add_argument("task", nargs='?', default="multi")
    args = parser.parse_args()

    main(args.mode, args.task)


