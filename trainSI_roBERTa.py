import sys
import glob
import os.path
import random
import numpy as np
import logging
import argparse
import tensorflow as tf
import sklearn.metrics
# import unicodedata
#from keras import backend as K
from sklearn.model_selection import train_test_split
from transformers import RobertaConfig, RobertaTokenizer, TFRobertaForTokenClassification

# from keras.preprocessing.sequence import pad_sequences
# input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", value=0, truncating="post", padding="post")

random.seed(2019)

logger = logging.getLogger('propaganda_train_SI_roBERTa')

PRETRAINED_MODEL = 'roberta-base'
MAX_TOKEN = 512
EPOCHS = 10
BATCH_SIZE = 8 

config = RobertaConfig.from_pretrained(PRETRAINED_MODEL)

tokenizer = RobertaTokenizer.from_pretrained(PRETRAINED_MODEL)

logger.info("Bert pretrained model: {0}".format(PRETRAINED_MODEL))


# load article files
def loadArticleFiles(article_dir):
    articles = {}

    article_files = glob.glob(os.path.join(article_dir, "*.txt"))
    for filename in article_files:
        with open(filename, "r", encoding="utf-8") as f:
            content = f.read()
            article_id = os.path.basename(filename).split(".")[0][7:]

            # normalizedContent = unicodedata.normalize('NFKD', content)#.encode('ascii', 'ignore')#.decode("utf-8", "surrogatepass")
            # # normalizedContent = unidecode.unidecode(content)  # repalce accent char to normal char
            # assert len(content) == len(normalizedContent), (article_id, len(content), len(normalizedContent))

            if 'uncased' in PRETRAINED_MODEL:
                content = content.lower()

                content = content.replace('á', 'a')
                content = content.replace('à', 'a')
                content = content.replace('ã', 'a')
                content = content.replace('ä', 'a')
                content = content.replace('õ', 'o')
                content = content.replace('ö', 'o')
                content = content.replace('ò', 'o')
                content = content.replace('ó', 'o')
                content = content.replace('ó', 'o')
                content = content.replace('ç', 'c')
                content = content.replace('é', 'e')
                content = content.replace('ê', 'e')
                content = content.replace('ñ', 'n')
                content = content.replace('ï', 'i')
                content = content.replace('í', 'i')
                content = content.replace('ü', 'u')
                
            articles[article_id] = {}
            articles[article_id]['content'] = content

    return articles

# load label file
def loadLabelFile(article_dir, labelFilename) :
    articles = loadArticleFiles(article_dir)
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

    return articles

# extract token id, tag, labels
def alignLabelsBySentence(articles):
    articleList = []
    maxTokenLen = 0

    for key in articles:
        article = articles[key]
        content = article['content']
            
        label_spans = []
        if 'spans' in article:
            label_spans = article['spans']

        sent_tokens = []
        sent_token_ids = []
        sent_token_spans = []
        sent_labels = []
        sents = content.split('\n')

        sentOffset = 0

        for sent in sents:
            tokens = tokenizer.tokenize(sent)
            # tokens.insert(0, '[CLS]')
            tokens.append('[SEP]')

            sent_tokens.append(tokens)
    
            token_spans = []
            token_ids = np.zeros(len(tokens), dtype='int32')
            token_tags = np.zeros(len(tokens), dtype=int) 
            end = 0
            
            logger.debug("{0}: {1} {2}".format(key, len(label_spans), label_spans))

            for i, token in enumerate(tokens):
                token_id = tokenizer.convert_tokens_to_ids(token)
                token_ids[i] = token_id

                if token in ['[CLS]', '[UNK]']:
                    token_spans.append((sentOffset, sentOffset))
                elif token == '[SEP]':
                    sentOffset = sentOffset + len(sent) + len('\n')
                    token_spans.append((sentOffset, sentOffset))
                else:
                    # token = replace_all(token)
                    prev_token = tokens[i-1] if i > 0 else ''
                    # print(prev_token, token)
                    CON_TOKENS = ['âĢ', 'ĠâĢ', 'Ã', 'Ë']

                    if token in CON_TOKENS:
                        token = ''

                    if prev_token in CON_TOKENS:
                        # print([prev_token, token])
                        token = tokenizer.convert_tokens_to_string([prev_token, token]).strip()
                    else:
                        token = tokenizer.convert_tokens_to_string([token]).strip()

                    start = sent.find(token, end) 

                    if start == -1:
                        # print("{0}\t{1} : //{2}//\t//{3}//\n{4}\n{5}\n{6}\n".format((key, i), (start, end), token, sent[:end], sent, tokens, tokens[i]))
                        start = end
                    else:
                        end = start + len(token)

                    # assert start > -1, "{0}\t{1} : //{2}//\t//{3}//\n{4}\n{5}\n{6}".format((key, i), (start, end), token, sent[:end], sent, tokens, tokens[i])

                    for label_span in label_spans:
                        if start + sentOffset >= label_span[0] and end + sentOffset <= label_span[1]:
                            assert token == sent[start:end], "tokens are not the same at {0}: {1}\t{2}".format((start, end), token, sent[start:end])
                            
                            token_tags[i] = 1
                            logger.debug("{0} {1} {2} {3} {4} {5}".format(i, label_span, start, end, token, sent[label_span[0]:label_span[1]]))

                    token_spans.append((start + sentOffset, end + sentOffset))

            sent_token_ids.append(token_ids)
            sent_token_spans.append(token_spans)
            sent_labels.append(token_tags)


            #### validate spans!
            # for i, token_span in enumerate(token_spans):
            #     # token = replace_all(tokens[i])
            #     if token_tags[i] > 0:
            #         # assert token == content[token_span[0]:token_span[1]], "{0} {1} {2}".format(token_span, token, content[token_span[0]:token_span[1]])
            #         print((key, i), token_span, tokens[i], articles[key]['content'][token_span[0]:token_span[1]], token_tags[i])

            maxTokenLen = len(tokens) if maxTokenLen < len(tokens) else maxTokenLen

            
        article['tokens'] = sent_tokens
        article['token_ids'] = sent_token_ids
        article['token_spans'] = sent_token_spans
        article['token_tags'] = sent_labels
                    
        articleList.append([key, article])


    print('max token length: {0}'.format(maxTokenLen))
    return articleList

def chunkData(articleList, mode='single', overlap=4):
    input_ids = []
    input_masks = []
    input_tags = []
    token_spans = []

    if mode == 'single':
        for key, article in articleList:
            for i, sent_token_ids in enumerate(article['token_ids']):
                sent_token_ids = np.insert(sent_token_ids, 0, 101)   #add [CLS] token_id = 101
                ids = np.zeros(MAX_TOKEN, dtype=int)
                ids[0:len(sent_token_ids)] = sent_token_ids
                input_ids.append(ids)

                mask = np.copy(ids)
                mask[mask>0] = 1
                input_masks.append(mask)

                sent_token_tags = article['token_tags'][i]
                sent_token_tags = np.insert(sent_token_tags, 0, 0)     #add [CLS] tag = 0
                tags = np.zeros(MAX_TOKEN, dtype=int)
                tags[0:len(sent_token_tags)] = sent_token_tags
                input_tags.append(tags)

                sent_token_spans = article['token_spans'][i]
                sent_token_spans.insert(0, (0, 0))     #add [CLS] span = (0, 0)
                token_spans.append(sent_token_spans)

    elif mode == 'max':
        for key, article in articleList:
            sumSentLen = 0
            startIdx = 0
            i = 0

            sent_token_ids = article['token_ids']
            # for i in range(startIdx, len(sent_token_ids)):
            while(i < len(sent_token_ids)):
                sumSentLen = sumSentLen + len(sent_token_ids[i])
                # print(i, sumSentLen)

                if sumSentLen > MAX_TOKEN - 1:  # save [CLS] space
                    # ids = sent_token_ids[startIdx:i-1]
                    # print(key, sumSentLen, len(sent_token_ids), startIdx, i-1)
                    token_ids = np.concatenate(sent_token_ids[startIdx:i-1], axis=None)
                    token_ids = np.insert(token_ids, 0, 101)    #add [CLS] token_id = 101
                    ids = np.zeros(MAX_TOKEN, dtype=int)
                    ids[0:len(token_ids)] = token_ids
                    input_ids.append(ids)

                    mask = np.copy(ids)
                    mask[mask>0] = 1
                    input_masks.append(mask)

                    token_tags = np.concatenate(article['token_tags'][startIdx:i-1], axis=None)
                    token_tags = np.insert(token_tags, 0, 0)     #add [CLS] tag = 0
                    tags = np.zeros(MAX_TOKEN, dtype=int)
                    tags[0:len(token_tags)] = token_tags
                    input_tags.append(tags)

                    sent_token_spans = [y for x in article['token_spans'][startIdx:i-1] for y in x]
                    sent_token_spans.insert(0, (sent_token_spans[0][0], sent_token_spans[0][0]))     #add [CLS] span = (0, 0)
                    token_spans.append(sent_token_spans)

                    startIdx = i - overlap if i - overlap > 0 else 0
                    i = startIdx
                    sumSentLen = 0
                else:
                    i = i + 1

            # print(key, len(ids), startIdx, len(sent_token_ids))
            # last chunk
            idx = len(sent_token_ids) - 1
            tokenSum = 0
            while tokenSum < MAX_TOKEN and idx > 0:
                # print(idx, tokenSum)
                tokenSum = tokenSum + len(sent_token_ids[idx])
                idx = idx - 1
            # print(idx, tokenSum)
            startIdx = idx + 2

            token_ids = np.concatenate(sent_token_ids[startIdx:], axis=None)
            token_ids = np.insert(token_ids, 0, 101)    #add [CLS] token_id = 101

            # print(len(token_ids))
            ids = np.zeros(MAX_TOKEN, dtype=int)
            ids[0:len(token_ids)] = token_ids
            input_ids.append(ids)

            mask = np.copy(ids)
            mask[mask>0] = 1
            input_masks.append(mask)

            token_tags = np.concatenate(article['token_tags'][startIdx:], axis=None)
            token_tags = np.insert(token_tags, 0, 0)     #add [CLS] tag = 0
            tags = np.zeros(MAX_TOKEN, dtype=int)
            tags[0:len(token_tags)] = token_tags
            input_tags.append(tags)

            sent_token_spans = [y for x in article['token_spans'][startIdx:] for y in x]
            sent_token_spans.insert(0, (sent_token_spans[0][0], sent_token_spans[0][0]))     #add [CLS] span = (0, 0)
            token_spans.append(sent_token_spans)

            assert len(token_ids) == len(sent_token_spans), 'Token ID length is not the same with span length'

    return input_ids, input_masks, input_tags, token_spans



# build data
def buildData(article_dir, label_file):    
    articles = loadLabelFile(article_dir, label_file)
    articleList = alignLabelsBySentence(articles)

    random.shuffle(articleList)
    input_ids, input_masks, input_tags, token_spans = chunkData(articleList, mode='max')

    # sample = 10
    # input_ids = input_ids[0:sample]
    # input_masks = input_masks[0:sample]
    # input_tags = input_tags[0:sample]
    # token_spans = token_spans[0:sample]

    x = dict(
       input_ids = np.array(input_ids, dtype=np.int32),
       attention_mask = np.array(input_masks, dtype=np.int32),
       token_type_ids = np.zeros(shape=(len(input_ids), MAX_TOKEN)))
    y = np.array(input_tags, dtype=np.int32)

    return x, y


def getroBERTaModel():
    config.num_labels = 2
    roBERTModel = TFRobertaForTokenClassification.from_pretrained(PRETRAINED_MODEL, config=config)
    optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5, epsilon=1e-08, clipnorm=1.0)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    roBERTModel.compile(optimizer=optimizer, loss=loss)

    return roBERTModel


def trainBert(train_x, train_y, val_x, val_y):
    checkpoint_path = "model-SI-roBERTa/cp-base-{epoch:04d}.ckpt"
    cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, verbose=1, save_weights_only=True, save_freq=1)

    bertModel = getroBERTaModel()

    # bertModel.load_weights("model-SI/best/cp-0010.ckpt")
    history = bertModel.fit(x=train_x, y=train_y, 
                  validation_data=(val_x, val_y),
                  epochs=EPOCHS,
                  batch_size=BATCH_SIZE,
                  callbacks=[cp_callback]
                  )

    # print(history.history['val_loss'])

    return history

# evaluate model and get train score and val score (f1, precision, recall)
def evaluate(modelPath, train_x, train_y, val_x, val_y, num=EPOCHS):

    def calculateScore(pred, labels):
        pred_flat = pred.flatten()
        labels_flat = labels.flatten()

        con1 = (pred_flat == 1)
        con2 = (labels_flat == 1)

        part = np.where(con1 & con2)
        correct = len(part[0])
        sum_pred = np.sum(pred_flat)
        sum_true = np.sum(labels_flat)

        precision = correct/sum_pred if correct > 0 else 0
        recall = correct/sum_true  if correct > 0 else 0
        f1 = 2 * ((precision * recall) / (precision + recall)) if (precision + recall) > 0 else 0
        return f1, precision, recall

    def getPredGold(model, x, y, npyFile):
        result = model.predict(x, batch_size=BATCH_SIZE)
        pred = np.argmax(result, axis=-1)
        np.save(npyFile, result)
        labels = x['attention_mask'] * y
        return pred, labels


    bertModel = getroBERTaModel()
    print("epoch\ttr_f1\ttr_precision\ttr_recall\tval_f1\tval_precision\tval_recall")

    for i in range (1, num + 1):
        path = modelPath.format(i)
        bertModel.load_weights(path)

        # Train score
        tr_pred, tr_labels = getPredGold(bertModel, train_x, train_y, "npy/SI-train-roBERTa-base-result-{0:04d}.npy".format(i))
        tr_f1, tr_precision, tr_recall = calculateScore(tr_pred, tr_labels)

        # Val score
        val_pred, val_labels = getPredGold(bertModel, val_x, val_y, "npy/SI-val-roBERTa-base-result-{0:04d}.npy".format(i))
        val_f1, val_precision, val_recall = calculateScore(val_pred, val_labels)

        print("{0}\t{1}\t{2}\t{3}\t{4}\t{5}\t{6}".format(i, tr_f1, tr_precision, tr_recall, val_f1, val_precision, val_recall))



def main(task='SI'):
    logger.setLevel(logging.INFO)
    np.set_printoptions(threshold=sys.maxsize)

    train_article_dir = "datasets/train-articles"
    dev_article_dir = "datasets/dev-articles"

    train_label_file = "datasets/train-task1-SI.labels"
    dev_label_file = "datasets/dev-task-TC-template.out"

    train_x, train_y = buildData(train_article_dir, train_label_file)
    print('({0}) train data loaded:'.format(len(train_y)))
    val_x, val_y = buildData(dev_article_dir, dev_label_file)
    print('({0}) validataion data loaded:'.format(len(val_y)))

    # print(train_x['input_ids'][0:10])
    # print(val_x['input_ids'][0:10])

    checkpoint_path = "model-SI-roBERTa/cp-base-{0:04d}.ckpt"

    trainBert(train_x, train_y, val_x, val_y)
    evaluate(checkpoint_path, train_x, train_y, val_x, val_y, num=EPOCHS)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("task", nargs='?', default="SI")
    args = parser.parse_args()

    print('mode=' + args.task)
    main(args.task)

