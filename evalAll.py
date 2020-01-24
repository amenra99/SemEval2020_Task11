import glob
import os.path
import random
import numpy as np
import logging
import tensorflow as tf
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, TFBertForTokenClassification, BertForTokenClassification

random.seed(2019)

logger = logging.getLogger('propaganda_model_eval')
logger.setLevel(logging.INFO)

train_article_folder = "datasets/train-articles"
dev_article_folder = "datasets/dev-articles"
article_pre = "article"

SI_label_file = "datasets/train-task1-SI.labels"
TC_label_file = "datasets/train-task2-TC.labels"
dev_label_file = "datasets/dev-task-TC-template.out"

PROP_CLASS = ['Appeal_to_Authority', 'Appeal_to_fear-prejudice', 'Bandwagon,Reductio_ad_hitlerum',
                'Black-and-White_Fallacy', 'Causal_Oversimplification', 'Doubt', 'Exaggeration,Minimisation',
                'Flag-Waving', 'Loaded_Language', 'Name_Calling,Labeling', 'Repetition', 'Slogans',
                'Thought-terminating_Cliches', 'Whataboutism,Straw_Men,Red_Herring']


PRETRAINED_MODEL = 'bert-base-uncased'
MAX_TOKEN = 512
OVERLAP = 0.4
EPOCHS = 10
BATCH_SIZE = 8

checkpoint_path = "model/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

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
            
            articles[article_id] = {}
            articles[article_id]['content'] = content

    return articles

# load label file
def loadLabelFile(articles, filename, task='SI') :
    with open(filename, 'r', encoding="utf-8") as f:
        spans = []
        propagandaTypes = []
        label_id = ''
        for line in f.readlines():
            line = line.strip().split('\t')
            if len(line) > 0:
                label_id = line[0]
                span = (int(line[-2]), int(line[-1]))    

                if 'span' in articles[label_id]:
                    articles[label_id]['spans'].append(span)
                else:
                    articles[label_id]['spans'] = [span]

                if task == 'TC':
                    prop_type = PROP_CLASS.index(line[1]) + 1
                    assert prop_type not in PROP_CLASS, "{0} is not class".format(line[1])

                    if 'class' in article[label_id]:
                        articles[label_id]['class'].append(prop_type)
                    else:
                        articles[label_id]['class'] = [prop_type]

    return articles


# extract token id, tag, labels
def alignLabels(articles, task='SI'):
    articleList = []

    for key in articles:
        article = articles[key]
        content = article['content']

        if 'uncased' in PRETRAINED_MODEL:
            content = content.lower()
            
        label_spans = []
        if 'spans' in article:
            label_spans = article['spans']

        prop_list = []
        if task == 'TC':
            prop_list = article['class']

        tokens = tokenizer.tokenize(content)
        token_spans = []
        token_ids = np.zeros(len(tokens), dtype='int32')
        token_tags = np.zeros(len(tokens), dtype=int)
        end = 0
        
        logger.debug("{0}: {1} {2}".format(key, len(label_spans), label_spans))

        for i, token in enumerate(tokens):
            token_id = tokenizer.convert_tokens_to_ids(token)
            token = token.replace('#', '')
            start = content.find(token, end)
            end = start + len(token)
            
            for label_span in label_spans:
                if start >= label_span[0] and end <= label_span[1]:
                    if token != content[start:end]:
                        logger.error("token error at {0}: {1}\t{2}".format((start, end), token, content[start:end]))
                    else:
                        if task == 'TC':
                            token_tags[i] = prop_list[i]
                        else:
                            token_tags[i] = 1
                        logger.debug("{0} {1} {2} {3} {4} {5}".format(i, label_span, start, end, token, content[label_span[0]:label_span[1]]))

            token_ids[i] = token_id
            token_spans.append((start, end))
            
        article['tokens'] = tokens
        article['token_ids'] = token_ids
        article['token_spans'] = token_spans
        article['token_tags'] = token_tags
            
        logger.debug(np.where(token_tags > 0))
        
        articleList.append([key, article])

    return articleList
    

# construct input data
# split with MAX_TOKENS and pad with 0
def getInputData(atricles):
    input_ids = []
    input_masks = []
    input_tags = []

    for article in atricles:
        articleId = article[0]
        article = article[1]
        tokens = article['tokens']

        logger.debug("({0}) Token length: {1}".format(articleId, len(tokens)))

        start = 0
        end = MAX_TOKEN
        step = int(MAX_TOKEN * OVERLAP)

        sentIndex = [i for i, token in enumerate(article['tokens']) if token in ['.', '?', '!']]

        mask_start = 0
        mask_end = MAX_TOKEN

        while end < len(tokens):
            mask_end = max([i for i in sentIndex if i < end]) + 1
            if start > 0:
                mask_start = min([i for i in sentIndex if i > start]) + 1

            token_mask = np.zeros(MAX_TOKEN, dtype=int)
            token_mask[mask_start - start:mask_end - start] = 1
            input_ids.append(article['token_ids'][start:end])
            input_masks.append(token_mask)
            input_tags.append(article['token_tags'][start:end])

            logger.debug("start: {0}, mask_start:{1}, maks_end:{2}, end:{3}".format(start, mask_start, mask_end, end))
            logger.debug(tokens[start:end])
            logger.debug(article['token_ids'][start:end])
            logger.debug(article['token_tags'][start:end])
            logger.debug(token_mask)

            start = end - step
            end = start + MAX_TOKEN

        # last sentence + padding with zeros
        last_ids = np.zeros(MAX_TOKEN, dtype=int)
        ids = article['token_ids'][start:]
        last_ids[0:len(ids)] = ids
        input_ids.append(last_ids)
        logger.debug(last_ids)

        last_tags = np.zeros(MAX_TOKEN, dtype=int)
        tags = article['token_tags'][start:]
        last_tags[0:len(tags)] = tags
        input_tags.append(last_tags)
        logger.debug(last_tags)

        mask_start = min([i for i in sentIndex if i > start]) + 1
        token_mask = np.zeros(MAX_TOKEN, dtype=int)
        token_mask[mask_start - start:len(tokens) - start] = 1
        input_masks.append(token_mask)
        logger.debug(token_mask)

    logger.debug("Data length: {0}".format(len(input_ids)))

    return input_ids, input_masks, input_tags


# build data
def buildData(article_dir, label_file, task='SI'):
    articles = loadArticleFiles(article_dir)
    articles = loadLabelFile(articles, label_file, task='SI')
    articleList = alignLabels(articles, task='SI')
    random.shuffle(articleList)
    ids, masks, tags = getInputData(articleList)

    x = dict(
       input_ids = np.array(ids, dtype=np.int32),
       attention_mask = np.array(masks, dtype=np.int32),
       token_type_ids = np.zeros(shape=(len(ids), MAX_TOKEN)))
    y = np.array(tags, dtype=np.int32)

    return x, y


# evaluate model and get train score and val score (f1, precision, recall)
def evaluate(modelPath, train_x, train_y, val_x, val_y, num=10, task='SI'):

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
        np.save(npyFile, tr_result)
        lebels = x['attention_mask'] * y

        return pred, labels

    num_labels = 2
    if task == 'TC':
        num_labels = len(PROP_CLASS) + 1

    bertModel = TFBertForTokenClassification.from_pretrained(PRETRAINED_MODEL, num_labels=num_labels)
    optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5, epsilon=1e-08, clipnorm=1.0)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    bertModel.compile(optimizer=optimizer, loss=loss)

    for i in range (1, num + 1):
        path = checkpoint_path.format(i)
        bertModel.load_weights(path)
        print("{0} loaded".format(path))

        print("type\tnum\tf1\tprecision\trecall")
        # Train score
        tr_pred, tr_labels = getPredGold(bertModel, train_x, train_y, "npy/train-result-{0:04d}.npy".format(i))
        f1, precision, recall = calculateScore(tr_pred, tr_labels)
        print("Train:\t{0}\t{1}\t{2}\t{3}".format(i, f1, precision, recall))

        # Val score
        val_pred, val_labels = getPredGold(bertModel, val_x, val_y, "npy/val-result-{0:04d}.npy".format(i))
        f1, precision, recall = calculateScore(val_pred, val_labels)
        print("Val:\t{0}\t{1}\t{2}\t{3}".format(i, f1, precision, recall))



def main():
    train_x, train_y = buildData(train_article_folder, SI_label_file, task='SI')
    val_x, val_y = buildData(dev_article_folder, dev_label_file, task='SI')

    evaluate(checkpoint_path, train_x, train_y, val_x, val_y, EPOCHS, task='SI')


if __name__ == "__main__":
    main()


