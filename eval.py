import glob
import os.path
import random
import numpy as np
import logging
import tensorflow as tf
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, TFBertForTokenClassification, BertForTokenClassification

random.seed(2019)

train_article_folder = "datasets/train-articles"
train_label_folder = "datasets/train-labels-task1-span-identification"
article_pre = "article"
SI_ext = ".task1-SI.labels"

logger = logging.getLogger('propaganda_detection')
logger.setLevel(logging.INFO)

PRETRAINED_MODEL = 'bert-base-uncased'
MAX_TOKEN = 512
OVERLAP = 0.4
EPOCHS = 2
BATCH_SIZE = 8

checkpoint_path = "model/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL)

logger.info("Bert pretrained model: {0}".format(PRETRAINED_MODEL))

articles = {}

# load files
article_files = glob.glob(os.path.join(train_article_folder, "*.txt"))
for filename in article_files:
    with open(filename, "r", encoding="utf-8") as f:
        content = f.read()
        article_id = os.path.basename(filename).split(".")[0][7:]
        
        articles[article_id] = {}
        articles[article_id]['content'] = content

label_files = glob.glob(os.path.join(train_label_folder, "*.labels"))
for filename in label_files:
    with open(filename, "r", encoding="utf-8") as f:
        spans = []
        propagandaTypes = []
        label_id = ''
        for line in f.readlines():
            line = line.strip().split('\t')
            if len(line) > 0:
                label_id = line[0]
                span = (int(line[-2]), int(line[-1]))
                spans.append(span)
#                 if len(line) > 3:  #type... sort??
#                     propagandaType.append([1])
            else:
                logger.error('not correct file: {0} at {1}'.format(line, filename))
        if len(spans) > 0:
            spans.sort()
            articles[label_id]['spans'] = spans
        else:
            logger.warning('file empty: ' + filename)


# extract token id, tag, labels
articleList = []

for key in articles:
    article = articles[key]
    
    content = article['content']
    if 'uncased' in PRETRAINED_MODEL:
        content = content.lower()
        
    label_spans = []
    if 'spans' in article:
        label_spans = article['spans']

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



random.shuffle(articleList)  # shuffle data
tr_articles, val_articles = train_test_split(articleList, random_state=2019, test_size=0.2) # split data into train & validation
logger.debug("Train articles: {0}\t Val article: {1}".format(len(tr_articles), len(val_articles)))
val_article_ids = [val[0] for val in val_articles]
print( val_article_ids )


# construct data for Bert format
tr_ids, tr_masks, tr_tags = getInputData(tr_articles)
val_ids, val_masks, val_tags = getInputData(val_articles)

# test small dataset
# tr_ids = tr_ids[0:10]
# tr_masks = tr_masks[0:10]
# tr_tags = tr_tags[0:10]

# val_ids = val_ids[0:3] 
# val_masks = val_masks[0:3]
# val_tags = val_tags[0:3]


# training dataset
train_x = dict(
   input_ids = np.array(tr_ids, dtype=np.int32),
   attention_mask = np.array(tr_masks, dtype=np.int32),
   token_type_ids = np.zeros(shape=(len(tr_ids), MAX_TOKEN)))
train_y = np.array(tr_tags, dtype=np.int32)

# validataion dataset
val_x = dict(
   input_ids = np.array(val_ids, dtype=np.int32),
   attention_mask = np.array(val_masks, dtype=np.int32),
   token_type_ids = np.zeros(shape=(len(val_ids), MAX_TOKEN)))
val_y = np.array(val_tags, dtype=np.int32)


# build model
bertModel = TFBertForTokenClassification.from_pretrained(PRETRAINED_MODEL, num_labels=2)
optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5, epsilon=1e-08, clipnorm=1.0)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
bertModel.compile(optimizer=optimizer, loss=loss)


# training
# cp_callback = tf.keras.callbacks.ModelCheckpoint(
#     checkpoint_path, verbose=1, save_weights_only=True,
#     save_freq=1)

# bertModel.fit(x=train_x, y=train_y, 
#               validation_data = (val_x, val_y),
#               epochs=EPOCHS, batch_size=BATCH_SIZE,
#               callbacks=[cp_callback])

# batch training
# bertModel.fit(x=train_x, y=train_y,
#               validation_data=(val_x, val_y),
#               epochs=EPOCHS,
#               # steps_per_epoch=115,
#               # validation_steps=7,
#               batch_size=BATCH_SIZE,
#               callbacks=[eval_metrics])


# bertModel.save_pretrained('./save/')


# predict
# checkpoint_path = "model/cp-0002.ckpt"
# bertModel.load_weights(checkpoint_path)
# logger.info("{0} loaded".format(checkpoint_path))

# result = bertModel(val_x)[0]
# np.save('result.npy', result)
# logger.info("result saved")


checkpoint_path = "model/cp-{0:04d}.ckpt"
npFile = "result-{0:04d}.npy"

for i in range (1, 11):
    path = checkpoint_path.format(i)
    bertModel.load_weights(path)
    print("{0} loaded".format(path))

    result = bertModel.predict(val_x, batch_size=BATCH_SIZE)
    path = npFile.format(i) 
    np.save(path, result)
    print("{0} saved".format(path))

    # calculate f1, precision, recall by token
    pred = np.argmax(result, axis=-1)
    pred_flat = pred.flatten()

    labels = np.array(val_masks) * val_y
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

    print(i, f1, precision, recall)

    # reconstruct test file






