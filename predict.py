import glob
import os.path
import random
import numpy as np
import logging
import tensorflow as tf
import re
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, TFBertForTokenClassification, BertForTokenClassification

random.seed(2019)

test_article_folder = "datasets/dev-articles"

logger = logging.getLogger('propaganda_predict')
logger.setLevel(logging.INFO)

PRETRAINED_MODEL = 'bert-base-uncased'
MAX_TOKEN = 512
OVERLAP = 0.4
EPOCHS = 2
BATCH_SIZE = 8

FILL_BLANK = True

checkpoint_path = "model/best/cp-0007.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL)

logger.info("Bert pretrained model: {0}".format(PRETRAINED_MODEL))

articles = {}

# load files
article_files = glob.glob(os.path.join(test_article_folder, "*.txt"))
for filename in article_files:
    with open(filename, "r", encoding="utf-8") as f:
        content = f.read()
        article_id = os.path.basename(filename).split(".")[0][7:]
        
        articles[article_id] = {}
        articles[article_id]['content'] = content



# extract token id, tag, labels
articleList = []

for key in articles:
    article = articles[key]
    
    content = article['content']
    if 'uncased' in PRETRAINED_MODEL:
        content = content.lower()

    tokens = tokenizer.tokenize(content)
    token_spans = []
    token_ids = np.zeros(len(tokens), dtype='int32')
    end = 0
    
    for i, token in enumerate(tokens):
        token_id = tokenizer.convert_tokens_to_ids(token)
        token = token.replace('#', '')
        start = content.find(token, end)
        end = start + len(token)
        
        token_ids[i] = token_id
        token_spans.append((start, end))
        
    article['tokens'] = tokens
    article['token_ids'] = token_ids
    article['token_spans'] = token_spans
            
    articleList.append([key, article])
    



# construct input data
# split with MAX_TOKENS and pad with 0
# construct input data
def getInputData(atricles):
    input_ids = []
    input_masks = []
    article_ids = []
    token_spans = []

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
                
            article_ids.append((articleId, start))
                
            token_mask = np.zeros(MAX_TOKEN, dtype=int)
            token_mask[mask_start - start:mask_end - start] = 1
            input_ids.append(article['token_ids'][start:end])
            input_masks.append(token_mask)
            token_spans.append(article['token_spans'][start:end])

            logger.debug("start: {0}, mask_start:{1}, maks_end:{2}, end:{3}".format(start, mask_start, mask_end, end))
            logger.debug(tokens[start:end])
            logger.debug(article['token_ids'][start:end])
            logger.debug(article['token_spans'][start:end])
            logger.debug(token_mask)

            start = end - step
            end = start + MAX_TOKEN

        # last sentence + padding with zeros
        article_ids.append((articleId, start))
        
        last_ids = np.zeros(MAX_TOKEN, dtype=int)
        ids = article['token_ids'][start:]
        last_ids[0:len(ids)] = ids
        input_ids.append(last_ids)
        logger.debug(last_ids)

        
        token_spans.append(article['token_spans'][start:])
        logger.debug(token_spans)

        mask_start = min([i for i in sentIndex if i > start]) + 1
        token_mask = np.zeros(MAX_TOKEN, dtype=int)
        token_mask[mask_start - start:len(tokens) - start] = 1
        input_masks.append(token_mask)
        logger.debug(token_mask)

    logger.debug("Data length: {0}".format(len(input_ids)))

    return input_ids, input_masks, (article_ids, token_spans)




# construct data for Bert format
test_ids, test_masks, id_spans = getInputData(articleList)


# test dataset
test_x = dict(
   input_ids = np.array(test_ids, dtype=np.int32),
   attention_mask = np.array(test_masks, dtype=np.int32),
   token_type_ids = np.zeros(shape=(len(test_ids), MAX_TOKEN)))


# build model and predict test data
bertModel = TFBertForTokenClassification.from_pretrained(PRETRAINED_MODEL, num_labels=2)
optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5, epsilon=1e-08, clipnorm=1.0)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
bertModel.compile(optimizer=optimizer, loss=loss)

bertModel.load_weights(checkpoint_path)
print("{0} loaded".format(checkpoint_path))
result = bertModel.predict(test_x, batch_size=BATCH_SIZE)
np.save('result.npy', result)
print("result.npy saved")

# result = np.load('npy/result-0007.npy') # load from npy file
pred = np.argmax(result, axis=-1)

assert len(pred) != len(articleList), "Testset and Prediction lengths are different"

# recunstruct test data into each article
article_span = {}

for i, (article_id, start) in enumerate(id_spans[0]):
    if article_id not in article_span:
        article_span[article_id] = []   # create new span list per each article
        
    for j, span in enumerate(id_spans[1][i]):  
        if start > 0:
            span_idx = start + j
            
            if span_idx >= len(article_span[article_id]):
                article_span[article_id].append((span, pred[i][j]))
            else:
                article_span[article_id][span_idx] = (article_span[article_id][span_idx][0], 
                                                      article_span[article_id][span_idx][1] or pred[i][j])
                                  
        else:
            article_span[article_id].append((span, pred[i][j]))


# wrtie submission file
with open('submission.txt', 'w') as f:
    for key in article_span:
    #     print(key)
        spans = article_span[key]
        tags = ''.join([str(span[1]) for span in spans])
        logger.debug("Original Prediction: ".format(tags))
        if FILL_BLANK:
            tags = re.sub('101', '111', tags)
            tags = re.sub('101', '111', tags)
            tags = re.sub('101', '111', tags)
            logger.debug("Blank Filled Prediction: ".format(tags))

        real_span = []
        start = 0
        end = 0
        for i, tag in enumerate(tags):
            if int(tag) > 0:
                if i-1 > 0 and spans[i-1][1] == 0: # 0 0 0 1
                    start = spans[i][0][0]
            else:
                if i-1 > 0 and spans[i-1][1] > 0: # 1 1 1 0
                    end = spans[i-1][0][1]
                    real_span.append((start, end))
                    print("{0}\t{1}\t{2}".format(key, start, end))
                    f.write("{0}\t{1}\t{2}\n".format(key, start, end))

