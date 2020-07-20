import sys
import glob
import os
import os.path
import random
import numpy as np
import logging
import argparse
import tensorflow as tf
import sklearn.metrics
import re
#from keras import backend as K
from sklearn.model_selection import train_test_split
from transformers import RobertaConfig, RobertaTokenizer, TFRobertaForTokenClassification

# from keras.preprocessing.sequence import pad_sequences
# input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", value=0, truncating="post", padding="post")

random.seed(2019)

logger = logging.getLogger('propaganda_predict_SI')

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


# extract token id, tag, labels
def alignLabelsBySentence(articles):
    articleList = []
    maxTokenLen = 0
    print(articles.keys())

    # keys = ['787730392', '999001211', '788847916', '761568202', '790499380', '730269378', '776385494', '832913653',
    # '788273361', '798244842', '738028498', '738447109', '779309765', '773650987', '999000878', '832908978', '787966255',
    # '999000874', '999000858', '832910505', '786004130', '778358096', '999001299', '999001256', '999001419', '832917778',
    # '832916492', '730246508', '999000851', '763280007', '832916508', '761722669', '832908905', '785331076', '787085939',
    # '761575506', '763412406', '761955563', '999001323', '789512681', '777720051', '832908730', '789753303', '778507244',
    # '783774960', '738361208', '777785889', '776126299', '782149032', '832913316', '789454337', '778094905', '790666929',
    # '738442776', '738781754', '772836731', '794141509', '788271400', '730081389', '740235127', '999001280', '778730964',
    # '784382409', '999001259', '738542398', '777869943', '999001290', '763114850', '781672902', '794344513', '832917532',
    # '784143418', '782448403', '730093263', '788626289']

    for key in articles:
    # for key in keys:
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
    articleIds = []

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
                    articleIds.append(key)

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
            articleIds.append(key)

            assert len(token_ids) == len(sent_token_spans), 'Token ID length is not the same with span length'

    return input_ids, input_masks, (articleIds, token_spans)



# build data
def buildData(article_dir, label_file):    
    articles = loadArticleFiles(article_dir)
    articleList = alignLabelsBySentence(articles)

    input_ids, input_masks, token_spans = chunkData(articleList, mode='max')

    x = dict(
       input_ids = np.array(input_ids, dtype=np.int32),
       attention_mask = np.array(input_masks, dtype=np.int32),
       token_type_ids = np.zeros(shape=(len(input_ids), MAX_TOKEN)))

    return x, token_spans, articles


def getroBERTaModel():
    config.num_labels = 2
    roBERTModel = TFRobertaForTokenClassification.from_pretrained(PRETRAINED_MODEL, config=config)
    optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5, epsilon=1e-08, clipnorm=1.0)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    roBERTModel.compile(optimizer=optimizer, loss=loss)

    return roBERTModel


def predictAllFromNPY(npyPath, x, num=10):
    bertModel = getroBERTaModel()

    preds = []

    for i in range(1, num + 1):
        npyFile = npyPath.format(i)
        result = np.load(npyFile)
        print(npyFile + " loaded")

        pred = np.argmax(result, axis=-1)
        preds.append(pred)

    return preds


def predictAll(modelPath, x, num=10):
    bertModel = getroBERTaModel()

    preds = []

    for i in range(1, num + 1):
        path = modelPath.format(i)
        bertModel.load_weights(path)
        print("{0} loaded".format(path))

        result = bertModel.predict(x, batch_size=BATCH_SIZE)
        npyFile = 'npy/SI-val-roBERTa-base-result-{0:02d}.npy'.format(i)
        np.save(npyFile, result)
        print(npyFile + " saved")

        pred = np.argmax(result, axis=-1)
        preds.append(pred)

    return preds


def predict(modelPath, x):
    bertModel = getroBERTaModel()

    bertModel.load_weights(modelPath)
    print("{0} loaded".format(modelPath))

    result = bertModel.predict(x, batch_size=BATCH_SIZE)
    npyFile = 'npy/SI-val-roBERTa-base-result.npy'
    np.save(npyFile, result)
    print(npyFile + " saved")
    pred = np.argmax(result, axis=-1)
    
    return pred


def reconstructSpans(id_spans, pred):
    articleIds, token_spans = id_spans
    article_span = {}

    # print('reconstructSpans: id - {0}\ttoken_spans - {1}\tpred - {2}'.format(len(articleIds), len(token_spans), len(pred)))
    assert len(articleIds) == len(token_spans), 'articleIds - token_spans data length is wrong {0} {1}'.format(len(articleIds), len(token_spans))
    assert len(articleIds) == len(pred), 'articleIds - pred data length is wrong {0} {1}'.format(len(articleIds), len(pred))

    for i, article_id in enumerate(articleIds):
        if article_id not in article_span:
            article_span[article_id] = {}   # create new span dict per each article
            
        for j, span in enumerate(token_spans[i]):
            if span in article_span[article_id]:
                # if article_span[article_id][span] != pred[i][j]:
                #     print(article_id, span, article_span[article_id][span], pred[i][j])

                article_span[article_id][span] = article_span[article_id][span] or pred[i][j]   # current lable or new tag
            else:
                article_span[article_id][span] = pred[i][j]

    return article_span



def writeSubmissionFile(fileName, article_span, articles, FILL_BLANK=True):
    # print('writeSubmissionFile: {0} {1}'.format(fileName, list(article_span.keys())))
    print('writeSubmissionFile: {0}'.format(fileName))

    # with open(fileName, 'w') as f:
    #     for article_id in article_span:
    #         spans = article_span[article_id]
    #         span_keys = list(spans.keys())
    #         span_keys.sort()

    #         tags = ''.join([str(spans[span_key]) for span_key in span_keys])
    #         logger.debug("Original Prediction: ".format(tags))
    #         if FILL_BLANK:
    #             tags = re.sub('101', '111', tags)
    #             tags = re.sub('101', '111', tags)
    #             tags = re.sub('101', '111', tags)
    #             logger.debug("Blank Filled Prediction: ".format(tags))

    #         real_span = []
    #         start = 0
    #         end = 0
    #         for i, tag in enumerate(tags):
    #             if int(tag) > 0:
    #                 if i == 0:
    #                     start = 0
    #                 elif i-1 > 0 and int(tags[i-1]) == 0: # if tag-1 is 0: 0 0 0 1
    #                     # print(i, span_keys[i], spans[span_keys[i]])
    #                     # print(tags)
    #                     # print(spans)
    #                     start = span_keys[i][0]  # chunk span start = current span start
    #             else:
    #                 if i-1 > 0 and int(tags[i-1]) > 0: # if tag-1 is 1: 1 1 1 0
    #                     end = span_keys[i-1][1]  # chunk span end = current span end
    #                     if end - start > 3:
    #                         real_span.append((start, end))
    #                         # print("{0}\t{1}\t{2}".format(article_id, start, end))
    #                         f.write("{0}\t{1}\t{2}\n".format(article_id, start, end))


    with open(fileName, 'w') as f:
        for article_id in article_span:
            content = articles[article_id]['content']
            spans = article_span[article_id]
            span_keys = list(spans.keys())
            span_keys.sort()

            tags = ''.join([str(spans[span_key]) for span_key in span_keys])
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
                    if i == 0:
                        start = 0
                    elif i-1 > 0 and int(tags[i-1]) == 0: # if tag-1 is 0: 0 0 0 1
                        # print(i, span_keys[i], spans[span_keys[i]])
                        # print(tags)
                        # print(spans)
                        start = span_keys[i][0]  # chunk span start = current span start
                else:
                    if i-1 > 0 and int(tags[i-1]) > 0: # if tag-1 is 1: 1 1 1 0
                        end = span_keys[i-1][1]  # chunk span end = current span end
                        # if content[end].strip() == "":
                        #     end = end - 1

                        if end - start > 0:
                            # include quetos
                            if start-1 > 0 and end+1 < len(content):
                                if content[start-1].isalpha():
                                    pos = start - 1
                                    while content[pos].isalpha():
                                        pos = pos - 1
                                    # print(start, content[start:end], pos, content[pos:end])
                                    start = pos

                                if content[end].strip() != "" and content[end+1].isalpha():
                                    pos = end + 1
                                    while content[pos].isalpha():
                                        pos = pos + 1
                                    # print(start, content[start:end], pos, content[start:pos])
                                    end = pos

                                # if content[start-1] in ['"', '\''] and content[start-1] == content[end+1]:
                                #     print(content[start-1], content[end+1])
                                #     start = start-1
                                #     end = end+1

                                # if content[start-1] == '“' and content[end+1] == '”':
                                #     print(content[start-1], content[end+1])
                                #     start = start-1
                                #     end = end+1

                                

                            real_span.append((start, end))
                            # print("{0}\t{1}\t{2}".format(article_id, start, end))


            # write submission file
            for start, end in real_span:
                if end - start > 2:
                    start = 0 if start < 0 else start
                    f.write("{0}\t{1}\t{2}\n".format(article_id, start, end))

                    # # check duplicated words
                    # findText = content[start:end]
                    # # if findText not in ['is', 'the', 'be', 'to', 'or', 'we', 'she', 'our', 'no', 'if', 'of', 'what', 'that']:
                    # if ' ' in findText:
                    #     res = [i for i in range(len(content)) if content.startswith(findText, i)] 
                    #     # print(findText, len(res))
                    #     if len(res) > 1:
                    #         for start in res:
                    #             end = start + len(findText)
                    #             # print(content[start:end])

                    #             isIncluded = False
                    #             for s, e in real_span:
                    #                 if start >= s and end <= e:
                    #                     isIncluded = True

                    #             if isIncluded == False:
                    #                 f.write("{0}\t{1}\t{2}\n".format(article_id, start, end))
                    #                 # print(findText)

                    # add frequent words!!!!!!
                    # frequentText = ['black', 'jihad', 'innocent', 'affidavit', 'killer']
                    # for findText in frequentText:
                    #     res = [i for i in range(len(content)) if content.startswith(findText, i)] 
                    #     # print(findText, len(res))
                    #     if len(res) > 1:
                    #         for start in res:
                    #             end = start + len(findText)
                    #             # print(content[start:end])

                    #             isIncluded = False
                    #             for s, e in real_span:
                    #                 if start >= s and end <= e:
                    #                     isIncluded = True

                    #             if isIncluded == False:
                    #                 f.write("{0}\t{1}\t{2}\n".format(article_id, start, end))
                    #                 # print(findText)

                    # add frequent words only found at least one in the document
                    # frequentText = ['black', 'jihad', 'innocent', 'affidavit', 'killer']
                    # for findText in frequentText:
                    #     hasText = False
                    #     for s, e in real_span:
                    #         realText = content[s:e]
                    #         if realText.find(findText) > 0:
                    #             hasText = True

                    #     if hasText:
                    #         res = [i for i in range(len(content)) if content.startswith(findText, i)] 
                    #         # print(findText, len(res))
                    #         if len(res) > 1:
                    #             for start in res:
                    #                 end = start + len(findText)
                    #                 # print(content[start:end])

                    #                 isIncluded = False
                    #                 for s, e in real_span:
                    #                     if start >= s and end <= e:
                    #                         isIncluded = True

                    #                 if isIncluded == False:
                    #                     f.write("{0}\t{1}\t{2}\n".format(article_id, start, end))
                    #                     # print(findText)




                

    print('{0} done'.format(fileName))
    # cmd = '/tools/task-SI_scorer.py -r {0} {1} -m'.format(fileName, GOLD_FOLDER)



def main(mode='multi', model='npy', num=10):
    logger.setLevel(logging.INFO)
    np.set_printoptions(threshold=sys.maxsize)

    train_article_dir = "datasets/train-articles"
    dev_article_dir = "datasets/dev-articles"

    train_label_file = "datasets/train-task1-SI.labels"
    dev_label_file = "datasets/dev-task-TC-template.out"

    checkpoint_path = "model-SI-roBERTa/cp-base-{0:04d}.ckpt"
    submissionFile = 'submission_SI/submission_SI_{0}.txt'
    npy_path = "npy/SI-val-roBERTa-base-result-{0:04d}.npy"

    test_x, token_spans, articles = buildData(dev_article_dir, dev_label_file)
    print('({0}) test data loaded:'.format(len(test_x['input_ids'])))

    # print(test_x['input_ids'][0:10])
    # print(token_spans[1][0:10])
    # print(token_spans[0][0:10])

    ## Predict multiple models
    if mode == 'multi':
        preds = predictAll(checkpoint_path, test_x, num=num) if model == 'model' else predictAllFromNPY(npy_path, test_x ,num=num)

        for i, pred in enumerate(preds):
            article_span = reconstructSpans(token_spans, pred)
            writeSubmissionFile(submissionFile.format(i + 1), article_span, articles)

        for i in range(2, num + 1):
            cmd = 'python3.7 ./tools/task-SI_scorer.py -s {0} -r ./datasets/dev-labels-SI -m'.format(submissionFile.format(i))
            os.system(cmd)

    else:   ### Predict single model
        pred = []
        if model == 'npy':
            # result = np.load('npy/SI-test-result.npy')
            result = np.load('npy/SI-test-large-result-18.npy')
            pred = np.argmax(result, axis=-1)
        else:
            pred = predict(checkpoint_path.format(6), test_x)

        article_span = reconstructSpans(token_spans, pred)
        writeSubmissionFile(submissionFile.format(0), article_span, articles)

        cmd = 'python3.7 ./tools/task-SI_scorer.py -s {0} -r ./datasets/dev-labels-SI -m'.format(submissionFile.format(0))
        os.system(cmd)

        cmd = 'python3.7 visualize_result.py'
        os.system(cmd)

    



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("task", nargs='?', default="SI")
    parser.add_argument("mode", nargs='?', default="multi")
    parser.add_argument("model", nargs='?', default="model")
    args = parser.parse_args()

    print('mode=' + args.mode)
    print('model=' + args.model)
    main(args.mode, args.model)

