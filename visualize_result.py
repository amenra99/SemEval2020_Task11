import sys
import glob
import os.path
import random
import numpy as np
import logging
import argparse

random.seed(2019)

logger = logging.getLogger('propaganda_train_TC')

PROP_CLASS = ['Appeal_to_Authority', 'Appeal_to_fear-prejudice', 'Bandwagon,Reductio_ad_hitlerum',
                'Black-and-White_Fallacy', 'Causal_Oversimplification', 'Doubt', 'Exaggeration,Minimisation',
                'Flag-Waving', 'Loaded_Language', 'Name_Calling,Labeling', 'Repetition', 'Slogans',
                'Thought-terminating_Cliches', 'Whataboutism,Straw_Men,Red_Herring']

MAX_TOKEN = 256

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
def loadLabels(article_dir, trueFile, predFile) :
    articles = loadArticleFiles(article_dir)

    with open(trueFile, 'r', encoding="utf-8") as f:
        spans = []
        for line in f.readlines():
            line = line.strip().split('\t')
            if len(line) > 0:
                label_id = line[0]
                span = (int(line[-2]), int(line[-1]))   

                if 'true_start' in articles[label_id]:
                    articles[label_id]['true_start'].append(span[0])
                else:
                    articles[label_id]['true_start'] = [span[0]]
                if 'true_end' in articles[label_id]:
                    articles[label_id]['true_end'].append(span[1])
                else:
                    articles[label_id]['true_end'] = [span[1]]

    with open(predFile, 'r', encoding="utf-8") as f:
        spans = []
        for line in f.readlines():
            line = line.strip().split('\t')
            if len(line) > 0:
                label_id = line[0]
                span = (int(line[-2]), int(line[-1]))   

                if 'pred_start' in articles[label_id]:
                    articles[label_id]['pred_start'].append(span[0])
                else:
                    articles[label_id]['pred_start'] = [span[0]]
                if 'pred_end' in articles[label_id]:
                    articles[label_id]['pred_end'].append(span[1])
                else:
                    articles[label_id]['pred_end'] = [span[1]]

    return articles


def buildResults(articles):
    MARK_START = '<MARK>'
    MARK_END = '</MARK>'
    PRED_START = "<SPAN>"
    PRED_END = '</SPAN>'

    with open('result.html', 'w', encoding="utf-8") as f:
        f.write('<HTML><meta charset="UTF-8"><STYLE>span {color:blue;font-weight:bold;}</STYLE><BODY>')

        for key in sorted(articles.keys()):
            f.write('<h3>{0}</h3><p>'.format(key))
            htmlText = []

            article = articles[key]
            content = article['content']
            trueStarts = article['true_start'] if 'true_start' in article else []
            trueEnds = article['true_end'] if 'true_end' in article else []
            predStarts = article['pred_start'] if 'pred_start' in article else []
            predEnds = article['pred_end'] if 'pred_end' in article else []

            for i, char in enumerate(content):
                for z in range(predStarts.count(i)):
                    htmlText.append(PRED_START)
                for z in range(predEnds.count(i)):
                    htmlText.append(PRED_END)                
                for z in range(trueStarts.count(i)):
                    htmlText.append(MARK_START)
                for z in range(trueEnds.count(i)):
                    htmlText.append(MARK_END)
                htmlText.append(char)
                

            f.write(''.join(htmlText).replace('\n', '<br>'))
            f.write('</p>')


        f.write('</BODY></HTML>')





# load label file
def loadTestLabels(article_dir, predFile) :
    articles = loadArticleFiles(article_dir)

    with open(predFile, 'r', encoding="utf-8") as f:
        spans = []
        for line in f.readlines():
            line = line.strip().split('\t')
            if len(line) > 0:
                label_id = line[0]
                span = (int(line[-2]), int(line[-1]))   

                if 'pred_start' in articles[label_id]:
                    articles[label_id]['pred_start'].append(span[0])
                else:
                    articles[label_id]['pred_start'] = [span[0]]
                if 'pred_end' in articles[label_id]:
                    articles[label_id]['pred_end'].append(span[1])
                else:
                    articles[label_id]['pred_end'] = [span[1]]

    return articles

def buildTestResults(articles):
    MARK_START = '<MARK>'
    MARK_END = '</MARK>'
    PRED_START = "<SPAN>"
    PRED_END = '</SPAN>'

    with open('result_test.html', 'w', encoding="utf-8") as f:
        f.write('<HTML><meta charset="UTF-8"><STYLE>span {color:blue;font-weight:bold;}</STYLE><BODY>')

        for key in sorted(articles.keys()):
            f.write('<h3>{0}</h3><p>'.format(key))
            htmlText = []

            article = articles[key]
            content = article['content']
            predStarts = article['pred_start'] if 'pred_start' in article else []
            predEnds = article['pred_end'] if 'pred_end' in article else []

            for i, char in enumerate(content):
                for z in range(predStarts.count(i)):
                    htmlText.append(PRED_START)
                for z in range(predEnds.count(i)):
                    htmlText.append(PRED_END)                
                htmlText.append(char)                

            f.write(''.join(htmlText).replace('\n', '<br>'))
            f.write('</p>')

        f.write('</BODY></HTML>')





def main():
    logger.setLevel(logging.INFO)
    np.set_printoptions(threshold=sys.maxsize)

    # train_article_dir = "datasets/train-articles"
    # dev_article_dir = "datasets/dev-articles"

    # SI_label_file = "datasets/train-task1-SI.labels"
    # TC_label_file = "datasets/train-task2-TC.labels"
    # dev_label_file = "datasets/dev-task-TC-template.out"
    # pred_file = "submission_SI/submission_SI_0.txt"

    # articles = loadLabels(dev_article_dir, dev_label_file, pred_file)
    # buildResults(articles)
    # print('{0} has created from {1}'.format('result.html', pred_file))

    test_article_dir = "datasets/test-articles"
    pred_file = "submission_SI/submission_SI_final_0.txt"
    articles = loadTestLabels(test_article_dir, pred_file)
    buildTestResults(articles)

    print('{0} has created from {1}'.format('result_test.html', pred_file))
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("task", nargs='?', default="SI")
    args = parser.parse_args()

    main()


