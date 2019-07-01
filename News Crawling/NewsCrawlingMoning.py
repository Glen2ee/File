from newspaper import Article
from konlpy.tag import Kkma
from konlpy.tag import Twitter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize
import numpy as np
import datetime
import re
import requests
import pandas as pd
import smtplib
from email.mime.text import MIMEText

smtp = smtplib.SMTP('smtp.gmail.com', 587)
smtp.ehlo()  # say Hello
smtp.starttls()  # TLS 사용시 필요
smtp.login('eamudo27@gmail.com', 'dlrlgus12#')


class SentenceTokenizer(object):
    def __init__(self):
        self.kkma = Kkma()
        self.twitter = Twitter()
        self.stopwords = ['중인', '만큼', '마찬가지', '꼬집었', "연합뉴스", "데일리", "동아일보", "중앙일보", "조선일보", "기자"
            , "아", "휴", "아이구", "아이쿠", "아이고", "어", "나", "우리", "저희", "따라", "의해", "을", "를", "에", "의", "가", "서울경제"
            , "이데일리", "기사", '[서울경제]', '( 서울= 연합뉴스)', '【 서울= 뉴시스】']

    def get_title(url):
        a = Article(url, language='ko')
        a.download()
        a.parse()
        return a.title

    def url2sentences(self, url):
        article = Article(url, language='ko')
        article.download()
        article.parse()
        sentences = self.kkma.sentences(article.text)
        for idx in range(0, len(sentences)):
            if len(sentences[idx]) <= 10:
                sentences[idx - 1] += (' ' + sentences[idx])
                sentences[idx] = ''
        return sentences

    def text2sentences(self, text):
        sentences = self.kkma.sentences(text)
        for idx in range(0, len(sentences)):
            if len(sentences[idx]) <= 10:
                sentences[idx - 1] += (' ' + sentences[idx])
                sentences[idx] = ''
        return sentences

    def get_nouns(self, sentences):
        nouns = []
        for sentence in sentences:
            if sentence is not '':
                nouns.append(' '.join([noun for noun in self.twitter.nouns(str(sentence))
                                       if noun not in self.stopwords and len(noun) > 1]))

        return nouns


class GraphMatrix(object):
    def __init__(self):
        self.tfidf = TfidfVectorizer()
        self.cnt_vec = CountVectorizer()
        self.graph_sentence = []

    def build_sent_graph(self, sentence):
        tfidf_mat = self.tfidf.fit_transform(sentence).toarray()
        self.graph_sentence = np.dot(tfidf_mat, tfidf_mat.T)
        return self.graph_sentence

    def build_words_graph(self, sentence):
        cnt_vec_mat = normalize(self.cnt_vec.fit_transform(sentence).toarray().astype(float), axis=0)
        vocab = self.cnt_vec.vocabulary_
        return np.dot(cnt_vec_mat.T, cnt_vec_mat), {vocab[word]: word for word in vocab}


class Rank(object):
    def get_ranks(self, graph, d=0.85):  # d = damping factor
        A = graph
        matrix_size = A.shape[0]
        for id in range(matrix_size):
            A[id, id] = 0  # diagonal 부분을 0으로
            link_sum = np.sum(A[:, id])  # A[:, id] = A[:][id]
            if link_sum != 0:
                A[:, id] /= link_sum
            A[:, id] *= -d
            A[id, id] = 1
        B = (1 - d) * np.ones((matrix_size, 1))
        ranks = np.linalg.solve(A, B)  # 연립방정식 Ax = b
        return {idx: r[0] for idx, r in enumerate(ranks)}


class TextRank(object):
    def __init__(self, text):
        self.sent_tokenize = SentenceTokenizer()
        if text[:5] in ('http:', 'https'):
            self.sentences = self.sent_tokenize.url2sentences(text)
        else:
            self.sentences = self.sent_tokenize.text2sentences(text)

        self.nouns = self.sent_tokenize.get_nouns(self.sentences)
        self.graph_matrix = GraphMatrix()
        self.sent_graph = self.graph_matrix.build_sent_graph(self.nouns)
        self.words_graph, self.idx2word = self.graph_matrix.build_words_graph(self.nouns)
        self.rank = Rank()
        self.sent_rank_idx = self.rank.get_ranks(self.sent_graph)
        self.sorted_sent_rank_idx = sorted(self.sent_rank_idx, key=lambda k: self.sent_rank_idx[k], reverse=True)
        self.word_rank_idx = self.rank.get_ranks(self.words_graph)
        self.sorted_word_rank_idx = sorted(self.word_rank_idx, key=lambda k: self.word_rank_idx[k], reverse=True)

    def summarize(self, sent_num=3):
        summary = []
        index = []
        for idx in self.sorted_sent_rank_idx[:sent_num]:
            index.append(idx)
        index.sort()
        for idx in index:
            summary.append(self.sentences[idx])
        return summary

    def keywords(self, word_num=10):
        rank = Rank()
        rank_idx = rank.get_ranks(self.words_graph)
        sorted_rank_idx = sorted(rank_idx, key=lambda k: rank_idx[k], reverse=True)
        keywords = []
        index = []
        for idx in sorted_rank_idx[:word_num]:
            index.append(idx)
        for idx in index:
            keywords.append(self.idx2word[idx])
        return keywords


class Search(object):
    def fun(msg):
        result_dic = {'Title': [], 'Content': [], 'Url': []}
        page_num = 1
        last_page = 1
        user_agent = "'Mozilla/5.0"
        headers = {"User-Agent": user_agent}
        today = datetime.date.today()
        yesterday = today - datetime.timedelta(1)
        yesterday = yesterday.strftime("%Y%m%d")
        eval_d = yesterday
        count = 0

        page_url = "https://news.naver.com/main/ranking/popularDay.nhn?rankingType=popular_day&sectionId=101&date=" + str(
            eval_d)
        print(eval_d)
        response = requests.get(page_url, headers=headers)
        html = response.text
        #     print(page_url)
        """
        주어진 HTML에서 기사 URL을 추출한다.
        """
        url_frags = re.findall('<a href="(.*?)"', html)
        urls = []

        for url_frag in url_frags:
            if "rankingSectionId=101" in url_frag and "aid" in url_frag and eval_d in url_frag:
                temp = ''
                temp = 'https://news.naver.com' + url_frag
                if temp not in urls:
                    urls.append(temp)
            if len(urls) == 30:
                break
        print(urls)
        print(len(urls))
        print('------' * 10)
        for item in urls:
            content = ''
            while 'amp;' in item:
                item = item.replace('amp;', '', 4)

            title = SentenceTokenizer.get_title(item)
            if title in result_dic['Title']:
                continue

            textrank = TextRank(item)
            for ix, row in enumerate(textrank.summarize(5)):
                space = 120

                if len(row) <= space:
                    content += '%d. %s ' % (ix + 1, row)
                    content += '\n'
                    content += '\n'
                else:

                    content += row[:space]
                    content += '\n'
                    while len(row[space:]) > 120:
                        content += row[space:space + 120]
                        content += '\n'
                        space += 120

                    content += row[space:]
                    content += '\n'
                    content += '\n'

            result_dic['Title'].append(title)
            result_dic['Content'].append(content)
            result_dic['Url'].append(item)

        result = pd.DataFrame(data=result_dic)
        print(result)

        total = ''
        for x in range(len(result)):
            total += 'Title : '
            total += result.loc[x]['Title']
            total += '\n'

            total += 'Content : '
            total += result.loc[x]['Content']
            total += '\n'

            total += 'Url : '
            total += result.loc[x]['Url']
            total += '\n'
            total += '\n'
        msg = MIMEText(total)
        msg['Subject'] = '%s news' % eval_d
        smtp.sendmail('eamudo27@gmail.com', 'znflxk@naver.com', msg.as_string())
        print(result)
        return result


# -----------------------------------------------------------------------------------------------------------------------------


import sys
from PyQt5.QtWidgets import *  # GUI 구축을 위한 라이브러리 import
from PyQt5 import uic  # GUI 구축을 위한 라이브러리 import
from PyQt5 import QtCore  # GUI 구축을 위한 라이브러리 import

form_class = uic.loadUiType("news.ui")[0]  # 미리 구현한 GUI 시스템 import


class MyWindow(QMainWindow, form_class):
    def __init__(self):
        # 생성자
        super().__init__()
        self.setupUi(self)  # GUI 환경 구축

        # 검색 엔진 적용
        self.engine = Search()
        self.result = self.engine.fun()
        self.setTableWidgetData()

    def setTableWidgetData(self):
        # Table 설정
        self.column_num = 3
        column_headers = ['Title', 'Content', 'Url']
        self.column_idx_lookup = {k: i for i, k in enumerate(self.result.columns)}
        self.tableWidget = self.tableWidget

        self.tableWidget.setRowCount(len(self.result))  # 최종 30개의 Row 추출
        self.tableWidget.setColumnCount(self.column_num)
        self.tableWidget.setHorizontalHeaderLabels(column_headers)
        result = self.result

        for k, v in result.items():  # 결과 추출
            col = self.column_idx_lookup[k]
            for row, val in enumerate(v):
                val = str(val)
                item = QTableWidgetItem(val)
                self.tableWidget.setItem(row, col, item)

        self.tableWidget.resizeColumnsToContents()  # Column 크기에 맞게 Resize
        self.tableWidget.resizeRowsToContents()  # Row 크기에 맞게 Resize


if __name__ == "__main__":  # 실행
    app = QApplication(sys.argv)
    myWindow = MyWindow()
    myWindow.show()
    sys.exit(app.exec_())
