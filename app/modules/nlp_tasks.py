import MeCab

'''
CountVectorizerというvectorizerを使うと、
全ての単語において、出現するかしないかを０１で表現したベクトルとなります。

ただ、実際カテゴリ判別をする際に「の」や「です」等は不要な気がしますね。これを考慮してくれるのがTfIdfです。
	Tf = Term Frequency
	Idf = Inverse Document Frequency

の略で、Tfはドキュメント内の単語の出現頻度、Idfは全ての文章内の単語の出現頻度の逆数です。
つまりTfIdfを使うと、いろいろな文章に出てくる単語は無視して、ある文章に何回も出てくる単語は重要な語として扱うというものです。
'''
#from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

import urllib.request
import bs4
from bs4 import BeautifulSoup

def _preproc_NLP( text ):
	CONTENT_WORD_POS = ('名詞', '動詞', '形容詞', '副詞')
	STOPWORDS_URL = 'http://svn.sourceforge.jp/svnroot/slothlib/CSharp/Version1/SlothLib/NLP/Filter/StopWord/word/Japanese.txt'

	# Processing for PoS(Part of Speech)
	tagger = MeCab.Tagger()
	words = []
	
	for line in tagger.parse( text ).splitlines()[:-1]:
		surface, feature = line.split('\t')
		
		if feature.startswith(CONTENT_WORD_POS) and ',非自立,' not in feature:
			words.append(surface)
	
	# Processing for Japanese Stop Words
	soup = bs4.BeautifulSoup(urllib.request.urlopen(STOPWORDS_URL).read(), "html.parser")
	ss = str(soup).splitlines()
	stopwords = list( filter( lambda a: a != '', ss ) )

	for w in words:
		if w in stopwords:
			words.remove(w)
	
	return words


def _split_to_words( text ):

	selected_words = _preproc_NLP( text )

	'''
	tagger = MeCab.Tagger( '-O wakati' )

	try:
		res = tagger.parse( text.strip() )
	except:
		return []
	'''

	return selected_words

def get_vector_by_text_list( _items ):
	count_vect = TfidfVectorizer( analyzer=_split_to_words )
	bow = count_vect.fit_transform( _items )
	X = bow.todense()

	return [X, count_vect]