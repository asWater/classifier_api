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
from bs4 import BeautifulSoup
import emoji
import re

def _normalize_text( text ):
	# Chanage Alphabet to small letters.
	text = text.lower()

	# Removing HTML tags
	text = BeautifulSoup( text, "html.parser" ).get_text()

	# Removing Emoji
	#text = ''.join(c for c in text if c not in emoji.UNICODE_EMOJI)
	text = ''.join(c for c in text if c not in emoji.EMOJI_DATA)

	# Removing other redundant letters
	#text = re.sub( r'[!-~]', "", text ) #半角記号,数字,英字
	#text = re.sub( r'[ -/:-@\[-~]', "", text) #半角記号 -> 小文字アルファベットも消されてしまう。
	text = re.sub( r'[!-/:-@[-`{-~]', "", text) #半角記号 (半角空白を除く)
	text = re.sub( r'[0-9０-９]', "", text) # 全角/半角 数字
	text = re.sub( r'[︰-＠]', "", text ) #全角記号。 こっちにしたほうがいいかも？ -> [！-／：-＠［-｀｛-～、-〜”’・]
	text = re.sub( '\n', " ", text ) #改行文字

	return text



def _preproc_NLP( text ):
	CONTENT_WORD_POS = ('名詞', '動詞', '形容詞', '副詞')
	STOPWORDS_URL = 'http://svn.sourceforge.jp/svnroot/slothlib/CSharp/Version1/SlothLib/NLP/Filter/StopWord/word/Japanese.txt'

	# Processing for PoS(Part of Speech)
	tagger = MeCab.Tagger()
	words = []
	
	'''
	!!! Following process issues "ValueError: too many values to unpack (expected 2)", so replaced to "parseToNode" !!!

	for line in tagger.parse( text ).splitlines()[:-1]:
		# !!! > The following statement issues the error "ValueError: too many values to unpack (expected 2)"
		surface, feature = line.split('\t')
		
		if feature.startswith(CONTENT_WORD_POS) and ',非自立,' not in feature:
			words.append(surface)

	'''
	# Replaced processing (changed to "parseToNode" from "parse")
	node = tagger.parseToNode( text )
	while node:
		if node.feature.startswith(CONTENT_WORD_POS) \
		   and '非自立' not in node.feature:
			words.append(node.surface)
		node = node.next
	
	
	# Processing for Japanese Stop Words
	# !!! Replaced due to the error "http.client.RemoteDisconnected: Remote end closed connection without response"

	#soup = BeautifulSoup(urllib.request.urlopen(STOPWORDS_URL).read(), "html.parser")
	#ss = str(soup).splitlines()
	f = open('app/modules/Japanese.txt', 'r', encoding='utf-8')
	ss = str(f.read()).splitlines()
	stopwords = list( filter( lambda a: a != '', ss ) )

	# listをforループで回してremoveしたら思い通りにならない (https://www.haya-programming.com/entry/2018/06/02/163415)
	removed_stopwords = [ x for x in words if x not in stopwords ]

	return removed_stopwords


def _split_to_words( text ):

	text = _normalize_text( text )
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
	#!!! Changed to "toarray" from "todense" due t othe error "TypeError: np.matrix is not supported. Please convert to a numpy array with np.asarray"
	X = bow.toarray()

	return [X, count_vect]