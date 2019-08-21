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
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

def _split_to_words( text ):
	tagger = MeCab.Tagger( '-O wakati' )

	try:
		res = tagger.parse( text.strip() )
	except:
		return []

	return res

def get_vector_by_text_list( _items ):
	count_vect = TfidfVectorizer( analyzer=_split_to_words )
	bow = count_vect.fit_transform( _items )
	X = bow.todense()

	return [X, count_vect]