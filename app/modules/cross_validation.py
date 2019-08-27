import warnings
import re
import pandas as pd
import numpy as np
from sklearn.utils.testing import all_estimators
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold, cross_val_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
#import MeCab
import sys
import time
from nlp_tasks import _split_to_words as s2w
#from modules import nlp_tasks._split_to_words as s2w


CROSS_VALIDATTION_N = 4
#CSV_FILE = 'data/text/corpus.csv'
#CSV_FILE = 'data/incident_for_english.csv'
CSV_FILE = 'data/incident_utf8.csv'
EXCLUDED_ESTIMATORS = [
	'ClassifierChain', # Required positional argument: 'base_estimator'
	'GaussianProcessClassifier', # Segmentation Fault
	'HistGradientBoostingClassifier', # Segmentation Fault
	'MultiOutputClassifier', # Missing 1 required positional argument: 'estimator'
	'NuSVC', # Missing 1 required positional argument: 'estimator'
	'OneVsOneClassifier', # Missing 1 required positional argument: 'estimator'
	'OneVsRestClassifier', # Missing 1 required positional argument: 'estimator'
	'OutputCodeClassifier', # Missing 1 required positional argument: 'estimator'
	'QuadraticDiscriminantAnalysis', # ValueError('y has only 1 sample in class 0, covariance is ill defined.',)
	'RadiusNeighborsClassifier', # ValueError('No neighbors found for test samples
	'VotingClassifier', # Missing 1 required positional argument: 'estimators'
]

warnings.filterwarnings( 'ignore' )

'''
wakati = MeCab.Tagger( '-O wakati' )

def tokenize( text ):
	# テキストを形態素解析し、形態素の配列を返す
	parsed_text = wakati.parse( text )
	word_list = parsed_text.split( ' ' )
	return word_list
'''

def get_textdata_and_labels( _data ):
	"""
	df = pd.DataFrame( _data )
	# CSV の種類により、どちらがカテゴリで、どちらがテキストか変わる。とりあえず2列のCSVファイルを想定。
	df.columns = [ 'text', 'category' ]
	#df.columns = [ 'category', 'title', 'text' ]
	"""

	# Reading CSV file with specifying the column names.
	#df = pd.read_csv( CSV_FILE, names=('text', 'category') )
	df = pd.read_csv( CSV_FILE, names=('category', 'title', 'text') )

	# Replace NaN value to space.
	# ??? Should I check whether the data has NaN by "df.isnull().any()" and if true, should we execute fillna?
	df.fillna( ' ', inplace=True )

	# テキストのBoW生成
	#count_vect = CountVectorizer( analyzer=s2w, max_df=0.5, max_features=1000 )
	count_vect = TfidfVectorizer( analyzer=s2w, max_features=1000 )
	
	#bow = count_vect.fit_transform( df['text'].tolist() )

	titleText = df['title'] + ' ' + df['text']
	bow = count_vect.fit_transform( titleText.tolist() )
	
	# numpy.matrixに変換する時はtodense()を使います
	X = bow.todense()

	# ラベルデータをベクトルに変換
	le = LabelEncoder()
	le.fit( df['category'] )
	Y = le.transform( df['category'] )

	return X, Y


import csv, io

def get_list_by_csv( file_path ):

	# CSVから配列に変換を行う
	csv_reader = csv.reader( io.open( file_path, 'r', encoding='utf_8' ), delimiter=',', quotechar='"' )

	return [row for row in csv_reader]

def main():
	_data = get_list_by_csv( CSV_FILE )
	#_data = get_list_by_csv( 'data/incident_for_english.csv' )
	# ここでcsvのデータから、入力・出力のベクトルデータを生成しています。 その際に、入力が日本語なので tokenize 関数によって分かち書きした各単語を1hotベクトルにし、入力としています。
	X,Y = get_textdata_and_labels( _data )

	kfold_cv = KFold( n_splits=4, shuffle=True )

	# all_estimators()でscikit-learnに実装されている全モデルが取得できます。
	for (name, Estimater) in all_estimators( type_filter="classifier" ):
		#if name == 'ClassifierChain': continue
		#if re.match( '^[A-S]|[a-h]', name ): continue
		if name in EXCLUDED_ESTIMATORS: continue
		if name != 'MLPClassifier': continue

		model = Estimater()

		# scoreメソッドがないモデルはcross_val_scoreで交差検定ができないので今回は弾きます。
		if 'score' not in dir( model ): continue

		try:
			print( '>>> Validation Start for {0} at {1}'.format( name, time.ctime() ) )
			startTime = time.time()
			# 学習データをCROSS_VALIDATION_N等分に分けて、そのうちの１セットは学習に使わずにモデルの評価のために使います。今回は4に設定しています。
			scores = cross_val_score( model, X, Y, cv=kfold_cv )
			#print( scores )
			print( name, "の正解率 ＝",scores )
			print( "正解率:", np.mean(scores), "標準偏差:", np.std(scores) )
			print( '<<< Validation Ended for {0} at {1}, Elapsed Time is {2} sec.'. format( name, time.ctime(), (time.time() - startTime) ))
			print()

		except:
			print( sys.exc_info() )
			pass

if __name__ == '__main__':
	main()


