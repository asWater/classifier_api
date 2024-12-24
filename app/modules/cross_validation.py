import warnings
import re
import pandas as pd
import numpy as np
from sklearn.utils.testing import all_estimators
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV, train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
import MeCab
import sys
import time
import csv
import io
from nlp_tasks import _split_to_words as s2w
#from modules import nlp_tasks._split_to_words as s2w


CROSS_VALIDATTION_N = 4
#CSV_FILE = 'data/text/corpus.csv'
#CSV_FILE = 'data/incident_for_english.csv'
CSV_FILE = 'data/incident_utf8.csv'

LINEAR_SVC = 'LinearSVC'
LOGISTIC_REG = 'LogisticRegression'

EXCLUDED_ESTIMATORS = [
	'ClassifierChain', # Required positional argument: 'base_estimator'
	'GaussianProcessClassifier', # Segmentation Fault
	'GradientBoostingClassifier', # Too long
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

USING_CLASSIFIER = [
	#'CalibratedClassifierCV', 	# 正解率: 0.8113011893328327, 標準偏差: 0.007860441673527077, Elapsed Time is 18.2694847583770 sec: <n_splits=5, max_feature=3000, ngram_range=(1,3)>
	LINEAR_SVC, 				# 正解率: 0.804869204495146,  標準偏差: 0.006491747986168883, Elapsed Time is 5.85333895683288 sec: <n_splits=5, max_feature=3000, ngram_range=(1,3)>
	LOGISTIC_REG, 				# 正解率: 0.7968593858752075, 標準偏差: 0.006940136565695386, Elapsed Time is 8.96686840057373 sec: <n_splits=5, max_feature=3000, ngram_range=(1,3)>
	#'LogisticRegressionCV', 	# 正解率: 0.8058198486546058, 標準偏差: 0.005686157087056435, Elapsed Time is 1601.21514654159 sec: <n_splits=5, max_feature=3000, ngram_range=(1,3)>
	#'MLPClassifier', 			# 正解率: 0.7818887678565178, 標準偏差: 0.004239074316626771, Elapsed Time is 757.170982360839 sec: <n_splits=5, max_feature=3000, ngram_range=(1,3)>
	#'RidgeClassifier',			# 正解率: 0.7985453711228366, 標準偏差: 0.003928558718265873, Elapsed Time is 14.4331545829772 sec: <n_splits=5, max_feature=3000, ngram_range=(1,3)>
	#'RidgeClassifierCV',		# 正解率: 0.798648856577381,  標準偏差: 0.008609893136342918, Elapsed Time is 180.402863025665 sec: <n_splits=5, max_feature=3000, ngram_range=(1,3)>
	#'SGDClassifier', 			# 正解率: 0.8022364634304179, 標準偏差: 0.010644152669136932, Elapsed Time is 55.6776697635650 sec: <n_splits=5, max_feature=3000, ngram_range=(1,3)>
]

warnings.filterwarnings( 'ignore' )

def tokenize( text ):
	wakati = MeCab.Tagger( '-O wakati' )
	# テキストを形態素解析し、形態素の配列を返す
	parsed_text = wakati.parse( text )
	word_list = parsed_text.split( ' ' )
	return word_list


def get_textdata_and_labels():
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
	#vect = CountVectorizer( analyzer=s2w, max_df=0.5, max_features=1000 )
	vect = TfidfVectorizer( analyzer=s2w, max_df=0.5, max_features=3000, ngram_range=(1, 2) )
	#vect = TfidfVectorizer( analyzer=tokenize, max_df=0.5, max_features=2000 )
	
	#bow = vect.fit_transform( df['text'].tolist() )

	titleText = df['title'] + ' ' + df['text']
	#bow = vect.fit_transform( titleText.tolist() )
	bow = vect.fit_transform( titleText )
	
	# numpy.matrixに変換する時はtodense()を使います
	#!!! Changed to "toarray" from "todense" due t othe error "TypeError: np.matrix is not supported. Please convert to a numpy array with np.asarray"
	X = bow.toarray()

	# ラベルデータをベクトルに変換
	le = LabelEncoder()
	le.fit( df['category'] )
	Y = le.transform( df['category'] )

	return X, Y


def getGridSearchResult( clf_name, model, X, Y ):
	param4LinearSVC = {
		'C':[0.001, 0.01, 0.1, 1, 10, 100],
		'loss':('hinge', 'squared_hinge')
	}
	param4LogReg = {
		'C':[0.001, 0.01, 0.1, 1, 10, 100],
		"penalty":["l1","l2"]
	}

	X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=0 )

	print("=== Grid Search for {} ===".format( clf_name ) )

	if clf_name == LINEAR_SVC:
		params = param4LinearSVC
	elif clf_name == LOGISTIC_REG:
		params = param4LogReg
	
	gridSearch = GridSearchCV( model, params, cv=5 )
	gridSearch.fit( X_train, y_train )

	print("Test Set Score: {:.2f}".format( gridSearch.score( X_test, y_test ) ))
	print("Best Parameters: {}".format( gridSearch.best_params_ ))
	print("Best cross-validation score: {:.2f}".format( gridSearch.best_score_ ))

	if clf_name == LINEAR_SVC:
		GS_C, GS_loss = gridSearch.best_params_.values()
		clf = LinearSVC( loss=GS_loss, C=GS_C )
	elif clf_name == LOGISTIC_REG:
		GS_C, GS_pena = gridSearch.best_params_.values()
		clf = LogisticRegression( C=GS_C, penalty=GS_pena )
	else:
		clf = model
	
	return clf


'''
def get_list_by_csv( file_path ):

	# CSVから配列に変換を行う
	csv_reader = csv.reader( io.open( file_path, 'r', encoding='utf_8' ), delimiter=',', quotechar='"' )

	return [row for row in csv_reader]
'''

def main():
	#_data = get_list_by_csv( CSV_FILE )
	#_data = get_list_by_csv( 'data/incident_for_english.csv' )
	# ここでcsvのデータから、入力・出力のベクトルデータを生成しています。 その際に、入力が日本語なので tokenize 関数によって分かち書きした各単語を1hotベクトルにし、入力としています。
	X,Y = get_textdata_and_labels()

	kfold_cv = KFold( n_splits=5, shuffle=True, random_state=0 )

	# all_estimators()でscikit-learnに実装されている全モデルが取得できます。
	for (name, Estimater) in all_estimators( type_filter="classifier" ):
		#if re.match( '^[A-S]|[a-h]', name ): continue
		#if name in EXCLUDED_ESTIMATORS: continue
		if name not in USING_CLASSIFIER: continue

		model = Estimater()

		# scoreメソッドがないモデルはcross_val_scoreで交差検定ができないので今回は弾きます。
		if 'score' not in dir( model ): continue

		# Grid Search
		model = getGridSearchResult( name, model, X, Y )

		try:
			print( '>>> Validation Start for {0} at {1}'.format( name, time.ctime() ) )
			print( '>>> Paramters: {}'.format( model ) )
			startTime = time.time()
			# 学習データをCROSS_VALIDATION_N等分に分けて、そのうちの１セットは学習に使わずにモデルの評価のために使います。今回は4に設定しています。
			scores = cross_val_score( model, X, Y, cv=kfold_cv )
			#print( scores )
			print( name, "の正解率 ＝",scores )
			print( "正解率（平均）:", np.mean(scores), "標準偏差:", np.std(scores) )
			print( '<<< Validation Ended for {0} at {1}, Elapsed Time is {2} sec.'. format( name, time.ctime(), (time.time() - startTime) ))
			print()

		except:
			print( sys.exc_info() )
			pass

if __name__ == '__main__':
	main()


