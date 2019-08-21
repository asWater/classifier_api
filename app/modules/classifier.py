import sys

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
#from sklearn.externals import joblib
import joblib
import os.path
from modules import nlp_tasks

from sklearn.neural_network import MLPClassifier  # アルゴリズムとしてmlpを使用

sys.modules['nlp_tasks'] = nlp_tasks

# Constants
#CSV_FILE = 'data/text/corpus.csv'
CSV_FILE = 'data/incident_utf8.csv'

class MyMLPClassifier():
	model = None
	model_name = 'mlp'

	def get_model_path( self, type='model' ):
		return 'app/models/' + self.model_name + '_' + type + '.pkl'

	def get_vector( self, text ):
		return self.vectorizer.transform( [text] )

	def load_model( self ):
		if os.path.exists( self.get_model_path() ) == False:
			raise Exception( 'No model file found!' )
		
		self.model = joblib.load( self.get_model_path() )
		self.classes = joblib.load( self.get_model_path( 'class' ) ).tolist()
		self.vectorizer = joblib.load( self.get_model_path( 'vect' ) )
		self.le = joblib.load( self.get_model_path( 'le' ) )

	def train( self, csvfile ):
		#df = pd.read_csv( csvfile, names=('text', 'category') )
		df = pd.read_csv( csvfile, names=('category', 'title', 'text') )

		# Replace NaN value to space.
		# ??? Should I check whether the data has NaN by "df.isnull().any()" and if true, should we execute fillna?
		df.fillna( ' ', inplace=True )

		#contents = df['text']
		contents = df['title'] + ' ' + df['text']

		X, vectorizer = nlp_tasks.get_vector_by_text_list( contents )

		# Loading Lables
		le = LabelEncoder()
		le.fit( df['category'] )
		Y = le.transform( df['category'] )

		model = MLPClassifier( max_iter=300, hidden_layer_sizes=(100,), verbose=10, )
		model.fit( X, Y )

		# Save Models
		joblib.dump( model, self.get_model_path() )
		joblib.dump( le.classes_, self.get_model_path( 'class' ) )
		joblib.dump( vectorizer, self.get_model_path( 'vect' ) )
		joblib.dump( le, self.get_model_path( 'le' ) )

		self.model = model
		self.classes = le.classes_.tolist()
		self.vectorizer = vectorizer

	def predict( self, query ):
		X = self.vectorizer.transform( [query] )
		key = self.model.predict( X )

		return self.classes[ key[0] ]


def train():
	classifier = MyMLPClassifier()
	classifier.train( CSV_FILE )

def predict( text ):
	classifier = MyMLPClassifier()
	classifier.load_model()
	result = classifier.predict( text )
	return result


if __name__ == '__main__':
	argvs = sys.argv
	argc = len(argvs)

	if argc <= 1:
		print( '!!! You have to specify the option "train" or "predict".!!!\n!!! Even if you specified more than 2 options for "train", only the first option is taken care. !!!' )
		sys.exit()
	else:
		option = argvs[1]

	if option == 'train':
		print( '### Trainig is started based on the file {} ###'.format( CSV_FILE ) )
		train()
	elif option == 'predict':
		if argc <= 2:
			print( '!!! You have to input the string that should be classified by this application !!!' )
			sys.exit()
		else:
			question = argvs[2]
			print( '>>> The string should be categorized is "{}" <<<'.format( question ) )
			predict( question )
	else:
		print( '!!! Option must be "train" or "predict".!!!\n!!! Even if you specified more than 2 options for "train", only the first option is taken care. !!!' )