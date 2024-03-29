# Module import 
from flask import Flask, render_template, request, redirect, url_for, jsonify
import os

# My own modules
from modules.classifier import predict, train

# 自身の名称を app という名前でインスタンス化する
app = Flask( __name__ )

# Port number is required to fetch from env variable
# http://docs.cloudfoundry.org/devguide/deploy-apps/environment-variable.html#PORT
cf_port = os.getenv("PORT")

# Route / & /entry ===============================================================
@app.route('/')
@app.route('/entry')
def entry_page() -> 'html':
	return render_template('entry.html',
							the_title='Welcome to My Classifier on the web!')

# Route /predict ================================================================
@app.route('/predict', methods=['POST'])
def do_predict() -> 'html':
	phrase = request.form['phrase']
	title = 'Here is the result.'
	results = predict( phrase )

	return render_template('results.html',
						   the_title = title,
						   the_phrase = phrase,
						   the_results = results,)

# Route /predict-api ================================================================
@app.route('/predict-api', methods=['POST'])
def do_predict_api():
	phrase = request.form['phrase']
	result = predict( phrase )
	return jsonify({
			'phrase': phrase,
			'category': result
		})

# Route /train ================================================================
@app.route('/train', methods=['GET'])
def do_train():
	result = train()
	return result


# Run the applicaiton ===============================================================
if __name__ == '__main__':
	if cf_port is None:
		app.run( host='0.0.0.0', port=5000, debug=True )
	else:
		app.run( host='0.0.0.0', port=int(cf_port), debug=True )