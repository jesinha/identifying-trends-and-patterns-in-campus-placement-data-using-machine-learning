from flask import Flask, request,render_template
import pickle
import sklearn


app = Flask(__name__)

model = pickle.load(open('placement.pkl','rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/predict')
def predict():
    return render_template('predict.html')

@app.route('/pred', methods=['post'])
def pred():
    sen1 = request.form['sen1']
    sen2 = request.form['sen2']
    sen3 = request.form['sen3']
    sen4 = request.form['sen4']
    sen5 = request.form['sen5']
    sen6 = request.form['sen6']
    variables = [[int(sen1), int(sen2), int(sen3), int(sen4),int(sen5), int(sen6)]]

    model.predict(variables)
    output = model.predict(variables)
    return render_template('submit.html',Y=output[0])


if __name__ == "__main__":
    app.run(debug=True)