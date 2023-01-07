import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI
import pickle
import onnxruntime as rt
from fastapi.templating import Jinja2Templates


#app = Flask(__name__)
# Create application object
app = FastAPI()

templates = Jinja2Templates(directory="templates")

## Load the model
rf_model = pickle.load(open('final_RF_model.pkl','rb'))
#scaler = pickle.load(open('scaling.pkl','rb'))


# API endpoints
@app.get('/')
def index():
    return templates.TemplateResponse('home.html')
	#return render_template('home.html')

#---------------------------------------
# Update the following API 
#---------------------------------------

@app.route('/predict_api', methods = ['POST'])
def predict_api():
    data = request.json['data']
    print(data)
    print(np.array(list(data.values())).reshape(1,-1))
    new_data = scaler.transform(np.array(list(data.values())).reshape(1,-1))
    output = reg_model.predict(new_data)
    print(output[0])
    return jsonify(output[0])

@app.route('/predict', methods = ['POST'])
def predict():
    data = [float(x) for x in request.form.values()]
    final_input = scaler.transform(np.array(data).reshape(1,-1))
    print(final_input)
    output = reg_model.predict(final_input)[0]
    return render_template("home.html", prediction_text = "Predicted price is {}".format(output))

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")