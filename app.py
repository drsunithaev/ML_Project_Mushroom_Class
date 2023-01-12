import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI
import pickle
import onnxruntime as rt
from fastapi.templating import Jinja2Templates
from variables import MushroomVariables

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

@app.post('/predict_api')
def predict_api(data : MushroomVariables):
    data = data.dict()

    # fetch input data using data variables
    cap_shape = data['cap_shape']
    cap_surface = data['cap_surface']
    cap_color = data['cap_color']
    bruises = data['bruises']
    odor = data['odor']
    gill_attachment = data['gill_attachment']
    gill_spacing = data['gill_spacing']
    gill_size = data['gill_size']
    gill_color = data['gill_color']
    stalk_shape = data['stalk_shape']
    stalk_root = data['stalk_root']
    stalk_surface_above_ring = data['stalk_surface_above_ring']
    stalk_surface_below_ring = data['stalk_surface_below_ring']
    stalk_color_above_ring = data['stalk_color_above_ring']
    stalk_color_below_ring = data['stalk_color_below_ring']
    veil_type = data['veil_type']
    veil_color = data['veil_color']
    ring_number = data['ring_number']
    ring_type = data['ring_type']
    spore_print_color = data['spore_print_color']
    population = data['population']
    habitat = data['habitat']

    data_to_pred = np.array([[cap_shape, cap_surface, cap_color,bruises , odor, gill_attachment, gill_spacing ,gill_size ,gill_color , stalk_shape, stalk_root , stalk_surface_above_ring , stalk_surface_below_ring , stalk_color_above_ring , stalk_color_below_ring, veil_type , veil_color , ring_number , ring_type , spore_print_color , population ,habitat]])
    output = rf_model.predict(data_to_pred)
    if output[0] ==1:
        prediction = "Edible"
    else:
        prediction = "Poisonous"
    
    return {'prediction': prediction}

'''
@app.route('/predict', methods = ['POST'])
def predict():
    data = [float(x) for x in request.form.values()]
    final_input = scaler.transform(np.array(data).reshape(1,-1))
    print(final_input)
    output = reg_model.predict(final_input)[0]
    return render_template("home.html", prediction_text = "Predicted price is {}".format(output))

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")

    '''