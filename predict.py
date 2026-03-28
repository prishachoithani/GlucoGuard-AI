import numpy as np

def predict_diabetes(model, scaler, input_data):
    input_data = np.array(input_data).reshape(1, -1)
    input_data = scaler.transform(input_data)

    prediction = model.predict(input_data) 

    return "Diabetic Risk" if prediction[0] == 1 else "No Risk"
