from flask import Flask, request, jsonify
import torch
from torch import nn
import joblib

class FraudDetectionModel(nn.Module):
    def __init__(self, input_size=30, hidden_sizes=[94, 42]):
        super(FraudDetectionModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1], 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)


app = Flask(__name__)

model = FraudDetectionModel()
model.load_state_dict(torch.load("fraud_detection_model.pth"))
model.eval()


scaler = joblib.load("scaler.pkl")

@app.route("/")
def home():
    return "Hello World"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        if not data or "inputs" not in data:
            return jsonify({"error": "Invalid input"}), 400

        if len(data["inputs"]) != 30: 
            return jsonify({"error": "Invalid number of input features"}), 400

        inputs = scaler.transform([data["inputs"]])
        inputs = torch.tensor(inputs, dtype=torch.float32)

   
        with torch.no_grad():
            probabilities = model(inputs).item()  
            label = 1 if probabilities > 0.5 else 0


        return jsonify({"probability": probabilities, "label": label})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
