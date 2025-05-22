import pickle
import numpy as np
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS  # Import CORS
from rdkit import Chem
from rdkit.Chem import AllChem


import pickle

# Load the trained Random Forest model
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

# Debugging: Check if the model is loaded properly
print("Loaded model type:", type(model))  # Should print: RandomForestClassifier

# Load the trained Random Forest model
model_path = "model.pkl"
with open(model_path, "rb") as file:
    model = pickle.load(file)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Function to convert SMILES to Morgan fingerprint and pIC50 value
def smiles_to_features(smiles, num_value):
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return None  # Invalid SMILES

    # Convert to Morgan fingerprint (881 bits)
    fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=881)
    fingerprint_array = np.array(fingerprint)

    # Convert numeric value (assumed to be IC50) to pIC50
    molar = num_value * (10 ** -9)  # Convert nM to M
    pIC50 = -np.log10(molar) if molar > 0 else 0  # Avoid log(0) error

    # Combine fingerprint with pIC50
    return np.hstack((fingerprint_array, [pIC50]))  # Shape: (882,)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # Validate input
        if "smiles" not in data or "num_value" not in data:
            return jsonify({"error": "Missing 'smiles' or 'num_value'"}), 400

        smiles = data["smiles"]
        num_value = float(data["num_value"])  # Ensure it's a float

        # Convert input into model-compatible features
        features = smiles_to_features(smiles, num_value)
        if features is None:
            return jsonify({"error": "Invalid SMILES string"}), 400

        # Reshape and make prediction
        features = features.reshape(1, -1)  # Ensure correct input shape
        prediction = model.predict(features)

        return jsonify({"prediction": prediction.tolist()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
if __name__ == "__main__":
    app.run(debug=True, port=5001)  # Change 5001 to any available port

