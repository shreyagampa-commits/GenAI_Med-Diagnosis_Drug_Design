from flask import Flask, jsonify, request
from flask_cors import CORS
import torch
import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw, Crippen, Descriptors
from io import BytesIO
import base64
from pubchempy import get_compounds

from stackRNN import StackAugmentedRNN  # Replace with actual module if different
from data import GeneratorData  # Replace with actual module if different

app = Flask(__name__)
CORS(app)

# === Configuration ===
model_path = 'latest'
gen_data_path = '123.smi'
tokens = ['<', '>', '#', '%', ')', '(', '+', '-', '/', '.', '1', '0', '3', '2', '5', '4', '7',
          '6', '9', '8', '=', 'A', '@', 'C', 'B', 'F', 'I', 'H', 'O', 'N', 'P', 'S', '[', ']',
          '\\', 'c', 'e', 'i', 'l', 'o', 'n', 'p', 's', 'r', '\n']

# === Data and Model Initialization ===
gen_data = GeneratorData(training_data_path=gen_data_path, delimiter='\t',
                         cols_to_read=[0], keep_header=True, tokens=tokens)

hidden_size = 1500
stack_width = 1500
stack_depth = 200
layer_type = 'GRU'
lr = 0.001
optimizer_instance = torch.optim.Adam

generator = StackAugmentedRNN(
    input_size=gen_data.n_characters,
    hidden_size=hidden_size,
    output_size=gen_data.n_characters,
    layer_type=layer_type,
    n_layers=1,
    is_bidirectional=False,
    has_stack=True,
    stack_width=stack_width,
    stack_depth=stack_depth,
    use_cuda=None,
    optimizer_instance=optimizer_instance,
    lr=lr
)

generator.load_model(model_path)

# === Utilities ===
def smiles_to_image_base64(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    img = Draw.MolToImage(mol, size=(300, 300))
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{img_str}"

def calculate_logp_and_pic50(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None
    logp = Crippen.MolLogP(mol)
    mw = Descriptors.MolWt(mol)
    pic50 = 10 - np.log10(mw + 1)  # Simple estimation
    return round(logp, 2), round(pic50, 2)

def name_to_smiles(name: str):
    try:
        compound = get_compounds(name, 'name')
        if compound and compound[0].isomeric_smiles:
            return compound[0].isomeric_smiles
    except:
        return None
    return None

# === Flask Route ===
@app.route("/generate", methods=["GET"])
def generate_smiles():
    name_input = request.args.get("smiles", "")
    num = int(request.args.get("num", 5))

    # Convert names to SMILES if input is alphabetical
    if all(ch.isalpha() or ch.isspace() for ch in name_input.strip()):
        smiles_input = name_to_smiles(name_input.strip())
        if not smiles_input:
            return jsonify({"error": "Invalid compound name"}), 400
    else:
        smiles_input = name_input

    results = []
    for _ in range(num * 2):  # Try more to ensure enough valid results
        generated = generator.evaluate(gen_data, predict_len=120, prime_str=smiles_input)[1:-1]
        image_data = smiles_to_image_base64(generated)
        if image_data:
            logp, pic50 = calculate_logp_and_pic50(generated)
            print(logp,pic50)
            results.append({
    "smiles": generated,
    "image": image_data,
    "logP": logp,          # ← Rename to logP
    "pIC50": pic50         # ← Rename to pIC50
})

        if len(results) == num:
            break
        print(results)

    return jsonify({"results": results})

if __name__ == "__main__":
    app.run(debug=True,port=5002)
