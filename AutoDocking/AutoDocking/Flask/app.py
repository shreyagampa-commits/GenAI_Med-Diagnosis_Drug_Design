from flask import Flask, request, jsonify
import requests
from rcsbsearchapi.search import TextQuery
from rcsbsearchapi import rcsb_attributes as attrs
from flask_cors import CORS

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing

# Helper function to download PDB and ligand files without saving them
def download_pdb_and_ligand(ECnumber, LIGAND_ID):
    # Create query for ECnumber and LIGAND_ID
    q1 = attrs.rcsb_polymer_entity.rcsb_ec_lineage.id == ECnumber
    q2 = TextQuery(LIGAND_ID)

    query = q1 & q2  # Combine the queries
    results = list(query())

    # If no results found, return None for both
    if not results:
        return None, None

    # Extract PDB and Ligand IDs (convert to lowercase)
    pdb_id = results[0].lower()
    ligand_id = LIGAND_ID.lower()

    # Download PDB and ligand data from the RCSB PDB server
    pdb_response = requests.get(f"https://files.rcsb.org/download/{pdb_id}.pdb")
    ligand_response = requests.get(f"https://files.rcsb.org/ligands/download/{ligand_id}_ideal.sdf")

    # Check if download was successful
    if pdb_response.status_code != 200 or ligand_response.status_code != 200:
        return None, None

    # Return the content of the PDB and ligand files
    pdb_content = pdb_response.text
    ligand_content = ligand_response.text

    return pdb_content, ligand_content

# Flask route to trigger the download process and generate data for the frontend
@app.route('/process', methods=['POST'])
def process_ligand():
    try:
        # Get data from the incoming JSON request
        data = request.get_json()
        ECnumber = data.get('ECnumber')
        LIGAND_ID = data.get('LIGAND_ID')

        # Ensure that the required fields are provided
        if not ECnumber or not LIGAND_ID:
            return jsonify({"error": "ECnumber and LIGAND_ID are required"}), 400

        # Call the helper function to download PDB and ligand data
        pdb_content, ligand_content = download_pdb_and_ligand(ECnumber, LIGAND_ID)

        # If either PDB or ligand data is not found, return an error
        if not pdb_content or not ligand_content:
            return jsonify({"error": "Failed to download PDB or ligand data"}), 400

        # Return PDB content to frontend for visualization
        return jsonify({
            "pdb_content": pdb_content
        }), 200

    except Exception as e:
        # Catch unexpected errors and return a generic error message
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

# Main entry point to run the Flask application
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5008)
