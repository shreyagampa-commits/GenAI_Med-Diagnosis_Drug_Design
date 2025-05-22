from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QLabel, QMainWindow, QApplication
from rdkit import Chem
from rdkit.Chem.Draw import MolToImage
import sys
import os

class SmilesVisualizer(QMainWindow):
    def __init__(self, smiles):
        super().__init__()
        
        # Create molecule from SMILES
        self.mol = Chem.MolFromSmiles(smiles)
        if not self.mol:
            raise ValueError(f"Invalid SMILES: {smiles}")
            
        # Setup UI
        self.label = QLabel()
        self.setCentralWidget(self.label)
        self.setWindowTitle("SMILES Visualizer")
        
        # Display molecule
        self.display_molecule()
        
    def display_molecule(self):
        """Display the molecule as an image in the window"""
        try:
            # Generate image using RDKit
            img = MolToImage(self.mol, size=(400, 400))
            
            # Save to temporary file
            temp_file = "temp_molecule.png"
            img.save(temp_file)
            
            # Load and display the image
            pixmap = QPixmap(temp_file)
            self.label.setPixmap(pixmap)
            
            # Clean up
            os.remove(temp_file)
            
            # Resize window to fit image
            self.resize(pixmap.width(), pixmap.height())
            
        except Exception as e:
            self.label.setText(f"Visualization Error: {str(e)}")

if __name__ == "__main__":
    # Example usage
    app = QApplication(sys.argv)
    window = SmilesVisualizer("CCO")  # Test with ethanol
    window.show()
    sys.exit(app.exec_())