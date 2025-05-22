// server/routes/protein.js
const express = require('express');
const axios = require('axios');
const router = express.Router();

router.post('/fold', async (req, res) => {
  try {
    const { sequence } = req.body;

    const response = await axios.post(
      'https://api.esmatlas.com/foldSequence/v1/pdb/',
      sequence,
      {
        headers: {
          'Content-Type': 'application/x-www-form-urlencoded',
        }
      }
    );

    const pdbString = response.data;
    res.json({ pdb: pdbString });
  } catch (error) {
    console.error('Protein Folding Error:', error.message);
    res.status(500).json({ error: 'Failed to predict protein structure' });
  }
});

module.exports = router;