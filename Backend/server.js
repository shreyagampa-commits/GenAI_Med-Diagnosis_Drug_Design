require("dotenv").config();
const express = require("express");
const cors = require("cors");
const mongoose = require("./config/db");
const authRoutes = require("./routes/authRoutes");
const proteinRoutes =  require("./routes/protein");

const app = express();
app.use(express.json());
app.use(cors());

app.use("/auth", authRoutes);
app.use("/api/protein",proteinRoutes);
const PORT = process.env.PORT || 5000;
app.listen(PORT, () => console.log(`Server running on port ${PORT}`));
