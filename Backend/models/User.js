

const mongoose =require('mongoose');
const userSchema = new mongoose.Schema({
  name: { type: String, required: true },
  email: { type: String, unique: true, required: true },
  password: { type: String, required: true },
  otp: { type: String, required: false },
  otpPhone: { type: String, required: false },  // Make it optional here
  phone: { type: String, required: true },
  verifiedEmail: { type: Boolean, default: false },
  verifiedPhone: { type: Boolean, default: false },
  otpExpiry: { type: Date },
});

module.exports = mongoose.model("User", userSchema);

