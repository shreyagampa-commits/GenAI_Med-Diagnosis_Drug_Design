const express = require("express");
const bcrypt = require("bcryptjs");
const jwt = require("jsonwebtoken");
const nodemailer = require("nodemailer");
const twilio = require("twilio"); // Import Twilio
const User = require("../models/User");
require("dotenv").config();

const router = express.Router();

// Email Transporter Setup
const transporter = nodemailer.createTransport({
  service: "gmail",
  auth: {
    user: process.env.EMAIL_USER,
    pass: process.env.EMAIL_PASS
  }
});

// Twilio setup
const client = twilio(process.env.TWILIO_SID, process.env.TWILIO_AUTH_TOKEN); // Twilio credentials from .env

// Signup Route - Sends OTP to Email and Phone
router.post("/signup", async (req, res) => {
  const { name, email, password, phone } = req.body;

  try {
    // Check if user already exists
    let user = await User.findOne({ email });
    if (user) {
      return res.status(400).json({ success: false, message: "User already exists" });
    }

    // Hash the password
    const hashedPassword = await bcrypt.hash(password, 10);
    
    // Generate OTPs
    const otpEmail = Math.floor(100000 + Math.random() * 900000).toString();
    const otpPhone = Math.floor(100000 + Math.random() * 900000).toString();
    
    // Set OTP expiry time (5 minutes)
    const otpExpiry = new Date();
    otpExpiry.setMinutes(otpExpiry.getMinutes() + 5);

    // Create new user object
    const newUser = new User({
      name,
      email,
      password: hashedPassword,
      otp: otpEmail,
      phone,
      otpPhone,
      otpExpiry,
    });

    // Send OTPs to email and phone before saving user
    try {
      // Send OTP via email
      await transporter.sendMail({
        from: process.env.EMAIL_USER,
        to: email,
        subject: "Verify Your Email",
        text: `Your OTP for email verification is ${otpEmail}`,
      });

      // Send OTP via SMS using Twilio
      await client.messages.create({
        body: `Your OTP for phone verification is ${otpPhone}`,
        from: process.env.TWILIO_PHONE_NUMBER, // Your Twilio phone number
        to: phone,
      });

      // Save user after sending OTPs
      await newUser.save();
      console.log("Saved user:", newUser);

      res.status(200).json({
        success: true,
        message: "OTP sent to email and phone",
      });
    } catch (otpError) {
      // If OTP sending fails, delete user record from DB
      await User.deleteOne({ email });
      return res.status(500).json({
        success: false,
        message: `Error sending OTPs: ${otpError.message}`,
      });
    }
  } catch (error) {
    res.status(500).json({
      success: false,
      message: error.message,
    });
  }
});

// Verify Email OTP Route
router.post("/verify-email", async (req, res) => {
  const { email, otp } = req.body;

  try {
    let user = await User.findOne({ email });
    if (!user) return res.status(400).json({ message: "User not found" });

    // Check Email OTP
    if (user.otp !== otp) return res.status(400).json({ message: "Invalid Email OTP" });
    if (user.otp !== otp) {
      console.log(user.otp);
      console.log(otp);
    }

    user.verifiedEmail = true; // Add a flag for email verification
    user.otp = null; // Clear the OTP after successful verification
    await user.save();

    res.status(200).json({ success: true, message: "Email verified successfully" });
  } catch (error) {
    res.status(500).json({ success: false, message: error.message });
  }
});

// Verify Phone OTP Route
router.post("/verify-phone", async (req, res) => {
  const { phone, otpPhone } = req.body;

  try {
    let user = await User.findOne({ phone });
    if (!user) return res.status(400).json({ message: "User not found" });

    // Check Phone OTP
   // const user = await User.findOne({ phone: req.body.phone });
    console.log("Fetched User:", user);

    if (user.otpPhone !== otpPhone) {
      console.log(user.otpPhone);
      console.log(otpPhone);
    }
    if (user.otpPhone !== otpPhone) return res.status(400).json({ message: "Invalid Phone OTP" });

    user.verifiedPhone = true; // Mark the phone as verified
    user.otpPhone = null; // Clear the OTP after successful verification
    await user.save();

    res.status(200).json({ success: true, message: "Phone verified successfully" });
  } catch (error) {
    res.status(500).json({ success: false, message: error.message });
  }
});



// Login Route
router.post("/login", async (req, res) => {
  const { email, password } = req.body;

  try {
    let user = await User.findOne({ email });
    console.log(user);
    if (!user) return res.status(400).json({ message: "User not found" });

    // Allow login if either email or phone is verified
    if (!user.verifiedEmail && !user.verifiedPhone) {
      return res.status(400).json({ message: "Email or phone must be verified to log in" });
    }

    const isMatch = await bcrypt.compare(password, user.password);
    if (!isMatch) return res.status(400).json({ message: "Invalid credentials" });

    const token = jwt.sign({ id: user._id }, process.env.JWT_SECRET, { expiresIn: "1h" });

    res.status(200).json({ message: "Login successful", token });
  } catch (error) {
    res.status(500).json({ message: error.message });
  }
});


module.exports = router;


