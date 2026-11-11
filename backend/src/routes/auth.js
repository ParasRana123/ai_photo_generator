const express = require("express");
const bcrypt = require("bcryptjs");
const jwt = require("jsonwebtoken");
const store = require("../models/store");
const { JWT_SECRET, JWT_EXPIRES_IN } = require("../config/env");
const { authenticateToken } = require("../middleware/auth");

const router = express.Router();

function issueToken(user) {
  return jwt.sign(
    { id: user.id, email: user.email, name: user.name },
    JWT_SECRET,
    { expiresIn: JWT_EXPIRES_IN }
  );
}

/**
 * POST /api/auth/register
 * Creates a new user account and returns a signed JWT.
 */
router.post("/register", async (req, res) => {
  try {
    const { name, email, password } = req.body || {};

    if (!name || !email || !password) {
      return res.status(400).json({ error: "Name, email, and password are required." });
    }
    if (password.length < 6) {
      return res.status(400).json({ error: "Password must be at least 6 characters long." });
    }

    const existing = store.findUserByEmail(email);
    if (existing) {
      return res.status(409).json({ error: "An account with this email already exists." });
    }

    const passwordHash = await bcrypt.hash(password, 10);
    const user = store.createUser({ name: name.trim(), email: email.trim(), passwordHash });
    const token = issueToken(user);

    return res.status(201).json({
      token,
      user: { id: user.id, name: user.name, email: user.email, createdAt: user.createdAt },
    });
  } catch (err) {
    return res.status(500).json({ error: "Failed to register user." });
  }
});

/**
 * POST /api/auth/login
 * Validates user credentials and returns a signed JWT.
 */
router.post("/login", async (req, res) => {
  try {
    const { email, password } = req.body || {};

    if (!email || !password) {
      return res.status(400).json({ error: "Email and password are required." });
    }

    const user = store.findUserByEmail(email);
    if (!user) {
      return res.status(401).json({ error: "Invalid email or password." });
    }

    const isMatch = await bcrypt.compare(password, user.passwordHash);
    if (!isMatch) {
      return res.status(401).json({ error: "Invalid email or password." });
    }

    const token = issueToken(user);
    return res.json({
      token,
      user: { id: user.id, name: user.name, email: user.email, createdAt: user.createdAt },
    });
  } catch (err) {
    return res.status(500).json({ error: "Login failed." });
  }
});

/**
 * GET /api/auth/me
 * Returns authenticated user profile.
 */
router.get("/me", authenticateToken, (req, res) => {
  const user = store.findUserById(req.user.id);
  if (!user) {
    return res.status(404).json({ error: "User not found." });
  }
  return res.json({
    user: { id: user.id, name: user.name, email: user.email, createdAt: user.createdAt },
  });
});

module.exports = router;
