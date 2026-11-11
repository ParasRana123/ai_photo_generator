const jwt = require("jsonwebtoken");
const { JWT_SECRET } = require("../config/env");

/**
 * Middleware verifying Bearer JWT tokens on protected Express endpoints.
 */
function authenticateToken(req, res, next) {
  const authHeader = req.headers["authorization"];
  const token = authHeader && authHeader.startsWith("Bearer ")
    ? authHeader.slice(7)
    : null;

  if (!token) {
    return res.status(401).json({ error: "Authentication token missing." });
  }

  try {
    const decoded = jwt.verify(token, JWT_SECRET);
    req.user = decoded;
    next();
  } catch (err) {
    return res.status(401).json({ error: "Invalid or expired authentication token." });
  }
}

module.exports = { authenticateToken };
