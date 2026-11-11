const express = require("express");
const cors = require("cors");
const fs = require("fs");
const { PORT, RAW_PHOTOS_DIR, GENERATED_DIR } = require("./config/env");

const app = express();

// Ensure workspace directories exist
[RAW_PHOTOS_DIR, GENERATED_DIR].forEach((dir) => {
  if (!fs.existsSync(dir)) {
    fs.mkdirSync(dir, { recursive: true });
  }
});

// Middleware
app.use(cors());
app.use(express.json({ limit: "25mb" }));
app.use(express.urlencoded({ extended: true }));

// Static serving for raw uploaded photos and generated outputs
app.use("/uploads", express.static(RAW_PHOTOS_DIR));
app.use("/outputs/generated", express.static(GENERATED_DIR));

// Health route
app.get("/api/health", (req, res) => {
  res.json({
    status: "ok",
    service: "express-api-gateway",
    timestamp: new Date().toISOString(),
  });
});

// Global error handler
app.use((err, req, res, next) => {
  console.error("[Express Error]", err.message);
  res.status(err.status || 500).json({
    error: err.message || "Internal server error",
  });
});

if (require.main === module) {
  app.listen(PORT, () => {
    console.log(`Express API Gateway running at http://localhost:${PORT}`);
  });
}

module.exports = app;
