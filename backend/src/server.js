const express = require("express");
const cors = require("cors");
const fs = require("fs");
const { PORT, RAW_PHOTOS_DIR, GENERATED_DIR, VIDEOS_DIR } = require("./config/env");
const authRoutes = require("./routes/auth");
const photoRoutes = require("./routes/photos");
const aiRoutes = require("./routes/ai");
const videoRoutes = require("./routes/video");

const app = express();

// Ensure workspace directories exist
[RAW_PHOTOS_DIR, GENERATED_DIR, VIDEOS_DIR].forEach((dir) => {
  if (!fs.existsSync(dir)) {
    fs.mkdirSync(dir, { recursive: true });
  }
});

// Middleware
app.use(cors());
app.use(express.json({ limit: "25mb" }));
app.use(express.urlencoded({ extended: true }));

// Static serving for raw uploaded photos, generated outputs, and videos
app.use("/uploads", express.static(RAW_PHOTOS_DIR));
app.use("/outputs/generated", express.static(GENERATED_DIR));
app.use("/outputs/videos", express.static(VIDEOS_DIR));

// Routes
app.use("/api/auth", authRoutes);
app.use("/api/photos", photoRoutes);
app.use("/api/ai", aiRoutes);
app.use("/api/video", videoRoutes);

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
