const express = require("express");
const multer = require("multer");
const path = require("path");
const fs = require("fs");
const { RAW_PHOTOS_DIR } = require("../config/env");
const { authenticateToken } = require("../middleware/auth");

const router = express.Router();

const ALLOWED_EXTENSIONS = new Set([".jpg", ".jpeg", ".png", ".webp"]);

const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    if (!fs.existsSync(RAW_PHOTOS_DIR)) {
      fs.mkdirSync(RAW_PHOTOS_DIR, { recursive: true });
    }
    cb(null, RAW_PHOTOS_DIR);
  },
  filename: (req, file, cb) => {
    const ext = path.extname(file.originalname).toLowerCase() || ".png";
    const safeName = file.originalname
      .replace(ext, "")
      .replace(/[^a-zA-Z0-9_-]/g, "_")
      .slice(0, 40);
    cb(null, `${Date.now()}_${safeName}${ext}`);
  },
});

const upload = multer({
  storage,
  limits: { fileSize: 15 * 1024 * 1024, files: 25 },
  fileFilter: (req, file, cb) => {
    const ext = path.extname(file.originalname).toLowerCase();
    if (!ALLOWED_EXTENSIONS.has(ext)) {
      return cb(new Error("Only JPG, PNG, and WEBP images are allowed."));
    }
    cb(null, true);
  },
});

function listRawPhotos() {
  if (!fs.existsSync(RAW_PHOTOS_DIR)) return [];
  return fs
    .readdirSync(RAW_PHOTOS_DIR)
    .filter((file) => ALLOWED_EXTENSIONS.has(path.extname(file).toLowerCase()))
    .map((file) => {
      const fullPath = path.join(RAW_PHOTOS_DIR, file);
      const stat = fs.statSync(fullPath);
      return {
        filename: file,
        url: `/uploads/${encodeURIComponent(file)}`,
        sizeBytes: stat.size,
        uploadedAt: stat.mtime.toISOString(),
      };
    })
    .sort((a, b) => (a.uploadedAt < b.uploadedAt ? 1 : -1));
}

/**
 * POST /api/photos/upload
 * Accepts 10-15+ personal photos for Phase 1 LoRA dataset preparation.
 */
router.post("/upload", authenticateToken, upload.array("photos", 25), (req, res) => {
  const uploaded = (req.files || []).map((f) => ({
    filename: f.filename,
    url: `/uploads/${encodeURIComponent(f.filename)}`,
    sizeBytes: f.size,
  }));

  const allPhotos = listRawPhotos();
  res.status(201).json({
    message: `Uploaded ${uploaded.length} photo(s) successfully.`,
    uploaded,
    totalCount: allPhotos.length,
    readyToTrain: allPhotos.length >= 10,
  });
});

/**
 * GET /api/photos
 * Lists all uploaded reference photos and dataset readiness.
 */
router.get("/", authenticateToken, (req, res) => {
  const photos = listRawPhotos();
  res.json({
    photos,
    totalCount: photos.length,
    recommendedMin: 10,
    recommendedMax: 15,
    readyToTrain: photos.length >= 10,
  });
});

/**
 * DELETE /api/photos/:filename
 * Deletes a specific uploaded photo safely.
 */
router.delete("/:filename", authenticateToken, (req, res) => {
  const safeFilename = path.basename(req.params.filename);
  const targetPath = path.join(RAW_PHOTOS_DIR, safeFilename);

  if (!fs.existsSync(targetPath)) {
    return res.status(404).json({ error: "Photo not found." });
  }

  fs.unlinkSync(targetPath);
  const remaining = listRawPhotos();
  res.json({
    message: `Deleted ${safeFilename}.`,
    totalCount: remaining.length,
    readyToTrain: remaining.length >= 10,
  });
});

module.exports = router;
