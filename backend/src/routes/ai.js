const express = require("express");
const axios = require("axios");
const fs = require("fs");
const path = require("path");
const { PYTHON_API_URL, GENERATED_DIR } = require("../config/env");
const { authenticateToken } = require("../middleware/auth");
const store = require("../models/store");

const router = express.Router();

const FALLBACK_STYLES = [
  { name: "photorealistic", display_name: "Photorealistic Portrait", description: "Crisp 85mm portrait photography with natural skin texture" },
  { name: "cinematic", display_name: "Cinematic 35mm Film", description: "Hollywood film still with atmospheric lighting" },
  { name: "studio_portrait", display_name: "High-End Studio Headshot", description: "Clean commercial studio portrait with softbox lighting" },
  { name: "cyberpunk", display_name: "Cyberpunk Neon Night", description: "Futuristic neon-lit night cityscape" },
  { name: "vintage_polaroid", display_name: "Vintage 90s Polaroid", description: "Nostalgic instant film photograph" },
  { name: "black_and_white", display_name: "Dramatic Monochrome", description: "Fine-art monochrome portrait" },
  { name: "oil_painting", display_name: "Classical Oil Painting", description: "Masterpiece Rembrandt oil portrait" },
];

/**
 * GET /api/ai/health
 * Queries the Python AI Engine for compute device (CUDA vs CPU) and cached models.
 */
router.get("/health", async (req, res) => {
  try {
    const response = await axios.get(`${PYTHON_API_URL}/health`, { timeout: 3000 });
    res.json(response.data);
  } catch {
    res.status(503).json({
      status: "unavailable",
      device: { cuda_available: false, device_type: "unknown", device_name: "Offline" },
      mock_mode_default: true,
      cached_models: [],
    });
  }
});

/**
 * POST /api/ai/train
 * Triggers LoRA model training on the Python FastAPI AI engine.
 */
router.post("/train", authenticateToken, async (req, res) => {
  try {
    const {
      trigger_word = "sks person",
      steps = 10,
      rank = 16,
      engine_mode = "instant_faceid",
      dry_run,
    } = req.body || {};
    const response = await axios.post(`${PYTHON_API_URL}/train`, {
      trigger_word,
      steps: Number(steps),
      rank: Number(rank),
      engine_mode,
      dry_run,
    });
    res.json(response.data);
  } catch (err) {
    if (err.response) {
      return res.status(err.response.status).json(err.response.data);
    }
    return res.status(503).json({
      error: "Python AI Engine unreachable. Please ensure python_api.py is running on port 8000.",
    });
  }
});

/**
 * GET /api/ai/train/status
 * Polls live training progress indicator from the Python AI engine.
 */
router.get("/train/status", authenticateToken, async (req, res) => {
  try {
    const response = await axios.get(`${PYTHON_API_URL}/train/status`);
    res.json(response.data);
  } catch (err) {
    return res.status(503).json({
      status: "unavailable",
      progress_percent: 0,
      message: "Python AI Engine offline",
    });
  }
});

/**
 * POST /api/ai/generate
 * Generates image(s) from text prompt using the trained LoRA model via Python AI engine.
 */
router.post("/generate", authenticateToken, async (req, res) => {
  try {
    const payload = req.body || {};
    if (!payload.prompt || !payload.prompt.trim()) {
      return res.status(400).json({ error: "Prompt is required to generate an image." });
    }

    const response = await axios.post(`${PYTHON_API_URL}/generate`, {
      prompt: payload.prompt.trim(),
      negative_prompt: payload.negative_prompt,
      engine_mode: payload.engine_mode || "sdxl_lora",
      base_model: payload.base_model || "stabilityai/stable-diffusion-xl-base-1.0",
      style_preset: payload.style_preset || "photorealistic",
      trigger_word: payload.trigger_word || "sks person",
      lora_scale: payload.lora_scale !== undefined ? Number(payload.lora_scale) : 0.85,
      ip_adapter_scale:
        payload.ip_adapter_scale !== undefined ? Number(payload.ip_adapter_scale) : 0.7,
      use_lcm: Boolean(payload.use_lcm),
      sampler: payload.sampler || (payload.use_lcm ? "lcm" : "dpm++_2m_karras"),
      face_id_path: payload.face_id_path || "checkpoints/person_face_id.safetensors",
      steps: payload.steps !== undefined ? Number(payload.steps) : 30,
      guidance_scale: payload.guidance_scale !== undefined ? Number(payload.guidance_scale) : 7.0,
      width: payload.width !== undefined ? Number(payload.width) : 1024,
      height: payload.height !== undefined ? Number(payload.height) : 1024,
      seed: payload.seed !== undefined && payload.seed !== "" ? Number(payload.seed) : null,
      num_images: payload.num_images ? Number(payload.num_images) : 1,
      dry_run: payload.dry_run,
    });

    const data = response.data;
    // Save generated item records into gallery store
    if (Array.isArray(data.images)) {
      data.images.forEach((img) => {
        store.addGalleryItem({
          userId: req.user.id,
          filename: img.filename,
          url: img.url,
          prompt: img.prompt,
          style_preset: img.style_preset,
          seed: img.seed,
        });
      });
    }

    res.json(data);
  } catch (err) {
    if (err.response) {
      return res.status(err.response.status).json(err.response.data);
    }
    return res.status(503).json({
      error: "Python AI Engine unreachable during image generation.",
    });
  }
});

/**
 * GET /api/ai/styles
 * Retrieves photographic style presets.
 */
router.get("/styles", async (req, res) => {
  try {
    const response = await axios.get(`${PYTHON_API_URL}/styles`, { timeout: 2500 });
    res.json(response.data);
  } catch {
    res.json({ styles: FALLBACK_STYLES });
  }
});

/**
 * GET /api/ai/gallery
 * Returns generated images merged from Python AI gallery and Express store.
 */
router.get("/gallery", authenticateToken, async (req, res) => {
  try {
    const response = await axios.get(`${PYTHON_API_URL}/gallery`, { timeout: 3000 });
    res.json(response.data);
  } catch {
    // Fallback to scanning local GENERATED_DIR directly
    if (!fs.existsSync(GENERATED_DIR)) {
      return res.json({ images: store.getGalleryItems() });
    }
    const files = fs
      .readdirSync(GENERATED_DIR)
      .filter((f) => f.toLowerCase().endsWith(".png"))
      .map((f) => {
        const stat = fs.statSync(path.join(GENERATED_DIR, f));
        const stored = store.getGalleryItems().find((g) => g.filename === f) || {};
        return {
          filename: f,
          url: `/outputs/generated/${f}`,
          prompt: stored.prompt || "",
          style_preset: stored.style_preset || "",
          seed: stored.seed || "",
          created_at: stat.mtime.toISOString(),
        };
      })
      .sort((a, b) => (a.created_at < b.created_at ? 1 : -1));

    res.json({ images: files });
  }
});

module.exports = router;
