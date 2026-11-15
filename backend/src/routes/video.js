const express = require("express");
const axios = require("axios");
const fs = require("fs");
const path = require("path");
const { PYTHON_API_URL, VIDEOS_DIR } = require("../config/env");
const { authenticateToken } = require("../middleware/auth");

const router = express.Router();

const FALLBACK_VIDEO_STYLES = [
  { name: "cinematic", display_name: "Cinematic 35mm", description: "Hollywood movie look, anamorphic lens flare, shallow depth of field, 24fps film cadence" },
  { name: "slow_motion", display_name: "High-Speed Slow Motion", description: "120fps high-speed capture, silky smooth fluid motion, micro-detail" },
  { name: "drone_aerial", display_name: "Drone Aerial Sweep", description: "High-altitude sweeping drone camera motion, sweeping vista landscape" },
  { name: "hyperlapse", display_name: "City Hyperlapse", description: "High-speed temporal compression, fast-moving clouds and streaking light trails" },
  { name: "candid_smartphone", display_name: "Candid Smartphone Video", description: "Authentic iPhone 4K HDR footage, handheld natural motion blur" },
  { name: "animated_3d", display_name: "Pixar 3D Animation", description: "Subsurface scattering, vibrant studio lighting, expressive character animation" },
  { name: "anime_shonen", display_name: "Modern Anime Film", description: "Makoto Shinkai aesthetic, luminous atmospheric skies, vivid saturated colors" },
  { name: "vintage_vhs", display_name: "Vintage VHS Tape", description: "1990s magnetic tape scanlines, chromatic aberration, retro analog artifacts" },
];

const FALLBACK_MOTIONS = [
  { name: "none", display_name: "Static Shot", description: "Camera locked off on a sturdy tripod with zero movement" },
  { name: "zoom_in", display_name: "Push-In Zoom", description: "Smooth cinematic zoom into the primary subject" },
  { name: "zoom_out", display_name: "Pull-Out Reveal", description: "Expansive camera pull backwards to reveal surroundings" },
  { name: "pan_left", display_name: "Horizontal Pan Left", description: "Camera pans smoothly across the horizontal horizon leftward" },
  { name: "pan_right", display_name: "Horizontal Pan Right", description: "Camera pans smoothly across the horizontal horizon rightward" },
  { name: "tilt_up", display_name: "Pedestal Tilt Up", description: "Camera tilts upwards towards the sky or subject's head" },
  { name: "tilt_down", display_name: "Pedestal Tilt Down", description: "Camera tilts downwards towards foreground or ground" },
  { name: "orbit_cw", display_name: "Circular Orbit", description: "Camera revolves 360-degrees around the focal subject" },
  { name: "roll_cw", display_name: "Dutch Roll Angle", description: "Cinematic barrel roll rotation along the camera optical axis" },
];

/**
 * Helper to call Python API with retry
 */
async function callPythonWithRetry(requestFn, retries = 2, delayMs = 1200) {
  for (let attempt = 0; attempt <= retries; attempt++) {
    try {
      return await requestFn();
    } catch (err) {
      const isNetworkErr =
        err.code === "ECONNREFUSED" ||
        err.code === "ETIMEDOUT" ||
        err.message?.includes("ECONNREFUSED") ||
        err.message?.includes("Network Error");
      if (isNetworkErr && attempt < retries) {
        await new Promise((resolve) => setTimeout(resolve, delayMs));
        continue;
      }
      throw err;
    }
  }
}

/**
 * POST /api/video/generate
 * Dispatches text-to-video generation to the Python AI engine.
 */
router.post("/generate", async (req, res, next) => {
  try {
    const {
      prompt,
      negative_prompt,
      camera_motion = "none",
      motion_intensity = 1.0,
      duration_seconds = 2.0,
      fps = 16,
      width = 512,
      height = 512,
      provider = "auto",
      seed,
      style_preset = "cinematic",
      export_gif = true,
      dry_run,
      api_key,
      hf_token,
    } = req.body || {};

    if (!prompt || typeof prompt !== "string" || !prompt.trim()) {
      return res.status(400).json({ error: "A valid prompt is required for video generation." });
    }

    const payload = {
      prompt: prompt.trim(),
      negative_prompt: negative_prompt || "",
      camera_motion,
      motion_intensity: Number(motion_intensity) || 1.0,
      duration_seconds: Number(duration_seconds) || 2.0,
      fps: Number(fps) || 16,
      width: Number(width) || 512,
      height: Number(height) || 512,
      provider,
      seed: seed !== undefined && seed !== "" ? Number(seed) : null,
      style_preset,
      export_gif: Boolean(export_gif),
      output_dir: "outputs/videos",
      dry_run: dry_run !== undefined ? Boolean(dry_run) : undefined,
      api_key: api_key || process.env.FAL_KEY || process.env.GEMINI_API_KEY || null,
      hf_token: hf_token || process.env.HF_TOKEN || null,
    };

    const response = await callPythonWithRetry(() =>
      axios.post(`${PYTHON_API_URL}/video/generate`, payload, {
        timeout: 240000, // 4 minutes timeout for video synthesis
        headers: { "Content-Type": "application/json" },
      })
    );

    res.json(response.data);
  } catch (err) {
    if (err.response) {
      return res.status(err.response.status).json({
        error: err.response.data?.detail || "Video generation failed on Python AI Engine.",
      });
    }
    if (err.code === "ECONNREFUSED") {
      return res.status(503).json({
        error: "Python AI Engine is unreachable. Please ensure python_api.py is running on port 8000.",
      });
    }
    next(err);
  }
});

/**
 * GET /api/video/gallery
 * Returns list of generated videos from Python engine or local videos folder.
 */
router.get("/gallery", async (req, res) => {
  try {
    const response = await axios.get(`${PYTHON_API_URL}/video/gallery`, { timeout: 4000 });
    return res.json(response.data);
  } catch {
    // Fallback: scan VIDEOS_DIR on disk
    if (!fs.existsSync(VIDEOS_DIR)) {
      return res.json({ videos: [] });
    }

    const files = fs.readdirSync(VIDEOS_DIR);
    const mp4Files = files
      .filter((f) => f.toLowerCase().endsWith(".mp4"))
      .map((f) => {
        const fullPath = path.join(VIDEOS_DIR, f);
        const stats = fs.statSync(fullPath);
        const base = f.slice(0, -4);
        const jsonPath = path.join(VIDEOS_DIR, `${base}.json`);
        const thumbPath = path.join(VIDEOS_DIR, `${base}.jpg`);
        const gifPath = path.join(VIDEOS_DIR, `${base}.gif`);

        let meta = {};
        if (fs.existsSync(jsonPath)) {
          try {
            meta = JSON.parse(fs.readFileSync(jsonPath, "utf-8"));
          } catch {}
        }

        return {
          filename: f,
          url: `/outputs/videos/${f}`,
          thumbnail_url: fs.existsSync(thumbPath) ? `/outputs/videos/${base}.jpg` : null,
          gif_url: fs.existsSync(gifPath) ? `/outputs/videos/${base}.gif` : null,
          prompt: meta.prompt || "",
          camera_motion: meta.camera_motion || "none",
          duration_seconds: meta.duration_seconds || 0,
          fps: meta.fps || 16,
          frame_count: meta.frame_count || 0,
          seed: meta.seed || null,
          provider: meta.provider || "unknown",
          created_at: stats.mtimeMs,
        };
      })
      .sort((a, b) => b.created_at - a.created_at);

    return res.json({ videos: mp4Files });
  }
});

/**
 * GET /api/video/motions
 * Returns available camera motion options.
 */
router.get("/motions", async (req, res) => {
  try {
    const response = await axios.get(`${PYTHON_API_URL}/video/motions`, { timeout: 2500 });
    res.json(response.data);
  } catch {
    res.json({ motions: FALLBACK_MOTIONS });
  }
});

/**
 * GET /api/video/styles
 * Returns cinematic style presets.
 */
router.get("/styles", async (req, res) => {
  try {
    const response = await axios.get(`${PYTHON_API_URL}/video/styles`, { timeout: 2500 });
    res.json(response.data);
  } catch {
    res.json({ styles: FALLBACK_VIDEO_STYLES });
  }
});

/**
 * GET /api/video/stream/:filename
 * Streams video with HTTP 206 Partial Content (range requests) for timeline scrubbing.
 */
router.get("/stream/:filename", (req, res) => {
  const safeFilename = path.basename(req.params.filename);
  const videoPath = path.join(VIDEOS_DIR, safeFilename);

  if (!fs.existsSync(videoPath)) {
    return res.status(404).json({ error: "Video file not found" });
  }

  const stat = fs.statSync(videoPath);
  const fileSize = stat.size;
  const range = req.headers.range;

  if (range) {
    const parts = range.replace(/bytes=/, "").split("-");
    const start = parseInt(parts[0], 10);
    const end = parts[1] ? parseInt(parts[1], 10) : fileSize - 1;

    if (start >= fileSize || end >= fileSize) {
      res.status(416).set("Content-Range", `bytes */${fileSize}`);
      return res.end();
    }

    const chunksize = end - start + 1;
    const stream = fs.createReadStream(videoPath, { start, end });
    res.writeHead(206, {
      "Content-Range": `bytes ${start}-${end}/${fileSize}`,
      "Accept-Ranges": "bytes",
      "Content-Length": chunksize,
      "Content-Type": "video/mp4",
    });
    stream.pipe(res);
  } else {
    res.writeHead(200, {
      "Content-Length": fileSize,
      "Accept-Ranges": "bytes",
      "Content-Type": "video/mp4",
    });
    fs.createReadStream(videoPath).pipe(res);
  }
});

/**
 * DELETE /api/video/:filename
 * Deletes generated video and companion files (thumbnail, gif, json sidecar).
 */
router.delete("/:filename", (req, res) => {
  const safeFilename = path.basename(req.params.filename);
  const base = safeFilename.replace(/\.[^/.]+$/, "");
  const extensions = [".mp4", ".jpg", ".gif", ".json"];

  let deletedCount = 0;
  extensions.forEach((ext) => {
    const target = path.join(VIDEOS_DIR, `${base}${ext}`);
    if (fs.existsSync(target)) {
      try {
        fs.unlinkSync(target);
        deletedCount++;
      } catch {}
    }
  });

  res.json({ status: "success", deleted_files: deletedCount, base_name: base });
});

module.exports = router;
