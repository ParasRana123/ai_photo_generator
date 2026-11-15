import React, { useState, useEffect } from "react";
import {
  Film,
  Video,
  Play,
  Sparkles,
  Camera,
  Sliders,
  Move,
  Clock,
  Layers,
  RefreshCw,
  Trash2,
  Download,
  AlertCircle,
  CheckCircle2,
  ChevronDown,
  ChevronUp,
  Cpu,
  Key,
  Maximize,
} from "lucide-react";
import VideoPlayer from "./VideoPlayer";
import { videoApi } from "../api/client";

const SUGGESTED_PROMPTS = [
  "A majestic golden retriever puppy running happily through sunlit golden autumn woods",
  "A sleek cyberpunk sports car cruising through a neon-lit rain-slicked Tokyo boulevard",
  "High-speed slow motion water droplet splashing into a tranquil emerald pool with concentric ripples",
  "Dramatic aerial drone sweep over misty pine-covered alpine peaks at sunrise",
  "A young handsome Indian man in a tailored navy suit smiling warmly in a luxurious modern lounge",
  "A glowing celestial butterfly fluttering through an enchanted bioluminescent fantasy forest",
];

const ASPECT_RATIOS = [
  { label: "16:9 Landscape", width: 640, height: 360, icon: "📺" },
  { label: "9:16 Reels / TikTok", width: 360, height: 640, icon: "📱" },
  { label: "1:1 Square", width: 512, height: 512, icon: "⏹️" },
];

export default function VideoStudio() {
  const [prompt, setPrompt] = useState("");
  const [negativePrompt, setNegativePrompt] = useState("");
  const [showAdvanced, setShowAdvanced] = useState(false);

  // Settings
  const [cameraMotion, setCameraMotion] = useState("zoom_in");
  const [motionIntensity, setMotionIntensity] = useState(1.0);
  const [stylePreset, setStylePreset] = useState("cinematic");
  const [duration, setDuration] = useState(2.0);
  const [fps, setFps] = useState(16);
  const [aspectRatio, setAspectRatio] = useState(ASPECT_RATIOS[0]);
  const [provider, setProvider] = useState("auto");
  const [apiKey, setApiKey] = useState("");
  const [hfToken, setHfToken] = useState("");

  // Options from backend
  const [availableMotions, setAvailableMotions] = useState([]);
  const [availableStyles, setAvailableStyles] = useState([]);

  // Generation state
  const [isGenerating, setIsGenerating] = useState(false);
  const [currentVideo, setCurrentVideo] = useState(null);
  const [error, setError] = useState("");
  const [successMsg, setSuccessMsg] = useState("");

  // Gallery
  const [gallery, setGallery] = useState([]);
  const [loadingGallery, setLoadingGallery] = useState(true);

  // Load motions, styles, and gallery on mount
  useEffect(() => {
    loadOptions();
    fetchGallery();
  }, []);

  const loadOptions = async () => {
    try {
      const [mRes, sRes] = await Promise.all([
        videoApi.getMotions().catch(() => ({ motions: [] })),
        videoApi.getStyles().catch(() => ({ styles: [] })),
      ]);
      if (mRes.motions?.length) setAvailableMotions(mRes.motions);
      if (sRes.styles?.length) setAvailableStyles(sRes.styles);
    } catch {}
  };

  const fetchGallery = async () => {
    setLoadingGallery(true);
    try {
      const data = await videoApi.getGallery();
      setGallery(data.videos || []);
    } catch (err) {
      console.warn("Failed to load video gallery:", err);
    } finally {
      setLoadingGallery(false);
    }
  };

  const handleGenerate = async (e) => {
    e?.preventDefault();
    if (!prompt.trim()) {
      setError("Please enter a video prompt description.");
      return;
    }

    setIsGenerating(true);
    setError("");
    setSuccessMsg("");

    try {
      const payload = {
        prompt: prompt.trim(),
        negative_prompt: negativePrompt.trim() || undefined,
        camera_motion: cameraMotion,
        motion_intensity: Number(motionIntensity),
        duration_seconds: Number(duration),
        fps: Number(fps),
        width: aspectRatio.width,
        height: aspectRatio.height,
        provider,
        style_preset: stylePreset,
        api_key: apiKey.trim() || undefined,
        hf_token: hfToken.trim() || undefined,
      };

      const result = await videoApi.generate(payload);
      if (result.video) {
        setCurrentVideo(result.video);
        setSuccessMsg(`Video generated successfully in ${result.video.generation_time_s?.toFixed(2) || "a few"}s!`);
        fetchGallery();
      }
    } catch (err) {
      setError(err.message || "Video generation failed. Please try again.");
    } finally {
      setIsGenerating(false);
    }
  };

  const handleDelete = async (filename) => {
    if (!window.confirm("Are you sure you want to delete this video?")) return;
    try {
      await videoApi.deleteVideo(filename);
      if (currentVideo?.filename === filename) {
        setCurrentVideo(null);
      }
      fetchGallery();
    } catch (err) {
      setError(`Failed to delete: ${err.message}`);
    }
  };

  return (
    <div className="space-y-6">
      {/* Header Banner */}
      <div
        className="card"
        style={{
          background: "linear-gradient(135deg, rgba(124, 58, 237, 0.15) 0%, rgba(15, 23, 42, 0.6) 100%)",
          borderColor: "rgba(124, 58, 237, 0.3)",
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: 14 }}>
          <div
            style={{
              width: 48,
              height: 48,
              borderRadius: 12,
              background: "linear-gradient(135deg, #7c3aed, #ec4899)",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              color: "#fff",
              boxShadow: "0 4px 14px rgba(124, 58, 237, 0.4)",
            }}
          >
            <Video size={24} />
          </div>
          <div>
            <h1 style={{ fontSize: "22px", fontWeight: 700, color: "#fff", display: "flex", alignItems: "center", gap: 8 }}>
              Text-to-Video AI Studio
              <span
                style={{
                  fontSize: "11px",
                  background: "#7c3aed",
                  color: "#fff",
                  padding: "2px 8px",
                  borderRadius: "999px",
                  fontWeight: 600,
                  textTransform: "uppercase",
                }}
              >
                ZeroGPU & Cloud
              </span>
            </h1>
            <p style={{ color: "#9ca3af", fontSize: "14px", marginTop: 2 }}>
              Generate fluid, photorealistic AI videos directly from text prompts with cinematic camera motion controls.
            </p>
          </div>
        </div>
      </div>

      {/* Main Grid: Left Controls / Right Preview */}
      <div style={{ display: "grid", gridTemplateColumns: "1.15fr 0.85fr", gap: "24px" }}>
        {/* Left Column: Generation Controls */}
        <div className="card space-y-5">
          {/* Prompt Section */}
          <div>
            <label style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 8, fontSize: 13, fontWeight: 600, color: "#e2e8f0" }}>
              <span style={{ display: "flex", alignItems: "center", gap: 6 }}>
                <Sparkles size={15} color="#a78bfa" />
                Video Prompt
              </span>
              <span style={{ fontSize: 12, color: "#6b7280" }}>{prompt.length} chars</span>
            </label>
            <textarea
              rows={3}
              className="input-textarea"
              placeholder="Describe your scene in rich detail (e.g. 'A sleek cyberpunk sports car speeding down rainy Tokyo street at night, neon reflections, anamorphic flares')..."
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              style={{
                width: "100%",
                background: "rgba(18, 21, 31, 0.8)",
                border: "1px solid rgba(255, 255, 255, 0.12)",
                borderRadius: "10px",
                color: "#fff",
                padding: "12px",
                fontSize: "14px",
                resize: "vertical",
              }}
            />

            {/* Quick Inspiration Pills */}
            <div style={{ marginTop: 8 }}>
              <span style={{ fontSize: 11, color: "#9ca3af", marginRight: 6 }}>Quick Ideas:</span>
              <div style={{ display: "flex", flexWrap: "wrap", gap: 6, marginTop: 4 }}>
                {SUGGESTED_PROMPTS.map((p, idx) => (
                  <button
                    key={idx}
                    type="button"
                    onClick={() => setPrompt(p)}
                    style={{
                      fontSize: 11,
                      background: "rgba(255, 255, 255, 0.05)",
                      border: "1px solid rgba(255, 255, 255, 0.08)",
                      borderRadius: "6px",
                      color: "#cbd5e1",
                      padding: "3px 8px",
                      cursor: "pointer",
                      textAlign: "left",
                      maxWidth: "280px",
                      overflow: "hidden",
                      textOverflow: "ellipsis",
                      whiteSpace: "nowrap",
                    }}
                    title={p}
                  >
                    {p}
                  </button>
                ))}
              </div>
            </div>
          </div>

          {/* Style Preset Selector */}
          <div>
            <label style={{ display: "block", marginBottom: 8, fontSize: 13, fontWeight: 600, color: "#e2e8f0" }}>
              Cinematic Style Preset
            </label>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(140px, 1fr))", gap: 8 }}>
              {(availableStyles.length ? availableStyles : [
                { name: "cinematic", display_name: "Cinematic 35mm" },
                { name: "slow_motion", display_name: "Slow Motion" },
                { name: "drone_aerial", display_name: "Drone Aerial" },
                { name: "hyperlapse", display_name: "City Hyperlapse" },
                { name: "candid_smartphone", display_name: "Smartphone 4K" },
                { name: "animated_3d", display_name: "Pixar 3D" },
              ]).map((style) => (
                <button
                  key={style.name}
                  type="button"
                  onClick={() => setStylePreset(style.name)}
                  style={{
                    padding: "8px 10px",
                    borderRadius: "8px",
                    border: stylePreset === style.name ? "1px solid #7c3aed" : "1px solid rgba(255, 255, 255, 0.08)",
                    background: stylePreset === style.name ? "rgba(124, 58, 237, 0.2)" : "rgba(255, 255, 255, 0.03)",
                    color: stylePreset === style.name ? "#fff" : "#9ca3af",
                    fontSize: "12px",
                    fontWeight: 500,
                    cursor: "pointer",
                    textAlign: "center",
                    transition: "all 0.15s ease",
                  }}
                >
                  {style.display_name}
                </button>
              ))}
            </div>
          </div>

          {/* Camera Motion Selector */}
          <div>
            <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 8 }}>
              <label style={{ fontSize: 13, fontWeight: 600, color: "#e2e8f0", display: "flex", alignItems: "center", gap: 6 }}>
                <Camera size={15} color="#ec4899" />
                Camera Motion Dynamic
              </label>
              <span style={{ fontSize: 11, color: "#9ca3af" }}>Intensity: {motionIntensity}x</span>
            </div>

            <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 8 }}>
              {[
                { name: "zoom_in", label: "Push-In Zoom", icon: "🔍" },
                { name: "zoom_out", label: "Pull-Out Reveal", icon: "🔎" },
                { name: "pan_left", label: "Pan Left", icon: "⬅️" },
                { name: "pan_right", label: "Pan Right", icon: "➡️" },
                { name: "tilt_up", label: "Tilt Up", icon: "⬆️" },
                { name: "orbit_cw", label: "Orbit 360°", icon: "🔄" },
              ].map((m) => (
                <button
                  key={m.name}
                  type="button"
                  onClick={() => setCameraMotion(m.name)}
                  style={{
                    padding: "10px 8px",
                    borderRadius: "8px",
                    border: cameraMotion === m.name ? "1px solid #ec4899" : "1px solid rgba(255, 255, 255, 0.08)",
                    background: cameraMotion === m.name ? "rgba(236, 72, 153, 0.15)" : "rgba(255, 255, 255, 0.03)",
                    color: cameraMotion === m.name ? "#fff" : "#9ca3af",
                    fontSize: "12px",
                    fontWeight: 500,
                    cursor: "pointer",
                    display: "flex",
                    flexDirection: "column",
                    alignItems: "center",
                    gap: 4,
                  }}
                >
                  <span style={{ fontSize: "16px" }}>{m.icon}</span>
                  <span>{m.label}</span>
                </button>
              ))}
            </div>

            {/* Intensity Slider */}
            <div style={{ marginTop: 10, display: "flex", alignItems: "center", gap: 12 }}>
              <input
                type="range"
                min="0.2"
                max="2.0"
                step="0.1"
                value={motionIntensity}
                onChange={(e) => setMotionIntensity(parseFloat(e.target.value))}
                style={{ flex: 1, accentColor: "#ec4899" }}
              />
              <span style={{ fontSize: 12, color: "#e2e8f0", minWidth: 36, textAlign: "right" }}>
                {motionIntensity}x
              </span>
            </div>
          </div>

          {/* Quick Specs: Aspect Ratio, Duration, FPS */}
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 12 }}>
            {/* Aspect Ratio */}
            <div>
              <label style={{ display: "block", fontSize: 11, fontWeight: 600, color: "#9ca3af", marginBottom: 6 }}>
                Format
              </label>
              <select
                value={aspectRatio.label}
                onChange={(e) => setAspectRatio(ASPECT_RATIOS.find((r) => r.label === e.target.value) || ASPECT_RATIOS[0])}
                style={selectStyle}
              >
                {ASPECT_RATIOS.map((r) => (
                  <option key={r.label} value={r.label}>
                    {r.icon} {r.label}
                  </option>
                ))}
              </select>
            </div>

            {/* Duration */}
            <div>
              <label style={{ display: "block", fontSize: 11, fontWeight: 600, color: "#9ca3af", marginBottom: 6 }}>
                Duration
              </label>
              <select
                value={duration}
                onChange={(e) => setDuration(parseFloat(e.target.value))}
                style={selectStyle}
              >
                <option value={1.5}>1.5s (Quick clip)</option>
                <option value={2.0}>2.0s (Standard)</option>
                <option value={3.0}>3.0s (Cinematic)</option>
                <option value={5.0}>5.0s (Extended)</option>
              </select>
            </div>

            {/* FPS */}
            <div>
              <label style={{ display: "block", fontSize: 11, fontWeight: 600, color: "#9ca3af", marginBottom: 6 }}>
                Frame Rate
              </label>
              <select
                value={fps}
                onChange={(e) => setFps(parseInt(e.target.value, 10))}
                style={selectStyle}
              >
                <option value={12}>12 fps (Fast)</option>
                <option value={16}>16 fps (Standard)</option>
                <option value={24}>24 fps (Cinema 35mm)</option>
              </select>
            </div>
          </div>

          {/* Advanced / Provider Dropdown Toggle */}
          <div>
            <button
              type="button"
              onClick={() => setShowAdvanced(!showAdvanced)}
              style={{
                background: "transparent",
                border: "none",
                color: "#a78bfa",
                fontSize: 12,
                cursor: "pointer",
                display: "flex",
                alignItems: "center",
                gap: 4,
                padding: 0,
              }}
            >
              <Sliders size={13} />
              {showAdvanced ? "Hide Provider & Advanced Settings" : "Configure AI Provider & API Keys"}
              {showAdvanced ? <ChevronUp size={13} /> : <ChevronDown size={13} />}
            </button>

            {showAdvanced && (
              <div
                style={{
                  marginTop: 12,
                  padding: 14,
                  background: "rgba(0, 0, 0, 0.3)",
                  borderRadius: 10,
                  border: "1px solid rgba(255, 255, 255, 0.08)",
                  display: "flex",
                  flexDirection: "column",
                  gap: 12,
                }}
              >
                <div>
                  <label style={{ display: "block", fontSize: 11, fontWeight: 600, color: "#cbd5e1", marginBottom: 4 }}>
                    AI Video Synthesis Provider
                  </label>
                  <select
                    value={provider}
                    onChange={(e) => setProvider(e.target.value)}
                    style={selectStyle}
                  >
                    <option value="auto">Auto (ZeroGPU with commercial fallback)</option>
                    <option value="cloud_zerogpu">Free Cloud ZeroGPU (HuggingFace Spaces)</option>
                    <option value="fal_kling">fal.ai Kling v1.5 (High Quality)</option>
                    <option value="fal_minimax">fal.ai Minimax Hailuo AI</option>
                    <option value="google_veo">Google DeepMind Veo Video</option>
                  </select>
                </div>

                <div>
                  <label style={{ display: "block", fontSize: 11, fontWeight: 600, color: "#cbd5e1", marginBottom: 4 }}>
                    fal.ai / Google API Key (optional for cloud tiers)
                  </label>
                  <input
                    type="password"
                    placeholder="Enter FAL_KEY or GEMINI_API_KEY..."
                    value={apiKey}
                    onChange={(e) => setApiKey(e.target.value)}
                    style={inputStyle}
                  />
                </div>

                <div>
                  <label style={{ display: "block", fontSize: 11, fontWeight: 600, color: "#cbd5e1", marginBottom: 4 }}>
                    Negative Prompt
                  </label>
                  <input
                    type="text"
                    placeholder="blurry, distorted, jitter, low quality, artifacts..."
                    value={negativePrompt}
                    onChange={(e) => setNegativePrompt(e.target.value)}
                    style={inputStyle}
                  />
                </div>
              </div>
            )}
          </div>

          {/* Error / Success Notifications */}
          {error && (
            <div
              style={{
                padding: "10px 14px",
                borderRadius: "8px",
                background: "rgba(239, 68, 68, 0.15)",
                border: "1px solid rgba(239, 68, 68, 0.3)",
                color: "#fca5a5",
                fontSize: "13px",
                display: "flex",
                alignItems: "center",
                gap: 8,
              }}
            >
              <AlertCircle size={16} />
              <span>{error}</span>
            </div>
          )}

          {successMsg && (
            <div
              style={{
                padding: "10px 14px",
                borderRadius: "8px",
                background: "rgba(16, 185, 129, 0.15)",
                border: "1px solid rgba(16, 185, 129, 0.3)",
                color: "#6ee7b7",
                fontSize: "13px",
                display: "flex",
                alignItems: "center",
                gap: 8,
              }}
            >
              <CheckCircle2 size={16} />
              <span>{successMsg}</span>
            </div>
          )}

          {/* Primary Action Button */}
          <button
            type="button"
            onClick={handleGenerate}
            disabled={isGenerating || !prompt.trim()}
            style={{
              width: "100%",
              padding: "14px 20px",
              borderRadius: "10px",
              background: isGenerating
                ? "rgba(124, 58, 237, 0.5)"
                : "linear-gradient(135deg, #7c3aed 0%, #ec4899 100%)",
              color: "#fff",
              border: "none",
              fontSize: "15px",
              fontWeight: 600,
              cursor: isGenerating ? "not-allowed" : "pointer",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              gap: 10,
              boxShadow: "0 4px 20px rgba(124, 58, 237, 0.35)",
              transition: "transform 0.15s ease",
            }}
          >
            {isGenerating ? (
              <>
                <div
                  className="spin"
                  style={{
                    width: 18,
                    height: 18,
                    border: "2px solid rgba(255, 255, 255, 0.3)",
                    borderTopColor: "#fff",
                    borderRadius: "50%",
                  }}
                />
                Synthesizing Video Frames & Encoding MP4...
              </>
            ) : (
              <>
                <Film size={18} />
                Generate AI Video
              </>
            )}
          </button>
        </div>

        {/* Right Column: Video Preview and Metadata */}
        <div className="card space-y-4" style={{ display: "flex", flexDirection: "column" }}>
          <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
            <h2 style={{ fontSize: "16px", fontWeight: 600, color: "#fff", display: "flex", alignItems: "center", gap: 6 }}>
              <Film size={17} color="#a78bfa" />
              Active Video Preview
            </h2>
            {currentVideo && (
              <span style={{ fontSize: "11px", color: "#10b981", fontWeight: 600, background: "rgba(16, 185, 129, 0.15)", padding: "2px 8px", borderRadius: "999px" }}>
                Ready to stream
              </span>
            )}
          </div>

          {currentVideo ? (
            <div className="space-y-4">
              <VideoPlayer
                src={currentVideo.url}
                poster={currentVideo.thumbnail_url}
                title={currentVideo.prompt}
                autoPlay={true}
                loop={true}
              />

              {/* Video Info Badges */}
              <div
                style={{
                  background: "rgba(255, 255, 255, 0.03)",
                  borderRadius: 10,
                  padding: "12px",
                  border: "1px solid rgba(255, 255, 255, 0.08)",
                  fontSize: 12,
                  display: "flex",
                  flexDirection: "column",
                  gap: 8,
                }}
              >
                <div style={{ color: "#e2e8f0", fontWeight: 500 }}>
                  "{currentVideo.prompt}"
                </div>
                <div style={{ display: "flex", flexWrap: "wrap", gap: 8, color: "#9ca3af" }}>
                  <span>🎬 {currentVideo.camera_motion}</span>
                  <span>⏱️ {currentVideo.duration_seconds}s</span>
                  <span>🎞️ {currentVideo.fps} fps</span>
                  <span>📦 {currentVideo.frame_count} frames</span>
                  {currentVideo.provider && <span>⚡ {currentVideo.provider}</span>}
                </div>
              </div>
            </div>
          ) : (
            <div
              style={{
                flex: 1,
                minHeight: 320,
                display: "flex",
                flexDirection: "column",
                alignItems: "center",
                justifyContent: "center",
                background: "rgba(0, 0, 0, 0.2)",
                borderRadius: 12,
                border: "2px dashed rgba(255, 255, 255, 0.1)",
                color: "#6b7280",
                padding: "24px",
                textAlign: "center",
              }}
            >
              <Film size={44} style={{ opacity: 0.4, marginBottom: 12 }} />
              <p style={{ fontSize: "14px", color: "#9ca3af", fontWeight: 500 }}>
                No video generated yet
              </p>
              <p style={{ fontSize: "12px", maxWidth: "260px", marginTop: 4 }}>
                Enter a prompt and click "Generate AI Video" to synthesize an MP4 video here.
              </p>
            </div>
          )}
        </div>
      </div>

      {/* Video Gallery Section */}
      <div className="card space-y-4">
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
          <div>
            <h2 style={{ fontSize: "18px", fontWeight: 600, color: "#fff", display: "flex", alignItems: "center", gap: 8 }}>
              <Layers size={18} color="#a78bfa" />
              Generated Video Library
            </h2>
            <p style={{ fontSize: "12px", color: "#9ca3af", marginTop: 2 }}>
              Browse, play, or download your rendered MP4 video animations.
            </p>
          </div>

          <button
            type="button"
            onClick={fetchGallery}
            disabled={loadingGallery}
            className="btn btn-secondary"
            style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 12 }}
          >
            <RefreshCw size={14} className={loadingGallery ? "spin" : ""} />
            Refresh Library
          </button>
        </div>

        {gallery.length === 0 ? (
          <div
            style={{
              padding: "40px 20px",
              textAlign: "center",
              color: "#6b7280",
              fontSize: "13px",
            }}
          >
            {loadingGallery ? "Loading library..." : "No rendered videos in gallery yet."}
          </div>
        ) : (
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "repeat(auto-fill, minmax(260px, 1fr))",
              gap: 16,
            }}
          >
            {gallery.map((vid) => (
              <div
                key={vid.filename}
                style={{
                  background: "rgba(18, 21, 31, 0.6)",
                  borderRadius: 12,
                  overflow: "hidden",
                  border: "1px solid rgba(255, 255, 255, 0.08)",
                  display: "flex",
                  flexDirection: "column",
                  transition: "transform 0.15s, border-color 0.15s",
                }}
              >
                {/* Media Preview Container */}
                <div
                  style={{
                    position: "relative",
                    aspectRatio: "16/9",
                    background: "#000",
                    cursor: "pointer",
                  }}
                  onClick={() => setCurrentVideo(vid)}
                >
                  <img
                    src={vid.gif_url || vid.thumbnail_url || vid.url}
                    alt={vid.prompt}
                    style={{
                      width: "100%",
                      height: "100%",
                      objectFit: "cover",
                    }}
                    onError={(e) => {
                      e.target.style.display = "none";
                    }}
                  />
                  <div
                    style={{
                      position: "absolute",
                      inset: 0,
                      background: "rgba(0, 0, 0, 0.3)",
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "center",
                      opacity: 0.85,
                    }}
                  >
                    <div
                      style={{
                        width: 40,
                        height: 40,
                        borderRadius: "50%",
                        background: "rgba(124, 58, 237, 0.8)",
                        display: "flex",
                        alignItems: "center",
                        justifyContent: "center",
                        color: "#fff",
                      }}
                    >
                      <Play size={18} fill="#fff" style={{ marginLeft: 2 }} />
                    </div>
                  </div>

                  {vid.duration_seconds > 0 && (
                    <span
                      style={{
                        position: "absolute",
                        bottom: 8,
                        right: 8,
                        background: "rgba(0, 0, 0, 0.8)",
                        color: "#fff",
                        fontSize: 10,
                        padding: "2px 6px",
                        borderRadius: 4,
                      }}
                    >
                      {vid.duration_seconds}s
                    </span>
                  )}
                </div>

                {/* Info Card Body */}
                <div style={{ padding: "12px", display: "flex", flexDirection: "column", gap: 8, flex: 1 }}>
                  <div
                    style={{
                      fontSize: 12,
                      color: "#e2e8f0",
                      fontWeight: 500,
                      maxHeight: "36px",
                      overflow: "hidden",
                      textOverflow: "ellipsis",
                      display: "-webkit-box",
                      WebkitLineClamp: 2,
                      WebkitBoxOrient: "vertical",
                    }}
                    title={vid.prompt}
                  >
                    {vid.prompt || "Generated Video"}
                  </div>

                  <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginTop: "auto", paddingTop: 8, borderTop: "1px solid rgba(255, 255, 255, 0.05)" }}>
                    <span style={{ fontSize: 11, color: "#9ca3af" }}>
                      {vid.camera_motion || "camera"}
                    </span>

                    <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                      <a
                        href={vid.url}
                        download={vid.filename}
                        title="Download Video"
                        style={{
                          color: "#cbd5e1",
                          padding: "4px",
                          borderRadius: 4,
                          display: "inline-flex",
                        }}
                      >
                        <Download size={14} />
                      </a>
                      <button
                        type="button"
                        onClick={() => handleDelete(vid.filename)}
                        title="Delete Video"
                        style={{
                          background: "transparent",
                          border: "none",
                          color: "#ef4444",
                          cursor: "pointer",
                          padding: "4px",
                          borderRadius: 4,
                          display: "inline-flex",
                        }}
                      >
                        <Trash2 size={14} />
                      </button>
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

const selectStyle = {
  width: "100%",
  background: "rgba(18, 21, 31, 0.8)",
  border: "1px solid rgba(255, 255, 255, 0.12)",
  borderRadius: "8px",
  color: "#fff",
  padding: "8px 10px",
  fontSize: "12px",
  cursor: "pointer",
};

const inputStyle = {
  width: "100%",
  background: "rgba(18, 21, 31, 0.8)",
  border: "1px solid rgba(255, 255, 255, 0.12)",
  borderRadius: "8px",
  color: "#fff",
  padding: "8px 10px",
  fontSize: "12px",
};
