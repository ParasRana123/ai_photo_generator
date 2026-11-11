import React, { useState, useEffect } from "react";
import { Wand2, Sparkles, Sliders, Loader2, Download, Images, AlertTriangle, Cpu } from "lucide-react";
import { aiApi } from "../api/client";

const PROMPT_IDEAS = [
  "sks person standing in rainy neon Tokyo at night, leather jacket",
  "sks person as a sharp executive in a glass office overlooking skyline",
  "sks person sipping espresso at a sunny Parisian sidewalk cafe",
  "sks person wearing a NASA space suit on the surface of Mars",
];

export default function PromptStudio({ onGenerated }) {
  const [styles, setStyles] = useState([]);
  const [health, setHealth] = useState(null);
  const [baseModel, setBaseModel] = useState("stabilityai/stable-diffusion-xl-base-1.0");
  const [prompt, setPrompt] = useState("sks person in a tailored navy suit, studio lighting");
  const [negativePrompt, setNegativePrompt] = useState(
    "blurry, low quality, distorted face, extra limbs, watermark"
  );
  const [selectedStyle, setSelectedStyle] = useState("photorealistic");
  const [loraScale, setLoraScale] = useState(0.85);
  const [guidanceScale, setGuidanceScale] = useState(7.0);
  const [steps, setSteps] = useState(30);
  const [width, setWidth] = useState(1024);
  const [height, setHeight] = useState(1024);
  const [seed, setSeed] = useState("");
  const [dryRun, setDryRun] = useState(true);
  const [generating, setGenerating] = useState(false);
  const [latestResult, setLatestResult] = useState(null);
  const [error, setError] = useState("");

  useEffect(() => {
    aiApi
      .getStyles()
      .then((data) => setStyles(data.styles || []))
      .catch(() => {});

    aiApi
      .getHealth()
      .then((h) => {
        setHealth(h);
        if (typeof h.mock_mode_default === "boolean") {
          setDryRun(h.mock_mode_default);
        }
      })
      .catch(() => {});
  }, []);

  const handleModelChange = (modelId) => {
    setBaseModel(modelId);
    if (modelId === "stabilityai/stable-diffusion-xl-base-1.0") {
      setWidth(1024);
      setHeight(1024);
      setSteps(30);
    } else {
      // Lighter 512x512 local CPU test model
      setWidth(512);
      setHeight(512);
      setSteps(15);
      setDryRun(false);
    }
  };

  const handleGenerate = async (e) => {
    e.preventDefault();
    if (!prompt.trim()) return;
    setError("");
    setGenerating(true);

    try {
      const data = await aiApi.generate({
        prompt,
        negative_prompt: negativePrompt,
        base_model: baseModel,
        style_preset: selectedStyle,
        lora_scale: Number(loraScale),
        guidance_scale: Number(guidanceScale),
        steps: Number(steps),
        width: Number(width),
        height: Number(height),
        seed: seed.trim() ? Number(seed) : null,
        num_images: 1,
        dry_run: dryRun,
      });

      setLatestResult(data);
      onGenerated?.(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setGenerating(false);
    }
  };

  const latestImage = latestResult?.images?.[0];
  const hasCuda = health?.device?.cuda_available;
  const cachedModels = health?.cached_models || [];
  const hasDreamlike = cachedModels.includes("dreamlike-art/dreamlike-diffusion-1.0");

  return (
    <div className="grid-2">
      {/* Left: Prompt Input & Settings */}
      <div className="card">
        <div className="card-header">
          <div>
            <h2 className="card-title">
              <Wand2 size={20} color="#a78bfa" />
              Prompt Studio
            </h2>
            <p className="card-subtitle">
              Describe any scene and select a photographic preset for your trained LoRA.
            </p>
          </div>

          {health && (
            <span className={`badge ${hasCuda ? "badge-success" : "badge-warning"}`}>
              <Cpu size={13} />
              {hasCuda ? `GPU: ${health.device.device_name}` : "CPU Mode (No CUDA GPU)"}
            </span>
          )}
        </div>

        {error && <div className="alert-error">{error}</div>}

        <form onSubmit={handleGenerate}>
          <div className="form-group">
            <label className="form-label">Diffusion Model</label>
            <select
              className="select"
              value={baseModel}
              onChange={(e) => handleModelChange(e.target.value)}
            >
              <option value="stabilityai/stable-diffusion-xl-base-1.0">
                SDXL 1.0 + Trained Face LoRA (1024x1024 — GPU Recommended)
              </option>
              <option value="dreamlike-art/dreamlike-diffusion-1.0">
                Dreamlike Diffusion 1.0 {hasDreamlike ? "(Cached Locally — Real CPU Test, 512x512)" : "(512x512)"}
              </option>
            </select>
          </div>

          <div className="form-group">
            <label className="form-label">
              <span>Creative Prompt</span>
              <span style={{ color: "var(--text-muted)", fontSize: "0.78rem" }}>
                Include trigger: sks person
              </span>
            </label>
            <textarea
              className="textarea"
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              placeholder="Describe the setting, clothing, lighting, and expression..."
              required
            />
          </div>

          {/* Prompt Idea Chips */}
          <div style={{ display: "flex", flexWrap: "wrap", gap: 6, marginBottom: 18 }}>
            {PROMPT_IDEAS.map((idea, idx) => (
              <button
                key={idx}
                type="button"
                className="btn btn-secondary"
                style={{ fontSize: "0.75rem", padding: "5px 10px" }}
                onClick={() => setPrompt(idea)}
              >
                + Idea {idx + 1}
              </button>
            ))}
          </div>

          {/* Style Presets */}
          <div className="form-group">
            <label className="form-label">Photographic Style Preset</label>
            <div className="style-grid">
              {styles.map((style) => (
                <button
                  key={style.name}
                  type="button"
                  className={`style-card ${selectedStyle === style.name ? "selected" : ""}`}
                  onClick={() => setSelectedStyle(style.name)}
                >
                  <div style={{ fontWeight: 600, fontSize: "0.84rem" }}>{style.display_name}</div>
                  <div
                    style={{
                      fontSize: "0.74rem",
                      color: "var(--text-secondary)",
                      marginTop: 4,
                      display: "-webkit-box",
                      WebkitLineClamp: 2,
                      WebkitBoxOrient: "vertical",
                      overflow: "hidden",
                    }}
                  >
                    {style.description}
                  </div>
                </button>
              ))}
            </div>
          </div>

          {/* Advanced Sliders */}
          <div
            style={{
              background: "var(--bg-base)",
              border: "1px solid var(--border-subtle)",
              borderRadius: "var(--radius-md)",
              padding: 14,
              marginBottom: 18,
            }}
          >
            <div
              style={{
                display: "flex",
                alignItems: "center",
                gap: 6,
                fontSize: "0.84rem",
                fontWeight: 600,
                marginBottom: 12,
                color: "var(--text-secondary)",
              }}
            >
              <Sliders size={14} /> Inference Parameters ({width}x{height})
            </div>

            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 14 }}>
              <div className="form-group" style={{ marginBottom: 0 }}>
                <label className="form-label">
                  <span>LoRA Strength</span>
                  <span style={{ fontFamily: "var(--font-mono)" }}>{loraScale}</span>
                </label>
                <input
                  type="range"
                  min={0.1}
                  max={1.4}
                  step={0.05}
                  value={loraScale}
                  onChange={(e) => setLoraScale(parseFloat(e.target.value))}
                />
              </div>

              <div className="form-group" style={{ marginBottom: 0 }}>
                <label className="form-label">
                  <span>Guidance (CFG)</span>
                  <span style={{ fontFamily: "var(--font-mono)" }}>{guidanceScale}</span>
                </label>
                <input
                  type="range"
                  min={1}
                  max={14}
                  step={0.5}
                  value={guidanceScale}
                  onChange={(e) => setGuidanceScale(parseFloat(e.target.value))}
                />
              </div>

              <div className="form-group" style={{ marginBottom: 0 }}>
                <label className="form-label">
                  <span>Inference Steps</span>
                  <span style={{ fontFamily: "var(--font-mono)" }}>{steps}</span>
                </label>
                <input
                  type="range"
                  min={5}
                  max={50}
                  step={5}
                  value={steps}
                  onChange={(e) => setSteps(parseInt(e.target.value, 10))}
                />
              </div>

              <div className="form-group" style={{ marginBottom: 0 }}>
                <label className="form-label">Seed (Optional)</label>
                <input
                  className="input"
                  type="number"
                  placeholder="Random"
                  value={seed}
                  onChange={(e) => setSeed(e.target.value)}
                  style={{ padding: "6px 10px" }}
                />
              </div>
            </div>
          </div>

          <label
            style={{
              display: "flex",
              alignItems: "center",
              gap: 10,
              fontSize: "0.86rem",
              color: "var(--text-secondary)",
              marginBottom: 10,
              cursor: "pointer",
            }}
          >
            <input
              type="checkbox"
              checked={dryRun}
              onChange={(e) => setDryRun(e.target.checked)}
            />
            Dry-Run Preview Mode (Instant placeholder card without running Diffusers)
          </label>

          {dryRun ? (
            <div
              style={{
                background: "rgba(245, 158, 11, 0.1)",
                border: "1px solid rgba(245, 158, 11, 0.3)",
                color: "#fcd34d",
                padding: "10px 12px",
                borderRadius: "var(--radius-sm)",
                fontSize: "0.8rem",
                marginBottom: 14,
                display: "flex",
                gap: 8,
                alignItems: "flex-start",
              }}
            >
              <AlertTriangle size={16} style={{ flexShrink: 0, marginTop: 2 }} />
              <span>
                Dry-Run Mode is checked, so clicking Generate returns an instant mock card.
                Uncheck Dry-Run for real Diffusers AI image generation (note: on CPU without CUDA, generation takes 1–3 mins for 512x512 or 15+ mins for SDXL).
              </span>
            </div>
          ) : !hasCuda && (
            <div
              style={{
                background: "rgba(99, 102, 241, 0.12)",
                border: "1px solid rgba(99, 102, 241, 0.35)",
                color: "#c7d2fe",
                padding: "10px 12px",
                borderRadius: "var(--radius-sm)",
                fontSize: "0.8rem",
                marginBottom: 14,
              }}
            >
              Real AI Diffusion running on <strong>CPU</strong>. Please wait 1–3 minutes (Dreamlike 512x512) or 15+ mins (SDXL) while denoising steps run.
            </div>
          )}

          <button
            type="submit"
            className="btn btn-primary"
            disabled={generating}
            style={{ width: "100%", padding: "12px 18px" }}
          >
            {generating ? (
              <>
                <Loader2 size={18} className="spin" />
                {dryRun ? "Generating Preview..." : "Running Diffusion Inference..."}
              </>
            ) : (
              <>
                <Sparkles size={18} /> Generate AI Photo
              </>
            )}
          </button>
        </form>
      </div>

      {/* Right: Live Result Preview */}
      <div className="card" style={{ display: "flex", flexDirection: "column" }}>
        <div className="card-header">
          <div>
            <h2 className="card-title">
              <Images size={20} color="#a78bfa" />
              Generated Output
            </h2>
            <p className="card-subtitle">Real-time result preview with embedded PNG metadata.</p>
          </div>
          {latestResult && (
            <span className={`badge ${latestResult.is_mock ? "badge-warning" : "badge-success"}`}>
              {latestResult.is_mock ? "Dry-Run Card" : "Real AI Render"}
            </span>
          )}
        </div>

        <div
          style={{
            flex: 1,
            minHeight: 360,
            background: "var(--bg-base)",
            borderRadius: "var(--radius-md)",
            border: "1px solid var(--border-subtle)",
            display: "grid",
            placeItems: "center",
            overflow: "hidden",
            position: "relative",
          }}
        >
          {generating ? (
            <div style={{ textAlign: "center", color: "var(--text-secondary)", padding: 24 }}>
              <Loader2 size={38} className="spin" color="#8b5cf6" />
              <p style={{ marginTop: 12, fontWeight: 500 }}>
                {dryRun ? "Rendering Preview Card..." : "Synthesizing Real AI Image..."}
              </p>
              {!dryRun && !hasCuda && (
                <p style={{ fontSize: "0.8rem", color: "var(--text-muted)", marginTop: 6 }}>
                  CPU inference in progress — check terminal logs for step progress.
                </p>
              )}
            </div>
          ) : latestImage ? (
            <img
              src={latestImage.url}
              alt={latestImage.prompt}
              style={{ width: "100%", height: "100%", objectFit: "contain" }}
            />
          ) : (
            <div style={{ textAlign: "center", padding: 24, color: "var(--text-muted)" }}>
              <Sparkles size={36} style={{ opacity: 0.4, marginBottom: 10 }} />
              <p>Enter a prompt and click &ldquo;Generate AI Photo&rdquo; to see your result here.</p>
            </div>
          )}
        </div>

        {latestImage && (
          <div
            style={{
              marginTop: 16,
              paddingTop: 14,
              borderTop: "1px solid var(--border-subtle)",
              display: "flex",
              justifyContent: "space-between",
              alignItems: "center",
              gap: 12,
            }}
          >
            <div style={{ fontSize: "0.82rem", color: "var(--text-secondary)" }}>
              Seed: <code style={{ color: "var(--text-primary)" }}>{latestImage.seed}</code> •{" "}
              {latestImage.duration_seconds}s
            </div>
            <a
              href={latestImage.url}
              download={latestImage.filename}
              className="btn btn-secondary"
              style={{ textDecoration: "none", padding: "7px 14px", fontSize: "0.84rem" }}
            >
              <Download size={15} /> Download PNG
            </a>
          </div>
        )}
      </div>
    </div>
  );
}
