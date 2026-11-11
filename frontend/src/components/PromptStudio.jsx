import React, { useState, useEffect } from "react";
import { Wand2, Sparkles, Sliders, Loader2, Download, Images } from "lucide-react";
import { aiApi } from "../api/client";

const PROMPT_IDEAS = [
  "sks person standing in rainy neon Tokyo at night, leather jacket",
  "sks person as a sharp executive in a glass office overlooking skyline",
  "sks person sipping espresso at a sunny Parisian sidewalk cafe",
  "sks person wearing a NASA space suit on the surface of Mars",
];

export default function PromptStudio({ onGenerated }) {
  const [styles, setStyles] = useState([]);
  const [prompt, setPrompt] = useState("sks person in a tailored navy suit, studio lighting");
  const [negativePrompt, setNegativePrompt] = useState(
    "blurry, low quality, distorted face, extra limbs, watermark"
  );
  const [selectedStyle, setSelectedStyle] = useState("photorealistic");
  const [loraScale, setLoraScale] = useState(0.85);
  const [guidanceScale, setGuidanceScale] = useState(7.0);
  const [steps, setSteps] = useState(30);
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
  }, []);

  const handleGenerate = async (e) => {
    e.preventDefault();
    if (!prompt.trim()) return;
    setError("");
    setGenerating(true);

    try {
      const data = await aiApi.generate({
        prompt,
        negative_prompt: negativePrompt,
        style_preset: selectedStyle,
        lora_scale: Number(loraScale),
        guidance_scale: Number(guidanceScale),
        steps: Number(steps),
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
        </div>

        {error && <div className="alert-error">{error}</div>}

        <form onSubmit={handleGenerate}>
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
              <Sliders size={14} /> Inference Parameters
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
                  min={10}
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
              marginBottom: 16,
              cursor: "pointer",
            }}
          >
            <input
              type="checkbox"
              checked={dryRun}
              onChange={(e) => setDryRun(e.target.checked)}
            />
            Dry-Run Preview Mode (Instant generation without CUDA GPU)
          </label>

          <button
            type="submit"
            className="btn btn-primary"
            disabled={generating}
            style={{ width: "100%", padding: "12px 18px" }}
          >
            {generating ? (
              <>
                <Loader2 size={18} className="spin" /> Generating Image...
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
            <div style={{ textAlign: "center", color: "var(--text-secondary)" }}>
              <Loader2 size={38} className="spin" color="#8b5cf6" />
              <p style={{ marginTop: 12, fontWeight: 500 }}>Running SDXL Diffusion...</p>
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
