import React, { useState, useEffect } from "react";
import {
  Wand2,
  Sparkles,
  Sliders,
  Loader2,
  Download,
  Images,
  AlertTriangle,
  Cpu,
  Zap,
  CheckCircle2,
} from "lucide-react";
import { aiApi } from "../api/client";

const PROMPT_IDEAS_FACEID = [
  "portrait of a man standing in rainy neon Tokyo at night, wearing a dark leather jacket",
  "portrait of a man as a sharp executive in a glass office overlooking city skyline",
  "candid photo of a man sipping espresso at a sunny Parisian sidewalk cafe",
  "cinematic photo of a man wearing a NASA astronaut suit on the surface of Mars",
];

const PROMPT_IDEAS_SDXL = [
  "sks person standing in rainy neon Tokyo at night, leather jacket",
  "sks person as a sharp executive in a glass office overlooking skyline",
  "sks person sipping espresso at a sunny Parisian sidewalk cafe",
  "sks person wearing a NASA space suit on the surface of Mars",
];

export default function PromptStudio({ onGenerated }) {
  const [styles, setStyles] = useState([]);
  const [health, setHealth] = useState(null);
  const [engineMode, setEngineMode] = useState("instant_faceid"); // 'instant_faceid' | 'sdxl_lora'
  const [baseModel, setBaseModel] = useState("dreamlike-art/dreamlike-diffusion-1.0");
  const [prompt, setPrompt] = useState(
    "portrait of a man in a tailored navy suit, studio lighting"
  );
  const [negativePrompt, setNegativePrompt] = useState(
    "monochrome, lowres, bad anatomy, worst quality, low quality, blurry, distorted face"
  );
  const [selectedStyle, setSelectedStyle] = useState("photorealistic");
  const [loraScale, setLoraScale] = useState(0.85);
  const [ipAdapterScale, setIpAdapterScale] = useState(0.7);
  const [guidanceScale, setGuidanceScale] = useState(1.5);
  const [steps, setSteps] = useState(6);
  const [width, setWidth] = useState(512);
  const [height, setHeight] = useState(512);
  const [useLcm, setUseLcm] = useState(true);
  const [seed, setSeed] = useState("");
  const [dryRun, setDryRun] = useState(false);
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
        if (engineMode === "instant_faceid") {
          setDryRun(false);
        } else if (typeof h.mock_mode_default === "boolean") {
          setDryRun(h.mock_mode_default);
        }
      })
      .catch(() => {});
  }, []);

  const handleEngineModeChange = (mode) => {
    setEngineMode(mode);
    if (mode === "instant_faceid") {
      setBaseModel("dreamlike-art/dreamlike-diffusion-1.0");
      setWidth(512);
      setHeight(512);
      setSteps(6);
      setGuidanceScale(1.5);
      setUseLcm(true);
      setDryRun(false);
    } else {
      setBaseModel("stabilityai/stable-diffusion-xl-base-1.0");
      setWidth(1024);
      setHeight(1024);
      setSteps(30);
      setGuidanceScale(7.0);
      setUseLcm(false);
      setDryRun(!health?.device?.cuda_available);
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
        engine_mode: engineMode,
        base_model: baseModel,
        style_preset: selectedStyle,
        lora_scale: Number(loraScale),
        ip_adapter_scale: Number(ipAdapterScale),
        use_lcm: useLcm,
        sampler: useLcm ? "lcm" : "dpm++_2m_karras",
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
  const faceIdReady = health?.face_id_ready;
  const ideas = engineMode === "instant_faceid" ? PROMPT_IDEAS_FACEID : PROMPT_IDEAS_SDXL;

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
              Synthesize personalized photos using Instant FaceID + LCM or SDXL LoRA.
            </p>
          </div>

          {health && (
            <span className={`badge ${hasCuda ? "badge-success" : "badge-warning"}`}>
              <Cpu size={13} />
              {hasCuda ? `GPU: ${health.device.device_name}` : "CPU Mode"}
            </span>
          )}
        </div>

        {error && <div className="alert-error">{error}</div>}

        <form onSubmit={handleGenerate}>
          {/* Engine Mode Selector */}
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "1fr 1fr",
              gap: 10,
              marginBottom: 16,
            }}
          >
            <button
              type="button"
              className={`style-card ${engineMode === "instant_faceid" ? "selected" : ""}`}
              onClick={() => handleEngineModeChange("instant_faceid")}
            >
              <div style={{ display: "flex", alignItems: "center", gap: 6, fontWeight: 600, fontSize: "0.86rem" }}>
                <Zap size={15} color="#a78bfa" /> Instant FaceID + LCM
              </div>
              <div style={{ fontSize: "0.74rem", color: "var(--text-secondary)", marginTop: 4 }}>
                6-Step Fast Open-Source • CPU &amp; GPU
              </div>
            </button>

            <button
              type="button"
              className={`style-card ${engineMode === "sdxl_lora" ? "selected" : ""}`}
              onClick={() => handleEngineModeChange("sdxl_lora")}
            >
              <div style={{ display: "flex", alignItems: "center", gap: 6, fontWeight: 600, fontSize: "0.86rem" }}>
                <Cpu size={15} color="#60a5fa" /> SDXL LoRA
              </div>
              <div style={{ fontSize: "0.74rem", color: "var(--text-secondary)", marginTop: 4 }}>
                1024x1024 Fine-Tuned • GPU Required
              </div>
            </button>
          </div>

          {engineMode === "instant_faceid" && (
            <div
              style={{
                display: "flex",
                justifyContent: "space-between",
                alignItems: "center",
                padding: "9px 12px",
                borderRadius: "var(--radius-sm)",
                background: faceIdReady
                  ? "rgba(16, 185, 129, 0.1)"
                  : "rgba(245, 158, 11, 0.1)",
                border: `1px solid ${
                  faceIdReady ? "rgba(16, 185, 129, 0.3)" : "rgba(245, 158, 11, 0.3)"
                }`,
                fontSize: "0.8rem",
                marginBottom: 14,
              }}
            >
              <span style={{ display: "flex", alignItems: "center", gap: 6 }}>
                <CheckCircle2 size={14} color={faceIdReady ? "#34d399" : "#fbbf24"} />
                {faceIdReady
                  ? "FaceID Identity Ready (person_face_id.safetensors)"
                  : "FaceID not enrolled yet — run Enroll FaceID on Tab 1 first!"}
              </span>
            </div>
          )}

          <div className="form-group">
            <label className="form-label">Base Diffusion Model</label>
            <select
              className="select"
              value={baseModel}
              onChange={(e) => setBaseModel(e.target.value)}
            >
              {engineMode === "instant_faceid" ? (
                <>
                  <option value="dreamlike-art/dreamlike-diffusion-1.0">
                    Dreamlike Diffusion 1.0 (Cached Local Model — 512x512)
                  </option>
                  <option value="SG161222/Realistic_Vision_V5.1_noVAE">
                    Realistic Vision V5.1 (Photorealistic SD 1.5 — Downloads ~2GB)
                  </option>
                </>
              ) : (
                <option value="stabilityai/stable-diffusion-xl-base-1.0">
                  SDXL 1.0 + Trained Face LoRA (1024x1024)
                </option>
              )}
            </select>
          </div>

          <div className="form-group">
            <label className="form-label">
              <span>Creative Prompt</span>
              {engineMode === "sdxl_lora" && (
                <span style={{ color: "var(--text-muted)", fontSize: "0.78rem" }}>
                  Include trigger: sks person
                </span>
              )}
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
            {ideas.map((idea, idx) => (
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
              <Sliders size={14} /> Inference Parameters ({width}x{height} •{" "}
              {useLcm ? "LCM Fast" : "Standard"})
            </div>

            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 14 }}>
              {engineMode === "instant_faceid" ? (
                <div className="form-group" style={{ marginBottom: 0 }}>
                  <label className="form-label">
                    <span>Face Fidelity</span>
                    <span style={{ fontFamily: "var(--font-mono)" }}>{ipAdapterScale}</span>
                  </label>
                  <input
                    type="range"
                    min={0.2}
                    max={1.2}
                    step={0.05}
                    value={ipAdapterScale}
                    onChange={(e) => setIpAdapterScale(parseFloat(e.target.value))}
                  />
                </div>
              ) : (
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
              )}

              <div className="form-group" style={{ marginBottom: 0 }}>
                <label className="form-label">
                  <span>Guidance (CFG)</span>
                  <span style={{ fontFamily: "var(--font-mono)" }}>{guidanceScale}</span>
                </label>
                <input
                  type="range"
                  min={1.0}
                  max={12.0}
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
                  min={4}
                  max={50}
                  step={1}
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
            Dry-Run Preview Mode (Instant wireframe card without running Diffusers)
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
                Dry-Run Mode is checked, so Generate produces a mock card. Uncheck for real AI generation!
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
              {engineMode === "instant_faceid"
                ? "Running Real IP-Adapter-FaceID + LCM on CPU (first load downloads ~360MB adapters, takes ~20–45s per image)."
                : "Running SDXL on CPU (takes 15–25+ mins per image). Switch to Instant FaceID + LCM for fast CPU results!"}
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
                {dryRun ? "Generating Preview..." : "Synthesizing Real AI Photo..."}
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
                {dryRun ? "Rendering Preview Card..." : "Synthesizing Personalized AI Image..."}
              </p>
              {!dryRun && !hasCuda && (
                <p style={{ fontSize: "0.8rem", color: "var(--text-muted)", marginTop: 6 }}>
                  {steps}-step LCM diffusion running on CPU — check terminal for download/step progress.
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
