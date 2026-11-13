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
  Cloud,
  CheckCircle2,
  KeyRound,
} from "lucide-react";
import { aiApi } from "../api/client";

const PROMPT_IDEAS_IDENTITY = [
  "candid smartphone photo of a young man smiling in a casual cotton shirt at a cafe, natural indoor light, realistic skin pores, shot on iPhone",
  "authentic candid photo of a young man celebrating Diwali with diyas at home, warm ambient glow, natural skin texture",
  "portrait of a young man in a tailored navy suit, soft natural studio lighting, photorealistic, 8k",
  "portrait of a young man standing in rainy neon Tokyo at night, wearing a dark leather jacket, cinematic lighting",
  "candid photo of a young man sipping espresso at a sunny Parisian sidewalk cafe, warm daylight",
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
  const [engineMode, setEngineMode] = useState("google_imagen3"); // 'google_imagen3' | 'fal_ai' | 'cloud_zerogpu' | 'instant_faceid' | 'sdxl_lora'
  const [geminiApiKey, setGeminiApiKey] = useState(() => localStorage.getItem("gemini_api_key") || "");
  const [falApiKey, setFalApiKey] = useState(() => localStorage.getItem("fal_api_key") || "");
  const [cloudProvider, setCloudProvider] = useState("auto");
  const [hfToken, setHfToken] = useState(() => localStorage.getItem("hf_free_token") || "");
  const [baseModel, setBaseModel] = useState("yanze/PuLID-FLUX");
  const [prompt, setPrompt] = useState(
    "portrait of a young man in a tailored navy suit, soft studio lighting, photorealistic, 8k"
  );
  const [negativePrompt, setNegativePrompt] = useState(
    "(lowres, low quality, worst quality:1.2), watermark, deformed, ugly, deformed eyes, blur, blurry, neon contours, neon outlines, sketch, drawing, painting, cartoon, anime, illustration, airbrushed, 3d render, oversaturated, CGI, plastic skin, extreme contrast, harsh lighting"
  );
  const [selectedStyle, setSelectedStyle] = useState("candid_smartphone");
  const [loraScale, setLoraScale] = useState(0.85);
  const [ipAdapterScale, setIpAdapterScale] = useState(0.7);
  const [idWeight, setIdWeight] = useState(1.0);
  const [guidanceScale, setGuidanceScale] = useState(4.0);
  const [steps, setSteps] = useState(20);
  const [width, setWidth] = useState(896);
  const [height, setHeight] = useState(1152);
  const [useLcm, setUseLcm] = useState(false);
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
        if (engineMode === "cloud_zerogpu") {
          setDryRun(false);
        } else if (typeof h.mock_mode_default === "boolean") {
          setDryRun(h.mock_mode_default);
        }
      })
      .catch(() => {});
  }, []);

  const handleEngineModeChange = (mode) => {
    setEngineMode(mode);
    setError("");
    if (mode === "google_imagen3") {
      setBaseModel("imagen-3.0-generate-002");
      setWidth(1024);
      setHeight(1024);
      setSteps(30);
      setGuidanceScale(4.0);
      setUseLcm(false);
      setDryRun(false);
    } else if (mode === "fal_ai") {
      setBaseModel("fal-ai/flux-pulid");
      setWidth(1024);
      setHeight(1024);
      setSteps(28);
      setGuidanceScale(4.0);
      setUseLcm(false);
      setDryRun(false);
    } else if (mode === "cloud_zerogpu") {
      setBaseModel("yanze/PuLID-FLUX");
      setWidth(896);
      setHeight(1152);
      setSteps(20);
      setGuidanceScale(4.0);
      setUseLcm(false);
      setDryRun(false);
    } else if (mode === "instant_faceid") {
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

  const handleTokenChange = (val) => {
    setHfToken(val);
    if (val.trim()) {
      localStorage.setItem("hf_free_token", val.trim());
    } else {
      localStorage.removeItem("hf_free_token");
    }
  };

  const handleGenerate = async (e) => {
    e.preventDefault();
    if (!prompt.trim()) return;
    setError("");
    setGenerating(true);

    try {
      const activeApiKey =
        engineMode === "google_imagen3"
          ? geminiApiKey.trim() || null
          : engineMode === "fal_ai"
          ? falApiKey.trim() || null
          : null;

      const data = await aiApi.generate({
        prompt,
        negative_prompt: negativePrompt,
        engine_mode: engineMode,
        base_model: baseModel,
        cloud_provider: cloudProvider,
        id_weight: Number(idWeight),
        hf_token: hfToken.trim() || null,
        api_key: activeApiKey,
        style_preset: selectedStyle,
        lora_scale: Number(loraScale),
        ip_adapter_scale: Number(ipAdapterScale),
        use_lcm: useLcm,
        sampler: useLcm ? "lcm" : "euler",
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
  const cloudProfileReady = health?.cloud_profile_ready;
  const ideas = engineMode === "sdxl_lora" ? PROMPT_IDEAS_SDXL : PROMPT_IDEAS_IDENTITY;

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
              Generate real personalized photos via Free Cloud ZeroGPU (fal.ai alternative) or local models.
            </p>
          </div>

          {engineMode === "cloud_zerogpu" ? (
            <span className="badge badge-success">
              <Cloud size={13} /> Free Cloud GPU
            </span>
          ) : (
            health && (
              <span className={`badge ${hasCuda ? "badge-success" : "badge-warning"}`}>
                <Cpu size={13} />
                {hasCuda ? `GPU: ${health.device.device_name}` : "Local CPU"}
              </span>
            )
          )}
        </div>

        {error && <div className="alert-error">{error}</div>}

        <form onSubmit={handleGenerate}>
          {/* Multi-Engine Selector */}
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "repeat(3, 1fr)",
              gap: 8,
              marginBottom: 16,
            }}
          >
            <button
              type="button"
              className={`style-card ${engineMode === "google_imagen3" ? "selected" : ""}`}
              onClick={() => handleEngineModeChange("google_imagen3")}
            >
              <div style={{ display: "flex", alignItems: "center", gap: 6, fontWeight: 600, fontSize: "0.82rem" }}>
                <Sparkles size={14} color="#38bdf8" /> Google Gemini
              </div>
              <div style={{ fontSize: "0.71rem", color: "var(--text-secondary)", marginTop: 4 }}>
                Imagen 3 (100% Real SOTA)
              </div>
            </button>

            <button
              type="button"
              className={`style-card ${engineMode === "fal_ai" ? "selected" : ""}`}
              onClick={() => handleEngineModeChange("fal_ai")}
            >
              <div style={{ display: "flex", alignItems: "center", gap: 6, fontWeight: 600, fontSize: "0.82rem" }}>
                <Zap size={14} color="#f59e0b" /> fal.ai FLUX
              </div>
              <div style={{ fontSize: "0.71rem", color: "var(--text-secondary)", marginTop: 4 }}>
                FLUX.1 PuLID (2s H100)
              </div>
            </button>

            <button
              type="button"
              className={`style-card ${engineMode === "cloud_zerogpu" ? "selected" : ""}`}
              onClick={() => handleEngineModeChange("cloud_zerogpu")}
            >
              <div style={{ display: "flex", alignItems: "center", gap: 6, fontWeight: 600, fontSize: "0.82rem" }}>
                <Cloud size={14} color="#34d399" /> Free Cloud GPU
              </div>
              <div style={{ fontSize: "0.71rem", color: "var(--text-secondary)", marginTop: 4 }}>
                InstantID / ZeroGPU ($0)
              </div>
            </button>
          </div>

          {engineMode === "google_imagen3" && (
            <div
              style={{
                padding: "12px 14px",
                borderRadius: "var(--radius-sm)",
                background: "rgba(56, 189, 248, 0.08)",
                border: "1px solid rgba(56, 189, 248, 0.28)",
                marginBottom: 16,
              }}
            >
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 8 }}>
                <span style={{ fontWeight: 600, fontSize: "0.82rem", color: "#7dd3fc", display: "flex", alignItems: "center", gap: 6 }}>
                  <Sparkles size={14} /> Google Gemini / Imagen 3 API (Photorealistic SOTA)
                </span>
                <a
                  href="https://aistudio.google.com/app/apikey"
                  target="_blank"
                  rel="noreferrer"
                  style={{ fontSize: "0.73rem", color: "#38bdf8", textDecoration: "underline" }}
                >
                  Get Free Key (Google AI Studio) ↗
                </a>
              </div>
              <div className="form-group" style={{ marginBottom: 0 }}>
                <input
                  className="input"
                  type="password"
                  placeholder="Paste your Gemini API Key (e.g. AIzaSy...)"
                  value={geminiApiKey}
                  onChange={(e) => {
                    setGeminiApiKey(e.target.value);
                    if (e.target.value.trim()) localStorage.setItem("gemini_api_key", e.target.value.trim());
                    else localStorage.removeItem("gemini_api_key");
                  }}
                />
              </div>
            </div>
          )}

          {engineMode === "fal_ai" && (
            <div
              style={{
                padding: "12px 14px",
                borderRadius: "var(--radius-sm)",
                background: "rgba(245, 158, 11, 0.08)",
                border: "1px solid rgba(245, 158, 11, 0.28)",
                marginBottom: 16,
              }}
            >
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 8 }}>
                <span style={{ fontWeight: 600, fontSize: "0.82rem", color: "#fcd34d", display: "flex", alignItems: "center", gap: 6 }}>
                  <Zap size={14} /> fal.ai FLUX.1 PuLID (2-Second Dedicated H100)
                </span>
                <a
                  href="https://fal.ai/dashboard/keys"
                  target="_blank"
                  rel="noreferrer"
                  style={{ fontSize: "0.73rem", color: "#fbbf24", textDecoration: "underline" }}
                >
                  Get Key (fal.ai) ↗
                </a>
              </div>
              <div className="form-group" style={{ marginBottom: 0 }}>
                <input
                  className="input"
                  type="password"
                  placeholder="Paste your fal.ai Key (e.g. 5b98...)"
                  value={falApiKey}
                  onChange={(e) => {
                    setFalApiKey(e.target.value);
                    if (e.target.value.trim()) localStorage.setItem("fal_api_key", e.target.value.trim());
                    else localStorage.removeItem("fal_api_key");
                  }}
                />
              </div>
            </div>
          )}

          {engineMode === "cloud_zerogpu" && (
            <>
              <div
                style={{
                  display: "flex",
                  justifyContent: "space-between",
                  alignItems: "center",
                  padding: "9px 12px",
                  borderRadius: "var(--radius-sm)",
                  background: "rgba(16, 185, 129, 0.08)",
                  border: "1px solid rgba(16, 185, 129, 0.28)",
                  fontSize: "0.8rem",
                  marginBottom: 14,
                }}
              >
                <span style={{ display: "flex", alignItems: "center", gap: 6, color: "#a7f3d0" }}>
                  <CheckCircle2 size={14} color="#34d399" />
                  {cloudProfileReady
                    ? "Cloud Face Profile Ready (using your sharpest uploaded portrait)"
                    : "Auto-selecting reference photos from your uploaded photos"}
                </span>
              </div>

              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
                <div className="form-group">
                  <label className="form-label">Cloud ZeroGPU Provider</label>
                  <select
                    className="select"
                    value={cloudProvider}
                    onChange={(e) => setCloudProvider(e.target.value)}
                  >
                    <option value="auto">Auto Failover (InstantID → PuLID-FLUX → PhotoMaker-V2)</option>
                    <option value="instantid">InstantID SDXL (Ultra-Realistic ~20s)</option>
                    <option value="pulid_flux">PuLID-FLUX (FLUX.1-dev Studio Grade)</option>
                    <option value="photomaker_v2">PhotoMaker-V2 (Multi-Photo Blending)</option>
                    <option value="pulid_sdxl">PuLID SDXL (Lightning Fast 4-Steps)</option>
                  </select>
                </div>

                <div className="form-group">
                  <label className="form-label">
                    <span style={{ display: "flex", alignItems: "center", gap: 5 }}>
                      <KeyRound size={13} /> Free HF Token (Optional)
                    </span>
                  </label>
                  <input
                    className="input"
                    type="password"
                    placeholder="hf_... (4x ZeroGPU quota)"
                    value={hfToken}
                    onChange={(e) => handleTokenChange(e.target.value)}
                  />
                </div>
              </div>
            </>
          )}

          {engineMode !== "cloud_zerogpu" && (
            <div className="form-group">
              <label className="form-label">Local Base Diffusion Model</label>
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
                      Realistic Vision V5.1 (SD 1.5)
                    </option>
                  </>
                ) : (
                  <option value="stabilityai/stable-diffusion-xl-base-1.0">
                    SDXL 1.0 + Trained Face LoRA (1024x1024)
                  </option>
                )}
              </select>
            </div>
          )}

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
              placeholder="Describe the setting, outfit, lighting, and mood..."
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
              <Sliders size={14} /> Inference Parameters ({width}x{height})
            </div>

            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 14 }}>
              {engineMode === "cloud_zerogpu" ? (
                <div className="form-group" style={{ marginBottom: 0 }}>
                  <label className="form-label">
                    <span>Face Identity Weight</span>
                    <span style={{ fontFamily: "var(--font-mono)" }}>{idWeight}</span>
                  </label>
                  <input
                    type="range"
                    min={0.3}
                    max={1.4}
                    step={0.05}
                    value={idWeight}
                    onChange={(e) => setIdWeight(parseFloat(e.target.value))}
                  />
                </div>
              ) : engineMode === "instant_faceid" ? (
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
                  max={35}
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
              marginBottom: 14,
              cursor: "pointer",
            }}
          >
            <input
              type="checkbox"
              checked={dryRun}
              onChange={(e) => setDryRun(e.target.checked)}
            />
            Dry-Run Preview Mode (Instant mock card for offline testing)
          </label>

          {dryRun && (
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
              <span>Dry-Run Mode is active (returns wireframe mock). Uncheck for real cloud AI generation!</span>
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
                {engineMode === "cloud_zerogpu" && !dryRun
                  ? "Generating on Free Cloud GPU (~20-40s)..."
                  : "Generating AI Photo..."}
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
              {latestResult.is_mock
                ? "Dry-Run Card"
                : latestImage?.cloud_space
                  ? `Cloud: ${latestImage.cloud_space}`
                  : "Real AI Render"}
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
                {dryRun
                  ? "Rendering Preview Card..."
                  : engineMode === "cloud_zerogpu"
                    ? "Synthesizing on Free HuggingFace ZeroGPU..."
                    : "Synthesizing AI Image..."}
              </p>
              {!dryRun && engineMode === "cloud_zerogpu" && (
                <p style={{ fontSize: "0.8rem", color: "var(--text-muted)", marginTop: 6 }}>
                  Zero load on your laptop — usually completes in 20–40 seconds!
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
