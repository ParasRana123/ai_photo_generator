import React, { useState, useEffect } from "react";
import { Cpu, Play, Zap, Loader2, ArrowRight, Activity } from "lucide-react";
import { aiApi } from "../api/client";

export default function TrainingProgress({ photoCount = 0, onTrainingComplete }) {
  const [engineMode, setEngineMode] = useState("instant_faceid"); // 'instant_faceid' | 'sdxl_lora'
  const [triggerWord, setTriggerWord] = useState("sks person");
  const [steps, setSteps] = useState(15);
  const [rank, setRank] = useState(16);
  const [dryRun, setDryRun] = useState(false);
  const [health, setHealth] = useState(null);
  const [job, setJob] = useState({
    status: "idle",
    engine_mode: "instant_faceid",
    progress_percent: 0,
    current_step: 0,
    total_steps: 0,
    current_loss: 0,
    faces_detected: 0,
    message: "Ready for Instant FaceID enrollment (~2-5s)",
    lora_path: null,
    face_id_path: null,
  });
  const [error, setError] = useState("");
  const [starting, setStarting] = useState(false);

  const isRunning = ["preprocessing", "caching", "training", "saving"].includes(job.status);

  const pollStatus = async () => {
    try {
      const data = await aiApi.getTrainStatus();
      setJob(data);
    } catch {
      // Ignore transient polling errors
    }
  };

  useEffect(() => {
    pollStatus();
    aiApi
      .getHealth()
      .then((h) => {
        setHealth(h);
        if (engineMode === "instant_faceid" && h.insightface_available) {
          setDryRun(false);
        } else if (typeof h.mock_mode_default === "boolean") {
          setDryRun(h.mock_mode_default);
        }
      })
      .catch(() => {});
  }, []);

  const handleModeChange = (mode) => {
    setEngineMode(mode);
    if (mode === "instant_faceid") {
      setDryRun(health?.insightface_available ? false : true);
    } else {
      setDryRun(!health?.device?.cuda_available);
    }
  };

  useEffect(() => {
    if (!isRunning) return;
    const timer = setInterval(pollStatus, 600);
    return () => clearInterval(timer);
  }, [isRunning]);

  const handleStartTraining = async (e) => {
    e.preventDefault();
    setError("");
    setStarting(true);
    try {
      const status = await aiApi.startTraining({
        trigger_word: triggerWord,
        steps: Number(steps),
        rank: Number(rank),
        engine_mode: engineMode,
        dry_run: dryRun,
      });
      setJob(status);
    } catch (err) {
      setError(err.message);
    } finally {
      setStarting(false);
    }
  };

  const statusBadgeClass = () => {
    if (job.status === "completed") return "badge-success";
    if (job.status === "failed") return "badge-danger";
    if (isRunning) return "badge-info";
    return "badge-warning";
  };

  return (
    <div className="card">
      <div className="card-header">
        <div>
          <h2 className="card-title">
            <Cpu size={20} color="#a78bfa" />
            2. Personalize AI Identity
          </h2>
          <p className="card-subtitle">
            Choose Instant FaceID (~seconds on CPU/GPU) or full SDXL LoRA training.
          </p>
        </div>

        <span className={`badge ${statusBadgeClass()}`}>
          {isRunning && <Loader2 size={13} className="spin" />}
          {job.status.toUpperCase()}
        </span>
      </div>

      {error && <div className="alert-error">{error}</div>}

      <form onSubmit={handleStartTraining}>
        {/* Engine Mode Switcher */}
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
            onClick={() => handleModeChange("instant_faceid")}
            disabled={isRunning}
          >
            <div style={{ display: "flex", alignItems: "center", gap: 6, fontWeight: 600, fontSize: "0.86rem" }}>
              <Zap size={15} color="#a78bfa" /> Instant FaceID
            </div>
            <div style={{ fontSize: "0.74rem", color: "var(--text-secondary)", marginTop: 4 }}>
              ArcFace Embedding • ~2–5s on CPU/GPU
            </div>
          </button>

          <button
            type="button"
            className={`style-card ${engineMode === "sdxl_lora" ? "selected" : ""}`}
            onClick={() => handleModeChange("sdxl_lora")}
            disabled={isRunning}
          >
            <div style={{ display: "flex", alignItems: "center", gap: 6, fontWeight: 600, fontSize: "0.86rem" }}>
              <Cpu size={15} color="#60a5fa" /> SDXL LoRA
            </div>
            <div style={{ fontSize: "0.74rem", color: "var(--text-secondary)", marginTop: 4 }}>
              Deep Weight Fine-Tuning • Needs GPU
            </div>
          </button>
        </div>

        {engineMode === "sdxl_lora" ? (
          <>
            <div className="form-group">
              <label className="form-label">Subject Trigger Token</label>
              <input
                className="input"
                type="text"
                value={triggerWord}
                onChange={(e) => setTriggerWord(e.target.value)}
                disabled={isRunning}
                required
              />
            </div>

            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
              <div className="form-group">
                <label className="form-label">Training Steps</label>
                <input
                  className="input"
                  type="number"
                  min={5}
                  max={2000}
                  value={steps}
                  onChange={(e) => setSteps(e.target.value)}
                  disabled={isRunning}
                />
              </div>

              <div className="form-group">
                <label className="form-label">LoRA Rank</label>
                <select
                  className="select"
                  value={rank}
                  onChange={(e) => setRank(e.target.value)}
                  disabled={isRunning}
                >
                  <option value={8}>Rank 8 (Lightweight)</option>
                  <option value={16}>Rank 16 (Recommended)</option>
                  <option value={32}>Rank 32 (High Detail)</option>
                </select>
              </div>
            </div>
          </>
        ) : (
          <div
            style={{
              background: "rgba(124, 58, 237, 0.08)",
              border: "1px solid rgba(124, 58, 237, 0.25)",
              borderRadius: "var(--radius-sm)",
              padding: "10px 12px",
              fontSize: "0.82rem",
              color: "#ddd6fe",
              marginBottom: 14,
            }}
          >
            Extracts and averages 512-dim InsightFace embeddings from your uploaded photos into{" "}
            <code>person_face_id.safetensors</code> — zero backpropagation required.
          </div>
        )}

        <label
          style={{
            display: "flex",
            alignItems: "center",
            gap: 10,
            fontSize: "0.86rem",
            color: "var(--text-secondary)",
            marginBottom: 18,
            cursor: "pointer",
          }}
        >
          <input
            type="checkbox"
            checked={dryRun}
            onChange={(e) => setDryRun(e.target.checked)}
            disabled={isRunning}
          />
          Simulation / Dry-Run Mode (Mock embeddings without running InsightFace/SDXL)
        </label>

        {/* Live Progress Indicator Panel */}
        <div
          style={{
            background: "var(--bg-base)",
            border: "1px solid var(--border-subtle)",
            borderRadius: "var(--radius-md)",
            padding: 16,
            marginBottom: 18,
          }}
        >
          <div
            style={{
              display: "flex",
              justifyContent: "space-between",
              alignItems: "center",
              marginBottom: 8,
              fontSize: "0.86rem",
            }}
          >
            <span style={{ display: "flex", alignItems: "center", gap: 6, color: "var(--text-primary)" }}>
              <Activity size={15} color="#8b5cf6" />
              {job.message}
            </span>
            <span style={{ fontFamily: "var(--font-mono)", fontWeight: 600 }}>
              {Math.round(job.progress_percent)}%
            </span>
          </div>

          <div className="progress-track">
            <div
              className="progress-fill"
              style={{ width: `${Math.min(100, Math.max(0, job.progress_percent))}%` }}
            />
          </div>

          {(isRunning || job.status === "completed") && (
            <div
              style={{
                display: "flex",
                justifyContent: "space-between",
                marginTop: 10,
                fontSize: "0.8rem",
                color: "var(--text-secondary)",
                fontFamily: "var(--font-mono)",
              }}
            >
              <span>
                Progress: {job.current_step}/{job.total_steps}
              </span>
              {job.engine_mode === "instant_faceid" ? (
                <span>Faces Enrolled: {job.faces_detected}</span>
              ) : (
                <span>Loss: {job.current_loss}</span>
              )}
            </div>
          )}
        </div>

        <div style={{ display: "flex", gap: 10 }}>
          <button
            type="submit"
            className="btn btn-primary"
            disabled={isRunning || starting}
            style={{ flex: 1 }}
          >
            {isRunning ? (
              <>
                <Loader2 size={16} className="spin" /> Processing Identity...
              </>
            ) : engineMode === "instant_faceid" ? (
              <>
                <Zap size={16} />
                {photoCount === 0 ? "Enroll Sample Photos (~2s)" : "Enroll FaceID (~2-5s)"}
              </>
            ) : (
              <>
                <Play size={16} />
                {photoCount === 0 ? "Train with Auto Sample Photos" : "Start LoRA Training"}
              </>
            )}
          </button>

          {job.status === "completed" && onTrainingComplete && (
            <button
              type="button"
              className="btn btn-secondary"
              onClick={onTrainingComplete}
            >
              Generate Photos <ArrowRight size={16} />
            </button>
          )}
        </div>
      </form>
    </div>
  );
}
