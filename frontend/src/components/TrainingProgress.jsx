import React, { useState, useEffect } from "react";
import { Cpu, Play, CheckCircle2, Loader2, ArrowRight, Activity } from "lucide-react";
import { aiApi } from "../api/client";

export default function TrainingProgress({ photoCount = 0, onTrainingComplete }) {
  const [triggerWord, setTriggerWord] = useState("sks person");
  const [steps, setSteps] = useState(15);
  const [rank, setRank] = useState(16);
  const [dryRun, setDryRun] = useState(true);
  const [job, setJob] = useState({
    status: "idle",
    progress_percent: 0,
    current_step: 0,
    total_steps: 0,
    current_loss: 0,
    message: "Ready to start SDXL LoRA training",
    lora_path: null,
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
        if (typeof h.mock_mode_default === "boolean") {
          setDryRun(h.mock_mode_default);
        }
      })
      .catch(() => {});
  }, []);

  useEffect(() => {
    if (!isRunning) return;
    const timer = setInterval(pollStatus, 800);
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
            2. Train Personal SDXL LoRA
          </h2>
          <p className="card-subtitle">
            Fine-tune your custom identity weights (.safetensors) via Diffusers.
          </p>
        </div>

        <span className={`badge ${statusBadgeClass()}`}>
          {isRunning && <Loader2 size={13} className="spin" />}
          {job.status.toUpperCase()}
        </span>
      </div>

      {error && <div className="alert-error">{error}</div>}

      <form onSubmit={handleStartTraining}>
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

        <label
          style={{
            display: "flex",
            alignItems: "center",
            gap: 10,
            fontSize: "0.88rem",
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
          Fast Local Verification Mode (Dry-Run without CUDA GPU)
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
                Step: {job.current_step}/{job.total_steps}
              </span>
              <span>Loss: {job.current_loss}</span>
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
                <Loader2 size={16} className="spin" /> Training in Progress...
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
