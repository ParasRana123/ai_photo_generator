import React, { useState } from "react";
import { Sparkles, Mail, Lock, User, ArrowRight, Loader2 } from "lucide-react";
import { useAuth } from "../context/AuthContext";

export default function AuthPage() {
  const { login, register } = useAuth();
  const [mode, setMode] = useState("login"); // 'login' | 'register'
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [submitting, setSubmitting] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError("");
    setSubmitting(true);
    try {
      if (mode === "login") {
        await login(email, password);
      } else {
        await register(name, email, password);
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setSubmitting(false);
    }
  };

  const handleDemoUser = async () => {
    setError("");
    setSubmitting(true);
    const demoEmail = "demo@persona.ai";
    const demoPass = "demo1234";
    try {
      await login(demoEmail, demoPass);
    } catch {
      try {
        await register("Demo Creator", demoEmail, demoPass);
      } catch (err) {
        setError(err.message);
      }
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="auth-wrapper">
      <div className="auth-card">
        <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 24 }}>
          <div className="brand-icon">
            <Sparkles size={20} color="#fff" />
          </div>
          <div>
            <h1 style={{ fontSize: "1.35rem", fontWeight: 700, lineHeight: 1.2 }}>
              PersonaAI Studio
            </h1>
            <p style={{ fontSize: "0.85rem", color: "var(--text-secondary)" }}>
              Personal SDXL LoRA Training &amp; Inference
            </p>
          </div>
        </div>

        <div
          style={{
            display: "grid",
            gridTemplateColumns: "1fr 1fr",
            gap: 6,
            background: "var(--bg-base)",
            padding: 4,
            borderRadius: 10,
            marginBottom: 22,
          }}
        >
          <button
            type="button"
            className={`nav-tab ${mode === "login" ? "active" : ""}`}
            style={{ justifyContent: "center" }}
            onClick={() => {
              setMode("login");
              setError("");
            }}
          >
            Sign In
          </button>
          <button
            type="button"
            className={`nav-tab ${mode === "register" ? "active" : ""}`}
            style={{ justifyContent: "center" }}
            onClick={() => {
              setMode("register");
              setError("");
            }}
          >
            Create Account
          </button>
        </div>

        {error && <div className="alert-error">{error}</div>}

        <form onSubmit={handleSubmit}>
          {mode === "register" && (
            <div className="form-group">
              <label className="form-label">Full Name</label>
              <div style={{ position: "relative" }}>
                <input
                  className="input"
                  type="text"
                  placeholder="Paras Rana"
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                  required
                />
              </div>
            </div>
          )}

          <div className="form-group">
            <label className="form-label">Email Address</label>
            <input
              className="input"
              type="email"
              placeholder="you@example.com"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              required
            />
          </div>

          <div className="form-group">
            <label className="form-label">Password</label>
            <input
              className="input"
              type="password"
              placeholder="••••••••"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              minLength={6}
              required
            />
          </div>

          <button
            className="btn btn-primary"
            type="submit"
            disabled={submitting}
            style={{ width: "100%", marginTop: 8, padding: "12px 18px" }}
          >
            {submitting ? (
              <Loader2 size={18} className="spin" />
            ) : (
              <>
                {mode === "login" ? "Sign In to Studio" : "Create Account"}
                <ArrowRight size={17} />
              </>
            )}
          </button>
        </form>

        <div
          style={{
            marginTop: 20,
            paddingTop: 18,
            borderTop: "1px solid var(--border-subtle)",
            textAlign: "center",
          }}
        >
          <button
            type="button"
            className="btn btn-secondary"
            style={{ width: "100%" }}
            onClick={handleDemoUser}
            disabled={submitting}
          >
            <User size={16} /> Quick Demo Account Sign-In
          </button>
        </div>
      </div>
    </div>
  );
}
