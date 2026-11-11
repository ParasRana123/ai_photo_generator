import React, { useState, useEffect } from "react";
import { Images, RefreshCw, Download, X, Sparkles } from "lucide-react";
import { aiApi } from "../api/client";

export default function ImageGallery({ refreshSignal = 0, onImagesLoaded }) {
  const [images, setImages] = useState([]);
  const [loading, setLoading] = useState(true);
  const [selected, setSelected] = useState(null);
  const [error, setError] = useState("");

  const fetchGallery = async () => {
    setLoading(true);
    setError("");
    try {
      const data = await aiApi.getGallery();
      const list = data.images || [];
      setImages(list);
      onImagesLoaded?.(list.length);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchGallery();
  }, [refreshSignal]);

  return (
    <div className="card">
      <div className="card-header">
        <div>
          <h2 className="card-title">
            <Images size={20} color="#a78bfa" />
            Generated Image Gallery
          </h2>
          <p className="card-subtitle">
            Browse all synthesized AI photos along with their generation seeds and prompts.
          </p>
        </div>

        <button
          type="button"
          className="btn btn-secondary"
          onClick={fetchGallery}
          disabled={loading}
        >
          <RefreshCw size={15} className={loading ? "spin" : ""} />
          Refresh
        </button>
      </div>

      {error && <div className="alert-error">{error}</div>}

      {!loading && images.length === 0 ? (
        <div
          style={{
            padding: "56px 20px",
            textAlign: "center",
            color: "var(--text-muted)",
            background: "var(--bg-base)",
            borderRadius: "var(--radius-md)",
            border: "1px dashed var(--border-subtle)",
          }}
        >
          <Sparkles size={38} style={{ opacity: 0.35, marginBottom: 12 }} />
          <p style={{ fontWeight: 500, color: "var(--text-secondary)" }}>No generated images yet.</p>
          <p style={{ fontSize: "0.86rem", marginTop: 4 }}>
            Head over to the Prompt Studio tab to synthesize your first photo!
          </p>
        </div>
      ) : (
        <div className="gallery-grid">
          {images.map((item) => (
            <div
              key={item.filename}
              className="gallery-item"
              onClick={() => setSelected(item)}
            >
              <img src={item.url} alt={item.prompt || item.filename} loading="lazy" />
              <div className="gallery-overlay">
                <div
                  style={{
                    display: "flex",
                    justifyContent: "space-between",
                    alignItems: "center",
                    marginBottom: 6,
                  }}
                >
                  <span className="badge badge-info" style={{ fontSize: "0.7rem" }}>
                    {item.style_preset || "Custom"}
                  </span>
                  {item.seed && (
                    <span
                      style={{
                        fontSize: "0.72rem",
                        color: "var(--text-muted)",
                        fontFamily: "var(--font-mono)",
                      }}
                    >
                      #{item.seed}
                    </span>
                  )}
                </div>
                <p
                  style={{
                    fontSize: "0.82rem",
                    color: "var(--text-secondary)",
                    display: "-webkit-box",
                    WebkitLineClamp: 2,
                    WebkitBoxOrient: "vertical",
                    overflow: "hidden",
                  }}
                >
                  {item.prompt || item.filename}
                </p>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Lightbox Modal */}
      {selected && (
        <div className="modal-backdrop" onClick={() => setSelected(null)}>
          <div className="modal-content" onClick={(e) => e.stopPropagation()}>
            <div style={{ background: "#000", display: "grid", placeItems: "center" }}>
              <img
                src={selected.url}
                alt={selected.prompt}
                style={{ width: "100%", maxHeight: "75vh", objectFit: "contain" }}
              />
            </div>

            <div
              style={{
                padding: 24,
                display: "flex",
                flexDirection: "column",
                justifyContent: "space-between",
                gap: 16,
              }}
            >
              <div>
                <div
                  style={{
                    display: "flex",
                    justifyContent: "space-between",
                    alignItems: "center",
                    marginBottom: 16,
                  }}
                >
                  <span className="badge badge-info">{selected.style_preset || "SDXL LoRA"}</span>
                  <button
                    type="button"
                    className="btn btn-ghost"
                    style={{ padding: 6 }}
                    onClick={() => setSelected(null)}
                  >
                    <X size={18} />
                  </button>
                </div>

                <h3 style={{ fontSize: "0.9rem", color: "var(--text-secondary)", marginBottom: 6 }}>
                  Prompt
                </h3>
                <p
                  style={{
                    fontSize: "0.9rem",
                    background: "var(--bg-base)",
                    padding: 12,
                    borderRadius: "var(--radius-sm)",
                    border: "1px solid var(--border-subtle)",
                    lineHeight: 1.5,
                  }}
                >
                  {selected.prompt || "No prompt metadata recorded"}
                </p>

                <div
                  style={{
                    marginTop: 16,
                    fontSize: "0.82rem",
                    color: "var(--text-secondary)",
                    display: "grid",
                    gap: 6,
                    fontFamily: "var(--font-mono)",
                  }}
                >
                  <div>File: {selected.filename}</div>
                  {selected.seed && <div>Seed: {selected.seed}</div>}
                </div>
              </div>

              <a
                href={selected.url}
                download={selected.filename}
                className="btn btn-primary"
                style={{ textDecoration: "none", width: "100%" }}
              >
                <Download size={16} /> Download High-Res PNG
              </a>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

