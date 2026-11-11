import React, { useState, useEffect, useRef } from "react";
import { UploadCloud, Trash2, CheckCircle2, AlertCircle, Loader2, Image as ImageIcon } from "lucide-react";
import { photosApi } from "../api/client";

export default function PhotoUpload({ onCountChange }) {
  const [photos, setPhotos] = useState([]);
  const [dragging, setDragging] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState("");
  const fileInputRef = useRef(null);

  const fetchPhotos = async () => {
    try {
      const data = await photosApi.list();
      setPhotos(data.photos || []);
      onCountChange?.(data.totalCount || 0);
    } catch (err) {
      setError(err.message);
    }
  };

  useEffect(() => {
    fetchPhotos();
  }, []);

  const handleFiles = async (fileList) => {
    if (!fileList || fileList.length === 0) return;
    setError("");
    setUploading(true);
    try {
      await photosApi.upload(fileList);
      await fetchPhotos();
    } catch (err) {
      setError(err.message);
    } finally {
      setUploading(false);
    }
  };

  const handleDelete = async (filename) => {
    setError("");
    try {
      await photosApi.remove(filename);
      await fetchPhotos();
    } catch (err) {
      setError(err.message);
    }
  };

  const ready = photos.length >= 10;

  return (
    <div className="card">
      <div className="card-header">
        <div>
          <h2 className="card-title">
            <ImageIcon size={20} color="#a78bfa" />
            1. Upload Reference Photos
          </h2>
          <p className="card-subtitle">
            Upload 10–15 clear photos of yourself with varied angles and lighting.
          </p>
        </div>

        <span className={`badge ${ready ? "badge-success" : "badge-warning"}`}>
          {ready ? <CheckCircle2 size={14} /> : <AlertCircle size={14} />}
          {photos.length} / 15 Photos {ready ? "• Ready" : "• Need 10+"}
        </span>
      </div>

      {error && <div className="alert-error">{error}</div>}

      <div
        className={`dropzone ${dragging ? "dragging" : ""}`}
        onClick={() => fileInputRef.current?.click()}
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragging(false);
          handleFiles(e.dataTransfer.files);
        }}
      >
        <input
          ref={fileInputRef}
          type="file"
          multiple
          accept="image/jpeg,image/png,image/webp"
          style={{ display: "none" }}
          onChange={(e) => handleFiles(e.target.files)}
        />

        {uploading ? (
          <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 10 }}>
            <Loader2 size={32} className="spin" color="#8b5cf6" />
            <p style={{ fontWeight: 500 }}>Uploading photos to training workspace...</p>
          </div>
        ) : (
          <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 10 }}>
            <UploadCloud size={36} color="#8b5cf6" />
            <div>
              <p style={{ fontWeight: 600 }}>Click or drag &amp; drop photos here</p>
              <p style={{ fontSize: "0.84rem", color: "var(--text-secondary)", marginTop: 4 }}>
                Supports JPG, PNG, WEBP up to 15MB each
              </p>
            </div>
          </div>
        )}
      </div>

      {photos.length > 0 && (
        <div className="photo-grid">
          {photos.map((photo) => (
            <div key={photo.filename} className="photo-thumb">
              <img src={photo.url} alt={photo.filename} />
              <button
                type="button"
                className="photo-delete-btn"
                title="Delete photo"
                onClick={(e) => {
                  e.stopPropagation();
                  handleDelete(photo.filename);
                }}
              >
                <Trash2 size={14} />
              </button>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
