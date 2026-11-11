import React from "react";
import { Sparkles, UploadCloud, Wand2, Images, LogOut, User } from "lucide-react";
import { useAuth } from "../context/AuthContext";

export default function Navbar({ activeTab, setActiveTab, galleryCount = 0 }) {
  const { user, logout } = useAuth();

  return (
    <header className="navbar">
      <div className="navbar-inner">
        <div className="brand">
          <div className="brand-icon">
            <Sparkles size={18} color="#fff" />
          </div>
          <span>PersonaAI Studio</span>
        </div>

        <nav className="nav-tabs" aria-label="Main workspace tabs">
          <button
            type="button"
            className={`nav-tab ${activeTab === "train" ? "active" : ""}`}
            onClick={() => setActiveTab("train")}
          >
            <UploadCloud size={16} />
            Upload &amp; Train
          </button>
          <button
            type="button"
            className={`nav-tab ${activeTab === "generate" ? "active" : ""}`}
            onClick={() => setActiveTab("generate")}
          >
            <Wand2 size={16} />
            Prompt Studio
          </button>
          <button
            type="button"
            className={`nav-tab ${activeTab === "gallery" ? "active" : ""}`}
            onClick={() => setActiveTab("gallery")}
          >
            <Images size={16} />
            Gallery
            {galleryCount > 0 && (
              <span
                style={{
                  fontSize: "0.72rem",
                  background: "rgba(255,255,255,0.18)",
                  padding: "1px 7px",
                  borderRadius: 999,
                }}
              >
                {galleryCount}
              </span>
            )}
          </button>
        </nav>

        <div className="user-pill">
          <div className="user-badge">
            <User size={15} />
            <span>{user?.name || user?.email}</span>
          </div>
          <button
            type="button"
            className="btn btn-ghost"
            onClick={logout}
            title="Sign out"
            style={{ padding: "8px 12px" }}
          >
            <LogOut size={16} />
          </button>
        </div>
      </div>
    </header>
  );
}
