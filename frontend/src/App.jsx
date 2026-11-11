import React, { useState, useEffect } from "react";
import { Loader2 } from "lucide-react";
import { AuthProvider, useAuth } from "./context/AuthContext";
import AuthPage from "./components/AuthPage";
import Navbar from "./components/Navbar";
import PhotoUpload from "./components/PhotoUpload";
import TrainingProgress from "./components/TrainingProgress";
import PromptStudio from "./components/PromptStudio";
import ImageGallery from "./components/ImageGallery";
import { aiApi } from "./api/client";

function StudioWorkspace() {
  const { user, loading } = useAuth();
  const [activeTab, setActiveTab] = useState("train"); // 'train' | 'generate' | 'gallery'
  const [photoCount, setPhotoCount] = useState(0);
  const [galleryRefresh, setGalleryRefresh] = useState(0);
  const [galleryCount, setGalleryCount] = useState(0);

  useEffect(() => {
    if (!user) return;
    aiApi
      .getGallery()
      .then((d) => setGalleryCount((d.images || []).length))
      .catch(() => {});
  }, [user, galleryRefresh]);

  if (loading) {
    return (
      <div style={{ minHeight: "100vh", display: "grid", placeItems: "center" }}>
        <Loader2 size={36} className="spin" color="#8b5cf6" />
      </div>
    );
  }

  if (!user) {
    return <AuthPage />;
  }

  return (
    <>
      <Navbar
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        galleryCount={galleryCount}
      />

      <main className="app-container">
        {activeTab === "train" && (
          <div className="grid-2">
            <PhotoUpload onCountChange={setPhotoCount} />
            <TrainingProgress
              photoCount={photoCount}
              onTrainingComplete={() => setActiveTab("generate")}
            />
          </div>
        )}

        {activeTab === "generate" && (
          <PromptStudio
            onGenerated={() => setGalleryRefresh((c) => c + 1)}
          />
        )}

        {activeTab === "gallery" && (
          <ImageGallery
            refreshSignal={galleryRefresh}
            onImagesLoaded={setGalleryCount}
          />
        )}
      </main>
    </>
  );
}

export default function App() {
  return (
    <AuthProvider>
      <StudioWorkspace />
    </AuthProvider>
  );
}

