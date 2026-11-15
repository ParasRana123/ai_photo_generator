import React, { useState, useRef, useEffect } from "react";
import {
  Play,
  Pause,
  RotateCcw,
  Volume2,
  VolumeX,
  Maximize2,
  Download,
  Film,
  Sparkles,
} from "lucide-react";

export default function VideoPlayer({
  src,
  poster,
  title,
  autoPlay = false,
  loop: initialLoop = true,
  onEnded,
  className = "",
  aspectRatio = "16/9",
}) {
  const videoRef = useRef(null);
  const progressBarRef = useRef(null);

  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [isMuted, setIsMuted] = useState(true);
  const [isLooping, setIsLooping] = useState(initialLoop);
  const [playbackRate, setPlaybackRate] = useState(1);
  const [showControls, setShowControls] = useState(true);
  const [isBuffering, setIsBuffering] = useState(false);
  const [hasError, setHasError] = useState(false);

  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;

    const handleLoadedMetadata = () => {
      setDuration(video.duration || 0);
      setHasError(false);
    };

    const handleTimeUpdate = () => {
      setCurrentTime(video.currentTime || 0);
    };

    const handleWaiting = () => setIsBuffering(true);
    const handlePlaying = () => {
      setIsBuffering(false);
      setIsPlaying(true);
    };
    const handlePause = () => setIsPlaying(false);
    const handleEnded = () => {
      setIsPlaying(false);
      onEnded?.();
    };
    const handleError = () => {
      setHasError(true);
      setIsBuffering(false);
    };

    video.addEventListener("loadedmetadata", handleLoadedMetadata);
    video.addEventListener("timeupdate", handleTimeUpdate);
    video.addEventListener("waiting", handleWaiting);
    video.addEventListener("playing", handlePlaying);
    video.addEventListener("pause", handlePause);
    video.addEventListener("ended", handleEnded);
    video.addEventListener("error", handleError);

    return () => {
      video.removeEventListener("loadedmetadata", handleLoadedMetadata);
      video.removeEventListener("timeupdate", handleTimeUpdate);
      video.removeEventListener("waiting", handleWaiting);
      video.removeEventListener("playing", handlePlaying);
      video.removeEventListener("pause", handlePause);
      video.removeEventListener("ended", handleEnded);
      video.removeEventListener("error", handleError);
    };
  }, [src, onEnded]);

  const togglePlay = () => {
    const video = videoRef.current;
    if (!video) return;
    if (video.paused || video.ended) {
      video.play().catch(() => {});
    } else {
      video.pause();
    }
  };

  const toggleMute = () => {
    const video = videoRef.current;
    if (!video) return;
    video.muted = !video.muted;
    setIsMuted(video.muted);
  };

  const toggleLoop = () => {
    const video = videoRef.current;
    if (!video) return;
    video.loop = !video.loop;
    setIsLooping(video.loop);
  };

  const handleSeek = (e) => {
    const bar = progressBarRef.current;
    const video = videoRef.current;
    if (!bar || !video || !duration) return;

    const rect = bar.getBoundingClientRect();
    const clickX = Math.max(0, Math.min(e.clientX - rect.left, rect.width));
    const newFraction = clickX / rect.width;
    const newTime = newFraction * duration;
    video.currentTime = newTime;
    setCurrentTime(newTime);
  };

  const cycleSpeed = () => {
    const rates = [0.5, 1, 1.5, 2];
    const nextIdx = (rates.indexOf(playbackRate) + 1) % rates.length;
    const nextRate = rates[nextIdx];
    if (videoRef.current) {
      videoRef.current.playbackRate = nextRate;
    }
    setPlaybackRate(nextRate);
  };

  const toggleFullscreen = () => {
    const video = videoRef.current;
    if (!video) return;
    if (!document.fullscreenElement) {
      if (video.requestFullscreen) video.requestFullscreen();
      else if (video.webkitRequestFullscreen) video.webkitRequestFullscreen();
    } else {
      if (document.exitFullscreen) document.exitFullscreen();
    }
  };

  const formatTime = (secs) => {
    if (!secs || isNaN(secs)) return "0:00";
    const mins = Math.floor(secs / 60);
    const s = Math.floor(secs % 60);
    return `${mins}:${s.toString().padStart(2, "0")}`;
  };

  const progressPercent = duration > 0 ? (currentTime / duration) * 100 : 0;

  return (
    <div
      className={`video-player-container ${className}`}
      onMouseEnter={() => setShowControls(true)}
      onMouseLeave={() => isPlaying && setShowControls(false)}
      style={{
        position: "relative",
        background: "#05070a",
        borderRadius: "14px",
        overflow: "hidden",
        border: "1px solid rgba(255, 255, 255, 0.1)",
        boxShadow: "0 20px 40px rgba(0, 0, 0, 0.5)",
      }}
    >
      {/* Video Element */}
      <video
        ref={videoRef}
        src={src}
        poster={poster}
        autoPlay={autoPlay}
        loop={isLooping}
        muted={isMuted}
        playsInline
        onClick={togglePlay}
        style={{
          width: "100%",
          maxHeight: "560px",
          display: "block",
          objectFit: "contain",
          background: "#000",
          cursor: "pointer",
        }}
      />

      {/* Buffering Spinner */}
      {isBuffering && (
        <div
          style={{
            position: "absolute",
            inset: 0,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            background: "rgba(0, 0, 0, 0.4)",
            pointerEvents: "none",
          }}
        >
          <div
            className="spin"
            style={{
              width: 44,
              height: 44,
              border: "3px solid rgba(167, 139, 250, 0.3)",
              borderTopColor: "#a78bfa",
              borderRadius: "50%",
            }}
          />
        </div>
      )}

      {/* Play/Pause Overlay Button when stopped */}
      {!isPlaying && !isBuffering && !hasError && (
        <button
          type="button"
          onClick={togglePlay}
          aria-label="Play video"
          style={{
            position: "absolute",
            top: "50%",
            left: "50%",
            transform: "translate(-50%, -50%)",
            width: 64,
            height: 64,
            borderRadius: "50%",
            background: "rgba(124, 58, 237, 0.85)",
            border: "2px solid rgba(255, 255, 255, 0.8)",
            color: "#fff",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            cursor: "pointer",
            boxShadow: "0 0 30px rgba(124, 58, 237, 0.6)",
            transition: "transform 0.2s, background 0.2s",
          }}
          onMouseEnter={(e) => (e.currentTarget.style.transform = "translate(-50%, -50%) scale(1.1)")}
          onMouseLeave={(e) => (e.currentTarget.style.transform = "translate(-50%, -50%) scale(1)")}
        >
          <Play size={28} fill="#fff" style={{ marginLeft: 3 }} />
        </button>
      )}

      {/* Title Badge */}
      {title && (
        <div
          style={{
            position: "absolute",
            top: 12,
            left: 12,
            background: "rgba(9, 11, 16, 0.75)",
            backdropFilter: "blur(8px)",
            padding: "4px 10px",
            borderRadius: "6px",
            fontSize: "12px",
            fontWeight: 500,
            color: "#e2e8f0",
            border: "1px solid rgba(255, 255, 255, 0.1)",
            display: "flex",
            alignItems: "center",
            gap: 6,
            maxWidth: "80%",
            overflow: "hidden",
            textOverflow: "ellipsis",
            whiteSpace: "nowrap",
          }}
        >
          <Film size={13} color="#a78bfa" />
          <span style={{ overflow: "hidden", textOverflow: "ellipsis" }}>{title}</span>
        </div>
      )}

      {/* Control Bar */}
      <div
        style={{
          position: "absolute",
          bottom: 0,
          left: 0,
          right: 0,
          background: "linear-gradient(to top, rgba(5, 7, 10, 0.95), rgba(5, 7, 10, 0.6) 70%, transparent)",
          padding: "16px 14px 10px",
          display: "flex",
          flexDirection: "column",
          gap: 8,
          opacity: showControls || !isPlaying ? 1 : 0,
          transition: "opacity 0.25s ease",
          pointerEvents: showControls || !isPlaying ? "auto" : "none",
        }}
      >
        {/* Timeline Scrubber */}
        <div
          ref={progressBarRef}
          onClick={handleSeek}
          style={{
            height: 6,
            background: "rgba(255, 255, 255, 0.2)",
            borderRadius: 3,
            cursor: "pointer",
            position: "relative",
            transition: "height 0.15s ease",
          }}
          onMouseEnter={(e) => (e.currentTarget.style.height = "9px")}
          onMouseLeave={(e) => (e.currentTarget.style.height = "6px")}
        >
          <div
            style={{
              position: "absolute",
              top: 0,
              left: 0,
              bottom: 0,
              width: `${progressPercent}%`,
              background: "linear-gradient(90deg, #7c3aed, #ec4899)",
              borderRadius: 3,
            }}
          />
          <div
            style={{
              position: "absolute",
              top: "50%",
              left: `${progressPercent}%`,
              transform: "translate(-50%, -50%)",
              width: 12,
              height: 12,
              borderRadius: "50%",
              background: "#fff",
              boxShadow: "0 0 6px rgba(0,0,0,0.5)",
            }}
          />
        </div>

        {/* Buttons Row */}
        <div
          style={{
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            gap: 12,
            color: "#f3f4f6",
            fontSize: 12,
          }}
        >
          {/* Left Controls */}
          <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
            <button
              type="button"
              onClick={togglePlay}
              aria-label={isPlaying ? "Pause" : "Play"}
              style={controlBtnStyle}
            >
              {isPlaying ? <Pause size={17} /> : <Play size={17} />}
            </button>

            <button
              type="button"
              onClick={() => {
                if (videoRef.current) videoRef.current.currentTime = 0;
              }}
              aria-label="Restart video"
              style={controlBtnStyle}
            >
              <RotateCcw size={15} />
            </button>

            <button
              type="button"
              onClick={toggleMute}
              aria-label={isMuted ? "Unmute" : "Mute"}
              style={controlBtnStyle}
            >
              {isMuted ? <VolumeX size={17} /> : <Volume2 size={17} />}
            </button>

            {/* Time Display */}
            <span style={{ color: "#9ca3af", fontVariantNumeric: "tabular-nums" }}>
              {formatTime(currentTime)} / {formatTime(duration)}
            </span>
          </div>

          {/* Right Controls */}
          <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
            {/* Speed selector */}
            <button
              type="button"
              onClick={cycleSpeed}
              title="Playback Speed"
              style={{
                ...controlBtnStyle,
                fontSize: 11,
                fontWeight: 600,
                padding: "2px 6px",
              }}
            >
              {playbackRate}x
            </button>

            {/* Loop Toggle */}
            <button
              type="button"
              onClick={toggleLoop}
              title={isLooping ? "Loop Enabled" : "Loop Disabled"}
              style={{
                ...controlBtnStyle,
                color: isLooping ? "#a78bfa" : "#9ca3af",
              }}
            >
              <Sparkles size={15} />
            </button>

            {/* Download */}
            {src && (
              <a
                href={src}
                download={title ? `${title}.mp4` : "generated_video.mp4"}
                title="Download MP4"
                style={{
                  ...controlBtnStyle,
                  display: "inline-flex",
                  textDecoration: "none",
                }}
              >
                <Download size={16} />
              </a>
            )}

            {/* Fullscreen */}
            <button
              type="button"
              onClick={toggleFullscreen}
              title="Fullscreen"
              style={controlBtnStyle}
            >
              <Maximize2 size={16} />
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

const controlBtnStyle = {
  background: "transparent",
  border: "none",
  color: "#f3f4f6",
  cursor: "pointer",
  padding: "4px 6px",
  borderRadius: "4px",
  display: "flex",
  alignItems: "center",
  justifyContent: "center",
  transition: "color 0.15s, background 0.15s",
};
