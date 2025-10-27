import React, { useEffect, useState } from "react";
import axios from "axios";
import { ToastContainer, toast } from "react-toastify";
import "react-toastify/dist/ReactToastify.css";
import "./App.css";

function App() {
  const [cameras, setCameras] = useState([]);
  const [events, setEvents] = useState([]);
  const [latest, setLatest] = useState(null);
  const [verificationStatus, setVerificationStatus] = useState(null);

  useEffect(() => {
    // Load cameras
    axios
      .get("http://localhost:8000/cameras")
      .then((res) => {
        setCameras(res.data);
      })
      .catch((error) => {
        console.error("Failed to load cameras:", error);
        toast.error("Failed to load camera data");
      });

    // Load events
    axios
      .get("http://localhost:8000/events")
      .then((res) => {
        setEvents(res.data);
      })
      .catch((error) => {
        console.error("Failed to load events:", error);
        toast.error("Failed to load event data");
      });

    // WebSocket for live intruder events
    const ws = new WebSocket("ws://localhost:8000/ws/events");

    ws.onopen = () => {
      console.log("WebSocket connected");
    };

    ws.onerror = (error) => {
      console.error("WebSocket error:", error);
      toast.error("Connection to server lost");
    };

    ws.onclose = () => {
      console.log("WebSocket disconnected");
    };

    ws.onmessage = (msg) => {
      try {
        const data = JSON.parse(msg.data);

        if (data.type === "verification") {
          setVerificationStatus(data);
          setTimeout(() => setVerificationStatus(null), 3000);
          return;
        }

        const event = data;
        setLatest(event);
        setEvents((prev) => [event, ...prev].slice(0, 50));
        setVerificationStatus(null);

        if (
          event.label &&
          event.label.toLowerCase().includes("intruder") &&
          !event.label.toLowerCase().includes("verifying")
        ) {
          toast.error(
            `🚨 Intruder detected at ${event.location} (Camera: ${event.camera_id})`,
            {
              position: "top-right",
              autoClose: 5000,
              hideProgressBar: false,
              closeOnClick: true,
              pauseOnHover: true,
              draggable: true,
              style: {
                backgroundColor: "#CFAB8D",
                color: "#000",
                fontWeight: "bold",
              },
            }
          );
        }
      } catch (error) {
        console.error("Error parsing WebSocket message:", error);
      }
    };

    return () => ws.close();
  }, []);

  // helper: pick camera feed url for a given cam object
  const feedUrl = (cam) => {
    // Extract camera number from cam.id (e.g., "cam1" -> "1")
    const camNumber = (cam.id || "cam1").replace("cam", "");
    const url = `http://localhost:8000/camera_feed/${camNumber}`;
    console.log("Camera feed URL:", url, "for camera:", cam);
    return url;
  };

  // Get first camera (fallback if not loaded)
  const cam1 = cameras[0] || { id: "cam1", location: "Camera 1" };
  
  console.log("Cameras loaded:", cameras);
  console.log("Using cam1:", cam1);

  return (
    <div className="app-container">
      <ToastContainer />

      {/* Header */}
      <div className="app-header">
        <div className="title-pill">
          <h1 className="app-title">OmniCam</h1>
        </div>
      </div>

      {/* Top summary row */}
      <div className="summary-bar">
        <div className="summary-left">
          <div className="summary-text">Intelligent Security System</div>
        </div>
        <div className="summary-right">
          <div className="summary-text">{cameras.length} Camera{cameras.length !== 1 ? 's' : ''}</div>
          <div className="summary-text">{events.length} Events</div>
        </div>
      </div>

      {/* Main grid - 2 columns on top, 1 full width on bottom */}
      <div className="main-grid">
        {/* Top-left: Large Live feed (Cam-1) */}
        <div className="panel frame-like camera-panel">
          <div className="panel-header">
            <div className="header-left">
              <div className="cam-label">Cam-1</div>
              <div className="location-label">{cam1.location}</div>
            </div>
            <div className="stream-label">Live stream</div>
          </div>

          <div className="panel-body camera-feed">
            <img
              src={feedUrl(cam1)}
              alt="Cam-1 feed"
              className="feed-image"
              key={cam1.id}
            />

            {/* Verifying overlay */}
            {verificationStatus && (
              <div className="verifying-badge">
                <div className="pulse-dot"></div>
                <span className="verifying-text">Verifying...</span>
              </div>
            )}
          </div>
        </div>

        {/* Top-right: Recent alerts preview */}
        <div className="panel frame-like event-panel">
          <div className="panel-header">
            <div className="location-label">Event history</div>
            <div className="subtext">Recent alerts & quick preview</div>
          </div>

          <div className="panel-body event-scroll">
            {/* Intruder preview card (prominent) */}
            {latest ? (
              <div className="intruder-preview">
                <div className="preview-image-container">
                  {latest.image_path ? (
                    <img
                      src={`http://localhost:8000/snapshot/${latest.id}`}
                      alt="Intruder snapshot"
                      className="preview-image"
                      onError={(e) => {
                        e.target.style.display = "none";
                      }}
                    />
                  ) : (
                    <div className="no-snapshot">No snapshot</div>
                  )}
                </div>

                <div className="preview-details">
                  <div className="detail-section">
                    <div className="detail-label">Location & Time</div>
                    <div className="detail-value">{latest.location}</div>
                    <div className="detail-timestamp">
                      {new Date(latest.timestamp).toLocaleString()}
                    </div>
                  </div>
                  <div className="detail-section">
                    <div className="detail-label">Camera</div>
                    <div className="detail-value">{latest.camera_id}</div>
                  </div>
                </div>
              </div>
            ) : (
              <div className="no-alerts">No recent alerts</div>
            )}

            {/* Compact list of recent events */}
            <div className="event-list">
              {events.slice(0, 6).map((e) => (
                <div key={e.id} className="event-item">
                  <div className="event-info">
                    <div className="event-label">{e.label}</div>
                    <div className="event-meta">
                      {e.location} • {new Date(e.timestamp).toLocaleString()}
                    </div>
                  </div>
                  <div className="event-badge">
                    {e.label.toLowerCase().includes("intruder") ? (
                      <span className="badge-intruder">INTRUDER</span>
                    ) : (
                      <span className="badge-authorized">AUTHORIZED</span>
                    )}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Bottom: Full Event History (full width, scrollable) */}
        <div className="panel frame-like history-panel full-width">
          <div className="panel-header">
            <div className="location-label">Event history</div>
            <div className="subtext">All recorded events</div>
          </div>

          <div className="panel-body history-scroll">
            {events.length === 0 ? (
              <div className="empty-state">
                <div className="empty-icon">📊</div>
                <div className="empty-text">No events recorded yet</div>
              </div>
            ) : (
              <div className="history-list">
                {events.map((e, idx) => (
                  <div key={e.id} className="history-item">
                    <div className="history-thumbnail">
                      {e.image_path ? (
                        <img
                          src={`http://localhost:8000/snapshot/${e.id}`}
                          alt="snapshot"
                          className="thumbnail-image"
                          onError={(ev) => {
                            ev.target.style.display = "none";
                          }}
                        />
                      ) : (
                        <div className="no-image">No image</div>
                      )}
                    </div>

                    <div className="history-details">
                      <div className="history-top">
                        <div className="history-label">{e.label}</div>
                        <div className="history-number">#{events.length - idx}</div>
                      </div>
                      <div className="history-location">
                        {e.camera_id} • {e.location}
                      </div>
                      <div className="history-timestamp">
                        {new Date(e.timestamp).toLocaleString()}
                      </div>
                    </div>

                    <div className="history-status">
                      {e.label.toLowerCase().includes("intruder") ? (
                        <span className="status-intruder">INTRUDER</span>
                      ) : (
                        <span className="status-authorized">AUTHORIZED</span>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;