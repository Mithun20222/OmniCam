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
  const [notifiedIntruders, setNotifiedIntruders] = useState(new Set());
  const [currentIntruderAlert, setCurrentIntruderAlert] = useState(null);

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

        // Only update the alert panel for NEW intruders with snapshots
        if (
          event.label &&
          event.label.toLowerCase().includes("intruder") &&
          !event.label.toLowerCase().includes("verifying") &&
          event.image_path // Only if there's a snapshot
        ) {
          // Extract intruder ID from label like "Intruder (intruder_0)"
          const intruderMatch = event.label.match(/intruder_\d+/);
          const intruderId = intruderMatch ? intruderMatch[0] : event.id;

          // Update the alert panel with this new intruder
          setCurrentIntruderAlert({
            ...event,
            intruderId: intruderId,
          });

          // Only notify if we haven't notified about this intruder yet
          setNotifiedIntruders((prev) => {
            if (!prev.has(intruderId)) {
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

              // Add to notified set
              const newSet = new Set(prev);
              newSet.add(intruderId);
              return newSet;
            }
            return prev;
          });
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
    return `http://localhost:8000/camera_feed/${camNumber}`;
  };

  // Get first camera (fallback if not loaded)
  const cam1 = cameras[0] || { id: "cam1", location: "Camera 1" };

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
          <div className="summary-text">
            {cameras.length} Camera{cameras.length !== 1 ? "s" : ""}
          </div>
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
            <iframe
              src={feedUrl(cam1)}
              title="Cam-1 feed"
              className="feed-iframe"
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
            <div className="location-label">Latest Alert</div>
            <div className="subtext">Most recent detection</div>
          </div>

          <div className="panel-body event-scroll">
            {/* Only show the LATEST intruder alert (stays until new intruder) */}
            {currentIntruderAlert ? (
              <div className="intruder-preview">
                <div className="preview-image-container">
                  {currentIntruderAlert.image_path ? (
                    <img
                      src={`http://localhost:8000/snapshot/${currentIntruderAlert.id}`}
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
                    <div className="detail-label">Event Type</div>
                    <div className="detail-value">
                      {currentIntruderAlert.label}
                    </div>
                  </div>
                  <div className="detail-section">
                    <div className="detail-label">Location</div>
                    <div className="detail-value">
                      {currentIntruderAlert.location}
                    </div>
                  </div>
                  <div className="detail-section">
                    <div className="detail-label">Time</div>
                    <div className="detail-timestamp">
                      {new Date(
                        currentIntruderAlert.timestamp
                      ).toLocaleString()}
                    </div>
                  </div>
                  <div className="detail-section">
                    <div className="detail-label">Camera</div>
                    <div className="detail-value">
                      {currentIntruderAlert.camera_id}
                    </div>
                  </div>
                </div>
              </div>
            ) : (
              <div className="no-alerts">No intruder alerts yet</div>
            )}
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
                        <div className="history-number">
                          #{events.length - idx}
                        </div>
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