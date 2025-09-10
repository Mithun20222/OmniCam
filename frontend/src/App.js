import React, { useEffect, useState } from "react";
import axios from "axios";
import { ToastContainer, toast } from "react-toastify";
import "react-toastify/dist/ReactToastify.css";

function App() {
  const [cameras, setCameras] = useState([]);
  const [events, setEvents] = useState([]);
  const [latest, setLatest] = useState(null);

  useEffect(() => {
    // Load cameras
    axios.get("http://localhost:8000/cameras").then((res) => {
      setCameras(res.data);
    });

    // Load events
    axios.get("http://localhost:8000/events").then((res) => {
      setEvents(res.data);
    });

    // WebSocket for live intruder events
    const ws = new WebSocket("ws://localhost:8000/ws/events");
    ws.onmessage = (msg) => {
      const event = JSON.parse(msg.data);
      setLatest(event);
      setEvents((prev) => [event, ...prev].slice(0, 50));

      // 🔔 Show toast only for intruders
      if (event.label.toLowerCase().includes("intruder")) {
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
    };

    return () => ws.close();
  }, []);

  return (
    <div className="p-6 min-h-screen" style={{ backgroundColor: "#BBDCE5" }}>
      {/* Toast Container */}
      <ToastContainer />

      {/* Header */}
      <h1
        className="text-3xl font-bold mb-6 p-4 rounded-lg shadow"
        style={{ backgroundColor: "#BBDCE5", color: "#333" }}
      >
        OmniCam Dashboard
      </h1>

      {/* Camera list */}
      <div className="grid grid-cols-2 gap-4">
        {cameras.map((cam) => (
          <div
            key={cam.id}
            className="p-4 rounded-lg shadow"
            style={{ backgroundColor: "#D9C4B0" }}
          >
            <h2 className="font-semibold text-lg">{cam.location}</h2>
            <p className="text-sm text-gray-700">ID: {cam.id}</p>
          </div>
        ))}
      </div>

      {/* Latest Intruder Popup */}
      {latest && (
        <div
          className="mt-8 p-6 rounded-lg shadow-lg border"
          style={{ backgroundColor: "#CFAB8D", borderColor: "#BBDCE5" }}
        >
          <h2 className="text-xl font-bold text-red-800 flex items-center">
            🚨 Intruder Alert!
          </h2>
          <p className="mt-2">
            Camera: <strong>{latest.camera_id}</strong> ({latest.location})
          </p>
          <p>Time: {new Date(latest.timestamp).toLocaleString()}</p>
          <img
            src={`http://localhost:8000/snapshot/${latest.id}`}
            alt="snapshot"
            className="mt-3 w-72 border rounded"
          />
        </div>
      )}

      {/* Event History */}
      <div className="mt-10">
        <h2 className="text-2xl font-bold mb-3">Event History</h2>
        <ul className="space-y-3">
          {events.map((e) => (
            <li
              key={e.id}
              className="flex items-center space-x-4 p-3 rounded-lg shadow"
              style={{ backgroundColor: "#D9C4B0" }}
            >
              <img
                src={`http://localhost:8000/snapshot/${e.id}`}
                alt="snapshot"
                className="w-24 border rounded"
              />
              <div>
                <p className="font-semibold">{e.label}</p>
                <p className="text-sm text-gray-700">
                  {e.camera_id} - {e.location}
                </p>
                <p className="text-xs text-gray-600">
                  {new Date(e.timestamp).toLocaleString()}
                </p>
              </div>
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
}

export default App;
