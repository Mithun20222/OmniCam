import React, { useState, useRef, useEffect } from "react";

const JoystickControl = () => {
  const [isDragging, setIsDragging] = useState(false);
  const [position, setPosition] = useState({ x: 0, y: 0 });
  const joystickRef = useRef(null);
  const knobRef = useRef(null);
  const animationFrameRef = useRef(null);

  const JOYSTICK_SIZE = 120;
  const KNOB_SIZE = 50;
  const MAX_DISTANCE = (JOYSTICK_SIZE - KNOB_SIZE) / 2;

  const sendServoCommand = (x, y) => {
    const threshold = 0.2;
    if (Math.abs(x) > threshold || Math.abs(y) > threshold) {
      // Determine direction based on which axis has larger magnitude
      let direction;
      if (Math.abs(x) > Math.abs(y)) {
        direction = x > 0 ? "PAN:RIGHT" : "PAN:LEFT";
      } else {
        direction = y > 0 ? "TILT:DOWN" : "TILT:UP";
      }

      fetch("http://localhost:8000/servo_control", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          direction: direction,
        }),
      }).catch(console.error);
    }
  };

  const updatePosition = (clientX, clientY) => {
    if (!joystickRef.current) return;

    const rect = joystickRef.current.getBoundingClientRect();
    const centerX = rect.left + rect.width / 2;
    const centerY = rect.top + rect.height / 2;

    let dx = clientX - centerX;
    let dy = clientY - centerY;
    const distance = Math.sqrt(dx * dx + dy * dy);

    if (distance > MAX_DISTANCE) {
      const angle = Math.atan2(dy, dx);
      dx = Math.cos(angle) * MAX_DISTANCE;
      dy = Math.sin(angle) * MAX_DISTANCE;
    }

    setPosition({ x: dx, y: dy });
    sendServoCommand(dx / MAX_DISTANCE, dy / MAX_DISTANCE);
  };

  const handleStart = (e) => {
    setIsDragging(true);
    const touch = e.touches ? e.touches[0] : e;
    updatePosition(touch.clientX, touch.clientY);
  };

  const handleMove = (e) => {
    if (!isDragging) return;
    e.preventDefault();
    const touch = e.touches ? e.touches[0] : e;
    if (animationFrameRef.current)
      cancelAnimationFrame(animationFrameRef.current);
    animationFrameRef.current = requestAnimationFrame(() => {
      updatePosition(touch.clientX, touch.clientY);
    });
  };

  const handleEnd = () => {
    setIsDragging(false);
    setPosition({ x: 0, y: 0 });
    sendServoCommand(0, 0);
  };

  useEffect(() => {
    if (!isDragging) return;
    const move = (e) => handleMove(e);
    const end = () => handleEnd();

    document.addEventListener("mousemove", move);
    document.addEventListener("mouseup", end);
    document.addEventListener("touchmove", move, { passive: false });
    document.addEventListener("touchend", end);

    return () => {
      document.removeEventListener("mousemove", move);
      document.removeEventListener("mouseup", end);
      document.removeEventListener("touchmove", move);
      document.removeEventListener("touchend", end);
    };
  }, [isDragging]);

  return (
    <div style={{ height: "100%", display: "flex", flexDirection: "column" }}>
      <div
        style={{
          padding: "12px 16px",
          borderBottom: "1px solid rgba(255,255,255,0.1)",
          background: "rgba(255,255,255,0.05)",
        }}
      >
        <div style={{ color: "white", fontWeight: 600 }}>Joystick</div>
        <div style={{ color: "#9ca3af", fontSize: "12px" }}>
          Manual servo control
        </div>
      </div>
      <div
        style={{
          flex: 1,
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          background: "rgba(0,0,0,0.3)",
        }}
      >
        <div
          ref={joystickRef}
          onMouseDown={handleStart}
          onTouchStart={handleStart}
          style={{
            width: "120px",
            height: "120px",
            borderRadius: "50%",
            background: "rgba(30, 30, 40, 0.8)",
            position: "relative",
            touchAction: "none",
            userSelect: "none",
            border: "2px solid rgba(255,255,255,0.2)",
          }}
        >
          <div
            ref={knobRef}
            style={{
              width: "50px",
              height: "50px",
              borderRadius: "50%",
              background: isDragging
                ? "linear-gradient(135deg, #60a5fa, #3b82f6)"
                : "linear-gradient(135deg, #94a3b8, #64748b)",
              border: "3px solid rgba(255,255,255,0.8)",
              position: "absolute",
              top: "50%",
              left: "50%",
              transform: `translate(calc(-50% + ${position.x}px), calc(-50% + ${position.y}px))`,
              transition: isDragging ? "none" : "transform 0.2s ease-out",
            }}
          />
        </div>
      </div>
      <div
        style={{
          padding: "8px 0",
          textAlign: "center",
          fontSize: "12px",
          color: isDragging ? "#60a5fa" : "#aaa",
          background: "rgba(0,0,0,0.3)",
        }}
      >
        {isDragging ? "Active" : "Standby"}
      </div>
    </div>
  );
};

export default JoystickControl;
