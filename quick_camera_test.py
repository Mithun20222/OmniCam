#!/usr/bin/env python3
"""
Quick Camera Test for OmniCam
Simple test to verify camera works without starting the full server.
"""

import cv2
import sys

def quick_test():
    """Quick test of camera 0 with different backends."""
    
    print("🔍 Quick Camera Test")
    print("=" * 30)
    
    # Test different backends
    backends = [
        (cv2.CAP_MSMF, "Microsoft Media Foundation"),
        (cv2.CAP_ANY, "Any Available"),
        (cv2.CAP_DSHOW, "DirectShow")
    ]
    
    for backend, name in backends:
        print(f"\nTesting {name}...")
        try:
            cap = cv2.VideoCapture(0, backend)
            if cap.isOpened():
                print(f"✅ Camera opened with {name}")
                
                # Try to read a frame
                ret, frame = cap.read()
                if ret and frame is not None:
                    print(f"✅ Frame captured: {frame.shape[1]}x{frame.shape[0]}")
                    print(f"✅ {name} is working!")
                    cap.release()
                    return True
                else:
                    print(f"❌ Camera opened but no frame captured")
                cap.release()
            else:
                print(f"❌ Camera failed to open with {name}")
        except Exception as e:
            print(f"❌ Error with {name}: {e}")
    
    print("\n❌ No working camera found!")
    return False

if __name__ == "__main__":
    success = quick_test()
    if success:
        print("\n🎉 Camera test successful! Your camera should work with the server.")
    else:
        print("\n💡 Troubleshooting tips:")
        print("1. Make sure your camera is connected")
        print("2. Check if another application is using the camera")
        print("3. Try running as administrator")
        print("4. Restart your computer")
    
    sys.exit(0 if success else 1)
