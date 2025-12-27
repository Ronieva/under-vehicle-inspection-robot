# Setup notes

## RealSense
Install Intel RealSense SDK for your OS and verify the camera is detected:
- `realsense-viewer`

## Serial permissions (Linux)
If you get permission errors on `/dev/ttyACM0`, add your user to the dialout group:
```bash
sudo usermod -aG dialout $USER
```
Log out and back in.

## OpenVINO
Install OpenVINO runtime (CPU).
