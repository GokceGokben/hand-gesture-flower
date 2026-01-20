# Flower Animation Hand Tracking with MediaPipe

An interactive Python application that brings a flower to life through your hand movements - open your hand to bloom the flower, close it to watch it fold.

## Description

This project uses MediaPipe's Hand Landmarker to create an immersive augmented reality experience where a flower animation responds to your hand gestures in real-time. The flower blooms and closes based on how open or closed your hand is, with smooth transitions that adapt to your movement speed. The animation tracks the middle finger knuckle position, creating the illusion that the flower is growing from your hand.

## Features

- **Gesture-Based Control**: Open your hand to bloom the flower, close it to watch it fold
- **Real-Time Hand Tracking**: Precise hand position and gesture detection using MediaPipe Hand Landmarker
- **Adaptive Animation Speed**: Animation transitions adapt to your hand movement speed for natural interaction
- **Smooth Interpolation**: Fluid frame transitions create seamless blooming and closing effects
- **Visual Feedback**: On-screen display shows hand openness ratio (0.0 = closed, 1.0 = fully open)
- **Transparent PNG Overlay**: Supports alpha channel blending for realistic flower placement

## Requirements

```
opencv-python>=4.5.0
mediapipe>=0.10.0
numpy>=1.19.0
pillow>=8.0.0
```

## Installation

1. Clone or download this repository:
```bash
git clone https://github.com/GokceGokben/hand-gesture-flower.git
cd "flower animation"
```

2. Install the required packages:
```bash
pip install opencv-python mediapipe numpy pillow
```

3. Extract frames from the flower GIF file:
```bash
python utility.py
```

This command will convert the flower animation GIF into individual PNG frames in the `frames/` directory. The model file (`hand_landmarker.task`) will be automatically downloaded on first run if not present.

## Usage

Run the main application:
```bash
python flower.py
```

**Controls:**
- **Open Hand**: Bloom the flower (spread your fingers apart)
- **Close Hand**: Fold the flower (bring your fingers together)
- **Move Hand**: The flower follows your hand position
- **Press 'q'**: Exit the application

**Tips:**
- Ensure good lighting for better hand detection
- Keep your hand within the camera frame
- Experiment with different hand opening speeds for various animation effects
- The debug display shows the current openness ratio (0.0 to 1.0)

## File Structure

```
flower animation/
├── assets/                # Demo materials and media files
│   └── demo.mp4
├── frames/                # Extracted PNG frames with alpha channel
│   ├── frame_000.png
│   ├── frame_001.png
│   └── ...
├── flower.gif             # Source flower animation GIF
├── flower.py              # Main application with hand tracking and animation logic
├── hand_landmarker.task   # MediaPipe hand tracking model (auto-downloaded)
├── README.md              # Documentation
└── utility.py             # Utility script to extract PNG frames from GIF
```

## How It Works

1. **Hand Detection**: MediaPipe detects 21 hand landmarks in real-time from the webcam feed
2. **Openness Calculation**: Measures average distance from wrist to fingertips (index, middle, ring, pinky)
3. **Frame Mapping**: Maps hand openness (0.0-1.0) to animation frames (closed to fully bloomed)
4. **Smooth Interpolation**: Uses adaptive smoothing based on hand movement speed
5. **Overlay Rendering**: Positions the flower at the middle finger knuckle with alpha blending

## Customization

You can customize various parameters in `flower.py`:

- **Flower Size**: Change `side = 300` (line ~119) to adjust flower dimensions
- **Detection Confidence**: Modify `min_hand_detection_confidence=0.8` (line ~25)
- **Smoothing Speed**: Adjust speed factor range `np.clip(frame_diff / 10, 0.15, 0.8)` (line ~111)
- **Calibration Range**: Tune `(avg_dist - 0.15) / 0.35` (line ~97) for your hand size

## Troubleshooting

**Camera not opening:**
- Check if another application is using the webcam
- Try changing `cv2.VideoCapture(0)` to `cv2.VideoCapture(1)` or higher

**Hand not detected:**
- Improve lighting conditions
- Lower `min_hand_detection_confidence` value
- Ensure your entire hand is visible in the frame

**Model download fails:**
- Manually download from: https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
- Place in project root as `hand_landmarker.task`

**No frames folder:**
- Run `python utility.py` first to extract frames from the GIF

## Credits

- **Flower GIF**: Shared by [isayevak](https://pinterest.com) on Pinterest
- **MediaPipe**: [Google MediaPipe Solutions](https://developers.google.com/mediapipe)
- **Hand Landmarker Model**: [MediaPipe Hand Landmark Detection](https://developers.google.com/mediapipe/solutions/vision/hand_landmarker)

## License

MIT License

Copyright (c) 2026 Gokce Gokben

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the issues page or submit a pull request.

## Future Enhancements

- Support for multiple hands with multiple flowers
- Different flower animations (roses, sunflowers, etc.)
- Background blur/replacement options
- Recording and saving gesture-controlled animations
- Mobile device support

---

**Enjoy making flowers bloom with your hands!**
