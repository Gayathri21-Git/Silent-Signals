📌 Silent Signals – Early Intent & Posture Detection System

Silent Signals is a real-time computer vision project that detects a person’s posture (such as Sitting or Standing) and identifies early intent cues before a full action occurs.
The system uses body landmarks and motion analysis to infer human behavior without requiring speech or wearable devices.

🎯 Project Objective

The goal of Silent Signals is to:
Detect human posture in real time
Identify early intent (possible action starting) using subtle movements
Provide a foundation for non-verbal human–computer interaction
This project demonstrates how intent can be predicted before a complete action, which is useful in accessibility, surveillance, healthcare, and smart environments.

🧠 Key Features

📷 Real-time webcam-based detection
🧍 Posture classification:
Sitting
Standing
⚡ Intent detection states:
NO ACTION
POSSIBLE INTENT
ACTION STARTING (with confidence score)
🧠 ML-based intent classification using landmark movement
🚫 No wearable sensors required

🛠️ Tech Stack
Python
OpenCV – video capture & visualization
MediaPipe – pose landmark extraction
TensorFlow / Keras – intent classification model
NumPy – numerical processing

🧪 How It Works (High Level)

Webcam captures live video
MediaPipe extracts body pose landmarks
Posture is determined using joint positions
Landmark movement over time is analyzed
ML model predicts intent state:
No Action
Possible Intent
Action Starting
Output is displayed in real time

▶️ How to Run the Project
1️⃣ Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate

2️⃣ Install dependencies
pip install -r requirements.txt

3️⃣ Run the program
python silent_signals.py

4️⃣ Controls
Press q to quit the application

📊 Sample Output
POSTURE: SITTING
INTENT: NO ACTION

POSTURE: STANDING
INTENT: POSSIBLE INTENT

POSTURE: STANDING
INTENT: ACTION STARTING
Confidence: 0.94

🚀 Applications

Accessibility systems
Smart surveillance
Human–computer interaction
Assistive technology
Behavioral analysis

📌 Limitations

Intent prediction is probabilistic
Requires visible upper body
Performance depends on lighting and camera position

🔮 Future Enhancements

Gesture-specific intent detection (e.g., hand raise, waving)
Multi-person support
Improved early prediction accuracy
Deployment as a desktop or mobile app

👩‍💻 Author

Gayathri
Internship Project – Silent Signals
GitHub: https://github.com/Gayathri21-Git

📜 License

This project is licensed under the MIT License.
