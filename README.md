# 🧠 NeuroSync — Real-Time Focus & Emotion Analyzer

## 🚀 How to Run

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/yourusername/NeuroSync.git
cd NeuroSync
```

### 2️⃣ Create and Activate Virtual Environment

* **Windows**

  ```bash
  python -m venv nsync_env
  nsync_env\Scripts\activate
  ```
* **Mac/Linux**

  ```bash
  python3 -m venv nsync_env
  source nsync_env/bin/activate
  ```

### 3️⃣ Install Required Packages

```bash
pip install -r requirements.txt
```

If you don’t have a `requirements.txt`, install manually:

```bash
pip install fastapi uvicorn opencv-python mediapipe numpy
```

### 4️⃣ Run the Server

```bash
uvicorn main:app --reload
```

You should see:

```
INFO:     Uvicorn running on http://127.0.0.1:8000
```

### 5️⃣ Test the API

* Open your browser and go to:

  * **Docs:** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
  * **Focus Live Feed:** [http://127.0.0.1:8000/focus/live](http://127.0.0.1:8000/focus/live)

### 6️⃣ Using the Live Feed

* Ensure your webcam is connected.
* The live feed will display your camera stream.
* A **Focus Score** and corresponding emoji (🧠🙂😐😴) appear on screen.
* Press **`q`** or stop the server to exit.

### ✅ Example Workflow

```bash
# Step 1: Activate env
nsync_env\Scripts\activate

# Step 2: Run app
uvicorn main:app --reload

# Step 3: Visit in browser
http://127.0.0.1:8000/focus/live
```

### 🧩 Troubleshooting

| Problem            | Solution                                                     |
| ------------------ | ------------------------------------------------------------ |
| Camera not opening | Check if another app is using it or restart your system.     |
| Low focus score    | Adjust lighting or camera angle.                             |
| Port in use        | Run on another port: `uvicorn main:app --port 8001 --reload` |

### 🧠 Project Structure

```
NeuroSync/
├── backend/
│   ├── routes/
│   │   ├── emotion.py
│   │   └── focus.py
│   ├── models/
│   └── utils/
├── main.py
├── requirements.txt
└── README.md
```

---

💡 **Tip:** You can extend the API by adding more endpoints under `backend/routes/` for tracking, logging, and analytics!
