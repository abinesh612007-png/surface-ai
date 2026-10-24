# SurfaceAI — Deployment & Hosting Guide

This guide explains how to deploy **SurfaceAI** to cloud hosting platforms with 1 click or using Docker.

---

## Option 1: Deploy for Free on Render.com (Recommended)

1. Push your repository to GitHub (already completed).
2. Go to **[Render.com](https://render.com/)** and sign in with your GitHub account.
3. Click **"New +"** and choose **"Web Service"**.
4. Select your repository: `surface-ai`.
5. Configure the service settings:
   - **Environment:** `Python`
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --threads 4 --timeout 120`
   - **Instance Type:** `Free` (or Starter for higher RAM)
6. Click **"Create Web Service"**.
7. Render will build and deploy the app. Once finished, you will receive a public HTTPS URL (e.g., `https://surface-ai.onrender.com`).

> **Note on Free Tier:** Free tier web services sleep after 15 minutes of inactivity and take ~30 seconds to wake up on the first request.

---

## Option 2: Deploy on Railway.app

1. Go to **[Railway.app](https://railway.app/)** and log in with GitHub.
2. Click **"New Project"** → **"Deploy from GitHub repo"**.
3. Select `surface-ai`.
4. Railway will automatically detect the `Procfile` and deploy your application with zero extra configuration.
5. In **Settings** → **Networking**, click **"Generate Domain"** to get your public live URL.

---

## Option 3: Deploy on Hugging Face Spaces (Free CPU / GPU)

1. Go to **[huggingface.co/spaces](https://huggingface.co/spaces)** and create a new Space.
2. Name your space (e.g., `surface-ai-defect-detector`).
3. Select **Docker** (Blank) SDK.
4. Clone or push the GitHub repository to the Hugging Face Space repository.
5. Hugging Face will automatically build the `Dockerfile` and launch the app on a high-availability cloud container.

---

## Option 4: Deploy with Docker on Any Cloud / VPS (AWS, GCP, DigitalOcean)

Build and run anywhere Docker is installed:

```bash
# 1. Build the Docker image
docker build -t surfaceai .

# 2. Run the container on port 5000
docker run -d -p 5000:5000 --name surfaceai-app surfaceai
```

Access the application in your browser at `http://localhost:5000` or `http://<your-server-ip>:5000`.

---

## Option 5: Run Production Locally with Waitress (Windows)

For Windows hosting without Docker:

```powershell
pip install -r requirements.txt
python -c "from waitress import serve; import app; print('Serving on http://127.0.0.1:5000'); serve(app.app, host='0.0.0.0', port=5000)"
```
