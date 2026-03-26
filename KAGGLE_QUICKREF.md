# Kaggle GPU Bridge — Quick Reference

## TL;DR: Get Running in 5 Minutes

### 1. Prep (Do once)
- Get ngrok auth token: https://dashboard.ngrok.com/auth/your-authtoken
- Get Gemini API key: https://aistudio.google.com/app/apikeys
- Create `.env` file with:
  ```
  GEMINI_API_KEY=your-gemini-key
  NGROK_AUTH_TOKEN=your-ngrok-token
  ```

### 2. Upload to Kaggle
- Create Kaggle Dataset `vital-guardian` with your entire FYP folder
- (Or create a separate `env-secrets` dataset with just `.env`)

### 3. Create Kaggle Notebook
- **New Notebook** → Python
- **Settings** → GPU: On (T4 or P100)
- **Add Data** → Select `vital-guardian` dataset
- **Add Data** → Select `.env` dataset (if separate)

### 4. Copy & Paste Notebook
- Go to `/kaggle_gpu_bridge.ipynb` in your repo
- Copy all cells into Kaggle notebook
- Change dataset paths if needed (Cell 1)

### 5. Run It
- Click **Run All**
- Cell 4 will show your public URL: `https://abc-123-def.ngrok.app`
- Open that URL in your browser ← **Done!**

---

## What You'll See

### On your laptop browser:
- **Real-time video feed** from Kaggle GPU
- **Live FPS counter** (should be 30-40 FPS with GPU)
- **Incident log** with Gemini verification
- **"Cognitive Core Cross-Validating..."** → AI verdict appears in ~3-5s

### On Kaggle notebook terminal:
```
[Gemini] Starting verification for Alert 1...
[Gemini] Result for Alert 1: CONFIRMED
```

### Performance gains:
| Component       | CPU      | GPU (Kaggle) | Speedup |
|-----------------|----------|--------------|---------|
| YOLOv8          | ~20ms    | ~5ms         | 4x      |
| 5 Fall models   | ~150ms   | ~15ms        | 10x     |
| 10 Seizure      | ~300ms   | ~30ms        | 10x     |
| **Total**       | **5 FPS**| **30-40 FPS**| **6-8x**|

---

## Workflow Checklist

- [ ] Ngrok auth token obtained and stored
- [ ] Gemini API key obtained and stored
- [ ] `.env` file created locally
- [ ] FYP repo uploaded to Kaggle as `vital-guardian` dataset
- [ ] `.env` uploaded to Kaggle (same or separate dataset)
- [ ] Kaggle notebook created with GPU enabled
- [ ] All cells pasted and dataset paths verified
- [ ] Cell 1 ran without errors
- [ ] Cell 2 verified keys are set
- [ ] Cell 3 confirmed GPU available
- [ ] Cell 4 produced public URL
- [ ] Opened public URL in browser → Dashboard visible
- [ ] Clicked "Pause" or "Skip" → Controls work
- [ ] Watched an alert fire → Incident card appeared
- [ ] Waited 3-5s → Gemini verdict appeared

---

## Common Issues

### "Module not found: scripts.demo.demo_server"
**Solution:** Verify dataset structure. Check Cell 1 output shows all folders exist.

### "GEMINI_API_KEY not set"
**Solution:** In Cell 2, uncomment and paste your key:
```python
os.environ["GEMINI_API_KEY"] = "paste-your-key-here"
```

### "NGROK_AUTH_TOKEN not set"
**Solution:** Same as above, or add to `.env`.

### "Connection refused" or URL doesn't work
**Solution:** 
- Make sure Cell 4 completed successfully and shows a URL
- If older than 2 hours, re-run Cell 4 to get a fresh URL

### Server is slow / FPS is low
**Solution:**
- Check that GPU is enabled in Kaggle settings
- If T4 not available, try P100 (less common, but faster)
- Reduce frame resolution: edit `DISPLAY_W` / `DISPLAY_H` in config
- Reduce sending rate: only send every 2nd-3rd frame to browser

### Gemini verification not working
**Solution:**
- Check Kaggle notebook terminal output for errors
- Verify `GEMINI_API_KEY` is correct (you can test with `google-genai` directly)
- If API fails, falls back to mock response automatically

---

## For Long-term Use

**Free ngrok caveat:** Tunnels expire after 2 hours on free tier.
- Just re-run Cell 4 to get a new URL
- Or upgrade to ngrok Pro ($15/mo) for persistent URLs

**Kaggle notebook time limit:** 12 hours per session (free tier).
- After 12h, notebook stops automatically
- But you can restart and get a new ngrok URL

---

## Production Alternative

If you want a real 24/7 server:
- Deploy to **RunPod** (GPU rental), **Paperspace**, or **AWS** with ngrok/CloudFlare Tunnel
- Cost: ~$0.44/hr (T4) to $2.50/hr (A100)

---

## Support Files

- **Setup guide:** `KAGGLE_SETUP.md`
- **Script template:** `kaggle_gpu_bridge.py`
- **Notebook template:** `kaggle_gpu_bridge.ipynb`
- **This file:** `KAGGLE_QUICKREF.md`
