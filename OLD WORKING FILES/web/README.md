# Hey Orac Web Interface

This directory contains the web interface for monitoring and configuring the Hey Orac wake word detection system.

## 📁 Directory Structure

```
web/
├── index.html              # Main interface
├── css/
│   └── style.css          # Stylesheet
├── js/
│   └── app.js            # JavaScript (future)
├── assets/
│   ├── images/           # Icons, logos, etc.
│   └── audio/           # Web audio files
│       └── beep.mp3     # Detection sounds
└── api/                  # API endpoints (future)
    └── config.json       # Configuration
```

## 🚀 How to View

### Option 1: Direct File
Open `index.html` directly in your browser:
```bash
open web/index.html
```

### Option 2: Local Server
Start a simple HTTP server:
```bash
cd web
python3 -m http.server 8000
```
Then visit: http://localhost:8000

### Option 3: Python Flask (Future)
When we add the backend API, you'll be able to run:
```bash
python3 -m flask run --port=8000
```

## 🎨 Features

- **Dark neon pixel theme** with scanlines effect
- **Volume meter** with LCD-style segments
- **Model configuration** with settings popup
- **Real-time monitoring** (demo mode)
- **Responsive design** for different screen sizes

## 🔧 Configuration

The interface currently uses demo data. When connected to the backend:
- Models will be discovered from the filesystem
- Real audio levels will be displayed
- Settings will be saved to the backend
- WebSocket connection for live updates

## 📱 Usage

1. **View current models** - See all available wake word models
2. **Activate models** - Click to activate/deactivate models
3. **Configure settings** - Click the cog icon to open settings
4. **Monitor volume** - Watch the volume meter for audio levels
5. **Adjust filters** - Use sliders to configure detection parameters

## 🎯 Next Steps

- [ ] Extract JavaScript to `js/app.js`
- [ ] Add backend API integration
- [ ] Add WebSocket for real-time updates
- [ ] Add model discovery from filesystem
- [ ] Add configuration persistence
- [ ] Add audio file upload for testing 