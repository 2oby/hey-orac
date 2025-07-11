# TMPFS Storage Setup for SD Card Protection

## Overview

This document explains how we've configured Docker containers to use RAM-based temporary storage (tmpfs) to protect the Raspberry Pi's SD card from excessive writes while maintaining proper permanent storage for important data.

## Why TMPFS?

### **SD Card Wear Problem**
- SD cards have limited write cycles (typically 10,000-100,000 cycles)
- Frequent writes (like RMS data every 80ms) can quickly wear out the SD card
- Failed SD cards can corrupt the entire system

### **TMPFS Benefits**
- **RAM-based**: No writes to SD card
- **Fast**: RAM speed access
- **Volatile**: Data lost on reboot (appropriate for temporary data)
- **Configurable**: Can set size limits

## Storage Architecture

### **Temporary Data (TMPFS - RAM)**
```
/tmp/rms_data/          # RMS audio levels (10MB)
/tmp/cache/             # Application cache (50MB)
/tmp/sessions/          # Web sessions (20MB)
/tmp/uploads/           # File uploads (30MB)
```

### **Permanent Data (SD Card)**
```
./logs/                 # Application logs
./src/config.yaml       # Configuration files
./models/               # ML models
./homeassistant/        # Home Assistant config (future)
```

## Docker Compose Configuration

### **TMPFS Volumes**
```yaml
volumes:
  # TMPFS volumes (RAM-based, protects SD card)
  rms-data:
    driver: tmpfs
    driver_opts:
      size: 10M
      mode: 0755
  
  temp-cache:
    driver: tmpfs
    driver_opts:
      size: 50M
      mode: 0755
  
  temp-sessions:
    driver: tmpfs
    driver_opts:
      size: 20M
      mode: 0755
  
  temp-uploads:
    driver: tmpfs
    driver_opts:
      size: 30M
      mode: 0755
```

### **Volume Mounts**
```yaml
volumes:
  # TMPFS volumes for temporary data
  - rms-data:/tmp/rms_data:rw
  - temp-cache:/tmp/cache:rw
  - temp-sessions:/tmp/sessions:rw
  - temp-uploads:/tmp/uploads:rw
  
  # Permanent storage (SD card)
  - ./logs:/app/logs
  - ./src/config.yaml:/app/config.yaml:ro
  - ./models:/app/models:ro
```

## When Files Are Written Back

### **TMPFS Volumes**
- **Never written back** - data is lost on container restart/reboot
- **Use cases**: Temporary data, caches, inter-process communication
- **Examples**: RMS data, web sessions, upload caches

### **Permanent Storage**
- **Immediate writes** - data persists across restarts
- **Use cases**: Configuration, logs, databases, user data
- **Examples**: Application logs, config files, ML models

### **Write Patterns**

#### **High-Frequency Writes (→ TMPFS)**
- RMS audio data: Every 80ms
- Web session data: Every request
- Cache files: Every operation
- Temporary uploads: During processing

#### **Low-Frequency Writes (→ SD Card)**
- Application logs: Every few seconds/minutes
- Configuration changes: On user action
- Database updates: Periodically
- Model files: Rarely

## Home Assistant Integration

### **Future Setup (Commented Out)**
```yaml
homeassistant:
  volumes:
    # Permanent storage for Home Assistant configuration
    - ./homeassistant:/config
    
    # TMPFS volumes for temporary data
    - ha-temp:/tmp:rw
    - ha-cache:/config/.storage:rw
    - ha-sessions:/config/.storage/sessions:rw
    - ha-uploads:/config/www/upload:rw
```

### **Home Assistant TMPFS Volumes**
```yaml
volumes:
  ha-temp:
    driver: tmpfs
    driver_opts:
      size: 100M
      mode: 0755
  
  ha-cache:
    driver: tmpfs
    driver_opts:
      size: 200M
      mode: 0755
  
  ha-sessions:
    driver: tmpfs
    driver_opts:
      size: 50M
      mode: 0755
  
  ha-uploads:
    driver: tmpfs
    driver_opts:
      size: 100M
      mode: 0755
```

## Monitoring and Maintenance

### **Check TMPFS Usage**
```bash
# Check tmpfs volumes
docker exec hey-orac df -h /tmp/rms_data
docker exec hey-orac df -h /tmp/cache

# Check memory usage
docker stats hey-orac
```

### **Check Permanent Storage**
```bash
# Check log directory
ls -la ./logs/

# Check configuration
ls -la ./src/config.yaml
```

### **Memory Usage**
- **Total TMPFS**: ~110MB (hey-orac) + 450MB (future Home Assistant)
- **Available RAM**: 8GB on Pi 5
- **Impact**: Minimal (< 10% of total RAM)

## Troubleshooting

### **TMPFS Full**
```bash
# Check tmpfs usage
docker exec hey-orac df -h /tmp/rms_data

# Increase size in docker-compose.yml
volumes:
  rms-data:
    driver: tmpfs
    driver_opts:
      size: 20M  # Increase from 10M
```

### **Data Loss on Restart**
- **Expected behavior** for tmpfs volumes
- **Permanent data** is preserved in bind mounts
- **Configuration** should be in permanent storage

### **Performance Issues**
```bash
# Check if tmpfs is being used
docker exec hey-orac mount | grep tmpfs

# Verify RMS data location
docker exec hey-orac ls -la /tmp/rms_data/
```

## Best Practices

### **What Goes in TMPFS**
- ✅ Temporary data (caches, sessions)
- ✅ High-frequency writes (RMS data)
- ✅ Inter-process communication
- ✅ Upload buffers

### **What Goes in Permanent Storage**
- ✅ Configuration files
- ✅ Application logs
- ✅ ML models
- ✅ User data
- ✅ Database files

### **Size Guidelines**
- **Small volumes** (10-50MB): RMS data, sessions
- **Medium volumes** (50-200MB): Caches, uploads
- **Large volumes** (200MB+): Only if needed

## Migration from SD Card to TMPFS

### **Before (SD Card Writes)**
```
/tmp/rms_monitor_data.json  # 1000+ writes/hour to SD card
```

### **After (RAM Only)**
```
/tmp/rms_data/rms_monitor_data.json  # 0 writes to SD card
```

### **Impact**
- **SD card writes**: Reduced by ~99%
- **Performance**: Improved (RAM speed)
- **Reliability**: Increased (no SD card wear)
- **Data persistence**: Lost on restart (appropriate for temporary data) 