"""
Heartbeat sender for Hey ORAC wake word service.
Sends periodic heartbeats to ORAC STT with topic information.
"""

import os
import socket
import json
import logging
import threading
import time
import requests
from datetime import datetime, timedelta
from typing import Dict, Optional, List, Any
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class ModelHeartbeat:
    """Heartbeat data for a single wake word model."""
    name: str
    topic: str
    type: str = "wake_word"
    status: str = "active"
    wake_word: str = ""
    trigger_count: int = 0
    last_triggered: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = {
            "name": self.name,
            "topic": self.topic,
            "type": self.type,
            "status": self.status,
            "metadata": {
                "wake_word": self.wake_word,
                "trigger_phrase": f"hey {self.wake_word}",
                "sensitivity": 0.5
            }
        }
        
        # Add last_triggered if available
        if self.last_triggered:
            data["last_triggered"] = self.last_triggered.isoformat()
            data["trigger_count"] = self.trigger_count
        
        return data


class HeartbeatSender:
    """Manages heartbeat sending to ORAC STT service."""
    
    def __init__(self, orac_stt_url: Optional[str] = None):
        """Initialize heartbeat sender.
        
        Args:
            orac_stt_url: URL of ORAC STT service (default from env or localhost)
        """
        self.orac_stt_url = orac_stt_url or os.getenv("ORAC_STT_URL", "http://localhost:7272")
        self.heartbeat_endpoint = f"{self.orac_stt_url}/stt/v1/heartbeat"
        self.instance_id = f"hey-orac-{socket.gethostname()}"
        
        # Threading control
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        
        # Heartbeat interval and backoff
        self.interval = 30  # seconds
        self.backoff_base = 2  # exponential backoff base
        self.backoff_max = 300  # max 5 minutes
        self.current_backoff = 0
        
        # Model tracking
        self._models: Dict[str, ModelHeartbeat] = {}
        
        # Metrics
        self._last_heartbeat_time: Optional[datetime] = None
        self._heartbeat_failures = 0
        self._total_heartbeats_sent = 0
        
        # Activation metrics (updated from detection events)
        self._activations_1h: Dict[str, List[datetime]] = {}
        
        logger.info(f"Initialized heartbeat sender for {self.instance_id}")
        logger.info(f"ORAC STT endpoint: {self.heartbeat_endpoint}")
    
    def register_model(self, name: str, wake_word: str, enabled: bool = True) -> None:
        """Register a wake word model for heartbeat tracking.
        
        Args:
            name: Model name (e.g., "hey_jarvis")
            wake_word: Wake word (e.g., "jarvis")
            enabled: Whether model is enabled
        """
        with self._lock:
            # Extract topic from wake word (lowercase)
            topic = wake_word.lower().replace("hey ", "").replace("_", "")
            
            self._models[name] = ModelHeartbeat(
                name=name,
                topic=topic,
                wake_word=wake_word,
                status="active" if enabled else "inactive"
            )
            
            # Initialize activation tracking
            if topic not in self._activations_1h:
                self._activations_1h[topic] = []
            
            logger.info(f"Registered model '{name}' with topic '{topic}' (wake word: '{wake_word}')")
    
    def record_activation(self, model_name: str) -> None:
        """Record a wake word activation for metrics.
        
        Args:
            model_name: Name of the model that was activated
        """
        with self._lock:
            if model_name in self._models:
                model = self._models[model_name]
                model.trigger_count += 1
                model.last_triggered = datetime.utcnow()
                
                # Track for hourly metrics
                topic = model.topic
                if topic not in self._activations_1h:
                    self._activations_1h[topic] = []
                self._activations_1h[topic].append(datetime.utcnow())
                
                # Clean old activations (older than 1 hour)
                cutoff = datetime.utcnow() - timedelta(hours=1)
                self._activations_1h[topic] = [
                    t for t in self._activations_1h[topic] if t > cutoff
                ]
                
                logger.debug(f"Recorded activation for {model_name} (topic: {topic})")
    
    def _calculate_metrics(self) -> Dict[str, Any]:
        """Calculate current metrics for heartbeat.
        
        Returns:
            Metrics dictionary
        """
        with self._lock:
            # Count activations in last hour
            total_activations = sum(len(acts) for acts in self._activations_1h.values())
            
            # Find last activation time
            all_activations = []
            for acts in self._activations_1h.values():
                all_activations.extend(acts)
            
            last_activation = None
            if all_activations:
                last_activation = max(all_activations).isoformat() + "Z"
            
            return {
                "activations_1h": total_activations,
                "last_activation": last_activation
            }
    
    def _send_heartbeat(self) -> bool:
        """Send heartbeat to ORAC STT.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            with self._lock:
                # Build heartbeat message
                models_data = [model.to_dict() for model in self._models.values()]
                
                heartbeat = {
                    "instance_id": self.instance_id,
                    "source": "hey_orac",
                    "models": models_data,
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                    "metrics": self._calculate_metrics()
                }
            
            # Send heartbeat
            response = requests.post(
                self.heartbeat_endpoint,
                json=heartbeat,
                timeout=5
            )
            
            if response.status_code == 200:
                self._last_heartbeat_time = datetime.utcnow()
                self._total_heartbeats_sent += 1
                self._heartbeat_failures = 0
                self.current_backoff = 0
                
                result = response.json()
                logger.debug(f"Heartbeat sent successfully: {result.get('message', 'OK')}")
                return True
            else:
                logger.warning(f"Heartbeat failed with status {response.status_code}: {response.text}")
                self._heartbeat_failures += 1
                return False
                
        except requests.exceptions.ConnectionError as e:
            logger.debug(f"Connection error sending heartbeat: {e}")
            self._heartbeat_failures += 1
            return False
        except requests.exceptions.Timeout:
            logger.warning("Heartbeat request timed out")
            self._heartbeat_failures += 1
            return False
        except Exception as e:
            logger.error(f"Unexpected error sending heartbeat: {e}")
            self._heartbeat_failures += 1
            return False
    
    def _heartbeat_loop(self) -> None:
        """Main heartbeat loop running in separate thread."""
        logger.info("Heartbeat loop started")
        
        # Send initial heartbeat immediately
        self._send_heartbeat()
        
        while self._running:
            try:
                # Calculate sleep time with backoff
                if self.current_backoff > 0:
                    sleep_time = min(self.current_backoff, self.backoff_max)
                else:
                    sleep_time = self.interval
                
                # Sleep with periodic checks for shutdown
                for _ in range(int(sleep_time)):
                    if not self._running:
                        break
                    time.sleep(1)
                
                if not self._running:
                    break
                
                # Send heartbeat
                if not self._send_heartbeat():
                    # Apply exponential backoff on failure
                    if self.current_backoff == 0:
                        self.current_backoff = self.backoff_base
                    else:
                        self.current_backoff *= self.backoff_base
                    
                    if self.current_backoff > self.backoff_max:
                        self.current_backoff = self.backoff_max
                    
                    logger.info(f"Applying backoff: {self.current_backoff} seconds")
                    
            except Exception as e:
                logger.error(f"Error in heartbeat loop: {e}")
                time.sleep(5)
        
        logger.info("Heartbeat loop stopped")
    
    def start(self) -> None:
        """Start sending heartbeats."""
        with self._lock:
            if self._running:
                logger.warning("Heartbeat sender already running")
                return
            
            self._running = True
            self._thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
            self._thread.start()
            logger.info("Heartbeat sender started")
    
    def stop(self) -> None:
        """Stop sending heartbeats."""
        with self._lock:
            if not self._running:
                return
            
            self._running = False
        
        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None
        
        logger.info("Heartbeat sender stopped")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status of heartbeat sender.
        
        Returns:
            Status dictionary
        """
        with self._lock:
            return {
                "running": self._running,
                "instance_id": self.instance_id,
                "orac_stt_url": self.orac_stt_url,
                "models": len(self._models),
                "last_heartbeat": self._last_heartbeat_time.isoformat() if self._last_heartbeat_time else None,
                "total_heartbeats": self._total_heartbeats_sent,
                "failures": self._heartbeat_failures,
                "current_backoff": self.current_backoff
            }