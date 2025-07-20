"""WebSocket broadcaster for real-time updates."""

import logging
import time
import threading
from datetime import datetime

logger = logging.getLogger(__name__)


class WebSocketBroadcaster:
    """Broadcasts real-time updates via WebSocket."""
    
    def __init__(self, socketio, shared_data, event_queue):
        self.socketio = socketio
        self.shared_data = shared_data
        self.event_queue = event_queue
        self.running = False
        self.thread = None
        
    def start(self):
        """Start the broadcaster thread."""
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self._broadcast_loop)
            self.thread.daemon = True
            self.thread.start()
            logger.info("WebSocket broadcaster started")
    
    def stop(self):
        """Stop the broadcaster thread."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1)
        logger.info("WebSocket broadcaster stopped")
    
    def _broadcast_loop(self):
        """Main broadcast loop."""
        last_rms_time = 0
        rms_interval = 0.5  # 2 Hz (2 times per second)
        
        while self.running:
            try:
                current_time = time.time()
                
                # Broadcast RMS at 2 Hz
                if current_time - last_rms_time >= rms_interval:
                    self._broadcast_rms()
                    last_rms_time = current_time
                
                # Process event queue
                self._process_events()
                
                # Broadcast status updates
                self._broadcast_status()
                
                # Small sleep to prevent CPU spinning
                time.sleep(0.01)
                
            except Exception as e:
                logger.error(f"Error in broadcast loop: {e}")
                time.sleep(0.1)
    
    def _broadcast_rms(self):
        """Broadcast current RMS value."""
        try:
            rms = self.shared_data.get('rms', 0.0)
            # Log every 10th broadcast to reduce log spam
            if not hasattr(self, '_rms_count'):
                self._rms_count = 0
            self._rms_count += 1
            
            if self._rms_count % 10 == 0:
                logger.info(f"Broadcasting RMS #{self._rms_count}: {rms} to room 'realtime'")
            
            self.socketio.emit('rms_update', {
                'rms': rms,
                'timestamp': datetime.utcnow().isoformat() + 'Z'
            }, room='realtime')
        except Exception as e:
            logger.error(f"Error broadcasting RMS: {e}")
    
    def _process_events(self):
        """Process and broadcast events from the queue."""
        try:
            # Process up to 10 events per iteration
            for _ in range(10):
                if self.event_queue.empty():
                    break
                    
                event = self.event_queue.get_nowait()
                
                if event['type'] == 'detection':
                    # Broadcast detection event
                    self.socketio.emit('detection', {
                        'model': event['model'],
                        'confidence': event['confidence'],
                        'timestamp': event['timestamp']
                    }, room='realtime')
                    
                    # Update recent detections
                    recent = self.shared_data.get('recent_detections', [])
                    recent.append({
                        'model': event['model'],
                        'confidence': event['confidence'],
                        'timestamp': event['timestamp']
                    })
                    # Keep only last 10 detections
                    self.shared_data['recent_detections'] = recent[-10:]
                    
                elif event['type'] == 'config_changed':
                    # Broadcast config change
                    self.socketio.emit('config_changed', {
                        'timestamp': datetime.utcnow().isoformat() + 'Z'
                    }, room='realtime')
                    
        except Exception as e:
            if str(e) != "'Queue' object has no attribute 'empty'":
                logger.error(f"Error processing events: {e}")
    
    def _broadcast_status(self):
        """Broadcast system status updates."""
        try:
            # Check if status changed
            if self.shared_data.get('status_changed', False):
                status_data = {
                    'listening': self.shared_data.get('is_listening', False),
                    'active': self.shared_data.get('is_active', False),
                    'status': self.shared_data.get('status', 'Not connected'),
                    'timestamp': datetime.utcnow().isoformat() + 'Z'
                }
                
                self.socketio.emit('status_update', status_data, room='realtime')
                self.shared_data['status_changed'] = False
                
        except Exception as e:
            logger.error(f"Error broadcasting status: {e}")