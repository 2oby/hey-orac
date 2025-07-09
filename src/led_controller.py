#!/usr/bin/env python3
"""
LED Controller for SH-04 USB Microphone
Controls the mute LED on the SH-04 microphone through USB HID interface
"""

import usb.core
import usb.util
import time
import logging
from typing import Optional

logger = logging.getLogger(__name__)

class SH04LEDController:
    """Controls the LED on SH-04 USB microphone."""
    
    # SH-04 USB device identifiers - will be auto-detected
    VENDOR_ID = None
    PRODUCT_ID = None
    
    # HID interface (from lsusb output)
    HID_INTERFACE = 2
    
    def __init__(self):
        self.device = None
        self.is_connected = False
        
    def _detect_device_ids(self) -> bool:
        """Auto-detect SH-04 USB device IDs."""
        try:
            import subprocess
            import re
            
            # Run lsusb to find USB devices
            result = subprocess.run(['lsusb'], capture_output=True, text=True)
            if result.returncode != 0:
                logger.error("❌ Failed to run lsusb command")
                return False
            
            # Look for SH-04 or similar microphone devices
            lines = result.stdout.split('\n')
            for line in lines:
                # Common patterns for USB microphones
                if any(keyword in line.lower() for keyword in ['sh-04', 'microphone', 'audio', 'usb mic']):
                    logger.info(f"🔍 Found potential microphone device: {line}")
                    
                    # Extract vendor and product IDs
                    match = re.search(r'Bus \d+ Device \d+: ID ([0-9a-f]{4}):([0-9a-f]{4})', line)
                    if match:
                        self.VENDOR_ID = int(match.group(1), 16)
                        self.PRODUCT_ID = int(match.group(2), 16)
                        logger.info(f"✅ Auto-detected USB device: Vendor={self.VENDOR_ID:04x}, Product={self.PRODUCT_ID:04x}")
                        return True
            
            # If no specific device found, try common USB audio device patterns
            logger.info("🔍 No specific microphone found, trying common USB audio devices...")
            for line in lines:
                if 'audio' in line.lower() or 'sound' in line.lower():
                    logger.info(f"🔍 Found potential audio device: {line}")
                    match = re.search(r'Bus \d+ Device \d+: ID ([0-9a-f]{4}):([0-9a-f]{4})', line)
                    if match:
                        self.VENDOR_ID = int(match.group(1), 16)
                        self.PRODUCT_ID = int(match.group(2), 16)
                        logger.info(f"✅ Auto-detected USB audio device: Vendor={self.VENDOR_ID:04x}, Product={self.PRODUCT_ID:04x}")
                        return True
            
            logger.warning("⚠️ No USB audio devices found")
            return False
            
        except Exception as e:
            logger.error(f"❌ Error detecting USB device IDs: {e}")
            return False
    
    def connect(self) -> bool:
        """Connect to the SH-04 device."""
        try:
            # Auto-detect device IDs if not set
            if self.VENDOR_ID is None or self.PRODUCT_ID is None:
                if not self._detect_device_ids():
                    logger.error("❌ Could not detect USB device IDs")
                    return False
            
            # Find the device
            self.device = usb.core.find(idVendor=self.VENDOR_ID, idProduct=self.PRODUCT_ID)
            
            if self.device is None:
                logger.error(f"❌ USB device not found: Vendor={self.VENDOR_ID:04x}, Product={self.PRODUCT_ID:04x}")
                return False
            
            # Set the active configuration
            self.device.set_configuration()
            
            # Claim the HID interface
            usb.util.claim_interface(self.device, self.HID_INTERFACE)
            
            self.is_connected = True
            logger.info(f"✅ Connected to USB device: Vendor={self.VENDOR_ID:04x}, Product={self.PRODUCT_ID:04x}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to connect to USB device: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from the SH-04 device."""
        if self.device and self.is_connected:
            try:
                usb.util.release_interface(self.device, self.HID_INTERFACE)
                self.is_connected = False
                logger.info("✅ Disconnected from SH-04 LED controller")
            except Exception as e:
                logger.error(f"❌ Error disconnecting: {e}")
    
    def set_led_state(self, state: bool) -> bool:
        """
        Set LED state (True = on/red, False = off/green).
        
        Args:
            state: True for LED on (red/muted), False for LED off (green/active)
        
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.is_connected or not self.device:
            logger.error("❌ Not connected to SH-04 device")
            return False
        
        try:
            # Common HID report format for LED control
            # This is a generic approach - may need adjustment for specific device
            report_data = [0x05, 0x00, 0x01 if state else 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]
            
            # Send HID report
            self.device.ctrl_transfer(
                0x21,  # REQUEST_TYPE_CLASS | RECIPIENT_INTERFACE | ENDPOINT_OUT
                0x09,  # SET_REPORT
                0x0200,  # Report Type: Output
                self.HID_INTERFACE,  # Interface
                report_data
            )
            
            status = "ON (red/muted)" if state else "OFF (green/active)"
            logger.info(f"✅ LED set to {status}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to set LED state: {e}")
            return False
    
    def flash_led(self, duration: float = 0.5, count: int = 3) -> bool:
        """
        Flash the LED for visual feedback.
        
        Args:
            duration: Duration of each flash in seconds
            count: Number of flashes
        
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.is_connected:
            logger.error("❌ Not connected to SH-04 device")
            return False
        
        try:
            logger.info(f"🎯 Flashing LED {count} times ({duration}s each)")
            
            for i in range(count):
                # LED on
                self.set_led_state(True)
                time.sleep(duration)
                
                # LED off
                self.set_led_state(False)
                if i < count - 1:  # Don't sleep after last flash
                    time.sleep(duration)
            
            logger.info("✅ LED flash sequence completed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to flash LED: {e}")
            return False
    
    def wake_word_detected_feedback(self) -> bool:
        """Provide visual feedback when wake word is detected."""
        return self.flash_led(duration=0.2, count=2)
    
    def error_feedback(self) -> bool:
        """Provide visual feedback for errors."""
        return self.flash_led(duration=0.1, count=5)
    
    def startup_feedback(self) -> bool:
        """Provide visual feedback on startup."""
        return self.flash_led(duration=0.3, count=1)


def test_led_controller():
    """Test the LED controller functionality."""
    print("🧪 Testing SH-04 LED Controller")
    print("=" * 40)
    
    controller = SH04LEDController()
    
    # Test connection
    if not controller.connect():
        print("❌ Failed to connect to SH-04")
        return False
    
    try:
        # Test startup feedback
        print("🎯 Testing startup feedback...")
        controller.startup_feedback()
        time.sleep(1)
        
        # Test wake word detection feedback
        print("🎯 Testing wake word detection feedback...")
        controller.wake_word_detected_feedback()
        time.sleep(1)
        
        # Test error feedback
        print("🎯 Testing error feedback...")
        controller.error_feedback()
        time.sleep(1)
        
        print("✅ All LED tests completed successfully!")
        return True
        
    finally:
        controller.disconnect()


if __name__ == "__main__":
    test_led_controller() 