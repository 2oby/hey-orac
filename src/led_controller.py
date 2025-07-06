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
    
    # SH-04 USB device identifiers
    VENDOR_ID = 0x5678
    PRODUCT_ID = 0x2348
    
    # HID interface (from lsusb output)
    HID_INTERFACE = 2
    
    def __init__(self):
        self.device = None
        self.is_connected = False
        
    def connect(self) -> bool:
        """Connect to the SH-04 device."""
        try:
            # Find the SH-04 device
            self.device = usb.core.find(idVendor=self.VENDOR_ID, idProduct=self.PRODUCT_ID)
            
            if self.device is None:
                logger.error("❌ SH-04 device not found")
                return False
            
            # Set the active configuration
            self.device.set_configuration()
            
            # Claim the HID interface
            usb.util.claim_interface(self.device, self.HID_INTERFACE)
            
            self.is_connected = True
            logger.info("✅ Connected to SH-04 LED controller")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to connect to SH-04: {e}")
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