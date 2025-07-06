#!/usr/bin/env python3
"""
Simple LED test script for SH-04 USB Microphone
Flashes the LED on and off for 10 seconds
"""

import usb.core
import usb.util
import time

def test_sh04_led():
    """Test LED control on SH-04 microphone."""
    print("🧪 Testing SH-04 LED Control")
    print("=" * 40)
    
    # SH-04 USB device identifiers
    VENDOR_ID = 0x5678
    PRODUCT_ID = 0x2348
    HID_INTERFACE = 2
    
    try:
        # Find the SH-04 device
        print("🔍 Looking for SH-04 device...")
        device = usb.core.find(idVendor=VENDOR_ID, idProduct=PRODUCT_ID)
        
        if device is None:
            print("❌ SH-04 device not found!")
            print("   Make sure the microphone is connected and recognized")
            return False
        
        print("✅ SH-04 device found!")
        
        # Set the active configuration
        device.set_configuration()
        print("✅ Device configuration set")
        
        # Claim the HID interface
        usb.util.claim_interface(device, HID_INTERFACE)
        print("✅ HID interface claimed")
        
        print("\n🎯 Starting LED flash test (10 seconds)...")
        print("   LED should flash between green (off) and red (on)")
        
        # Flash LED for 10 seconds
        start_time = time.time()
        flash_count = 0
        
        while time.time() - start_time < 10:
            # LED ON (red/muted)
            try:
                report_data = [0x05, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00]
                device.ctrl_transfer(0x21, 0x09, 0x0200, HID_INTERFACE, report_data)
                print(f"   {flash_count+1:2d}. LED ON  (red) - {time.time() - start_time:.1f}s")
            except Exception as e:
                print(f"   ❌ Error setting LED ON: {e}")
            
            time.sleep(0.5)
            
            # LED OFF (green/active)
            try:
                report_data = [0x05, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]
                device.ctrl_transfer(0x21, 0x09, 0x0200, HID_INTERFACE, report_data)
                print(f"   {flash_count+1:2d}. LED OFF (green) - {time.time() - start_time:.1f}s")
            except Exception as e:
                print(f"   ❌ Error setting LED OFF: {e}")
            
            time.sleep(0.5)
            flash_count += 1
        
        # Release the interface
        usb.util.release_interface(device, HID_INTERFACE)
        print("\n✅ LED test completed!")
        print(f"   Total flashes: {flash_count}")
        print("   LED should be back to green (normal state)")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during LED test: {e}")
        return False

if __name__ == "__main__":
    test_sh04_led() 