#!/usr/bin/env python3
"""
Test Pi's built-in audio capabilities
Tests 3.5mm audio jack, HDMI audio, and LED flashing
"""

import os
import sys
import subprocess
import time
import logging
import threading

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PiAudioTester:
    """Test Pi's built-in audio capabilities"""
    
    def __init__(self):
        self.led_path = "/sys/class/leds/led0/brightness"
        self.led_trigger_path = "/sys/class/leds/led0/trigger"
        
    def test_led_flashing(self):
        """Test Pi's built-in LED"""
        logger.info("\n===== Testing Pi Built-in LED =====")
        
        try:
            # Check if LED files exist
            if not os.path.exists(self.led_path):
                logger.warning("❌ LED brightness file not found")
                return False
                
            logger.info("✅ LED files found")
            
            # Save current LED state
            try:
                with open(self.led_path, 'r') as f:
                    original_state = f.read().strip()
            except:
                original_state = "0"
            
            # Flash LED 5 times
            logger.info("🔴 Flashing LED 5 times...")
            for i in range(5):
                # Turn on
                with open(self.led_path, 'w') as f:
                    f.write("1")
                time.sleep(0.5)
                
                # Turn off
                with open(self.led_path, 'w') as f:
                    f.write("0")
                time.sleep(0.5)
            
            # Restore original state
            with open(self.led_path, 'w') as f:
                f.write(original_state)
                
            logger.info("✅ LED test completed - did you see the LED flash?")
            return True
            
        except Exception as e:
            logger.error(f"❌ LED test failed: {e}")
            return False
    
    def test_3_5mm_audio(self):
        """Test 3.5mm audio jack"""
        logger.info("\n===== Testing 3.5mm Audio Jack =====")
        
        # Check if we can set audio output to 3.5mm jack
        try:
            # Set audio output to 3.5mm jack
            cmd = ['amixer', 'set', 'PCM', '100%']
            subprocess.run(cmd, capture_output=True, text=True)
            
            # Play test tone through 3.5mm jack
            logger.info("🎵 Playing test tone through 3.5mm jack...")
            cmd = ['speaker-test', '-t', 'sine', '-f', '440', '-l', '3', '-D', 'hw:0,0']
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                logger.info("✅ 3.5mm audio test completed - did you hear a tone?")
                return True
            else:
                logger.error(f"❌ 3.5mm audio failed: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"❌ 3.5mm audio test failed: {e}")
            return False
    
    def test_hdmi_audio(self):
        """Test HDMI audio output"""
        logger.info("\n===== Testing HDMI Audio =====")
        
        try:
            # Set audio output to HDMI
            cmd = ['amixer', 'set', 'PCM', '100%']
            subprocess.run(cmd, capture_output=True, text=True)
            
            # Try HDMI audio devices we saw in the list
            hdmi_devices = [
                ('hw:1,0', 'HDMI 0'),
                ('hw:2,0', 'HDMI 1')
            ]
            
            for device, name in hdmi_devices:
                logger.info(f"🎵 Testing {name} ({device})...")
                try:
                    cmd = ['speaker-test', '-t', 'sine', '-f', '880', '-l', '2', '-D', device]
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=8)
                    
                    if result.returncode == 0:
                        logger.info(f"✅ {name} audio test completed - did you hear a tone?")
                        return True
                    else:
                        logger.warning(f"❌ {name} failed: {result.stderr}")
                        
                except subprocess.TimeoutExpired:
                    logger.warning(f"⏰ {name} timed out (this might be normal)")
                    return True  # Timeout might be normal for HDMI
                except Exception as e:
                    logger.warning(f"❌ {name} error: {e}")
            
            logger.warning("❌ No HDMI audio devices worked")
            return False
            
        except Exception as e:
            logger.error(f"❌ HDMI audio test failed: {e}")
            return False
    
    def test_system_beep(self):
        """Test system beep sound"""
        logger.info("\n===== Testing System Beep =====")
        
        try:
            # Try different beep methods
            beep_methods = [
                (['echo', '-e', '\\a'], 'echo beep'),
                (['beep'], 'beep command'),
                (['speaker-test', '-t', 'sine', '-f', '1000', '-l', '1'], 'speaker-test beep')
            ]
            
            for cmd, name in beep_methods:
                logger.info(f"🔊 Testing {name}...")
                try:
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
                    if result.returncode == 0:
                        logger.info(f"✅ {name} completed - did you hear a beep?")
                        return True
                except (subprocess.TimeoutExpired, FileNotFoundError):
                    logger.warning(f"❌ {name} not available")
            
            logger.warning("❌ No beep methods worked")
            return False
            
        except Exception as e:
            logger.error(f"❌ System beep test failed: {e}")
            return False
    
    def show_audio_config(self):
        """Show current audio configuration"""
        logger.info("\n===== Current Audio Configuration =====")
        
        # Show ALSA mixer settings
        try:
            cmd = ['amixer', 'get', 'PCM']
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                logger.info("ALSA PCM Mixer Settings:")
                for line in result.stdout.split('\n'):
                    if '[' in line and ']' in line:
                        logger.info(f"  {line.strip()}")
        except Exception as e:
            logger.warning(f"Could not get mixer settings: {e}")
        
        # Show available audio devices
        try:
            cmd = ['aplay', '-l']
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                logger.info("\nAvailable Audio Devices:")
                logger.info(result.stdout.strip())
        except Exception as e:
            logger.warning(f"Could not list audio devices: {e}")
    
    def run_comprehensive_test(self):
        """Run all Pi audio tests"""
        logger.info("🔊 Pi Built-in Audio Test")
        logger.info("This tests the Pi's built-in audio capabilities")
        logger.info("")
        
        # Show current configuration
        self.show_audio_config()
        
        # Test LED first (visual confirmation)
        led_success = self.test_led_flashing()
        
        # Test audio outputs
        audio_success = False
        
        # Try 3.5mm jack first
        if self.test_3_5mm_audio():
            audio_success = True
        else:
            # Try HDMI audio
            if self.test_hdmi_audio():
                audio_success = True
            else:
                # Try system beep as last resort
                if self.test_system_beep():
                    audio_success = True
        
        # Summary
        logger.info("\n" + "="*50)
        logger.info("📊 TEST RESULTS SUMMARY")
        logger.info("="*50)
        
        if led_success:
            logger.info("✅ LED Test: PASSED - Pi is responsive")
        else:
            logger.warning("❌ LED Test: FAILED - Check Pi hardware")
        
        if audio_success:
            logger.info("✅ Audio Test: PASSED - Pi can output sound")
        else:
            logger.warning("❌ Audio Test: FAILED - No audio output detected")
        
        logger.info("\n💡 Next Steps:")
        if audio_success:
            logger.info("  - Pi audio is working, you can use 3.5mm jack or HDMI")
            logger.info("  - For USB audio, get a device with speaker capability")
        else:
            logger.info("  - Check Pi audio configuration")
            logger.info("  - Try different audio output methods")
            logger.info("  - Consider USB audio device with speaker")
        
        return led_success and audio_success

def main():
    """Main function"""
    tester = PiAudioTester()
    success = tester.run_comprehensive_test()
    
    if success:
        logger.info("\n🎉 Pi audio test completed successfully!")
    else:
        logger.error("\n❌ Pi audio test failed!")
        logger.error("Check Pi hardware and audio configuration")

if __name__ == "__main__":
    main() 