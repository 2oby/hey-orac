#include <portaudio.h>
#include <stdio.h>
#include <stdlib.h>

int main() {
    printf("🔍 Testing PortAudio Directly (C)\n");
    printf("================================\n");
    
    // Initialize PortAudio
    PaError err = Pa_Initialize();
    if (err != paNoError) {
        printf("❌ PortAudio initialization error: %s\n", Pa_GetErrorText(err));
        return 1;
    }
    
    printf("✅ PortAudio initialized successfully!\n");
    printf("📦 PortAudio version: %s\n", Pa_GetVersionText());
    
    // Get device count
    int numDevices = Pa_GetDeviceCount();
    printf("📊 Number of devices: %d\n", numDevices);
    
    if (numDevices > 0) {
        printf("\n📋 Available devices:\n");
        for (int i = 0; i < numDevices; i++) {
            const PaDeviceInfo *deviceInfo = Pa_GetDeviceInfo(i);
            if (deviceInfo != NULL) {
                printf("  Device %d: %s\n", i, deviceInfo->name);
                printf("    Max input channels: %d\n", deviceInfo->maxInputChannels);
                printf("    Max output channels: %d\n", deviceInfo->maxOutputChannels);
                printf("    Default sample rate: %f\n", deviceInfo->defaultSampleRate);
                printf("    Host API: %d\n", deviceInfo->hostApi);
                
                // Check if it's a USB device
                const char *name = deviceInfo->name;
                int is_usb = 0;
                if (strstr(name, "USB") || strstr(name, "usb") || 
                    strstr(name, "SH-04") || strstr(name, "sh-04")) {
                    is_usb = 1;
                }
                printf("    USB Device: %s\n", is_usb ? "✅ Yes" : "❌ No");
                printf("\n");
            } else {
                printf("  Device %d: Error getting device info\n", i);
            }
        }
        
        // Try to open a stream
        printf("🔧 Testing stream opening...\n");
        PaStream *stream;
        err = Pa_OpenDefaultStream(&stream, 
                                  1,    // input channels
                                  0,    // output channels
                                  paInt16, // sample format
                                  16000, // sample rate
                                  1024,  // frames per buffer
                                  NULL,  // stream callback
                                  NULL); // user data
        
        if (err != paNoError) {
            printf("❌ Failed to open default stream: %s\n", Pa_GetErrorText(err));
        } else {
            printf("✅ Successfully opened default stream!\n");
            
            // Start the stream
            err = Pa_StartStream(stream);
            if (err != paNoError) {
                printf("❌ Failed to start stream: %s\n", Pa_GetErrorText(err));
            } else {
                printf("✅ Stream started successfully!\n");
                
                // Stop and close the stream
                Pa_StopStream(stream);
                Pa_CloseStream(stream);
                printf("✅ Stream closed successfully!\n");
            }
        }
    } else {
        printf("❌ No devices detected by PortAudio!\n");
    }
    
    // List host APIs
    int numHostApis = Pa_GetHostApiCount();
    printf("\n🎵 Host APIs available: %d\n", numHostApis);
    
    for (int i = 0; i < numHostApis; i++) {
        const PaHostApiInfo *hostApiInfo = Pa_GetHostApiInfo(i);
        if (hostApiInfo != NULL) {
            printf("  API %d: %s\n", i, hostApiInfo->name);
            printf("    Device count: %d\n", hostApiInfo->deviceCount);
            printf("    Default input device: %d\n", hostApiInfo->defaultInputDevice);
            printf("    Default output device: %d\n", hostApiInfo->defaultOutputDevice);
            
            // Check if this is ALSA
            if (strstr(hostApiInfo->name, "ALSA") || strstr(hostApiInfo->name, "alsa")) {
                printf("    ✅ This is the ALSA API!\n");
            }
        }
    }
    
    // Terminate PortAudio
    Pa_Terminate();
    printf("\n✅ PortAudio terminated successfully!\n");
    
    return 0;
} 