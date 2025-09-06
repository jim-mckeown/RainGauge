#include <WiFi.h>
#include <WiFiUdp.h>
#include <driver/i2s.h>

// WiFi configuration
const char* ssid = "SSID";
const char* password = "Password";
const char* pc_ip = "192.168.1.00"; // Replace with your PC's IP
const int udp_port = 12345;

WiFiUDP udp;

// I2S configuration - 32kHz sample rate
#define I2S_SAMPLE_RATE 32000
#define I2S_READ_LEN 512
#define I2S_CHANNEL_NUM 1

// Gain configuration - adjust this value!
int gain_multiplier = 15; // Start with 15x amplification, adjust as needed

// Statistics
unsigned long packet_count = 0;
unsigned long last_stats_time = 0;

void setup() {
  Serial.begin(115200);
  
  // Connect to WiFi
  Serial.println("Connecting to WiFi...");
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\nConnected to WiFi");
  Serial.print("IP Address: ");
  Serial.println(WiFi.localIP());

  // Initialize I2S
  i2s_config_t i2s_config = {
    .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
    .sample_rate = I2S_SAMPLE_RATE,
    .bits_per_sample = I2S_BITS_PER_SAMPLE_16BIT,
    .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,
    .communication_format = I2S_COMM_FORMAT_I2S,
    .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
    .dma_buf_count = 8,
    .dma_buf_len = 1024,
    .use_apll = false,
    .tx_desc_auto_clear = false,
    .fixed_mclk = 0
  };
  
  i2s_pin_config_t pin_config = {
    .bck_io_num = 33,    // BCKL
    .ws_io_num = 25,     // LRCL
    .data_out_num = I2S_PIN_NO_CHANGE,
    .data_in_num = 32   // DIN
  };
  
  i2s_driver_install(I2S_NUM_0, &i2s_config, 0, NULL);
  i2s_set_pin(I2S_NUM_0, &pin_config);
  
  // Start UDP
  udp.begin(udp_port);
  
  Serial.println("I2S and UDP initialized");
  Serial.printf("Sample rate: %d Hz\n", I2S_SAMPLE_RATE);
  Serial.printf("Gain multiplier: %dx\n", gain_multiplier);
  Serial.printf("Target: %s:%d\n", pc_ip, udp_port);
  Serial.println("Ready to stream audio");
}

void applyGain(int16_t* buffer, size_t sample_count) {
  for (int i = 0; i < sample_count; i++) {
    int32_t amplified_sample = (int32_t)buffer[i] * gain_multiplier;
    
    // Clamp to prevent overflow (16-bit range)
    if (amplified_sample > 32767) {
      amplified_sample = 32767;
    } else if (amplified_sample < -32768) {
      amplified_sample = -32768;
    }
    
    buffer[i] = (int16_t)amplified_sample;
  }
}

void printStatistics(int16_t* buffer, size_t sample_count) {
  // Calculate statistics
  int16_t max_val = 0;
  int16_t min_val = 0;
  long sum = 0;
  
  for (int i = 0; i < sample_count; i++) {
    if (buffer[i] > max_val) max_val = buffer[i];
    if (buffer[i] < min_val) min_val = buffer[i];
    sum += abs(buffer[i]);
  }
  
  int avg_amplitude = sum / sample_count;
  
  Serial.printf("Samples: %d, Max: %d, Min: %d, Avg: %d, Gain: %dx\n",
                sample_count, max_val, min_val, avg_amplitude, gain_multiplier);
}

void loop() {
  int16_t buffer[I2S_READ_LEN];
  size_t bytes_read;
  
  // Read from I2S
  i2s_read(I2S_NUM_0, (void*)buffer, sizeof(buffer), &bytes_read, portMAX_DELAY);
  
  if (bytes_read > 0) {
    size_t sample_count = bytes_read / sizeof(int16_t);
    
    // Apply gain amplification
    applyGain(buffer, sample_count);
    
    // Send via UDP
    udp.beginPacket(pc_ip, udp_port);
    udp.write((uint8_t*)buffer, bytes_read);
    udp.endPacket();
    
    packet_count++;
    
    // Print statistics every second
    unsigned long current_time = millis();
    if (current_time - last_stats_time >= 1000) {
      printStatistics(buffer, sample_count);
      Serial.printf("Packets sent: %d, Rate: %.1f pkt/s\n", 
                    packet_count, packet_count / ((current_time - last_stats_time) / 1000.0));
      packet_count = 0;
      last_stats_time = current_time;
    }
  }
  
  // Small delay to prevent watchdog
  delay(1);
}

// Optional: Add serial commands to adjust gain on the fly
void serialEvent() {
  while (Serial.available()) {
    String command = Serial.readStringUntil('\n');
    command.trim();
    
    if (command.startsWith("gain ")) {
      int new_gain = command.substring(5).toInt();
      if (new_gain >= 1 && new_gain <= 50) {
        gain_multiplier = new_gain;
        Serial.printf("Gain set to: %dx\n", gain_multiplier);
      } else {
        Serial.println("Gain must be between 1 and 50");
      }
    }
    else if (command.equals("stats")) {
      Serial.printf("Current gain: %dx\n", gain_multiplier);
      Serial.printf("Sample rate: %d Hz\n", I2S_SAMPLE_RATE);
    }
    else if (command.equals("help")) {
      Serial.println("Available commands:");
      Serial.println("gain [1-50] - Set amplification level");
      Serial.println("stats - Show current settings");
      Serial.println("help - Show this help");
    }
  }
}