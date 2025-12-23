import pyaudio
import numpy as np
import time
import queue
import logging
from auditory_watchdog.config import SAMPLE_RATE, CHUNK_SIZE, NUM_CHANNELS, AUDIO_STRIDE_SAMPLES

logger = logging.getLogger(__name__)

class AudioStream:
    """
    Manages the continuous rolling audio buffer from the laptop microphone.
    """
    def __init__(self):
        self.p = pyaudio.PyAudio()
        self.stream = None
        # Use maxsize=1 so we always fetch the Freshest chunk, dropping older ones if processing is too slow.
        self.audio_queue = queue.Queue(maxsize=1) 
        # Maintain a 3-second continuous rolling buffer 
        self.rolling_buffer = np.zeros(CHUNK_SIZE, dtype=np.float32)
        self.is_running = False

    def _callback(self, in_data, frame_count, time_info, status):
        """
        Callback to non-blockingly read audio strides into the rolling buffer.
        """
        if self.is_running:
            # Convert binary data to numpy float32 array (-1.0 to 1.0)
            new_data = np.frombuffer(in_data, dtype=np.int16).astype(np.float32) / 32768.0
            
            # Shift the rolling buffer to the left by len(new_data)
            self.rolling_buffer = np.roll(self.rolling_buffer, -len(new_data))
            # Insert the newly captured samples at the end
            self.rolling_buffer[-len(new_data):] = new_data
            
            try:
                # Put a COPY of the entire 3-second rolling buffer into the queue AND the new 1-second stride
                if self.audio_queue.full():
                    self.audio_queue.get_nowait()
                self.audio_queue.put_nowait((self.rolling_buffer.copy(), new_data.copy()))
            except queue.Full:
                pass
        return (in_data, pyaudio.paContinue)

    def start_stream(self):
        """
        Starts the continuous microphone recording.
        """
        self.is_running = True
        try:
            self.stream = self.p.open(
                format=pyaudio.paInt16,
                channels=NUM_CHANNELS,
                rate=SAMPLE_RATE,
                input=True,
                frames_per_buffer=AUDIO_STRIDE_SAMPLES,
                stream_callback=self._callback
            )
            self.stream.start_stream()
            logger.info("Audio stream started.")
        except Exception as e:
            logger.error(f"Failed to start audio stream: {e}")
            self.is_running = False

    def stop_stream(self):
        """
        Stops the microphone recording.
        """
        self.is_running = False
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            logger.info("Audio stream stopped.")

    def get_latest_chunk(self, timeout=None):
        """
        Retrieves the latest 3-second audio chunk from the queue.
        """
        try:
            return self.audio_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def terminate(self):
        self.stop_stream()
        self.p.terminate()

if __name__ == "__main__":
    # Simple test to ensure microphone capture works
    logging.basicConfig(level=logging.INFO)
    print("Testing audio stream... Speak into the microphone for 10 seconds.")
    audio = AudioStream()
    audio.start_stream()
    
    start_time = time.time()
    chunks_received = 0
    while time.time() - start_time < 10:
        chunk = audio.get_latest_chunk(timeout=1.0)
        if chunk is not None:
            chunks_received += 1
            vol = np.abs(chunk).mean()
            print(f"Captured chunk {chunks_received} of shape {chunk.shape}, average volume: {vol:.4f}")
            
    audio.terminate()
    print("Test finished.")
