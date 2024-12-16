import tensorflow as tf
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer
import soundfile as sf

# Load the model and tokenizer
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h", from_pt=True)  # Load model from PyTorch checkpoint
tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h")

# Load audio file
audio_input = "/response.mp3"

# Read audio file
speech_array, sampling_rate = sf.read(audio_input)

# Convert speech_array to the appropriate format
input_values = tokenizer(speech_array, return_tensors='tf', sampling_rate=sampling_rate).input_values

# Perform inference
logits = model(input_values).logits

# Decode the predicted text
predicted_ids = tf.argmax(logits, axis=-1)
transcription = tokenizer.batch_decode(predicted_ids.numpy())

print(transcription)
