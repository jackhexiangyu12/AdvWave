import json
import base64
import io
from pydub import AudioSegment
from IPython.display import Audio, display


def audioEncode(json_file_path):
    # Initialize an empty AudioSegment for concatenation
    combined_audio = AudioSegment.silent(duration=0)

    # Initialize an empty string for concatenating text
    combined_text = ""

    # Open and read the JSON file
    with open(json_file_path, 'r') as file:
        data = json.load(file)

        for entry in data:
            if entry['event'] == 'audio_buffer_add':
                # Decode the base64 audio data
                audio_bytes = base64.b64decode(entry['data'])

                # Convert the bytes to an audio segment
                audio = AudioSegment.from_raw(io.BytesIO(audio_bytes), sample_width=2, frame_rate=24000, channels=1)

                # Concatenate the current audio buffer with the combined audio
                combined_audio += audio

            elif entry['event'] == 'text_buffer_add':
                # Concatenate the text data
                combined_text += entry['data']


    return combined_text, combined_audio