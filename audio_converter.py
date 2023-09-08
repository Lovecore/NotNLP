import argparse
import os
from pydub import AudioSegment

def convert_to_mono(input_file):
    try:
        output_file = os.path.splitext(input_file)[0] + ".wav"
        print(f"Converting {input_file} to mono .wav format...")
        sound = AudioSegment.from_file(input_file)

        # Explicitly convert to mono, google needs this ha.
        sound = sound.set_channels(1)

        sound.export(output_file, format="wav")
        print(f"Mono conversion complete. Saved as {output_file}")
        return True
    except Exception as e:
        print(f"An error occurred during mono conversion: {e}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert an audio file to mono .wav format")
    parser.add_argument("input_file", help="Path to the input audio file")
    args = parser.parse_args()

    convert_to_mono(args.input_file)
