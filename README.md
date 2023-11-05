# Long Audio Transcription

This repository contains a Python script, `longFormTerminal.py`, which is designed to transcribe audio files into text using the Whisper model by OpenAI. The script leverages the power of deep learning models to process and transcribe long-form audio.

## Installation

Before you can run the `longFormTerminal.py` script, you need to have Python installed on your system. Additionally, you need several dependencies, most notably the `transformers`, `torch`, and `librosa` libraries.

To install these dependencies, you can use the following pip command:

```bash
pip install torch librosa transformers
```

Please note that `torch` installation may vary depending on your system and whether you're utilizing CUDA for GPU acceleration. Refer to [PyTorch's official site](https://pytorch.org/) for detailed installation instructions tailored to your setup.

## Usage

The script is executed from the command line and requires an audio file path as its input.

```bash
python longFormTerminal.py <path_to_your_audio_file>
```

For example:

```bash
python longFormTerminal.py audio/speech.wav
```

### Command Line Arguments

- `audio_file_path` (required): The path to the audio file that you want to transcribe.

### Output

After the script has finished running, it will generate a file named `transcript.txt` in the same directory where the script is located. This file will contain the transcribed text of the input audio file.

### GPU Support

The script automatically detects if CUDA is available and, if so, will utilize the GPU to accelerate the transcription process.

### Note

- The script currently uses the `distil-whisper/distil-large-v2` model by default.
- The maximum number of new tokens for the model is set to 128 and the chunk length for processing audio is set to 15 seconds.
- If running on a CPU, the script will optimize memory usage accordingly.

## Contributing

Feel free to fork the project, submit pull requests, or send suggestions to improve the script.

## License

Specify your license or if the project is open-source, include a line such as:

"This project is open-source and available under the [MIT License](LICENSE.md)."

## Disclaimer

This script is provided as-is without any warranty. The author is not responsible for any misuse or damages caused by the use of this script.

Ensure to comply with the terms of service for the model and respect copyright and privacy laws when transcribing audio content.