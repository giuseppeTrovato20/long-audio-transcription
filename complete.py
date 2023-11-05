import streamlit as st
import torch
import librosa
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import base64
from io import BytesIO
from langchain import OpenAI
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain



# Add a function to get or set the API key in the session state
def get_or_set_api_key():
    if 'openai_api_key' not in st.session_state:
        # Prompt the user to enter their OpenAI API key
        st.session_state['openai_api_key'] = st.text_input("Enter your OpenAI API key")
    return st.session_state['openai_api_key']


def generate_response(txt):
    # Instantiate the LLM model
    llm = OpenAI(temperature=0, openai_api_key=openai_api_key)
    # Split text
    text_splitter = CharacterTextSplitter()
    texts = text_splitter.split_text(txt)
    # Create multiple documents
    docs = [Document(page_content=t) for t in texts]
    # Text summarization
    chain = load_summarize_chain(llm, chain_type='map_reduce')
    return chain.run(docs)

# Function to download a string as a text file
def get_text_download_link(text, filename, text_to_display):
    b64 = base64.b64encode(text.encode()).decode()
    href = f'<a href="data:file/txt;base64,{b64}" download="{filename}">{text_to_display}</a>'
    return href

# Function to transcribe audio
def transcribe_audio(audio_file):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model_id = "distil-whisper/distil-large-v2"

    # Load the model and processor
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    )
    model.to(device)
    processor = AutoProcessor.from_pretrained(model_id)

    # Create the pipeline
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        max_new_tokens=128,
        chunk_length_s=15,
        batch_size=16,
        torch_dtype=torch_dtype,
        device=device,
    )

    # Load the audio and transcribe
    audio_array, sampling_rate = librosa.load(audio_file, sr=None)
    result = pipe({"raw": audio_array, "sampling_rate": sampling_rate})
    transcript = result["text"]
    return transcript

# Streamlit UI
st.set_page_config(page_title='Audio Transcription and Summarization')
st.title('Audio Transcription and Summarization')
# Get or set the API key
openai_api_key = get_or_set_api_key()

# Only proceed if the API key has been entered
if openai_api_key:

    # File uploader
    uploaded_file = st.file_uploader("Choose an audio file", type=['wav', 'mp3'])

    # Transcription and summarization
    if uploaded_file is not None:
        with st.spinner('Transcribing...'):
            transcript = transcribe_audio(uploaded_file)
        st.success('Transcription Complete!')

        # Display transcript and download link
        st.subheader('Transcript')
        st.write(transcript)
        st.markdown(get_text_download_link(transcript, 'transcript.txt', 'Download Transcript'), unsafe_allow_html=True)

        # Summarization button
        if st.button('Summarize Transcript'):
            with st.spinner('Summarizing...'):
                # Assuming the generate_response function from the previous code is defined and works here.
                summary = generate_response(transcript)
                st.subheader('Summary')
                st.write(summary)
