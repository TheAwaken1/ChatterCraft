import gradio as gr
import torch
import torchaudio # For audio processing
import tempfile
import os
import sys
import re
import nltk
import gc
import time # Added for time.sleep
import traceback # Added for detailed error printing
from pathlib import Path
from datetime import datetime
from huggingface_hub import snapshot_download
from huggingface_hub.utils import HfHubHTTPError # Corrected import
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import numpy as np
import warnings
import ssl
from spellchecker import SpellChecker

warnings.filterwarnings("ignore")

# Enable CUDA debugging
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

try:
    from chatterbox.tts import ChatterboxTTS
except ImportError:
    print("Error: chatterbox-tts not installed. Please run: pip install chatterbox-tts")
    exit(1)

# Set NLTK data path
NLTK_DATA_PATH = os.path.join(os.path.dirname(__file__), "nltk_data")
if not os.path.exists(NLTK_DATA_PATH):
    os.makedirs(NLTK_DATA_PATH)
nltk.data.path.append(NLTK_DATA_PATH)

# Disable SSL verification for NLTK download if needed
try:
    _create_unverified_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_context

def ensure_punkt_data(max_retries=3, delay=5):
    """Ensure NLTK punkt data is available, with retries."""
    for attempt in range(max_retries):
        try:
            nltk.data.find('tokenizers/punkt')
            print("NLTK 'punkt' data found.")
            return True
        except LookupError:
            print(f"NLTK 'punkt' data not found. Attempting download (attempt {attempt + 1}/{max_retries})...")
            try:
                nltk.download('punkt', download_dir=NLTK_DATA_PATH, quiet=True)
                nltk.data.find('tokenizers/punkt') # Verify after download
                print("NLTK 'punkt' data downloaded successfully.")
                return True
            except Exception as e:
                print(f"Error downloading NLTK 'punkt': {e}")
                if attempt < max_retries - 1:
                    print(f"Retrying in {delay} seconds...")
                    time.sleep(delay)
    raise RuntimeError("Could not load NLTK 'punkt' data after maximum retries.")

ensure_punkt_data()

# Model Definitions
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
AVAILABLE_MODELS = {
    "Phi-3-mini-4k-instruct": { # Using Phi-3 Mini as requested
        "repo_id": "microsoft/Phi-3-mini-4k-instruct",
        "local_path": os.path.join(MODEL_DIR, "Phi-3-mini-4k-instruct"),
        "context_length": 4096,
        "quantized_by_us": False # Or True if you decide to quantize it by default
    }
}
model_choices = list(AVAILABLE_MODELS.keys()) # Will be ["Phi-3-mini-4k-instruct"]
if not model_choices:
    raise ValueError("No models defined in AVAILABLE_MODELS. Please add at least one model.")
DEFAULT_MODEL_NAME = model_choices[0] # Will be "Phi-3-mini-4k-instruct"

# Global state for LLM
current_llm_model = None
current_tokenizer = None
current_llm_name = None
USE_CUDA_AVAILABLE = torch.cuda.is_available()
DEVICE = "cuda" if USE_CUDA_AVAILABLE else "cpu"
print(f"Using device: {DEVICE}")


def download_model(repo_id, local_path):
    """Downloads model from Hugging Face Hub if not present locally."""
    if not os.path.exists(local_path):
        print(f"Downloading model {repo_id} to {local_path}...")
        try:
            snapshot_download(
                repo_id=repo_id,
                local_dir=local_path,
                local_dir_use_symlinks=False,
                cache_dir=os.path.join(BASE_DIR, "cache") # Define cache for snapshot_download
            )
            print(f"Model {repo_id} downloaded successfully.")
        except Exception as e:
            print(f"Error downloading model {repo_id}: {e}")
            raise RuntimeError(f"Failed to download model {repo_id}: {e}")
    else:
        print(f"Model {repo_id} already exists at {local_path}.")

def unload_llm_model():
    """Unloads the current LLM model and clears GPU cache."""
    global current_llm_model, current_tokenizer, current_llm_name
    if current_llm_model is not None:
        llm_name_unloading = current_llm_name
        print(f"Unloading previous LLM: {llm_name_unloading}")
        del current_llm_model
        del current_tokenizer
        current_llm_model = None
        current_tokenizer = None
        current_llm_name = None
        gc.collect()
        if USE_CUDA_AVAILABLE:
            torch.cuda.empty_cache()
            print("Cleared PyTorch CUDA cache after LLM unload.")
    else:
        print("No LLM model currently loaded to unload.")

def get_llm(model_name_to_load):
    """Loads the selected LLM model, downloading if necessary."""
    global current_llm_model, current_tokenizer, current_llm_name
    if model_name_to_load == current_llm_name and current_llm_model is not None:
        print(f"LLM '{model_name_to_load}' is already loaded.")
        return current_llm_model, current_tokenizer

    if current_llm_name is not None and current_llm_name != model_name_to_load:
        unload_llm_model()

    if model_name_to_load not in AVAILABLE_MODELS:
        raise ValueError(f"Unknown model selected: {model_name_to_load}")

    model_config = AVAILABLE_MODELS[model_name_to_load]
    repo_id = model_config["repo_id"]
    local_path = model_config["local_path"]
    should_apply_bnb_quantization = model_config.get("quantized_by_us", False)

    download_model(repo_id, local_path) # Ensure model is downloaded
    print(f"Loading LLM model: {model_name_to_load} from: {local_path}")

    try:
        model_load_kwargs = {
            "device_map": "auto",
            "trust_remote_code": True, # Important for many models like Phi-3
        }

        # Set attention implementation
        if "Phi-3" in model_name_to_load:
            # Flash Attention 2 is recommended for Phi-3 if available
            try:
                # Test if flash_attn is importable and usable before setting
                import flash_attn
                model_load_kwargs["attn_implementation"] = "flash_attention_2"
                print("INFO: Using 'flash_attention_2' for Phi-3 model.")
            except ImportError:
                print("INFO: 'flash_attn' not found. Using 'eager' attention for Phi-3 model.")
                model_load_kwargs["attn_implementation"] = "eager"
            except Exception as e_flash: # Catch other potential errors with flash_attn
                print(f"INFO: Error enabling 'flash_attention_2' ({e_flash}). Using 'eager' attention for Phi-3 model.")
                model_load_kwargs["attn_implementation"] = "eager"
        else:
            model_load_kwargs["attn_implementation"] = "eager"


        if should_apply_bnb_quantization:
            print(f"INFO: Applying BitsAndBytes 4-bit quantization to {model_name_to_load}")
            compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            print(f"INFO: Using compute_dtype: {compute_dtype} for BitsAndBytes.")
            quantization_config_bnb = BitsAndBytesConfig( # Renamed to avoid conflict with 'config' variable
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=compute_dtype
            )
            model_load_kwargs["quantization_config"] = quantization_config_bnb
        else:
            print(f"INFO: Loading {model_name_to_load} without BitsAndBytes 4-bit quantization.")
            model_load_kwargs["torch_dtype"] = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            print(f"INFO: Loading model with torch_dtype: {model_load_kwargs['torch_dtype']}")

        model = AutoModelForCausalLM.from_pretrained(
            local_path,
            **model_load_kwargs
        )

        tokenizer = AutoTokenizer.from_pretrained(local_path, trust_remote_code=True)

        # Pad token handling
        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token_id is not None:
                tokenizer.pad_token_id = tokenizer.eos_token_id
                print(f"INFO: tokenizer.pad_token_id was None, set to eos_token_id: {tokenizer.pad_token_id}")
            else:
                # This state is problematic; models need a pad token for batching/padding.
                print("CRITICAL WARNING: tokenizer.pad_token_id AND tokenizer.eos_token_id are None. Attempting to add a pad token.")
                tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                # Ensure model's embedding layer is resized if a new token is added
                # model.resize_token_embeddings(len(tokenizer)) # May be needed if token actually added
                print(f"INFO: Added a new pad token '[PAD]'. New pad_token_id: {tokenizer.pad_token_id}")
        
        if tokenizer.pad_token is None and tokenizer.pad_token_id is not None: # Ensure pad_token attribute is also set
            tokenizer.pad_token = tokenizer.convert_ids_to_tokens(tokenizer.pad_token_id)

        tokenizer.padding_side = "right" # Commonly used, ensure consistency

        print(f"DEBUG: Loaded Tokenizer details: eos_token='{tokenizer.eos_token}', eos_token_id={tokenizer.eos_token_id}, pad_token='{tokenizer.pad_token}', pad_token_id={tokenizer.pad_token_id}")

        current_llm_model = model
        current_tokenizer = tokenizer
        current_llm_name = model_name_to_load
        print(f"LLM model '{model_name_to_load}' loaded successfully.")
        return current_llm_model, current_tokenizer
    except Exception as e:
        print(f"Error loading LLM model '{model_name_to_load}': {e}")
        traceback.print_exc()
        raise RuntimeError(f"Failed to load LLM model '{model_name_to_load}': {e}")

# This is the LLM story generation function
def generate_story_llm(prompt, model_name, target_word_count, max_tokens=1000):
    """Generates story text in English using the selected LLM with proper punctuation and spelling correction."""
    try:
        model, tokenizer = get_llm(model_name)
        model_config_llm = AVAILABLE_MODELS[model_name]
        context_length = model_config_llm["context_length"]

        base_instruction = (
            f"You are a creative assistant who writes engaging short stories based on user prompts. "
            f"Write a story approximately {target_word_count} words long in English, with a rich narrative, vivid descriptions, "
            f"and a complete arc. Ensure the story concludes naturally within the word limit. "
            f"Use proper sentence punctuation (periods, exclamation points, or question marks). "
            f"Do not include meta-commentary, explanations, or outlines. Focus solely on the narrative."
        )
        user_content = f"Write a short story based on this prompt: {prompt}"
        messages = [
            {"role": "system", "content": base_instruction},
            {"role": "user", "content": user_content}
        ]
        print(f"Using standard chat format. System prompt: {base_instruction[:100]}...")

        if tokenizer.chat_template is None:
            print(f"Warning: Tokenizer for {model_name} lacks a chat_template. Using manual formatting.")
            prompt_text = f"{messages[0]['content']}\n\n{messages[1]['content']}\n\nStory:"
            inputs_dict = tokenizer(prompt_text, return_tensors="pt", padding=True, truncation=True, max_length=context_length - max_tokens)
            input_ids = inputs_dict.input_ids.to(DEVICE)
            attention_mask = inputs_dict.attention_mask.to(DEVICE)
        else:
            try:
                inputs = tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_tensors="pt"
                )
                input_ids = inputs.to(DEVICE)
                attention_mask = input_ids.ne(tokenizer.pad_token_id).to(DEVICE) if tokenizer.pad_token_id is not None else torch.ones_like(input_ids).to(DEVICE)
            except Exception as e_template:
                print(f"Error applying chat template for {model_name}: {e_template}")
                print("Falling back to basic tokenization.")
                prompt_str = f"{messages[0]['content']}\n\n{messages[1]['content']}\n\nStory:"
                tokenized_inputs = tokenizer(prompt_str, return_tensors="pt", padding=True, truncation=True, max_length=context_length - max_tokens)
                input_ids = tokenized_inputs.input_ids.to(DEVICE)
                attention_mask = tokenized_inputs.attention_mask.to(DEVICE)

        if input_ids.numel() == 0 or torch.isnan(input_ids).any() or torch.isinf(input_ids).any():
            print(f"DEBUG: Invalid input_ids detected: numel={input_ids.numel()}, isnan={(torch.isnan(input_ids).any())}, isinf={(torch.isinf(input_ids).any())}")
            return f"Failed to generate story: Invalid input tokens prepared. Please try a different prompt."

        print(f"DEBUG Pre-LLM-Generation: input_ids shape: {input_ids.shape}, min: {input_ids.min().item()}, max: {input_ids.max().item()}, contains_zero: {(input_ids == 0).any().item()}")
        print(f"DEBUG Pre-LLM-Generation: attention_mask shape: {attention_mask.shape}, sum: {attention_mask.sum().item()}")
        print(f"DEBUG Pre-LLM-Generation: Passing pad_token_id={tokenizer.pad_token_id} (effective: {tokenizer.eos_token_id if tokenizer.pad_token_id is None else tokenizer.pad_token_id}) to model.generate")

        print(f"Generating story with {model_name} (Target: ~{target_word_count} words, max_tokens={max_tokens}). Input shape: {input_ids.shape}")
        
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

        gen_pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        if gen_pad_token_id is None:
            print("CRITICAL: No pad_token_id available for model.generate.")
            gen_pad_token_id = 0

        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_tokens,
            temperature=0.7,
            top_p=0.95,
            do_sample=True,
            pad_token_id=gen_pad_token_id,
            num_beams=1,
            no_repeat_ngram_size=3
        )

        num_input_tokens = input_ids.shape[1]
        generated_token_ids = outputs[0][num_input_tokens:]
        story_text = tokenizer.decode(generated_token_ids, skip_special_tokens=True).strip()

        if not story_text:
            print("Warning: Generated story is empty.")
            return f"Failed to generate story: Empty output. Please try a different prompt."

        # --- Spelling Correction ---
        spell = SpellChecker()
        # Add proper nouns (e.g., character names) to avoid correction
        proper_nouns = {"Rat", "Natters"}  # Adjust based on prompt or extract dynamically
        spell.word_frequency.load_words(proper_nouns)
        
        words = story_text.split()
        corrected_words = []
        for word in words:
            # Preserve punctuation at word end
            match = re.match(r'^(\w+)([.,!?]?)$', word)
            if match:
                w, punct = match.groups()
                if w.lower() not in proper_nouns and len(w) > 2:  # Skip short words
                    corrected = spell.correction(w.lower())
                    if corrected:
                        # Maintain original case
                        if w.islower():
                            w = corrected
                        elif w.istitle():
                            w = corrected.capitalize()
                        elif w.isupper():
                            w = corrected.upper()
                corrected_words.append(w + punct)
            else:
                corrected_words.append(word)
        story_text = " ".join(corrected_words)

        # --- Sentence Punctuation and Structure ---
        # Try NLTK sentence tokenization
        sentences = nltk.sent_tokenize(story_text)
        if not sentences or len(sentences) == 1 and len(story_text.split()) > 20:
            # Fallback: Split on capitalization boundaries
            print("Warning: NLTK failed to tokenize sentences. Using capitalization fallback.")
            sentences = re.split(r'(?<=[a-z])\s+([A-Z][a-z])', story_text)
            # Rejoin split parts (e.g., "sentence. New" → ["sentence.", "New"])
            sentences = [sentences[i] + (sentences[i+1] if i+1 < len(sentences) else "") for i in range(0, len(sentences), 2)]

        corrected_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            # Ensure sentence ends with punctuation
            if sentence[-1] not in ".!?":
                sentence += "."
            # Basic structure check: at least 3 words (subject, verb, object)
            if len(sentence.split()) < 3:
                print(f"Warning: Short sentence detected: '{sentence}'. May be incomplete.")
            corrected_sentences.append(sentence)

        if not corrected_sentences:
            print("Warning: No valid sentences after processing.")
            story_text = story_text.strip() + "."
        else:
            story_text = " ".join(corrected_sentences)

        # --- Final Cleanup ---
        story_text = re.sub(r'\s+', ' ', story_text)  # Normalize spaces
        story_text = re.sub(r'([.!?])\s*([A-Z])', r'\1 \2', story_text)  # Space after punctuation
        story_text = story_text.strip()

        actual_words = len(story_text.split())
        print(f"Generated story: {actual_words} words")
        
        if actual_words < 30:
            print(f"Warning: Generated story is very short ({actual_words} words). Text: '{story_text[:100]}...'")
        
        print(f"Final corrected story: {story_text[:200]}...")
        return story_text
    except Exception as e:
        print(f"Error during story generation: {e}")
        import traceback
        traceback.print_exc()
        return f"Failed to generate story due to an internal error: {str(e)}. Please check logs and try again."

class StoryCraftChatterboxApp:
    def __init__(self):
        self.tts_model = None # Renamed from self.model to avoid confusion with LLM
        self.device = self.get_optimal_device() # TTS device
        self.tts_model_id = "ResembleAI/chatterbox" # Renamed
        self.tts_cache_dir = Path.home() / ".cache" / "chatterbox" # Renamed
        self.tts_model_loaded = False # Renamed
        self.chunk_size = 250 # Max characters per chunk for TTS
        self.max_chunk_words = 40 # TTS chunking by words as a fallback

        self.tts_cache_dir.mkdir(parents=True, exist_ok=True)
        self.initialize_tts_model() # Renamed

    def get_optimal_device(self): # This is for TTS, can be different from LLM DEVICE
        if not torch.cuda.is_available():
            print("TTS Optimal Device: CPU (CUDA not available)")
            return "cpu"
        try:
            # A light CUDA test
            test_tensor = torch.tensor([1.0]).to("cuda")
            del test_tensor
            torch.cuda.empty_cache()
            print("TTS Optimal Device: CUDA")
            return "cuda"
        except Exception as e:
            print(f"CUDA test failed for TTS, falling back to CPU: {e}")
            return "cpu"

    def check_tts_model_exists(self): # Renamed
        model_files = ["s3gen.pt", "conds.pt"] # Key files for Chatterbox
        for filename in model_files:
            file_path = self.tts_cache_dir / filename
            if not file_path.exists():
                print(f"TTS model file not found: {file_path}")
                return False
        print("All TTS model files found.")
        return True

    def download_tts_model_with_progress(self): # Renamed
        print(f"Downloading Chatterbox TTS model files to {self.tts_cache_dir}...")
        try:
            snapshot_download(
                repo_id=self.tts_model_id,
                cache_dir=str(self.tts_cache_dir.parent), # HF expects parent of model dir
                local_dir=str(self.tts_cache_dir),
                local_dir_use_symlinks=False,
            )
            print("Chatterbox TTS model download completed.")
            return True
        except HfHubHTTPError as e: # More specific error
            print(f"Error downloading Chatterbox TTS model from Hugging Face: {e}")
            return False
        except Exception as e:
            print(f"Unexpected error during Chatterbox TTS model download: {e}")
            return False

    def initialize_tts_model(self): # Renamed
        if not self.check_tts_model_exists():
            print("Chatterbox TTS model not found locally. Attempting download...")
            if not self.download_tts_model_with_progress():
                raise Exception("Failed to download Chatterbox TTS model files.")
        self.load_tts_model() # Renamed

    def load_tts_model(self): # Renamed
        print(f"Loading Chatterbox TTS model on TTS device: {self.device}...")
        try:
            # Set cache environment variables specifically for Hugging Face related downloads by Chatterbox if any
            os.environ["TRANSFORMERS_CACHE"] = str(self.tts_cache_dir.parent)
            os.environ["HF_HOME"] = str(self.tts_cache_dir.parent)

            self.tts_model = ChatterboxTTS.from_pretrained(device=self.device)
            self.tts_model_loaded = True
            print("Chatterbox TTS model loaded. Performing a test generation...")
            # Test generation
            test_wav = self.tts_model.generate("Hello world, this is a test.", temperature=0.7, exaggeration=0.5)
            if test_wav is None or not isinstance(test_wav, torch.Tensor) or not test_wav.abs().sum().item():
                # Check if tensor and not silent
                raise Exception("Chatterbox TTS model validation failed: Generated silent or invalid audio.")
            print(f"Chatterbox TTS model test generation successful. Audio shape: {test_wav.shape}")
            del test_wav
            if self.device == "cuda":
                torch.cuda.empty_cache()
        except Exception as e:
            print(f"Error loading Chatterbox TTS model: {e}")
            traceback.print_exc()
            self.tts_model_loaded = False
            # Re-raise to prevent app from starting with a broken TTS
            raise e


    def split_text_into_chunks(self, text): # Max characters per chunk for TTS
        # Simple split by sentences, then by character limit
        sentences = nltk.sent_tokenize(text.strip())
        chunks = []
        current_chunk = ""
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            if len(current_chunk) + len(sentence) + 1 <= self.chunk_size:
                current_chunk += (" " if current_chunk else "") + sentence
            else:
                # If current_chunk is too large itself, split it
                if len(current_chunk) > self.chunk_size:
                    # Force split current_chunk by words if it's too long alone
                    words_in_current = current_chunk.split()
                    temp_sub_chunk = ""
                    for word in words_in_current:
                        if len(temp_sub_chunk) + len(word) + 1 <= self.chunk_size:
                            temp_sub_chunk += (" " if temp_sub_chunk else "") + word
                        else:
                            if temp_sub_chunk: chunks.append(temp_sub_chunk)
                            temp_sub_chunk = word
                    if temp_sub_chunk: chunks.append(temp_sub_chunk)
                elif current_chunk: # Current chunk is valid, add it
                    chunks.append(current_chunk)
                current_chunk = sentence # Start new chunk with current sentence

        # Add the last processed chunk
        if current_chunk:
            # If last chunk is too large, split it
            if len(current_chunk) > self.chunk_size:
                words_in_current = current_chunk.split()
                temp_sub_chunk = ""
                for word in words_in_current:
                    if len(temp_sub_chunk) + len(word) + 1 <= self.chunk_size:
                        temp_sub_chunk += (" " if temp_sub_chunk else "") + word
                    else:
                        if temp_sub_chunk: chunks.append(temp_sub_chunk)
                        temp_sub_chunk = word
                if temp_sub_chunk: chunks.append(temp_sub_chunk)
            else:
                chunks.append(current_chunk)
        
        return chunks if chunks else [text]


    def concatenate_audio_files(self, audio_tensors, sample_rate): # Expects list of tensors
        if not audio_tensors:
            return None
        if len(audio_tensors) == 1:
            return audio_tensors[0]
        
        # Ensure all tensors are 2D [channels, samples] and on the same device
        processed_tensors = []
        for i, audio_tensor in enumerate(audio_tensors):
            if audio_tensor.dim() == 1:
                audio_tensor = audio_tensor.unsqueeze(0) # Add channel dim
            if audio_tensor.shape[0] > 2 : # If more than stereo, try to take first channel or mean
                 print(f"Warning: audio tensor {i} has shape {audio_tensor.shape}, taking mean over channels.")
                 audio_tensor = torch.mean(audio_tensor, dim=0, keepdim=True)
            elif audio_tensor.shape[0] == 2: # Stereo to mono by taking mean
                 audio_tensor = torch.mean(audio_tensor, dim=0, keepdim=True)

            processed_tensors.append(audio_tensor.to(self.device)) # Move to TTS device

        try:
            concatenated = torch.cat(processed_tensors, dim=1)
        except Exception as e:
            print(f"Error during torch.cat: {e}")
            # Fallback: sum lengths and create silent audio of that duration
            total_samples = sum(t.shape[1] for t in processed_tensors)
            concatenated = torch.zeros((1, total_samples), device=self.device)
        return concatenated

    def generate_tts_audio(self, text, audio_upload_data=None, exaggeration=0.3, temperature=0.5):
        if not self.tts_model_loaded:
            print("TTS model not loaded, cannot generate audio.")
            return (None, None), "TTS Model not loaded"
        if not text or not text.strip():
            print("Empty text provided for TTS.")
            return (None, None), "Please enter text to synthesize"

        print(f"Starting TTS generation for text: '{text[:100]}...'")
        audio_prompt_path_for_tts = None
        temp_audio_prompt_file_to_delete = None
        CHATTERBOX_REF_SR = 24000

        try:
            # --- Voice Sample Preprocessing ---
            if audio_upload_data is not None:
                upload_sr, upload_audio_np = audio_upload_data
                print(f"Processing uploaded voice sample. Original SR: {upload_sr}, Shape: {upload_audio_np.shape}, dtype: {upload_audio_np.dtype}")

                if upload_audio_np.size == 0:
                    print("Warning: Uploaded voice sample is empty. Proceeding without voice cloning.")
                    audio_upload_data = None
                else:
                    # Convert to tensor
                    if upload_audio_np.dtype == np.int16:
                        audio_tensor = torch.from_numpy(upload_audio_np.astype(np.float32) / 32767.0)
                    elif upload_audio_np.dtype == np.int32:
                        audio_tensor = torch.from_numpy(upload_audio_np.astype(np.float32) / 2147483648.0)
                    elif upload_audio_np.dtype == np.float32:
                        audio_tensor = torch.from_numpy(upload_audio_np)
                    else:
                        audio_tensor = torch.from_numpy(upload_audio_np.astype(np.float32))

                    # Ensure correct shape
                    if audio_tensor.dim() == 1:
                        audio_tensor = audio_tensor.unsqueeze(0)  # [1, samples]
                    elif audio_tensor.dim() == 2 and audio_tensor.shape[0] > audio_tensor.shape[1]:
                        audio_tensor = audio_tensor.T  # Transpose if [samples, channels]

                    # Convert to mono
                    if audio_tensor.shape[0] > 1:
                        print(f"Converting stereo to mono (channels: {audio_tensor.shape[0]}).")
                        audio_tensor = torch.mean(audio_tensor, dim=0, keepdim=True)

                    # Validate tensor
                    if audio_tensor.numel() == 0 or torch.isnan(audio_tensor).any():
                        print("Error: Invalid audio tensor after conversion.")
                        audio_upload_data = None
                    else:
                        # Check signal strength
                        signal_power = torch.mean(audio_tensor**2)
                        if signal_power < 1e-6:
                            print("Warning: Voice sample has low signal strength. Disabling cloning.")
                            audio_upload_data = None

                        # Resample
                        if audio_upload_data is not None and upload_sr != CHATTERBOX_REF_SR:
                            print(f"Resampling from {upload_sr} Hz to {CHATTERBOX_REF_SR} Hz.")
                            resampler = torchaudio.transforms.Resample(
                                orig_freq=upload_sr,
                                new_freq=CHATTERBOX_REF_SR,
                                resampling_method='sinc_interpolation'
                            )
                            audio_tensor = resampler(audio_tensor)
                            print(f"Post-resampling shape: {audio_tensor.shape}")

                        # Verify duration
                        if audio_upload_data is not None:
                            if audio_tensor.shape[1] == 0:
                                print("Error: Empty tensor after resampling. Disabling cloning.")
                                audio_upload_data = None
                            else:
                                duration_s = audio_tensor.shape[1] / CHATTERBOX_REF_SR
                                print(f"Calculated duration: {duration_s:.2f}s")
                                if duration_s < 1.0:
                                    print(f"Warning: Voice sample too short ({duration_s:.2f}s). Minimum 1s required. Disabling cloning.")
                                    audio_upload_data = None
                                elif duration_s > 30.0:
                                    print(f"Warning: Voice sample long ({duration_s:.2f}s). Trimming to 30s.")
                                    audio_tensor = audio_tensor[:, :int(CHATTERBOX_REF_SR * 30.0)]

                        # Save to temporary file
                        if audio_upload_data is not None:
                            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_f:
                                torchaudio.save(tmp_f.name, audio_tensor, CHATTERBOX_REF_SR)
                                audio_prompt_path_for_tts = tmp_f.name
                                temp_audio_prompt_file_to_delete = audio_prompt_path_for_tts
                            print(f"Saved temporary voice prompt: {audio_prompt_path_for_tts}")

            # --- Text Chunking and TTS Generation ---
            text_chunks = self.split_text_into_chunks(text)
            print(f"Split text into {len(text_chunks)} chunks for TTS.")
            generated_audio_tensors = []

            tts_kwargs = {
                "exaggeration": float(exaggeration),
                "temperature": float(temperature)
            }
            if audio_prompt_path_for_tts:
                tts_kwargs["audio_prompt_path"] = audio_prompt_path_for_tts
                print(f"Using audio prompt: {audio_prompt_path_for_tts}")
            else:
                print("Using default voice.")

            for i, chunk_text in enumerate(text_chunks):
                if not chunk_text.strip():
                    continue
                print(f"Generating TTS for chunk {i+1}/{len(text_chunks)}: '{chunk_text[:50]}'")
                try:
                    wav_tensor = self.tts_model.generate(chunk_text, **tts_kwargs)
                    if wav_tensor is None or wav_tensor.numel() == 0 or not wav_tensor.abs().sum().any():
                        print(f"Warning: TTS generated empty audio for chunk {i+1}. Skipping.")
                        continue

                    if wav_tensor.dim() == 1:
                        wav_tensor = wav_tensor.unsqueeze(0)
                    if wav_tensor.shape[0] > 1:
                        wav_tensor = torch.mean(wav_tensor, dim=0, keepdim=True)

                    generated_audio_tensors.append(wav_tensor.cpu())
                    print(f"Generated chunk {i+1} successfully. Shape: {wav_tensor.shape}")
                except Exception as chunk_e:
                    print(f"Error generating TTS for chunk {i+1}: {chunk_e}")
                    import traceback
                    traceback.print_exc()
                    if audio_prompt_path_for_tts:
                        print(f"Retrying chunk {i+1} without audio prompt...")
                        try:
                            retry_kwargs = tts_kwargs.copy()
                            del retry_kwargs["audio_prompt_path"]
                            wav_tensor = self.tts_model.generate(chunk_text, **retry_kwargs)
                            if wav_tensor is not None and wav_tensor.numel() > 0 and wav_tensor.abs().sum().any():
                                if wav_tensor.dim() == 1:
                                    wav_tensor = wav_tensor.unsqueeze(0)
                                if wav_tensor.shape[0] > 1:
                                    wav_tensor = torch.mean(wav_tensor, dim=0, keepdim=True)
                                generated_audio_tensors.append(wav_tensor.cpu())
                                print(f"Retry successful for chunk {i+1}.")
                            else:
                                print(f"Retry failed for chunk {i+1}.")
                        except Exception as retry_e:
                            print(f"Retry error: {retry_e}")

            if not generated_audio_tensors:
                print("No audio chunks generated.")
                return (None, None), "No audio chunks generated. Please check TTS model or input text."

            # --- Concatenation and Output ---
            print("Concatenating audio chunks...")
            final_audio_tensor = self.concatenate_audio_files(generated_audio_tensors, self.tts_model.sr)
            if final_audio_tensor is None or final_audio_tensor.numel() == 0 or not final_audio_tensor.abs().sum().any():
                print("Warning: Final audio is silent or empty.")
                return (None, None), "Generated audio is empty."

            final_audio_np = final_audio_tensor.cpu().squeeze().numpy()
            if final_audio_np.ndim > 1:
                final_audio_np = final_audio_np[0]

            audio_data_int16 = np.clip(final_audio_np * 32767.0, -32768.0, 32767.0).astype(np.int16)
            print(f"TTS generation successful. Duration: {len(audio_data_int16)/self.tts_model.sr:.2f}s")
            status_msg = f"Generated audio from {len(text_chunks)} chunk(s)"
            status_msg += " with uploaded voice sample." if audio_prompt_path_for_tts else " with default voice."
            return (self.tts_model.sr, audio_data_int16), status_msg

        except Exception as e:
            print(f"Error during TTS generation: {e}")
            import traceback
            traceback.print_exc()
            return (None, None), f"Failed to generate audio: {str(e)}"
        finally:
            if temp_audio_prompt_file_to_delete and os.path.exists(temp_audio_prompt_file_to_delete):
                try:
                    os.unlink(temp_audio_prompt_file_to_delete)
                    print(f"Deleted temporary audio prompt: {temp_audio_prompt_file_to_delete}")
                except Exception as e_del:
                    print(f"Error deleting temporary audio prompt: {e_del}")

    def process_audio_only(self, story_text, audio_upload_data, temperature, exaggeration):
        status_text = ""
        # Use TTS model's sample rate as default, or a common one like 24000 if model not loaded
        default_sr = self.tts_model.sr if self.tts_model_loaded and hasattr(self.tts_model, 'sr') else 24000
        # Ensure audio_output_value is (sample_rate, numpy_array)
        audio_output_value = (default_sr, np.zeros(int(default_sr * 0.1), dtype=np.int16))
        download_file_path = None

        try:
            if not story_text or not story_text.strip():
                status_text = "Error: Story text is empty for audio regeneration.\n"
                yield status_text, audio_output_value, None # Yield valid empty audio tuple
                return

            status_text += "Regenerating speech from textbox content...\n"
            yield status_text, audio_output_value, None # Update status, yield placeholder audio

            # Call the renamed TTS audio generation method
            generated_speech_tuple, tts_status = self.generate_tts_audio(
                story_text, audio_upload_data, exaggeration, temperature
            )
            
            status_text += tts_status + "\n"

            if generated_speech_tuple and generated_speech_tuple[0] is not None and generated_speech_tuple[1] is not None and generated_speech_tuple[1].size > 0:
                audio_output_value = generated_speech_tuple # This is (sample_rate, numpy_array)
                status_text += "Speech regeneration successful.\n"
            else:
                status_text += "Speech regeneration failed or produced silent audio.\n"
                # audio_output_value remains the default silent audio
            
            yield status_text, audio_output_value, None # Update with generated audio (or silent on fail)

            # Save to file if audio is valid
            if audio_output_value and audio_output_value[1] is not None and audio_output_value[1].size > 0:
                sample_rate, audio_data_np = audio_output_value # audio_data_np is already int16
                
                output_dir = "output"
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_filename = f"regenerated_audio_{timestamp}.wav"
                output_filepath = os.path.join(output_dir, output_filename)
                
                # Torchaudio expects tensor, float32, range [-1, 1] for saving usually
                # So, convert back from int16 for saving with torchaudio, or use scipy.io.wavfile
                audio_tensor_to_save = torch.from_numpy(audio_data_np.astype(np.float32) / 32767.0).unsqueeze(0)
                if audio_tensor_to_save.dim() == 1: audio_tensor_to_save = audio_tensor_to_save.unsqueeze(0)

                torchaudio.save(output_filepath, audio_tensor_to_save, sample_rate)
                print(f"Saved regenerated audio: {output_filepath}, max amplitude in file: {torch.abs(audio_tensor_to_save).max().item():.4f}")
                download_file_path = output_filepath
                status_text += f"Final audio saved: {output_filepath}\n"
            
            status_text += "Audio regeneration process finished.\n"
            yield status_text, audio_output_value, download_file_path

        except Exception as e:
            error_msg = f"Error during audio regeneration: {e}"
            print(error_msg)
            traceback.print_exc()
            status_text += error_msg + "\n"
            yield status_text, audio_output_value, None # Yield placeholder audio on error


    def generate_story_audio(self, story_mode, prompt, manual_story_text, model_name_llm, # Renamed for clarity
                            audio_upload_data, target_minutes, temperature_tts, exaggeration_tts): # Renamed
        status_text = ""
        story_text_to_output = "Error: Story processing did not complete."
        # Default audio output: (sample_rate, numpy_array)
        default_sr = self.tts_model.sr if self.tts_model_loaded and hasattr(self.tts_model, 'sr') else 24000
        audio_output_value = (default_sr, np.zeros(int(default_sr * 0.1), dtype=np.int16))
        download_file_path = None
        
        try:
            status_text += f"Selected Mode: {story_mode}\n"
            yield status_text, story_text_to_output, audio_output_value, None # Initial update

            if story_mode == "Generate story with AI":
                status_text += f"Selected LLM: {model_name_llm}\n"
                WPM = 150 # Words Per Minute estimation
                target_word_count = int(target_minutes * WPM)
                # Adjust max_tokens for LLM generation
                # Example: allow up to 1.5 tokens per word, capped at a model's practical limit or context window portion
                llm_config = AVAILABLE_MODELS.get(model_name_llm, {})
                max_context_tokens = llm_config.get("context_length", 4096)
                # Reserve some tokens for the prompt itself
                prompt_token_estimate = 256 # Rough estimate for system prompt and user prompt
                available_for_generation = max_context_tokens - prompt_token_estimate
                
                max_tokens_llm = min(int(target_word_count * 1.5), available_for_generation - 100) # -100 buffer
                max_tokens_llm = max(100, min(max_tokens_llm, 2048)) # Ensure at least 100, cap at 2048 for sanity

                status_text += f"Generating story with LLM (Target: ~{target_word_count} words, Max New Tokens: {max_tokens_llm})...\n"
                current_story_text = "AI is thinking..." # Placeholder for story_output_textbox
                yield status_text, current_story_text, audio_output_value, None
                
                # Call the global LLM generation function
                story_text_to_output = generate_story_llm(
                    prompt, model_name_llm, target_word_count, max_tokens_llm
                )

                if "Failed to generate story" in story_text_to_output or "Error:" in story_text_to_output :
                    status_text += f"LLM Error: {story_text_to_output}\n"
                    # story_text_to_output is already the error message
                    yield status_text, story_text_to_output, audio_output_value, None
                    return # Stop processing if LLM failed

                actual_words = len(story_text_to_output.split())
                status_text += f"AI Story generated ({actual_words} words).\n"

            elif story_mode == "Use my own story text":
                if not manual_story_text or not manual_story_text.strip():
                    status_text += "Error: 'Use my own story text' mode selected, but the story textbox is empty.\n"
                    story_text_to_output = "Error: Story textbox is empty for 'Use my own story text' mode."
                    yield status_text, story_text_to_output, audio_output_value, None
                    return
                
                story_text_to_output = manual_story_text # Use the text from the interactive textbox
                actual_words = len(story_text_to_output.split())
                status_text += f"Using user-provided story text ({actual_words} words).\n"
            
            else: # Should not happen with Radio button
                status_text += f"Error: Unknown story mode selected: {story_mode}\n"
                story_text_to_output = f"Error: Unknown story mode: {story_mode}"
                yield status_text, story_text_to_output, audio_output_value, None
                return

            yield status_text, story_text_to_output, audio_output_value, None # Update textbox with final story text

            # --- TTS Generation ---
            status_text += f"Generating speech for the story using Chatterbox TTS...\n"
            yield status_text, story_text_to_output, audio_output_value, None

            generated_speech_tuple, tts_status = self.generate_tts_audio( # Call the correct method
                story_text_to_output, audio_upload_data, exaggeration_tts, temperature_tts
            )
            status_text += tts_status + "\n"

            if generated_speech_tuple and generated_speech_tuple[0] is not None and generated_speech_tuple[1] is not None and generated_speech_tuple[1].size > 0:
                audio_output_value = generated_speech_tuple # (sample_rate, numpy_array)
                status_text += "Speech generation successful.\n"
            else:
                status_text += "Speech generation failed or produced silent audio.\n"
                # audio_output_value remains the default silent audio
            
            yield status_text, story_text_to_output, audio_output_value, None # Update with generated audio

            # --- Save Audio and Provide Download ---
            if audio_output_value and audio_output_value[1] is not None and audio_output_value[1].size > 0:
                sample_rate, audio_data_np = audio_output_value # audio_data_np is int16
                
                output_dir_path = Path(BASE_DIR) / "output" # Use Path object
                output_dir_path.mkdir(parents=True, exist_ok=True) # Ensure output dir exists
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_filename = f"story_audio_{timestamp}.wav"
                output_filepath = str(output_dir_path / output_filename) # Convert Path to string for torchaudio
                
                # Convert int16 numpy array back to float32 tensor for torchaudio.save
                audio_tensor_to_save = torch.from_numpy(audio_data_np.astype(np.float32) / 32767.0)
                if audio_tensor_to_save.dim() == 1: audio_tensor_to_save = audio_tensor_to_save.unsqueeze(0) # Ensure [C, T]

                torchaudio.save(output_filepath, audio_tensor_to_save, sample_rate)
                
                # For Gradio Audio component, we need to provide a filepath string.
                # The component itself will handle making it playable.
                # The audio_output_value is (sample_rate, numpy_array) which Gradio Audio can also handle if type="numpy"
                # but for download, we need the filepath.
                
                print(f"Saved final audio to: {output_filepath}")
                story_word_count = len(story_text_to_output.split())
                audio_duration_final = len(audio_data_np) / sample_rate if audio_data_np is not None else 0
                status_text += f"Final audio saved ({story_word_count} words, {audio_duration_final:.1f}s): {output_filepath}\n"
                download_file_path = output_filepath
            else:
                status_text += "Warning: Final audio data is empty or invalid. Skipping save.\n"

            status_text += "Process finished.\n"
            yield status_text, story_text_to_output, audio_output_value, download_file_path

        except Exception as e:
            error_message = f"An unexpected error occurred: {e}"
            print(error_message)
            traceback.print_exc()
            status_text += f"ERROR: {error_message}\n"
            # Ensure story_text_to_output is a string
            final_story_text_for_error = story_text_to_output if isinstance(story_text_to_output, str) else "Error during processing."
            yield status_text, final_story_text_for_error, audio_output_value, None


    def update_ui_visibility_based_on_story_mode(self, story_mode):
        ai_controls_active = (story_mode == "Generate story with AI")
        prompt_placeholder_text = (
            "e.g., A brave knight searching for a lost dragon..."
            if ai_controls_active
            else "Not used when 'Use my own story text' is selected. Type story in story output."
        )
        # The LLM model display (model_select_input Textbox) should remain non-interactive.
        # Its visibility/relevance is still tied to AI mode.
        return (
            gr.update(interactive=ai_controls_active, placeholder=prompt_placeholder_text), # prompt_input
            gr.update(interactive=False), # model_select_input (Textbox) - always non-interactive
                                          # but its relevance/associated controls are active with AI mode.
            gr.update(interactive=ai_controls_active)  # duration_slider
        )

    def create_interface(self):
        with gr.Blocks(theme=gr.themes.Soft()) as interface:
            gr.Markdown("# 🎙️ ChatterCraft") # Title
            gr.Markdown("Generate a story with AI or type your own, then convert it to speech using Chatterbox TTS voice cloning.") # Subtitle

            with gr.Accordion("Quick Guide", open=False):
                 gr.Markdown(
                    """
                    **Welcome to ChatterCraft AI!**
                    1.  **Choose Story Source:**
                        * `Generate story with AI`: The app creates a story from your prompt.
                        * `Use my own story text`: Type/paste your story into the "Story Text" box.
                    2.  **Configure Settings:**
                        * If AI generating: Fill "Story Prompt", set "Target Audio Duration".
                        * Upload a "Voice Sample" (WAV/MP3, 5-30s clear speech) for voice cloning.
                        * Adjust "TTS Variation" (randomness) and "TTS Emotion".
                    3.  **Generate!**
                        * Click **"Generate Story & Audio"**.
                    4.  **Edit & Regenerate Audio:**
                        * Edit text in the "Story Text (Editable)" box.
                        * Click **"Regenerate Audio from Textbox"** for new audio of the edited text.
                    **Notes:**
                    * The LLM (story generation) uses the global `DEVICE` (CPU or GPU).
                    * The Chatterbox TTS (speech generation) uses its own `self.device` (CPU or GPU).
                    * Output audio is saved in an `output` folder in the app directory.
                    """
                )


            with gr.Row():
                with gr.Column(scale=2): # Left column for inputs
                    gr.Markdown("### 1. Story Input Mode")
                    story_mode_input = gr.Radio(
                        label="Choose Story Source",
                        choices=["Generate story with AI", "Use my own story text"],
                        value="Generate story with AI",
                        elem_id="story_mode_radio"
                    )

                    gr.Markdown("### 2. Story Generation Settings")
                    prompt_input = gr.Textbox(
                        lines=5,
                        label="Story Prompt",
                        placeholder="e.g., A brave knight searching for a lost dragon..."
                    )
                    # --- MODIFIED PART ---
                    # Replace Dropdown with a non-interactive Textbox to display the LLM model
                    model_select_input = gr.Textbox(
                        label="LLM Model", # Your desired label
                        value=DEFAULT_MODEL_NAME, # Displays the name of the single model
                        interactive=False # Makes it non-editable
                    )
                    # --- END MODIFIED PART ---
                    duration_slider = gr.Slider(
                        minimum=0.5, maximum=5.0, step=0.25, value=1.0, 
                        label="Approx. Target Story Duration (Minutes, for AI Story)"
                    )

                    gr.Markdown("### 3. Voice Settings (TTS)")
                    audio_upload_input = gr.Audio(
                        label="Voice Sample for Cloning (WAV/MP3, 5-30s of clear speech)",
                        type="numpy", 
                        elem_id="voice_sample_upload"
                    )
                    
                    gr.Markdown("### 4. TTS Tuning")
                    temperature_slider_tts = gr.Slider( 
                        minimum=0.1, maximum=1.0, step=0.05, value=0.7,
                        label="TTS Variation (Higher = More Random)"
                    )
                    exaggeration_slider_tts = gr.Slider( 
                        minimum=0.0, maximum=1.0, step=0.05, value=0.5,
                        label="TTS Emotion (0.0 = Subtle, 1.0 = Dramatic)"
                    )
                    
                    generate_btn = gr.Button("🚀 Generate Story & Audio", variant="primary", elem_id="generate_button")

                # ... (Rest of the Column for outputs and event handlers remain the same) ...
                # The event handlers will correctly use the value of `model_select_input` (which is now
                # the fixed DEFAULT_MODEL_NAME from the non-interactive Textbox).

                with gr.Column(scale=3): # Right column for outputs
                    status_textbox = gr.Textbox(
                        label="Status Log",
                        interactive=False,
                        lines=8, 
                        placeholder="Status updates will appear here..."
                    )
                    gr.Markdown("### Story Output & Editing")
                    story_output_textbox = gr.Textbox(
                        label="Story Text (Editable)",
                        interactive=True,
                        lines=15, 
                        placeholder="AI-generated story will appear here, or you can type/paste your own."
                    )
                    gr.Markdown("### Audio Output")
                    audio_output = gr.Audio(
                        label="Generated Audio Output",
                        type="numpy", 
                        interactive=False 
                    )
                    regenerate_audio_button = gr.Button("🔁 Regenerate Audio from Textbox", elem_id="regenerate_audio_button")
                    download_output = gr.File(label="Download Generated Audio File (.wav)")

            # Event Listeners
            story_mode_input.change(
                fn=self.update_ui_visibility_based_on_story_mode,
                inputs=[story_mode_input],
                # The model_select_input Textbox is now the second output component
                outputs=[prompt_input, model_select_input, duration_slider]
            )

            generate_btn.click(
                fn=self.generate_story_audio,
                inputs=[
                    story_mode_input,
                    prompt_input,
                    story_output_textbox, 
                    model_select_input,   # This will now pass the value of the Textbox
                    audio_upload_input,   
                    duration_slider,      
                    temperature_slider_tts, 
                    exaggeration_slider_tts 
                ],
                outputs=[status_textbox, story_output_textbox, audio_output, download_output]
            )

            regenerate_audio_button.click(
                fn=self.process_audio_only,
                inputs=[
                    story_output_textbox, 
                    audio_upload_input,   
                    temperature_slider_tts, 
                    exaggeration_slider_tts 
                ],
                outputs=[status_textbox, audio_output, download_output] 
            )
        return interface

    def cleanup(self):
        print("Cleaning up resources...")
        if hasattr(self, 'tts_model') and self.tts_model:
            del self.tts_model
            self.tts_model = None
            print("TTS model unloaded.")
        unload_llm_model() # Unloads global LLM
        if USE_CUDA_AVAILABLE or self.device == "cuda": # Check both global and TTS device
            torch.cuda.empty_cache()
            print("Cleared PyTorch CUDA cache during cleanup.")
        gc.collect()
        print("Cleanup finished.")


def main():
    print("Initializing ChatterCraft AI...")

    app_instance = None
    try:
        print("[DEBUG: Creating StoryCraftChatterboxApp instance]")
        app_instance = StoryCraftChatterboxApp()
        print("[DEBUG: Creating Gradio interface]")
        interface = app_instance.create_interface()

        print("=" * 50)
        print("Starting interface...")
        print("Local URL: http://localhost:7861")
        print("=" * 50)

        print("[DEBUG: Launching Gradio interface on port 7861]")
        interface.launch(
            server_name="0.0.0.0",
            server_port=7861,
            share=False,
            show_error=True,
            max_threads=50,
            # Remove max_file_size (not a valid parameter for launch)
            # request_timeout is also not directly supported; use server config if needed
        )

    except KeyboardInterrupt:
        print("\nShutting down...")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
    finally:
        if app_instance:
            print("[DEBUG: Cleaning up app instance]")
            app_instance.cleanup()


if __name__ == "__main__":
    main()