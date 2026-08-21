"""Inference engine — load models and generate responses.

==================================================================
WHAT THIS FILE DOES (read this first)
==================================================================
This file is the BRIDGE between our Python code and the actual LLM.
It knows how to:
  1. Load an LLM (either a fast quantized .gguf file OR a full HuggingFace model)
  2. Send chat messages to the model and get back responses
  3. Free memory when the model is no longer needed

Think of it as the "remote control" for the LLM. The rest of the
codebase talks to this class; it talks to the model.

==================================================================
KEY CONCEPTS
==================================================================
- GGUF: a file format for LLMs that uses quantization (storing model
  weights as low-precision numbers like 4-bit instead of 16-bit).
  This makes models 4-8x smaller with only a small quality loss.
  Example: a 14B model goes from 28GB (FP16) to 8GB (Q4_K_M).
- llama-cpp-python: a Python wrapper around llama.cpp, a C++ library
  that runs GGUF models on CPU or GPU.
- HuggingFace transformers: a Python library that loads full-precision
  models in PyTorch format. Slower but more flexible.
- GPU offloading: n_gpu_layers=99 means "put as many layers as possible
  on the GPU". 99 is a magic number meaning "all of them, if they fit".
- Context window (n_ctx): how many tokens the model can "see" at once.
  8192 is a good balance for 14B models on 8GB VRAM.
"""

# ─── IMPORTS ─────────────────────────────────────────────────────────────
# time: Python's built-in module for measuring elapsed time. We use it
# to record how long generation took (useful for benchmarking).
# Path: represents a filesystem path. More convenient than raw strings
# for path operations like checking if a file exists or getting the
# file extension.
from pathlib import Path
import time


# ═════════════════════════════════════════════════════════════════════════
# MAIN CLASS: InferenceEngine
# ═════════════════════════════════════════════════════════════════════════
class InferenceEngine:
    """Model inference engine — supports both GGUF and HuggingFace safetensors.

    This class encapsulates the model loading and generation logic.
    It can load either format and automatically uses the right backend.

    Attributes
    ----------
    model : Llama | AutoModelForCausalLM | None
        The loaded model object. None when no model is loaded.
    tokenizer : AutoTokenizer | None
        The tokenizer (only used for HuggingFace models). None for GGUF.
    model_path : str | None
        Path to the loaded model. None when no model is loaded.
    is_gguf : bool
        True if a GGUF model is loaded, False if HuggingFace.
    config : object | None
        Optional configuration object (from the inference server config).
    """

    def __init__(self):
        """Initialize the engine with no model loaded.

        All attributes start as None or False. The engine is a "shell"
        until you call load() to bring a model into memory.
        """
        # Set all state to "empty" — no model loaded yet.
        self.model = None  # The actual model object
        self.tokenizer = None  # Tokenizer (HG only)
        self.model_path = None  # Where the model files live
        self.is_gguf = False  # Format detector
        self.config = None  # Optional config object

    def load(self, model_path: str, config=None):
        """Load a model from the given path.

        This is the main entry point for loading. It auto-detects the
        format (GGUF or HuggingFace) and dispatches to the right loader.

        Parameters
        ----------
        model_path : str
            Either a path to a .gguf file OR a path to a HuggingFace
            model directory (containing config.json, model.safetensors, etc.)
        config : object | None
            Optional config object with model settings (n_ctx, n_gpu_layers).

        Raises
        ------
        ValueError
            If the path is neither a .gguf file nor a directory.
        """
        # First, unload any existing model to free memory. This is
        # important: loading a second model without unloading the first
        # would use twice as much RAM/VRAM.
        self.unload()

        # Convert the string path to a Path object for easier manipulation.
        path = Path(model_path)

        # ── Branch 1: GGUF model ──
        # If the path is a file AND ends with .gguf, load it via llama-cpp.
        if path.is_file() and path.suffix == ".gguf":
            self._load_gguf(str(path), config)
        # ── Branch 2: HuggingFace model ──
        # If the path is a directory, assume it's a HuggingFace model.
        elif path.is_dir():
            self._load_hf(str(path))
        # ── Branch 3: Unsupported format ──
        else:
            raise ValueError(f"Unsupported model format: {model_path}")

        # Remember the model path so we can report it in status() and
        # include it in benchmark results.
        self.model_path = model_path

    def _load_gguf(self, gguf_path: str, config=None):
        """Load a GGUF model via llama-cpp-python.

        Internal method called by load() when a .gguf file is detected.

        Parameters
        ----------
        gguf_path : str
            Path to the .gguf file.
        config : object | None
            Optional config with n_gpu_layers and n_ctx overrides.
        """
        # Lazy import: only load llama_cpp when we actually need it.
        # This keeps the import time fast for code paths that don't use GGUF.
        from llama_cpp import Llama

        # ── Default settings ──
        # n_gpu_layers=99 is a magic number meaning "all layers on GPU
        # if they fit". llama.cpp layers 0..N across CPU and GPU.
        # n_ctx=8192 is the context window — how many tokens the model
        # can see/process at once.
        n_gpu_layers = 99
        n_ctx = 8192

        # If a config object is provided, override the defaults with
        # the user's preferred settings.
        if config:
            n_gpu_layers = config.model.n_gpu_layers
            n_ctx = config.model.n_ctx

        # ── Load the model ──
        # This is the expensive step — reads the GGUF file, allocates
        # memory, and prepares the model for inference.
        self.model = Llama(
            model_path=gguf_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            verbose=False,  # Don't spam llama.cpp logging to stdout
        )
        # Mark this as a GGUF model so we know which generator to use.
        self.is_gguf = True

    def _load_hf(self, model_path: str):
        """Load a HuggingFace model.

        Internal method called by load() when a directory is detected.

        Parameters
        ----------
        model_path : str
            Path to the HuggingFace model directory.
        """
        # Lazy imports — transformers and torch are heavy libraries.
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        # Load the tokenizer first. The tokenizer converts text to
        # token IDs (numbers) that the model can process.
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,  # Allow custom tokenizer code
        )

        # If the tokenizer has no padding token, use the EOS token.
        # This is needed because some models weren't trained with padding
        # but we need a pad_token for batched generation.
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load the model itself. torch.float16 uses 16-bit floats
        # (half precision) to save memory. device_map="auto" lets
        # transformers decide where to put layers (GPU/CPU).
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        # Mark this as a HuggingFace model.
        self.is_gguf = False

    def unload(self):
        """Unload the model and free memory.

        This is critical for running multiple models on the same hardware
        or for freeing memory when the engine is no longer needed.
        """
        import torch

        # Drop all references to the model objects. Python's garbage
        # collector will eventually free the memory, but we can be
        # more explicit by also clearing CUDA's cache.
        self.model = None
        self.tokenizer = None
        self.model_path = None
        self.is_gguf = False

        # If we have a CUDA GPU, clear its cache. This releases the
        # VRAM that was holding the model weights.
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def generate(
        self,
        messages: list,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop: list | None = None,
    ) -> dict:
        """Generate a response from a list of messages.

        The main entry point for generating text. Takes a list of
        chat messages and returns the model's response plus timing info.

        Parameters
        ----------
        messages : list[dict]
            Chat history. Each dict has "role" and "content".
            Example: [{"role": "user", "content": "Hello!"}]
        max_tokens : int
            Maximum number of tokens to generate. Default 1024.
        temperature : float
            Sampling temperature. 0=deterministic, 1=creative. Default 0.7.
        top_p : float
            Nucleus sampling threshold. 0.9 is a common default.
        stop : list[str] | None
            Optional list of stop strings. Generation stops when
            the model produces any of these strings.

        Returns
        -------
        dict
            {
                "response": str,     # The generated text
                "time_ms": float,    # How long it took (ms)
                "model": str,        # The model path
            }

        Raises
        ------
        RuntimeError
            If no model is loaded.
        """
        # ── Guard clause: ensure a model is loaded ──
        if self.model is None:
            raise RuntimeError("No model loaded")

        # ── Record start time for benchmarking ──
        start = time.time()

        # ── Dispatch to the right generator ──
        if self.is_gguf:
            response = self._generate_gguf(messages, max_tokens, temperature, top_p, stop)
        else:
            response = self._generate_hf(messages, max_tokens, temperature, top_p)

        # ── Calculate elapsed time ──
        # time.time() returns seconds since the epoch. We compute the
        # difference and convert to milliseconds (* 1000).
        elapsed = (time.time() - start) * 1000

        # ── Return structured result ──
        return {
            "response": response,
            "time_ms": round(elapsed, 1),  # Round to 1 decimal
            "model": self.model_path,
        }

    def _generate_hf(self, messages, max_tokens, temperature, top_p):
        """Generate text with a HuggingFace model.

        Internal method for HuggingFace-format models.

        Parameters
        ----------
        messages : list[dict]
            Chat messages.
        max_tokens : int
            Max tokens to generate.
        temperature : float
            Sampling temperature.
        top_p : float
            Nucleus sampling threshold.

        Returns
        -------
        str
            The generated text.
        """
        import torch

        # ── Format messages using the model's chat template ──
        # Each model has its own chat template (stored in the tokenizer).
        # apply_chat_template converts the message list into a single
        # formatted string that the model expects.
        # tokenize=False returns the string (not token IDs).
        # add_generation_prompt=True adds the marker that tells the model
        # to start its response.
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # ── Tokenize the text and move to the model's device (GPU/CPU) ──
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        # return_tensors="pt" returns PyTorch tensors (not lists).

        # ── Generate ──
        # torch.no_grad() disables gradient computation — we don't need
        # gradients for inference, only for training. This saves memory.
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,  # Unpack the input tensors as keyword arguments
                max_new_tokens=max_tokens,
                # max(temperature, 0.01) ensures temperature is at least 0.01
                # because temperature=0 causes errors in some HF models.
                temperature=max(temperature, 0.01),
                top_p=top_p,
                # do_sample=True if temperature > 0 (random sampling),
                # do_sample=False if temperature=0 (greedy decoding).
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        # ── Decode the output ──
        # outputs[0] is the full sequence (input + generated).
        # We slice off the input tokens to get just the generated part.
        generated = outputs[0][inputs["input_ids"].shape[-1] :]
        # Decode the token IDs back to text, skipping special tokens like <|endoftext|>.
        return self.tokenizer.decode(generated, skip_special_tokens=True)

    def _generate_gguf(self, messages, max_tokens, temperature, top_p, stop):
        """Generate text with a GGUF model using native chat completion.

        Internal method for GGUF models. Tries the native chat completion
        API first (which auto-detects the model's template) and falls
        back to manual prompt construction if that fails.

        Parameters
        ----------
        messages : list[dict]
            Chat messages.
        max_tokens : int
            Max tokens to generate.
        temperature : float
            Sampling temperature.
        top_p : float
            Nucleus sampling threshold.
        stop : list[str] | None
            Optional stop strings.

        Returns
        -------
        str
            The generated text.
        """
        try:
            # ── Try native chat completion ──
            # llama-cpp-python can auto-detect the model's chat template
            # from the GGUF metadata and use it. This is the cleanest
            # path because it respects the model's intended format.
            output = self.model.create_chat_completion(
                messages=messages,
                max_tokens=max_tokens,
                temperature=max(temperature, 0.01),
                top_p=top_p,
            )
            # Extract the response text from the OpenAI-style response structure.
            # output["choices"] is a list; we take the first (and usually only) choice.
            # output["choices"][0]["message"] is the assistant's message dict.
            # .get("content", "") returns the content or empty string if missing.
            return output["choices"][0]["message"].get("content", "")
        except Exception:  # noqa: BLE001
            # ── Fallback: manual prompt construction ──
            # If native chat completion fails (e.g., old GGUF without
            # template metadata), we build the Gemma-style prompt manually.
            # This is the format that Google's Gemma models use.
            prompt = ""
            for msg in messages:
                role = msg["role"]
                content = msg["content"]
                # System and user messages both go in the "user" turn
                # in Gemma's format (Gemma doesn't have a separate system role).
                if role == "system" or role == "user":
                    prompt += f"<start_of_turn>user\n{content}<end_of_turn>\n"
                elif role == "assistant":
                    prompt += f"<start_of_turn>model\n{content}<end_of_turn>\n"
            # Add the final assistant turn marker to prompt the model to respond.
            prompt += "<start_of_turn>model\n"

            # Call the model directly with the manual prompt.
            output = self.model(
                prompt,
                max_tokens=max_tokens,
                temperature=max(temperature, 0.01),
                top_p=top_p,
                # Stop at <end_of_turn> (Gemma's turn marker) or any user-specified stop.
                stop=["<end_of_turn>"] + (stop or []),
            )
            # Extract the response text. .strip() removes leading/trailing whitespace.
            return output["choices"][0]["text"].strip()

    def status(self) -> dict:
        """Get the engine's current status.

        Returns
        -------
        dict
            {
                "loaded": bool,       # True if a model is loaded
                "model_path": str,    # Path to the loaded model
                "is_gguf": bool,      # True if GGUF format
            }
        """
        return {
            "loaded": self.model is not None,
            "model_path": self.model_path,
            "is_gguf": self.is_gguf,
        }
