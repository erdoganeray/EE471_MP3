"""The three AI models behind the app, each wrapped in a small class.

  ImageGenerator  text -> image   stabilityai/sd-turbo                (diffusers)
  Chatbot         chat replies    HuggingFaceTB/SmolLM2-360M-Instruct (transformers)
  SpeechToText    audio -> text   openai/whisper-large-v3-turbo       (transformers)

Every model is loaded once, on first use (or by warm_up() at start-up), and then reused.
A lock per model keeps two requests from using the same model at the same time.
Devices can be changed with environment variables, e.g. ASR_DEVICE=cuda.
"""
from __future__ import annotations

import base64
import io
import os
import threading

import torch

HAS_GPU = torch.cuda.is_available()


def pick_device(env_name: str, default: str) -> str:
    device = os.getenv(env_name, default)
    return device if (device == "cpu" or HAS_GPU) else "cpu"


class LazyModel:
    """Loads the model the first time it is needed; one request at a time per model."""

    name = ""

    def __init__(self) -> None:
        self._model = None
        self._lock = threading.Lock()

    def load(self):
        raise NotImplementedError

    @property
    def model(self):
        if self._model is None:
            self._model = self.load()
        return self._model

    @property
    def ready(self) -> bool:
        return self._model is not None


class ImageGenerator(LazyModel):
    name = "stabilityai/sd-turbo"

    def __init__(self) -> None:
        super().__init__()
        self.device = pick_device("IMAGE_DEVICE", "cuda")

    def load(self):
        from diffusers import AutoPipelineForText2Image

        gpu = self.device == "cuda"
        pipe = AutoPipelineForText2Image.from_pretrained(
            self.name, torch_dtype=torch.float16 if gpu else torch.float32, variant="fp16")
        if gpu:
            # keeps only the part that is running (text encoder / UNet / VAE) on the 4 GB GPU
            pipe.enable_model_cpu_offload()
        pipe.set_progress_bar_config(disable=True)
        return pipe

    def generate(self, prompt: str, steps: int = 2, seed: int | None = None) -> str:
        """Returns the image as a base64 PNG data URL."""
        generator = torch.Generator("cpu").manual_seed(seed) if seed is not None else None
        with self._lock:
            # SD-Turbo is distilled for 1-4 steps and is trained without classifier-free guidance
            image = self.model(prompt=prompt, num_inference_steps=steps, guidance_scale=0.0,
                               width=512, height=512, generator=generator).images[0]
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


class Chatbot(LazyModel):
    name = "HuggingFaceTB/SmolLM2-360M-Instruct"
    system_prompt = (
        "You are Dream Canvas, a friendly assistant in an image generation app. Keep answers short, "
        "at most 2 sentences. When the user asks for an image idea or a prompt, answer with only the prompt: "
        "one vivid sentence under 35 words that names the subject, setting, art style and lighting. "
        "No labels, no lists, no explanations."
    )
    # one example exchange (few-shot): a 360M model copies the format much better than it follows rules
    example = [
        {"role": "user", "content": "Give me a prompt for a nice image."},
        {"role": "assistant", "content": "A lone lighthouse on a stormy cliff at sunset, crashing waves, "
                                         "dramatic orange sky, oil painting style, cinematic lighting."},
    ]
    max_turns = 10  # only the last messages are sent to the model, the small context stays focused

    def __init__(self) -> None:
        super().__init__()
        # 360M parameters: about 5 s per reply on the CPU, the same as on the GPU, so keep the VRAM free
        self.device = pick_device("CHAT_DEVICE", "cpu")

    def load(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(self.name)
        model = AutoModelForCausalLM.from_pretrained(
            self.name, torch_dtype=torch.float16 if self.device == "cuda" else torch.float32).to(self.device)
        return tokenizer, model

    def reply(self, history: list[dict]) -> str:
        """history: [{"role": "user"|"assistant", "content": "..."}, ...] ending with the user's message."""
        tokenizer, model = self.model
        messages = [{"role": "system", "content": self.system_prompt}] + self.example + history[-self.max_turns:]
        input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True,
                                                  return_tensors="pt").to(self.device)
        with self._lock, torch.no_grad():
            output = model.generate(input_ids, attention_mask=torch.ones_like(input_ids), max_new_tokens=100,
                                    do_sample=True, temperature=0.6, top_p=0.9, repetition_penalty=1.1,
                                    pad_token_id=tokenizer.eos_token_id)
        return tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True).strip()


class SpeechToText(LazyModel):
    name = "openai/whisper-large-v3-turbo"

    def __init__(self) -> None:
        super().__init__()
        # GPU fp16: ~0.5 s for 13 s of audio (CPU: ~11 s), uses ~1.7 GB of VRAM
        self.device = pick_device("ASR_DEVICE", "cuda")

    def load(self):
        from transformers import pipeline

        gpu = self.device == "cuda"
        return pipeline("automatic-speech-recognition", model=self.name, device=0 if gpu else -1,
                        torch_dtype=torch.float16 if gpu else torch.float32)

    def transcribe(self, audio_path: str, language: str | None = "english") -> str:
        kwargs = {"generate_kwargs": {"language": language, "task": "transcribe"}} if language else {}
        with self._lock:
            return self.model(audio_path, **kwargs)["text"].strip()


class AIModels:
    """All three models together; the Flask app only talks to this object."""

    def __init__(self) -> None:
        self.image = ImageGenerator()
        self.chat = Chatbot()
        self.speech = SpeechToText()

    def warm_up(self) -> None:
        """Load every model in a background thread so the first click is fast."""
        def _load():
            for m in (self.chat, self.speech, self.image):
                try:
                    m.model
                    print(f"[warm-up] {m.name} ready on {m.device}", flush=True)
                except Exception as e:  # keep the app running, the error shows on first use
                    print(f"[warm-up] {m.name} failed: {e}", flush=True)
        threading.Thread(target=_load, daemon=True).start()

    def status(self) -> dict:
        return {key: {"model": m.name, "device": m.device, "ready": m.ready}
                for key, m in (("image", self.image), ("chat", self.chat), ("speech", self.speech))}
