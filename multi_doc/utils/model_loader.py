import os
import sys
import json
from dotenv import load_dotenv, find_dotenv
from multi_doc.utils.config_loader import load_config
from multi_doc.logger import GLOBAL_LOGGER as log
from multi_doc.exception.custom_exception import DocumentPortalException


class ApiKeyManager:
    """
    Collects API keys from the ECS secret (JSON) or individual env vars.
    Keys are validated lazily: only the key for the provider you actually use is required.
    """

    KNOWN_KEYS = ["GROQ_API_KEY", "GOOGLE_API_KEY"]

    def __init__(self):
        self.api_keys = {}
        raw = os.getenv("apikeyliveclass")

        if raw:
            try:
                parsed = json.loads(raw)
                if not isinstance(parsed, dict):
                    raise ValueError("API_KEYS is not a valid JSON object")
                self.api_keys = parsed
                log.info("Loaded API keys from ECS secret")
            except Exception as e:
                log.warning("Failed to parse API keys as JSON", error=str(e))

        for key in self.KNOWN_KEYS:
            if not self.api_keys.get(key):
                env_val = os.getenv(key)
                if env_val:
                    self.api_keys[key] = env_val

        
        log.info("API keys available", keys=sorted(k for k, v in self.api_keys.items() if v))

    def get(self, key: str) -> str:
        val = self.api_keys.get(key)
        if not val:
            log.error("Missing required API key", key=key)
            raise KeyError(f"API key for {key} is missing. Add it to .env or your secrets store.")
        return val


class ModelLoader:
    """
    Loads embedding models and LLMs based on config and environment.
    Default setup: local HuggingFace embeddings + Groq LLM.
    """

    def __init__(self):
        if os.getenv("ENV", "local").lower() != "production":
            load_dotenv(find_dotenv(usecwd=True), override=True)
            log.info("Running in LOCAL mode: .env loaded")
        else:
            log.info("Running in PRODUCTION mode")

        self.api_key_mgr = ApiKeyManager()
        self.config = load_config()
        log.info("YAML config loaded", config_keys=list(self.config.keys()))

    def load_embeddings(self):
        """
        Load and return the embedding model selected in config.
        Providers: "huggingface" (local, no key) or "google" (needs GOOGLE_API_KEY).
        """
        try:
            emb_cfg = self.config["embedding_model"]
            provider = emb_cfg.get("provider", "huggingface")
            model_name = emb_cfg["model_name"]
            log.info("Loading embedding model", provider=provider, model=model_name)

            if provider == "huggingface":
                from langchain_huggingface import HuggingFaceEmbeddings

                return HuggingFaceEmbeddings(
                    model_name=model_name,
                    encode_kwargs={"normalize_embeddings": emb_cfg.get("normalize", True)},
                )

            if provider == "google":
                from langchain_google_genai import GoogleGenerativeAIEmbeddings

                return GoogleGenerativeAIEmbeddings(
                    model=model_name,
                    google_api_key=self.api_key_mgr.get("GOOGLE_API_KEY"),  
                )

            raise ValueError(f"Unsupported embedding provider: {provider}")

        except Exception as e:
            log.error("Error loading embedding model", error=str(e))
            raise DocumentPortalException("Failed to load embedding model", e) from e

    def load_llm(self):
        """
        Load and return the configured LLM.
        Selection order: LLM_PROVIDER env var > active_llm in config > "groq".
        """
        llm_block = self.config["llm"]
        provider_key = os.getenv("LLM_PROVIDER") or self.config.get("active_llm", "groq")

        if provider_key not in llm_block:
            log.error("LLM provider not found in config", provider=provider_key)
            raise ValueError(f"LLM provider '{provider_key}' not found in config")

        llm_config = llm_block[provider_key]
        provider = llm_config.get("provider")
        model_name = llm_config.get("model_name")
        temperature = llm_config.get("temperature", 0)
        max_tokens = llm_config.get("max_output_tokens", 2048)

        log.info("Loading LLM", provider=provider, model=model_name)

        try:
            if provider == "groq":
                from langchain_groq import ChatGroq

                return ChatGroq(
                    model=model_name,
                    api_key=self.api_key_mgr.get("GROQ_API_KEY"),  # type: ignore
                    temperature=temperature,
                    max_tokens=max_tokens,
                )

            if provider == "google":
                from langchain_google_genai import ChatGoogleGenerativeAI

                return ChatGoogleGenerativeAI(
                    model=model_name,
                    google_api_key=self.api_key_mgr.get("GOOGLE_API_KEY"),
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                )

        except Exception as e:
            log.error("Error loading LLM", provider=provider, error=str(e))
            raise DocumentPortalException("Failed to load LLM", e) from e

        log.error("Unsupported LLM provider", provider=provider)
        raise ValueError(f"Unsupported LLM provider: {provider}")


if __name__ == "__main__":
    loader = ModelLoader()

    
    embeddings = loader.load_embeddings()
    print(f"Embedding Model Loaded: {embeddings}")
    result = embeddings.embed_query("Hello, how are you?")
    print(f"Embedding dimensions: {len(result)}")

    
    llm = loader.load_llm()
    print(f"LLM Loaded: {llm}")
    result = llm.invoke("Hello, how are you?")
    print(f"LLM Result: {result.content}")