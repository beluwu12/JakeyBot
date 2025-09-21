import aiohttp
import logging
import openai
from os import environ
from azure.storage.blob.aio import BlobServiceClient
from discord.ext import bridge
from google import genai


class ServicesInitBot(bridge.Bot):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    async def start_services(self):
        # Gemini API Client
        self._gemini_api_client = genai.Client(api_key=environ.get("GEMINI_API_KEY"))
        logging.info("Gemini API client initialized successfully")

        # OpenAI client (solo si hay API key)
        _base_url = environ.get("OPENAI_API_ENDPOINT")
        if environ.get("OPENAI_USE_AZURE_OPENAI") and _base_url:
            _default_query = {"api-version": "preview"}
            logging.info("Using Azure OpenAI endpoint for OpenAI models... Using nextgen API")
        else:
            _default_query = None

        api_key = environ.get("OPENAI_API_KEY")
        if api_key:
            self._openai_client = openai.AsyncOpenAI(
                api_key=api_key,
                base_url=_base_url,
                default_query=_default_query
            )
            logging.info("OpenAI client initialized successfully")
        else:
            self._openai_client = None
            logging.warning("OPENAI_API_KEY not set, skipping OpenAI client initialization")

        # OpenRouter
        if environ.get("OPENROUTER_API_KEY"):
            self._openai_client_openrouter = openai.AsyncOpenAI(
                api_key=environ.get("OPENROUTER_API_KEY"),
                base_url="https://openrouter.ai/api/v1"
            )
            logging.info("OpenAI client for OpenRouter initialized successfully")

        # Groq
        if environ.get("GROQ_API_KEY"):
            self._openai_client_groq = openai.AsyncOpenAI(
                api_key=environ.get("GROQ_API_KEY"),
                base_url="https://api.groq.com/openai/v1"
            )
            logging.info("OpenAI client for Groq initialized successfully")

        # aiohttp session
        self._aiohttp_main_client_session = aiohttp.ClientSession()
        logging.info("aiohttp client session initialized successfully")

        # Azure Blob Storage Client
        try:
            conn_str = environ.get("AZURE_STORAGE_CONNECTION_STRING")
            if conn_str:
                self._azure_blob_service_client = BlobServiceClient.from_connection_string(conn_str)
                logging.info("Azure Blob Storage client initialized successfully")
            else:
                logging.warning("AZURE_STORAGE_CONNECTION_STRING not set, skipping Azure Blob Storage client")
        except Exception as e:
            logging.error("Failed to initialize Azure Blob Storage client: %s, skipping....", e)

    async def stop_services(self):
        # Close aiohttp client sessions
        if hasattr(self, "_aiohttp_main_client_session"):
            await self._aiohttp_main_client_session.close()
            logging.info("aiohttp client session closed successfully")

        # Close Azure Blob Storage client
        if hasattr(self, "_azure_blob_service_client"):
            try:
                await self._azure_blob_service_client.close()
                logging.info("Azure Blob Storage client closed successfully")
            except Exception as e:
                logging.error("Failed to close Azure Blob Storage client: %s", e)
