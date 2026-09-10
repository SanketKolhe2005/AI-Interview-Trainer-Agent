"""
llm/granite_client.py
=====================
IBM Granite LLM client via IBM watsonx.ai.

This module provides a clean ``GraniteClient`` class that wraps the
``ibm_watsonx_ai`` SDK so that all other modules in the project use a
single, well-tested interface for text generation.

IBM watsonx.ai SDK reference
-----------------------------
- ``ibm_watsonx_ai.Credentials``               : Authentication object
- ``ibm_watsonx_ai.foundation_models.ModelInference`` : Inference engine
- ``ibm_watsonx_ai.metanames.GenTextParamsMetaNames`` : Generation parameter keys

Required environment variables (set in .env — never hardcode)
-------------------------------------------------------------
  WATSONX_API_KEY     IBM Cloud API key
  WATSONX_PROJECT_ID  watsonx.ai project ID
  WATSONX_URL         Service endpoint (default: https://us-south.ml.cloud.ibm.com)
  GRANITE_MODEL_ID    Model ID (default: ibm/granite-3-8b-instruct)

Usage
-----
    from llm.granite_client import GraniteClient

    client = GraniteClient()
    response = client.generate("Explain recursion in simple terms.")
    print(response)

    # Verify credentials before launching the app:
    client.test_connection()
"""

import logging
from typing import Optional

from config.settings import settings

logger = logging.getLogger(__name__)


class GraniteClient:
    """Wrapper around IBM watsonx.ai ``ModelInference`` for IBM Granite models.

    The client is lazy-initialised: the SDK ``ModelInference`` object is not
    created until the first call to :meth:`generate` or :meth:`test_connection`.
    This avoids network calls at import time and makes unit-testing easy via
    mocking.

    Attributes
    ----------
    model_id : str
        The Granite model ID loaded from ``settings.GRANITE_MODEL_ID``.
    _model : ModelInference or None
        Lazy-loaded SDK inference object; ``None`` until first use.
    """

    def __init__(self) -> None:
        self.model_id: str = settings.GRANITE_MODEL_ID
        self._model = None  # lazy-loaded on first use

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_model(self):
        """Return the cached ``ModelInference`` instance, creating it if needed.

        Raises
        ------
        ValueError
            If ``WATSONX_API_KEY`` or ``WATSONX_PROJECT_ID`` are not set.
        RuntimeError
            If the SDK raises an authentication or initialisation error.
        """
        if self._model is not None:
            return self._model

        # Fail fast with a clear message if credentials are missing
        missing = []
        if not settings.WATSONX_API_KEY:
            missing.append("WATSONX_API_KEY")
        if not settings.WATSONX_PROJECT_ID:
            missing.append("WATSONX_PROJECT_ID")
        if missing:
            raise ValueError(
                f"Missing required IBM watsonx.ai credentials: {', '.join(missing)}.\n"
                "Copy .env.example to .env and fill in your IBM Cloud credentials.\n"
                "See README.md -> Setup & Installation for instructions."
            )

        try:
            from ibm_watsonx_ai import Credentials
            from ibm_watsonx_ai.foundation_models import ModelInference

            credentials = Credentials(
                url=settings.WATSONX_URL,
                api_key=settings.WATSONX_API_KEY,
            )

            self._model = ModelInference(
                model_id=self.model_id,
                credentials=credentials,
                project_id=settings.WATSONX_PROJECT_ID,
                # validate=False skips a metadata call on init — we validate
                # explicitly in test_connection() instead.
                validate=False,
            )

            logger.info(
                "GraniteClient initialised. Model: %s, URL: %s",
                self.model_id,
                settings.WATSONX_URL,
            )

        except ImportError as exc:
            raise RuntimeError(
                "ibm-watsonx-ai package is not installed. "
                "Run: pip install ibm-watsonx-ai"
            ) from exc
        except Exception as exc:
            raise RuntimeError(
                f"Failed to initialise IBM watsonx.ai ModelInference: {exc}"
            ) from exc

        return self._model

    @staticmethod
    def _build_params(max_new_tokens: int, temperature: float, top_p: float) -> dict:
        """Build the generation parameters dict using SDK MetaNames keys.

        Using ``GenTextParamsMetaNames`` constants (rather than raw strings)
        guards against future SDK key renames.
        """
        from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams

        return {
            GenParams.MAX_NEW_TOKENS: max_new_tokens,
            GenParams.TEMPERATURE: temperature,
            GenParams.TOP_P: top_p,
            GenParams.DECODING_METHOD: "sample",
            GenParams.REPETITION_PENALTY: 1.1,
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = settings.MAX_NEW_TOKENS,
        temperature: float = settings.TEMPERATURE,
        top_p: float = settings.TOP_P,
    ) -> str:
        """Generate a text completion for *prompt* using IBM Granite.

        Parameters
        ----------
        prompt : str
            The full prompt string to send to the model.  The caller is
            responsible for constructing a well-formatted prompt (see the
            ``prompts/`` directory for templates).
        max_new_tokens : int
            Maximum number of tokens the model may generate.  Defaults to
            ``settings.MAX_NEW_TOKENS`` (512).
        temperature : float
            Sampling temperature.  Higher values produce more varied output;
            lower values are more deterministic.  Defaults to
            ``settings.TEMPERATURE`` (0.7).
        top_p : float
            Nucleus-sampling probability mass.  Defaults to
            ``settings.TOP_P`` (0.9).

        Returns
        -------
        str
            The generated text (stripped of leading/trailing whitespace).
            Returns an empty string if the model returns no output.

        Raises
        ------
        ValueError
            If *prompt* is empty, or if required credentials are missing.
        RuntimeError
            On SDK initialisation failure, authentication errors, or
            unrecoverable API errors.
        Exception
            Quota, rate-limit, or timeout errors from the watsonx.ai API are
            logged and re-raised so the caller can decide how to handle them.
        """
        if not prompt or not prompt.strip():
            raise ValueError("generate() called with an empty prompt.")

        model = self._get_model()
        params = self._build_params(max_new_tokens, temperature, top_p)

        logger.debug(
            "Sending prompt to %s (max_new_tokens=%d, temperature=%.2f)",
            self.model_id,
            max_new_tokens,
            temperature,
        )

        try:
            response = model.generate(prompt=prompt, params=params)
        except Exception as exc:
            error_msg = str(exc)
            # Provide human-readable guidance for the most common failure modes
            if "401" in error_msg or "Unauthorized" in error_msg or "authentication" in error_msg.lower():
                raise RuntimeError(
                    "IBM watsonx.ai authentication failed (HTTP 401). "
                    "Verify WATSONX_API_KEY is valid and has not expired."
                ) from exc
            if "403" in error_msg or "Forbidden" in error_msg:
                raise RuntimeError(
                    "IBM watsonx.ai access denied (HTTP 403). "
                    "Verify WATSONX_PROJECT_ID is correct and your API key "
                    "has access to the project."
                ) from exc
            if "429" in error_msg or "quota" in error_msg.lower() or "rate limit" in error_msg.lower():
                raise RuntimeError(
                    "IBM watsonx.ai rate limit or quota exceeded (HTTP 429). "
                    "Wait before retrying, or check your IBM Cloud Lite usage limits."
                ) from exc
            if "timeout" in error_msg.lower() or "timed out" in error_msg.lower():
                raise RuntimeError(
                    "IBM watsonx.ai request timed out. "
                    "Check your network connection and try again."
                ) from exc
            # Unknown error — re-raise with original message
            logger.error("watsonx.ai generate() failed: %s", error_msg)
            raise

        # Extract generated text from the response dict
        generated_text = self._extract_text(response)
        logger.debug("Received %d chars from Granite.", len(generated_text))
        return generated_text

    @staticmethod
    def _extract_text(response: dict) -> str:
        """Pull the generated text string out of the watsonx.ai response dict.

        The SDK returns a nested dict of the form::

            {
                "results": [
                    {
                        "generated_text": "...",
                        "generated_token_count": N,
                        "stop_reason": "eos_token"
                    }
                ],
                "model_id": "...",
                ...
            }

        Parameters
        ----------
        response : dict
            Raw response object from ``ModelInference.generate()``.

        Returns
        -------
        str
            The generated text, stripped of surrounding whitespace.
        """
        try:
            results = response.get("results", [])
            if results:
                text = results[0].get("generated_text", "")
                return text.strip()
        except (AttributeError, IndexError, KeyError) as exc:
            logger.warning("Could not extract generated_text from response: %s", exc)
        return ""

    def test_connection(self) -> bool:
        """Send a minimal prompt to IBM Granite to verify credentials and connectivity.

        This method is safe to call at application startup.  It consumes only
        a handful of tokens.

        Returns
        -------
        bool
            ``True`` if the connection and generation succeeded.

        Raises
        ------
        ValueError
            If credentials are missing from the environment.
        RuntimeError
            If the connection or generation test fails.
        """
        logger.info("Testing IBM watsonx.ai connection (model: %s)...", self.model_id)
        test_prompt = "Reply with exactly one word: Ready"
        try:
            response = self.generate(
                prompt=test_prompt,
                max_new_tokens=10,
                temperature=0.0,
                top_p=1.0,
            )
            logger.info(
                "IBM watsonx.ai connection test PASSED. Model response: '%s'",
                response,
            )
            return True
        except Exception as exc:
            logger.error("IBM watsonx.ai connection test FAILED: %s", exc)
            raise
