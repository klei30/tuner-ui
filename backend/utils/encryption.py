"""
Encryption utilities for sensitive data like HuggingFace tokens.
"""

from cryptography.fernet import Fernet
import os
import logging

logger = logging.getLogger(__name__)

# Load encryption key from environment
ENCRYPTION_KEY = os.getenv("ENCRYPTION_KEY")

if not ENCRYPTION_KEY:
    # Generate a key if not set (for development only)
    logger.warning("ENCRYPTION_KEY not set. Generating temporary key. DO NOT USE IN PRODUCTION!")
    ENCRYPTION_KEY = Fernet.generate_key().decode()
    logger.warning("Set ENCRYPTION_KEY in backend/.env before saving HuggingFace tokens.")

try:
    cipher = Fernet(ENCRYPTION_KEY.encode())
except Exception as e:
    logger.error(f"Failed to initialize cipher: {e}")
    raise ValueError("Invalid ENCRYPTION_KEY. Generate a new one with: python -c 'from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())'")


def encrypt_token(token: str) -> str:
    """
    Encrypt a token (like HuggingFace API token).

    Args:
        token: Plain text token to encrypt

    Returns:
        Encrypted token as base64 string
    """
    if not token:
        raise ValueError("Token cannot be empty")

    try:
        encrypted = cipher.encrypt(token.encode())
        return encrypted.decode()
    except Exception as e:
        logger.error(f"Failed to encrypt token: {e}")
        raise


def decrypt_token(encrypted_token: str) -> str:
    """
    Decrypt an encrypted token.

    Args:
        encrypted_token: Encrypted token as base64 string

    Returns:
        Decrypted plain text token
    """
    if not encrypted_token:
        raise ValueError("Encrypted token cannot be empty")

    try:
        decrypted = cipher.decrypt(encrypted_token.encode())
        return decrypted.decode()
    except Exception as e:
        logger.error(f"Failed to decrypt token: {e}")
        raise ValueError("Invalid or corrupted encrypted token")
