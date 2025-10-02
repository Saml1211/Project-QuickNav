"""
Secure credential management for QuickNav.

Uses OS-level credential storage:
- Windows: Windows Credential Manager
- macOS: Keychain
- Linux: python-keyring with best available backend
"""

import logging
import os
import sys
from typing import Dict, Optional, List

logger = logging.getLogger(__name__)

# Try to import keyring for cross-platform credential storage
try:
    import keyring
    import keyring.errors
    KEYRING_AVAILABLE = True
    logger.info("Keyring library available for secure credential storage")
except ImportError:
    keyring = None
    KEYRING_AVAILABLE = False
    logger.warning("Keyring library not available, falling back to environment variables")

# Service name for keyring
SERVICE_NAME = "QuickNav-AI-Keys"


class SecureCredentialManager:
    """Manages secure storage of API keys and other sensitive credentials."""
    
    def __init__(self):
        self.use_keyring = KEYRING_AVAILABLE
        if not self.use_keyring:
            logger.warning("Secure credential storage not available, using environment variables")
    
    def store_api_key(self, provider: str, api_key: str) -> bool:
        """
        Store an API key securely.
        
        Args:
            provider: Provider name (e.g., 'openai', 'anthropic')
            api_key: The API key to store
            
        Returns:
            True if stored successfully, False otherwise
        """
        if not api_key or not provider:
            return False
            
        provider = provider.lower()
        username = f"api_key_{provider}"
        
        try:
            if self.use_keyring:
                keyring.set_password(SERVICE_NAME, username, api_key)
                logger.info(f"API key for {provider} stored securely in keyring")
            else:
                # Fallback to environment variable
                env_key = f"{provider.upper()}_API_KEY"
                os.environ[env_key] = api_key
                logger.warning(f"API key for {provider} stored in environment variable {env_key}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to store API key for {provider}: {e}")
            return False
    
    def get_api_key(self, provider: str) -> Optional[str]:
        """
        Retrieve an API key.
        
        Args:
            provider: Provider name (e.g., 'openai', 'anthropic')
            
        Returns:
            API key if found, None otherwise
        """
        if not provider:
            return None
            
        provider = provider.lower()
        username = f"api_key_{provider}"
        
        try:
            if self.use_keyring:
                api_key = keyring.get_password(SERVICE_NAME, username)
                if api_key:
                    logger.debug(f"Retrieved API key for {provider} from keyring")
                    return api_key
            
            # Fallback to environment variable
            env_key = f"{provider.upper()}_API_KEY"
            api_key = os.environ.get(env_key)
            if api_key:
                logger.debug(f"Retrieved API key for {provider} from environment variable")
                return api_key
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to retrieve API key for {provider}: {e}")
            return None
    
    def delete_api_key(self, provider: str) -> bool:
        """
        Delete an API key.
        
        Args:
            provider: Provider name (e.g., 'openai', 'anthropic')
            
        Returns:
            True if deleted successfully, False otherwise
        """
        if not provider:
            return False
            
        provider = provider.lower()
        username = f"api_key_{provider}"
        
        try:
            if self.use_keyring:
                try:
                    keyring.delete_password(SERVICE_NAME, username)
                    logger.info(f"API key for {provider} deleted from keyring")
                except keyring.errors.PasswordDeleteError:
                    logger.info(f"No API key found for {provider} in keyring")
            
            # Also remove from environment variable if present
            env_key = f"{provider.upper()}_API_KEY"
            if env_key in os.environ:
                del os.environ[env_key]
                logger.info(f"API key for {provider} removed from environment")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete API key for {provider}: {e}")
            return False
    
    def list_stored_providers(self) -> List[str]:
        """
        List all providers with stored API keys.
        
        Returns:
            List of provider names
        """
        providers = []
        
        # Common AI providers to check
        common_providers = ['openai', 'anthropic', 'azure', 'google', 'cohere', 'huggingface']
        
        for provider in common_providers:
            if self.get_api_key(provider):
                providers.append(provider)
        
        return providers
    
    def migrate_from_plaintext(self, api_keys_dict: Dict[str, str]) -> bool:
        """
        Migrate API keys from plaintext storage to secure storage.
        
        Args:
            api_keys_dict: Dictionary of provider -> api_key mappings
            
        Returns:
            True if migration was successful for all keys
        """
        if not api_keys_dict:
            return True
            
        success = True
        migrated_count = 0
        
        for provider, api_key in api_keys_dict.items():
            if api_key and api_key.strip():
                if self.store_api_key(provider, api_key.strip()):
                    migrated_count += 1
                    logger.info(f"Migrated API key for {provider} to secure storage")
                else:
                    success = False
                    logger.error(f"Failed to migrate API key for {provider}")
        
        logger.info(f"Migration complete: {migrated_count}/{len(api_keys_dict)} API keys migrated")
        return success
    
    def validate_keyring_backend(self) -> Dict[str, any]:
        """
        Validate the keyring backend and return information about it.
        
        Returns:
            Dictionary with backend information
        """
        if not self.use_keyring:
            return {
                'available': False,
                'backend': None,
                'secure': False,
                'message': 'Keyring library not available'
            }
        
        try:
            backend = keyring.get_keyring()
            backend_name = backend.__class__.__name__
            
            # Test if we can store/retrieve a test credential
            test_key = "quicknav_test_credential"
            test_value = "test_value"
            
            keyring.set_password(SERVICE_NAME, test_key, test_value)
            retrieved = keyring.get_password(SERVICE_NAME, test_key)
            
            if retrieved == test_value:
                # Clean up test credential
                try:
                    keyring.delete_password(SERVICE_NAME, test_key)
                except:
                    pass
                
                return {
                    'available': True,
                    'backend': backend_name,
                    'secure': True,
                    'message': f'Using secure keyring backend: {backend_name}'
                }
            else:
                return {
                    'available': True,
                    'backend': backend_name,
                    'secure': False,
                    'message': f'Keyring backend {backend_name} not working properly'
                }
                
        except Exception as e:
            return {
                'available': False,
                'backend': None,
                'secure': False,
                'message': f'Keyring validation failed: {e}'
            }


# Global instance
_credential_manager = None


def get_credential_manager() -> SecureCredentialManager:
    """Get the global credential manager instance."""
    global _credential_manager
    if _credential_manager is None:
        _credential_manager = SecureCredentialManager()
    return _credential_manager