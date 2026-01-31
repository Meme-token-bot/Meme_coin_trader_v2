"""
AWS SECRETS MANAGER INTEGRATION
================================

Securely retrieves all API keys and credentials from AWS Secrets Manager.
Falls back to .env for local development.

SECRETS STRUCTURE:
- prod/solana-bot/keys      : All API keys (HELIUS_KEY, TELEGRAM_BOT_TOKEN, etc.)
- prod/solana-bot/hot-wallet: Solana wallet private key

USAGE:
    from core.secrets_manager import get_secrets, get_secret, init_secrets
    
    # Initialize at startup (sets environment variables)
    init_secrets()
    
    # Then use normally
    helius_key = os.getenv('HELIUS_KEY')
    
    # Or get directly
    private_key = get_secret('SOLANA_PRIVATE_KEY')

AWS CONFIGURATION:
    Ensure your EC2 instance has an IAM role with secretsmanager:GetSecretValue
    permission for the specified secrets.

Author: Trading Bot System
"""

import os
import json
import logging
from typing import Dict, Optional

logger = logging.getLogger("SecretsManager")

# =============================================================================
# CONFIGURATION
# =============================================================================

# AWS Secrets Manager secret names (matching your AWS setup)
SECRETS_CONFIG = {
    'keys': 'prod/solana-bot/keys',           # API keys and tokens
    'wallet': 'prod/solana-bot/hot-wallet',   # Solana private key
}

# AWS Region (us-west-2 based on your ARN)
AWS_REGION = os.getenv('AWS_REGION', 'us-west-2')

# Set to True to use AWS Secrets Manager, False for .env fallback
USE_AWS_SECRETS = os.getenv('USE_AWS_SECRETS', 'true').lower() == 'true'


# =============================================================================
# AWS SECRETS MANAGER CLIENT
# =============================================================================

def _get_boto3_client():
    """Get boto3 secrets manager client"""
    try:
        import boto3
        from botocore.config import Config
        
        config = Config(
            region_name=AWS_REGION,
            retries={'max_attempts': 3, 'mode': 'standard'}
        )
        
        return boto3.client('secretsmanager', config=config)
    except ImportError:
        logger.warning("boto3 not installed. Run: pip install boto3 --break-system-packages")
        return None
    except Exception as e:
        logger.error(f"Failed to create boto3 client: {e}")
        return None


def _fetch_secret_from_aws(secret_name: str) -> Optional[Dict]:
    """Fetch a secret from AWS Secrets Manager"""
    client = _get_boto3_client()
    
    if not client:
        return None
    
    try:
        response = client.get_secret_value(SecretId=secret_name)
        
        # Parse the secret string (JSON format)
        if 'SecretString' in response:
            secret_data = json.loads(response['SecretString'])
            logger.info(f"✅ Loaded secret: {secret_name}")
            return secret_data
        else:
            # Binary secret
            logger.warning(f"Secret {secret_name} is binary, not supported")
            return None
            
    except client.exceptions.ResourceNotFoundException:
        logger.error(f"❌ Secret not found: {secret_name}")
        return None
    except client.exceptions.AccessDeniedException:
        logger.error(f"❌ Access denied to secret: {secret_name}")
        return None
    except Exception as e:
        logger.error(f"❌ Error fetching secret {secret_name}: {e}")
        return None


# =============================================================================
# SECRETS LOADER
# =============================================================================

class SecretsLoader:
    """
    Loads and caches secrets from AWS Secrets Manager.
    Falls back to environment variables for local development.
    """
    
    def __init__(self):
        self._secrets: Dict[str, str] = {}
        self._loaded = False
    
    def load(self, force_reload: bool = False) -> Dict[str, str]:
        """
        Load all secrets from AWS Secrets Manager.
        
        Returns dict with all secrets merged together.
        """
        if self._loaded and not force_reload:
            return self._secrets
        
        self._secrets = {}
        
        if USE_AWS_SECRETS:
            logger.info("🔐 Loading secrets from AWS Secrets Manager...")
            
            # Load API keys and tokens
            keys_secret = _fetch_secret_from_aws(SECRETS_CONFIG['keys'])
            if keys_secret:
                self._secrets.update(keys_secret)
            
            # Load wallet private key
            wallet_secret = _fetch_secret_from_aws(SECRETS_CONFIG['wallet'])
            if wallet_secret:
                # The wallet secret might have the key under different names
                if 'SOLANA_PRIVATE_KEY' in wallet_secret:
                    self._secrets['SOLANA_PRIVATE_KEY'] = wallet_secret['SOLANA_PRIVATE_KEY']
                elif 'private_key' in wallet_secret:
                    self._secrets['SOLANA_PRIVATE_KEY'] = wallet_secret['private_key']
                elif 'PRIVATE_KEY' in wallet_secret:
                    self._secrets['SOLANA_PRIVATE_KEY'] = wallet_secret['PRIVATE_KEY']
                else:
                    # If it's a single value secret, use the first value
                    for key, value in wallet_secret.items():
                        self._secrets['SOLANA_PRIVATE_KEY'] = value
                        break
            
            if self._secrets:
                logger.info(f"✅ Loaded {len(self._secrets)} secrets from AWS")
            else:
                logger.warning("⚠️ No secrets loaded from AWS, falling back to .env")
                self._load_from_env()
        else:
            logger.info("📁 Loading secrets from environment variables (.env)")
            self._load_from_env()
        
        self._loaded = True
        return self._secrets
    
    def _load_from_env(self):
        """Load secrets from environment variables (fallback)"""
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass
        
        # Map of secret names we need
        env_vars = [
            'HELIUS_KEY',
            'HELIUS_WEBHOOK_ID',
            'HELIUS_DISCOVERY_KEY',
            'TELEGRAM_BOT_TOKEN',
            'TELEGRAM_CHAT_ID',
            'WEBHOOK_URL',
            'ANTHROPIC_API_KEY',
            'BIRDEYE_API_KEY',
            'KRAKEN_API',
            'KRAKEN_SECRET',
            'SOLANA_PRIVATE_KEY',
            'SOLANA_PUBLIC_KEY',
            'ENABLE_LIVE_TRADING',
            'POSITION_SIZE_SOL',
            'MAX_OPEN_POSITIONS',
            'MAX_DAILY_LOSS_SOL',
        ]
        
        for var in env_vars:
            value = os.getenv(var)
            if value:
                self._secrets[var] = value
        
        logger.info(f"📁 Loaded {len(self._secrets)} values from environment")
    
    def get(self, key: str, default: str = None) -> Optional[str]:
        """Get a specific secret value"""
        if not self._loaded:
            self.load()
        
        return self._secrets.get(key, default)
    
    def get_all(self) -> Dict[str, str]:
        """Get all secrets"""
        if not self._loaded:
            self.load()
        
        return self._secrets.copy()
    
    def set_in_environment(self):
        """
        Set all secrets as environment variables.
        Useful for compatibility with existing code that uses os.getenv().
        """
        if not self._loaded:
            self.load()
        
        for key, value in self._secrets.items():
            os.environ[key] = value
        
        logger.info(f"🔄 Set {len(self._secrets)} secrets in environment")


# =============================================================================
# GLOBAL INSTANCE AND HELPER FUNCTIONS
# =============================================================================

_secrets_loader = SecretsLoader()


def get_secrets(force_reload: bool = False) -> Dict[str, str]:
    """
    Get all secrets from AWS Secrets Manager.
    
    Returns:
        Dict with all secret key-value pairs
    """
    return _secrets_loader.load(force_reload)


def get_secret(key: str, default: str = None) -> Optional[str]:
    """
    Get a specific secret value.
    
    Args:
        key: Secret key name (e.g., 'HELIUS_KEY')
        default: Default value if not found
    
    Returns:
        Secret value or default
    """
    return _secrets_loader.get(key, default)


def init_secrets():
    """
    Initialize secrets and set them in the environment.
    Call this at the start of your application.
    
    Usage:
        from core.secrets_manager import init_secrets
        init_secrets()  # Now os.getenv() will work for all secrets
    """
    _secrets_loader.load()
    _secrets_loader.set_in_environment()
    return _secrets_loader.get_all()


# =============================================================================
# VALIDATION
# =============================================================================

def validate_secrets() -> Dict[str, bool]:
    """
    Validate that all required secrets are present.
    
    Returns:
        Dict mapping secret name to whether it's present
    """
    required_secrets = [
        'HELIUS_KEY',
        'TELEGRAM_BOT_TOKEN',
        'TELEGRAM_CHAT_ID',
    ]
    
    optional_secrets = [
        'ANTHROPIC_API_KEY',
        'BIRDEYE_API_KEY',
        'SOLANA_PRIVATE_KEY',
        'KRAKEN_API',
        'KRAKEN_SECRET',
        'WEBHOOK_URL',
    ]
    
    secrets = get_secrets()
    
    results = {}
    
    print("\n🔐 SECRETS VALIDATION")
    print("=" * 50)
    
    print("\nRequired:")
    all_required_present = True
    for key in required_secrets:
        present = key in secrets and secrets[key]
        results[key] = present
        status = "✅" if present else "❌ MISSING"
        if not present:
            all_required_present = False
        print(f"  {status} {key}")
    
    print("\nOptional:")
    for key in optional_secrets:
        present = key in secrets and secrets[key]
        results[key] = present
        status = "✅" if present else "⚠️ Not set"
        if present:
            masked = secrets[key][:8] + "..." if len(secrets[key]) > 8 else "***"
            print(f"  {status} {key}: {masked}")
        else:
            print(f"  {status} {key}")
    
    print("\n" + "=" * 50)
    
    if all_required_present:
        print("✅ All required secrets present")
    else:
        print("❌ Missing required secrets!")
    
    return results


# =============================================================================
# CLI
# =============================================================================

def main():
    """CLI for testing secrets manager"""
    import argparse
    
    parser = argparse.ArgumentParser(description="AWS Secrets Manager Integration")
    parser.add_argument('command', choices=['validate', 'list', 'test'],
                       help='Command to run')
    parser.add_argument('--local', action='store_true',
                       help='Use local .env instead of AWS')
    
    args = parser.parse_args()
    
    if args.local:
        global USE_AWS_SECRETS
        USE_AWS_SECRETS = False
    
    if args.command == 'validate':
        validate_secrets()
    
    elif args.command == 'list':
        secrets = get_secrets()
        print("\n🔐 LOADED SECRETS")
        print("=" * 50)
        for key in sorted(secrets.keys()):
            value = secrets[key]
            if len(value) > 12:
                masked = value[:4] + "..." + value[-4:]
            else:
                masked = "***"
            print(f"  {key}: {masked}")
        print(f"\nTotal: {len(secrets)} secrets")
    
    elif args.command == 'test':
        print("\n🧪 TESTING AWS SECRETS MANAGER CONNECTION")
        print("=" * 50)
        
        try:
            import boto3
            print("✅ boto3 installed")
        except ImportError:
            print("❌ boto3 not installed")
            print("   Run: pip install boto3 --break-system-packages")
            return
        
        client = _get_boto3_client()
        if client:
            print("✅ AWS client created")
        else:
            print("❌ Failed to create AWS client")
            return
        
        print(f"\nTesting secret: {SECRETS_CONFIG['keys']}")
        keys = _fetch_secret_from_aws(SECRETS_CONFIG['keys'])
        if keys:
            print(f"✅ Successfully loaded {len(keys)} keys")
        else:
            print("❌ Failed to load keys secret")
        
        print(f"\nTesting secret: {SECRETS_CONFIG['wallet']}")
        wallet = _fetch_secret_from_aws(SECRETS_CONFIG['wallet'])
        if wallet:
            print("✅ Successfully loaded wallet secret")
        else:
            print("❌ Failed to load wallet secret")


if __name__ == "__main__":
    main()
