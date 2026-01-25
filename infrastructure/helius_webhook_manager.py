"""
Helius Webhook Manager
Automatically manages webhook subscriptions for discovered wallets

NO CREDITS USED - These are management API calls, not RPC calls!
"""

import requests
import os
from typing import List, Dict, Optional
from datetime import datetime


class HeliusWebhookManager:
    """
    Manages Helius webhooks - adding/removing wallet addresses
    
    IMPORTANT: These API calls are FREE - they don't use RPC credits!
    """
    
    def __init__(self, api_key: str, webhook_id: str = None):
        """
        Initialize webhook manager
        
        Args:
            api_key: Your Helius API key
            webhook_id: Your existing webhook ID (get from Helius dashboard)
        """
        self.api_key = api_key
        self.webhook_id = webhook_id
        self.base_url = "https://api.helius.xyz/v0/webhooks"
        
        print(f"✅ Helius Webhook Manager initialized")
        if webhook_id:
            print(f"   Webhook ID: {webhook_id}")
        else:
            print(f"   ⚠️  No webhook ID provided - will need to create or specify one")
    
    def list_webhooks(self) -> List[Dict]:
        """
        List all your webhooks
        
        Returns: List of webhook objects
        """
        try:
            url = f"{self.base_url}?api-key={self.api_key}"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            webhooks = response.json()
            print(f"\n📋 Found {len(webhooks)} webhook(s):")
            for webhook in webhooks:
                print(f"   ID: {webhook.get('webhookID')}")
                print(f"   URL: {webhook.get('webhookURL')}")
                print(f"   Type: {webhook.get('webhookType')}")
                print(f"   Addresses: {len(webhook.get('accountAddresses', []))}")
                print()
            
            return webhooks
        
        except Exception as e:
            print(f"❌ Error listing webhooks: {e}")
            return []
    
    def get_webhook(self, webhook_id: str = None) -> Optional[Dict]:
        """
        Get details of a specific webhook
        
        Args:
            webhook_id: Webhook ID (uses self.webhook_id if not provided)
        
        Returns: Webhook object
        """
        webhook_id = webhook_id or self.webhook_id
        if not webhook_id:
            print("❌ No webhook ID provided")
            return None
        
        try:
            url = f"{self.base_url}/{webhook_id}?api-key={self.api_key}"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            webhook = response.json()
            print(f"\n📌 Webhook Details:")
            print(f"   ID: {webhook.get('webhookID')}")
            print(f"   URL: {webhook.get('webhookURL')}")
            print(f"   Type: {webhook.get('webhookType')}")
            print(f"   Addresses tracked: {len(webhook.get('accountAddresses', []))}")
            
            return webhook
        
        except Exception as e:
            print(f"❌ Error getting webhook: {e}")
            return None
    
    def add_addresses(self, addresses: List[str], webhook_id: str = None) -> bool:
        """
        Add wallet addresses to webhook
        
        Args:
            addresses: List of wallet addresses to add
            webhook_id: Webhook ID (uses self.webhook_id if not provided)
        
        Returns: True if successful
        
        IMPORTANT: This does NOT cost any RPC credits!
        """
        webhook_id = webhook_id or self.webhook_id
        if not webhook_id:
            print("❌ No webhook ID provided")
            return False
        
        if not addresses:
            print("⚠️  No addresses to add")
            return True
        
        try:
            url = f"{self.base_url}/{webhook_id}?api-key={self.api_key}"
            
            # Get current webhook to preserve settings
            current = self.get_webhook(webhook_id)
            if not current:
                return False
            
            # Get existing addresses
            existing_addresses = set(current.get('accountAddresses', []))
            
            # Filter out addresses that are already tracked
            new_addresses = [addr for addr in addresses if addr not in existing_addresses]
            
            if not new_addresses:
                print(f"ℹ️  All {len(addresses)} address(es) already tracked")
                return True
            
            # Combine with existing addresses
            all_addresses = list(existing_addresses) + new_addresses
            
            # Update webhook
            payload = {
                "webhookURL": current.get('webhookURL'),
                "accountAddresses": all_addresses,
                "webhookType": current.get('webhookType', 'enhanced'),
                "transactionTypes": current.get('transactionTypes', ['SWAP'])
            }
            
            response = requests.put(url, json=payload, timeout=10)
            response.raise_for_status()
            
            print(f"\n✅ Successfully added {len(new_addresses)} address(es) to webhook")
            print(f"   Total addresses now tracked: {len(all_addresses)}")
            
            for addr in new_addresses:
                print(f"   + {addr[:8]}...")
            
            return True
        
        except Exception as e:
            print(f"❌ Error adding addresses: {e}")
            return False
    
    def remove_addresses(self, addresses: List[str], webhook_id: str = None) -> bool:
        """
        Remove wallet addresses from webhook
        
        Args:
            addresses: List of wallet addresses to remove
            webhook_id: Webhook ID (uses self.webhook_id if not provided)
        
        Returns: True if successful
        """
        webhook_id = webhook_id or self.webhook_id
        if not webhook_id:
            print("❌ No webhook ID provided")
            return False
        
        try:
            url = f"{self.base_url}/{webhook_id}?api-key={self.api_key}"
            
            # Get current webhook
            current = self.get_webhook(webhook_id)
            if not current:
                return False
            
            # Remove specified addresses
            existing_addresses = set(current.get('accountAddresses', []))
            remaining_addresses = [addr for addr in existing_addresses if addr not in addresses]
            
            # Update webhook
            payload = {
                "webhookURL": current.get('webhookURL'),
                "accountAddresses": remaining_addresses,
                "webhookType": current.get('webhookType', 'enhanced'),
                "transactionTypes": current.get('transactionTypes', ['SWAP'])
            }
            
            response = requests.put(url, json=payload, timeout=10)
            response.raise_for_status()
            
            removed_count = len(existing_addresses) - len(remaining_addresses)
            print(f"\n✅ Successfully removed {removed_count} address(es)")
            print(f"   Total addresses now tracked: {len(remaining_addresses)}")
            
            return True
        
        except Exception as e:
            print(f"❌ Error removing addresses: {e}")
            return False
    
    def create_webhook(self, webhook_url: str, addresses: List[str] = None,
                      transaction_types: List[str] = None) -> Optional[str]:
        """
        Create a new webhook
        
        Args:
            webhook_url: Your server's webhook endpoint URL
            addresses: Initial wallet addresses to track
            transaction_types: Types of transactions to track (default: ['SWAP'])
        
        Returns: Webhook ID if successful
        """
        addresses = addresses or []
        transaction_types = transaction_types or ['SWAP']
        
        try:
            url = f"{self.base_url}?api-key={self.api_key}"
            
            payload = {
                "webhookURL": webhook_url,
                "accountAddresses": addresses,
                "webhookType": "enhanced",
                "transactionTypes": transaction_types
            }
            
            response = requests.post(url, json=payload, timeout=10)
            response.raise_for_status()
            
            result = response.json()
            webhook_id = result.get('webhookID')
            
            print(f"\n✅ Webhook created successfully!")
            print(f"   Webhook ID: {webhook_id}")
            print(f"   URL: {webhook_url}")
            print(f"   Addresses: {len(addresses)}")
            
            self.webhook_id = webhook_id
            return webhook_id
        
        except Exception as e:
            print(f"❌ Error creating webhook: {e}")
            return None
    
    def delete_webhook(self, webhook_id: str = None) -> bool:
        """
        Delete a webhook
        
        Args:
            webhook_id: Webhook ID (uses self.webhook_id if not provided)
        
        Returns: True if successful
        """
        webhook_id = webhook_id or self.webhook_id
        if not webhook_id:
            print("❌ No webhook ID provided")
            return False
        
        try:
            url = f"{self.base_url}/{webhook_id}?api-key={self.api_key}"
            response = requests.delete(url, timeout=10)
            response.raise_for_status()
            
            print(f"\n✅ Webhook {webhook_id} deleted")
            return True
        
        except Exception as e:
            print(f"❌ Error deleting webhook: {e}")
            return False
    
    def sync_with_database(self, db, webhook_id: str = None) -> bool:
        """
        Sync webhook with all verified wallets in database
        
        Args:
            db: DatabaseV2 instance
            webhook_id: Webhook ID (uses self.webhook_id if not provided)
        
        Returns: True if successful
        """
        webhook_id = webhook_id or self.webhook_id
        if not webhook_id:
            print("❌ No webhook ID provided")
            return False
        
        print(f"\n🔄 Syncing webhook with database...")
        
        # Get all verified wallets from database
        wallets = db.get_all_verified_wallets()
        addresses = [w['address'] for w in wallets]
        
        print(f"   Found {len(addresses)} wallet(s) in database")
        
        # Get current webhook addresses
        current = self.get_webhook(webhook_id)
        if not current:
            return False
        
        current_addresses = set(current.get('accountAddresses', []))
        
        # Find addresses that need to be added
        to_add = [addr for addr in addresses if addr not in current_addresses]
        
        if to_add:
            print(f"   Adding {len(to_add)} missing address(es)...")
            return self.add_addresses(to_add, webhook_id)
        else:
            print(f"   ✅ Webhook already in sync!")
            return True


# =============================================================================
# INTEGRATION HELPERS
# =============================================================================

def auto_add_discovered_wallet(webhook_manager: HeliusWebhookManager, 
                               wallet_address: str) -> bool:
    """
    Automatically add a newly discovered wallet to the webhook
    
    Call this function after verifying a new wallet during discovery.
    
    Args:
        webhook_manager: HeliusWebhookManager instance
        wallet_address: Wallet address to add
    
    Returns: True if successful
    """
    if not webhook_manager.webhook_id:
        print("⚠️  Webhook manager not configured - skipping auto-add")
        return False
    
    return webhook_manager.add_addresses([wallet_address])


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    import sys
    from dotenv import load_dotenv
    load_dotenv()
    
    print("\n" + "="*70)
    print("🔧 HELIUS WEBHOOK MANAGER")
    print("="*70)
    
    api_key = os.getenv('HELIUS_KEY')
    webhook_id = os.getenv('HELIUS_WEBHOOK_ID')  # Add this to your .env
    
    if not api_key:
        print("❌ HELIUS_KEY not found in environment")
        sys.exit(1)
    
    manager = HeliusWebhookManager(api_key, webhook_id)
    
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == 'list':
            manager.list_webhooks()
        
        elif command == 'get':
            if not webhook_id:
                print("❌ HELIUS_WEBHOOK_ID not set in .env")
            else:
                manager.get_webhook()
        
        elif command == 'sync':
            if not webhook_id:
                print("❌ HELIUS_WEBHOOK_ID not set in .env")
            else:
                from core.database_v2 import DatabaseV2
                db = DatabaseV2()
                manager.sync_with_database(db)
        
        elif command == 'add':
            if not webhook_id:
                print("❌ HELIUS_WEBHOOK_ID not set in .env")
            elif len(sys.argv) < 3:
                print("❌ Usage: python helius_webhook_manager.py add <wallet_address>")
            else:
                addresses = sys.argv[2:]
                manager.add_addresses(addresses)
        
        elif command == 'help':
            print("""
Usage:
  python helius_webhook_manager.py list          - List all webhooks
  python helius_webhook_manager.py get           - Get current webhook details
  python helius_webhook_manager.py sync          - Sync webhook with database
  python helius_webhook_manager.py add <addr>    - Add wallet address(es)
  
Environment variables needed:
  HELIUS_KEY          - Your Helius API key
  HELIUS_WEBHOOK_ID   - Your webhook ID (get from 'list' command)

Examples:
  python helius_webhook_manager.py list
  python helius_webhook_manager.py sync
  python helius_webhook_manager.py add ABC123... DEF456...
            """)
        
        else:
            print(f"Unknown command: {command}")
            print("Use 'help' for usage info")
    else:
        # Default: list webhooks
        manager.list_webhooks()
