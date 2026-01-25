"""
Multi-Webhook Manager
Automatically creates and manages multiple Helius webhooks to handle scaling beyond 25 wallets

Features:
- Auto-creates new webhooks when capacity reached
- Distributes wallets evenly across webhooks
- Tracks webhook assignments in database
- Handles webhook failures gracefully
- Rate limiting to avoid 429 errors
- Retry logic with exponential backoff

UPDATED: Added rate limiting and retry logic to handle Helius API limits
"""

import requests
import os
import time
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from database_v2 import DatabaseV2


class MultiWebhookManager:
    """
    Manages multiple Helius webhooks with automatic scaling.
    
    Helius limit: 25 addresses per webhook
    This manager creates new webhooks automatically when needed.
    """
    
    MAX_ADDRESSES_PER_WEBHOOK = 25
    
    # Rate limiting settings
    MIN_DELAY_BETWEEN_CALLS = 0.5  # 500ms between API calls
    MAX_RETRIES = 3
    RETRY_BASE_DELAY = 2.0  # Base delay for exponential backoff
    
    def __init__(self, api_key: str, webhook_url: str, db: DatabaseV2):
        """
        Initialize multi-webhook manager
        
        Args:
            api_key: Your Helius API key
            webhook_url: Your server's webhook endpoint URL
            db: Database instance for tracking webhook assignments
        """
        self.api_key = api_key
        self.webhook_url = webhook_url
        self.db = db
        self.base_url = "https://api.helius.xyz/v0/webhooks"
        self._last_api_call = 0  # Timestamp of last API call
        
        # Ensure webhook tracking table exists
        self._init_webhook_tracking()
        
        # Load existing webhooks
        self.webhooks = self._load_webhooks()
        
        print(f"✅ Multi-Webhook Manager initialized")
        print(f"   Active webhooks: {len(self.webhooks)}")
        print(f"   Total capacity: {len(self.webhooks) * self.MAX_ADDRESSES_PER_WEBHOOK} wallets")
    
    def _rate_limit(self):
        """Enforce minimum delay between API calls"""
        now = time.time()
        elapsed = now - self._last_api_call
        if elapsed < self.MIN_DELAY_BETWEEN_CALLS:
            sleep_time = self.MIN_DELAY_BETWEEN_CALLS - elapsed
            time.sleep(sleep_time)
        self._last_api_call = time.time()
    
    def _api_call_with_retry(self, method: str, url: str, **kwargs) -> Optional[requests.Response]:
        """
        Make an API call with rate limiting and retry logic.
        
        Args:
            method: HTTP method ('get', 'post', 'put', 'delete')
            url: Full URL to call
            **kwargs: Additional arguments for requests
        
        Returns:
            Response object or None on failure
        """
        kwargs.setdefault('timeout', 15)
        
        for attempt in range(self.MAX_RETRIES):
            self._rate_limit()
            
            try:
                if method.lower() == 'get':
                    response = requests.get(url, **kwargs)
                elif method.lower() == 'post':
                    response = requests.post(url, **kwargs)
                elif method.lower() == 'put':
                    response = requests.put(url, **kwargs)
                elif method.lower() == 'delete':
                    response = requests.delete(url, **kwargs)
                else:
                    raise ValueError(f"Unknown method: {method}")
                
                # Success
                if response.status_code < 400:
                    return response
                
                # Rate limited - retry with backoff
                if response.status_code == 429:
                    delay = self.RETRY_BASE_DELAY * (2 ** attempt)
                    print(f"   ⏳ Rate limited, waiting {delay:.1f}s (attempt {attempt + 1}/{self.MAX_RETRIES})")
                    time.sleep(delay)
                    continue
                
                # Other error
                response.raise_for_status()
                
            except requests.exceptions.RequestException as e:
                if attempt < self.MAX_RETRIES - 1:
                    delay = self.RETRY_BASE_DELAY * (2 ** attempt)
                    print(f"   ⚠️ API error, retrying in {delay:.1f}s: {e}")
                    time.sleep(delay)
                else:
                    print(f"   ❌ API call failed after {self.MAX_RETRIES} attempts: {e}")
                    return None
        
        return None
    
    def _init_webhook_tracking(self):
        """Create table for tracking webhook assignments"""
        with self.db.connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS webhook_registry (
                    webhook_id TEXT PRIMARY KEY,
                    webhook_url TEXT,
                    created_at TIMESTAMP,
                    wallet_count INTEGER DEFAULT 0,
                    status TEXT DEFAULT 'active',
                    last_updated TIMESTAMP
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS wallet_webhook_assignments (
                    wallet_address TEXT PRIMARY KEY,
                    webhook_id TEXT,
                    assigned_at TIMESTAMP,
                    FOREIGN KEY (webhook_id) REFERENCES webhook_registry(webhook_id)
                )
            """)
            
            # Create index for fast lookups
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_webhook_assignments 
                ON wallet_webhook_assignments(webhook_id)
            """)
    
    def _load_webhooks(self) -> Dict[str, Dict]:
        """Load webhook registry from database"""
        with self.db.connection() as conn:
            rows = conn.execute("""
                SELECT * FROM webhook_registry 
                WHERE status = 'active'
                ORDER BY created_at
            """).fetchall()
            
            webhooks = {}
            for row in rows:
                webhook_id = row['webhook_id']
                webhooks[webhook_id] = {
                    'id': webhook_id,
                    'url': row['webhook_url'],
                    'created_at': row['created_at'],
                    'wallet_count': row['wallet_count'],
                    'status': row['status']
                }
            
            return webhooks
    
    def _get_webhook_details(self, webhook_id: str) -> Optional[Dict]:
        """Get webhook details from Helius API"""
        url = f"{self.base_url}/{webhook_id}?api-key={self.api_key}"
        response = self._api_call_with_retry('get', url)
        
        if response:
            return response.json()
        return None
    
    def _create_new_webhook(self) -> Optional[str]:
        """Create a new webhook with Helius"""
        url = f"{self.base_url}?api-key={self.api_key}"
        
        payload = {
            "webhookURL": self.webhook_url,
            "accountAddresses": [],  # Start empty
            "webhookType": "enhanced",
            "transactionTypes": ["SWAP"]
        }
        
        response = self._api_call_with_retry('post', url, json=payload)
        
        if not response:
            print(f"   ❌ Failed to create new webhook")
            return None
        
        result = response.json()
        webhook_id = result.get('webhookID')
        
        if webhook_id:
            # Save to database
            with self.db.connection() as conn:
                conn.execute("""
                    INSERT INTO webhook_registry 
                    (webhook_id, webhook_url, created_at, wallet_count, status, last_updated)
                    VALUES (?, ?, ?, 0, 'active', ?)
                """, (webhook_id, self.webhook_url, datetime.now(), datetime.now()))
            
            # Add to local registry
            self.webhooks[webhook_id] = {
                'id': webhook_id,
                'url': self.webhook_url,
                'created_at': datetime.now(),
                'wallet_count': 0,
                'status': 'active'
            }
            
            print(f"\n   ✅ Created new webhook: {webhook_id[:12]}...")
            print(f"      Total webhooks: {len(self.webhooks)}")
            print(f"      New capacity: {len(self.webhooks) * self.MAX_ADDRESSES_PER_WEBHOOK} wallets")
            
            return webhook_id
        
        return None
    
    def _find_available_webhook(self) -> Optional[str]:
        """Find a webhook with capacity, or create a new one"""
        # Check existing webhooks for capacity
        for webhook_id, webhook in self.webhooks.items():
            if webhook['wallet_count'] < self.MAX_ADDRESSES_PER_WEBHOOK:
                return webhook_id
        
        # All full - create new webhook
        print(f"\n   📊 All webhooks at capacity ({self.MAX_ADDRESSES_PER_WEBHOOK} each)")
        print(f"   🆕 Creating new webhook...")
        
        return self._create_new_webhook()
    
    def _add_address_to_webhook(self, webhook_id: str, address: str) -> bool:
        """Add a single address to a specific webhook"""
        # Get current webhook state
        current = self._get_webhook_details(webhook_id)
        if not current:
            return False
        
        existing_addresses = current.get('accountAddresses', [])
        
        # Check if already exists
        if address in existing_addresses:
            return True
        
        # Add new address
        all_addresses = existing_addresses + [address]
        
        # Update webhook
        url = f"{self.base_url}/{webhook_id}?api-key={self.api_key}"
        payload = {
            "webhookURL": current.get('webhookURL'),
            "accountAddresses": all_addresses,
            "webhookType": current.get('webhookType', 'enhanced'),
            "transactionTypes": current.get('transactionTypes', ['SWAP'])
        }
        
        response = self._api_call_with_retry('put', url, json=payload)
        
        if response:
            return True
        return False
    
    def add_wallet(self, wallet_address: str) -> bool:
        """
        Add a wallet to the webhook system.
        Automatically finds/creates appropriate webhook.
        
        Args:
            wallet_address: Solana wallet address to track
        
        Returns:
            True if successfully added
        """
        # Check if already assigned
        with self.db.connection() as conn:
            existing = conn.execute("""
                SELECT webhook_id FROM wallet_webhook_assignments 
                WHERE wallet_address = ?
            """, (wallet_address,)).fetchone()
            
            if existing:
                print(f"   ℹ️  {wallet_address[:8]}... already assigned to webhook")
                return True
        
        # Find available webhook
        webhook_id = self._find_available_webhook()
        
        if not webhook_id:
            print(f"   ❌ Could not find or create webhook for {wallet_address[:8]}...")
            return False
        
        # Add address to Helius webhook
        if not self._add_address_to_webhook(webhook_id, wallet_address):
            return False
        
        # Record assignment in database
        with self.db.connection() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO wallet_webhook_assignments 
                (wallet_address, webhook_id, assigned_at)
                VALUES (?, ?, ?)
            """, (wallet_address, webhook_id, datetime.now()))
            
            # Update webhook wallet count
            conn.execute("""
                UPDATE webhook_registry 
                SET wallet_count = wallet_count + 1,
                    last_updated = ?
                WHERE webhook_id = ?
            """, (datetime.now(), webhook_id))
        
        # Update local cache
        self.webhooks[webhook_id]['wallet_count'] += 1
        
        print(f"   ✅ {wallet_address[:8]}... → Webhook {webhook_id[:8]}... ({self.webhooks[webhook_id]['wallet_count']}/{self.MAX_ADDRESSES_PER_WEBHOOK})")
        
        return True
    
    def add_wallets_batch(self, wallet_addresses: List[str]) -> Dict[str, int]:
        """
        Add multiple wallets efficiently.
        
        Args:
            wallet_addresses: List of wallet addresses
        
        Returns:
            Dict with stats: {'added': count, 'failed': count, 'skipped': count}
        """
        stats = {'added': 0, 'failed': 0, 'skipped': 0}
        
        print(f"\n📦 Adding {len(wallet_addresses)} wallet(s) to webhook system...")
        
        for address in wallet_addresses:
            # Check if already assigned
            with self.db.connection() as conn:
                existing = conn.execute("""
                    SELECT webhook_id FROM wallet_webhook_assignments 
                    WHERE wallet_address = ?
                """, (address,)).fetchone()
                
                if existing:
                    stats['skipped'] += 1
                    continue
            
            # Add to webhook system
            if self.add_wallet(address):
                stats['added'] += 1
            else:
                stats['failed'] += 1
        
        print(f"\n✅ Batch complete: {stats['added']} added, {stats['skipped']} skipped, {stats['failed']} failed")
        
        if stats['failed'] > 0:
            print(f"   ⚠️  {stats['failed']} wallet(s) failed to add - will retry on next sync")
        
        return stats
    
    def remove_wallet(self, wallet_address: str) -> bool:
        """Remove a wallet from the webhook system"""
        # Get webhook assignment
        with self.db.connection() as conn:
            assignment = conn.execute("""
                SELECT webhook_id FROM wallet_webhook_assignments 
                WHERE wallet_address = ?
            """, (wallet_address,)).fetchone()
            
            if not assignment:
                print(f"   ℹ️  {wallet_address[:8]}... not found in webhook system")
                return False
            
            webhook_id = assignment['webhook_id']
        
        # Get current webhook state
        current = self._get_webhook_details(webhook_id)
        
        if not current:
            return False
        
        # Remove address
        addresses = [addr for addr in current.get('accountAddresses', []) 
                    if addr != wallet_address]
        
        # Update webhook
        url = f"{self.base_url}/{webhook_id}?api-key={self.api_key}"
        payload = {
            "webhookURL": current.get('webhookURL'),
            "accountAddresses": addresses,
            "webhookType": current.get('webhookType', 'enhanced'),
            "transactionTypes": current.get('transactionTypes', ['SWAP'])
        }
        
        response = self._api_call_with_retry('put', url, json=payload)
        
        if not response:
            return False
        
        # Update database
        with self.db.connection() as conn:
            conn.execute("""
                DELETE FROM wallet_webhook_assignments 
                WHERE wallet_address = ?
            """, (wallet_address,))
            
            conn.execute("""
                UPDATE webhook_registry 
                SET wallet_count = wallet_count - 1,
                    last_updated = ?
                WHERE webhook_id = ?
            """, (datetime.now(), webhook_id))
        
        # Update local cache
        if webhook_id in self.webhooks:
            self.webhooks[webhook_id]['wallet_count'] -= 1
        
        print(f"   ✅ Removed {wallet_address[:8]}... from webhook system")
        return True
    
    def sync_with_database(self) -> Dict:
        """
        Sync all verified wallets from database to webhook system.
        Creates new webhooks as needed.
        """
        print(f"\n🔄 Syncing database wallets with webhook system...")
        
        # Get all verified wallets
        wallets = self.db.get_all_verified_wallets()
        wallet_addresses = [w['address'] for w in wallets]
        
        print(f"   Found {len(wallet_addresses)} verified wallet(s) in database")
        
        # Get currently assigned wallets
        with self.db.connection() as conn:
            assigned = conn.execute("""
                SELECT wallet_address FROM wallet_webhook_assignments
            """).fetchall()
            assigned_set = {row['wallet_address'] for row in assigned}
        
        # Find wallets that need to be added
        to_add = [addr for addr in wallet_addresses if addr not in assigned_set]
        
        if not to_add:
            print(f"   ✅ All wallets already synced!")
            return {'total': len(wallet_addresses), 'added': 0, 'already_synced': len(wallet_addresses)}
        
        print(f"   Adding {len(to_add)} new wallet(s)...")
        stats = self.add_wallets_batch(to_add)
        stats['total'] = len(wallet_addresses)
        stats['already_synced'] = len(assigned_set)
        
        return stats
    
    def get_status(self) -> Dict:
        """Get status of webhook system"""
        with self.db.connection() as conn:
            total_assigned = conn.execute("""
                SELECT COUNT(*) FROM wallet_webhook_assignments
            """).fetchone()[0]
            
            webhook_stats = conn.execute("""
                SELECT 
                    COUNT(*) as active_webhooks,
                    SUM(wallet_count) as total_wallets,
                    AVG(wallet_count) as avg_per_webhook,
                    MAX(wallet_count) as max_per_webhook
                FROM webhook_registry
                WHERE status = 'active'
            """).fetchone()
        
        return {
            'active_webhooks': webhook_stats['active_webhooks'] or 0,
            'total_wallets_assigned': total_assigned,
            'avg_wallets_per_webhook': webhook_stats['avg_per_webhook'] or 0,
            'max_wallets_per_webhook': webhook_stats['max_per_webhook'] or 0,
            'total_capacity': (webhook_stats['active_webhooks'] or 0) * self.MAX_ADDRESSES_PER_WEBHOOK,
            'utilization_pct': (total_assigned / max(1, (webhook_stats['active_webhooks'] or 0) * self.MAX_ADDRESSES_PER_WEBHOOK)) * 100
        }
    
    def print_status(self):
        """Print detailed status"""
        status = self.get_status()
        
        print(f"\n{'='*60}")
        print("WEBHOOK SYSTEM STATUS")
        print(f"{'='*60}")
        print(f"Active webhooks: {status['active_webhooks']}")
        print(f"Total wallets tracked: {status['total_wallets_assigned']}")
        print(f"Average per webhook: {status['avg_wallets_per_webhook']:.1f}")
        print(f"Max per webhook: {status['max_wallets_per_webhook']}/{self.MAX_ADDRESSES_PER_WEBHOOK}")
        print(f"Total capacity: {status['total_capacity']} wallets")
        print(f"Utilization: {status['utilization_pct']:.1f}%")
        print(f"{'='*60}\n")
    
    def list_webhooks(self):
        """List all managed webhooks"""
        print(f"\n{'='*60}")
        print("MANAGED WEBHOOKS")
        print(f"{'='*60}")
        
        for webhook_id, webhook in self.webhooks.items():
            print(f"\nWebhook ID: {webhook_id}")
            print(f"  URL: {webhook['url']}")
            print(f"  Wallets: {webhook['wallet_count']}/{self.MAX_ADDRESSES_PER_WEBHOOK}")
            print(f"  Created: {webhook['created_at']}")
            print(f"  Status: {webhook['status']}")


# =============================================================================
# INTEGRATION WITH HISTORIAN
# =============================================================================

def auto_add_discovered_wallet(multi_webhook_manager: MultiWebhookManager, 
                               wallet_address: str,
                               notifier=None) -> bool:
    """
    Automatically add a newly discovered wallet to the webhook system.
    
    Call this after verifying a new wallet during discovery.
    
    Args:
        multi_webhook_manager: MultiWebhookManager instance
        wallet_address: Wallet address to add
        notifier: Optional notifier for alerts
    
    Returns:
        True if successful
    """
    success = multi_webhook_manager.add_wallet(wallet_address)
    
    if success and notifier:
        # Send notification that wallet was added
        status = multi_webhook_manager.get_status()
        notifier.send(f"""✅ Wallet Auto-Added to Webhook System

Address: <code>{wallet_address}</code>

System Status:
• Active webhooks: {status['active_webhooks']}
• Total wallets: {status['total_wallets_assigned']}
• Capacity: {status['utilization_pct']:.0f}%""")
    
    return success


# =============================================================================
# CLI INTERFACE
# =============================================================================

if __name__ == "__main__":
    import sys
    from dotenv import load_dotenv
    load_dotenv()
    
    print("\n" + "="*60)
    print("🔧 MULTI-WEBHOOK MANAGER")
    print("="*60)
    
    api_key = os.getenv('HELIUS_KEY')
    webhook_url = os.getenv('WEBHOOK_URL', 'http://your-server:5000/webhook/helius')
    
    if not api_key:
        print("❌ HELIUS_KEY not found in .env")
        sys.exit(1)
    
    db = DatabaseV2()
    manager = MultiWebhookManager(api_key, webhook_url, db)
    
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == 'status':
            manager.print_status()
        
        elif command == 'list':
            manager.list_webhooks()
        
        elif command == 'sync':
            stats = manager.sync_with_database()
            print(f"\n✅ Sync complete!")
            print(f"   Total in DB: {stats['total']}")
            print(f"   Already synced: {stats['already_synced']}")
            print(f"   Newly added: {stats['added']}")
            print(f"   Failed: {stats['failed']}")
        
        elif command == 'add' and len(sys.argv) > 2:
            addresses = sys.argv[2:]
            stats = manager.add_wallets_batch(addresses)
        
        elif command == 'remove' and len(sys.argv) > 2:
            address = sys.argv[2]
            manager.remove_wallet(address)
        
        elif command == 'help':
            print("""
Usage:
  python multi_webhook_manager.py status           - Show system status
  python multi_webhook_manager.py list             - List all webhooks
  python multi_webhook_manager.py sync             - Sync all DB wallets
  python multi_webhook_manager.py add <addr...>    - Add wallet(s)
  python multi_webhook_manager.py remove <addr>    - Remove wallet

Environment variables needed:
  HELIUS_KEY    - Your Helius API key
  WEBHOOK_URL   - Your server's webhook endpoint (optional)

Examples:
  python multi_webhook_manager.py status
  python multi_webhook_manager.py sync
  python multi_webhook_manager.py add ABC123... DEF456...
            """)
        else:
            print(f"Unknown command: {command}")
            print("Use 'help' for usage info")
    else:
        # Default: show status
        manager.print_status()
