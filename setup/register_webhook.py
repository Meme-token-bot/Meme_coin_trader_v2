#!/usr/bin/env python3
"""
Register a manually created Helius webhook with the database.

Usage:
    python setup/register_webhook.py list
    python setup/register_webhook.py sync
    python setup/register_webhook.py <webhook_id>
    python setup/register_webhook.py add <id> <addr1> [addr2...]
"""

# ============================================================================
# PATH SETUP - Required for scripts in subdirectories
# ============================================================================
import sys
from pathlib import Path

# Add project root to Python path (go up one level from setup/)
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'core'))
sys.path.insert(0, str(project_root / 'infrastructure'))
# ============================================================================

import os
import requests
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()

from core.database_v2 import DatabaseV2

HELIUS_KEY = os.getenv('HELIUS_KEY')
WEBHOOK_URL = os.getenv('WEBHOOK_URL')


def list_helius_webhooks():
    """List all webhooks from Helius API"""
    print("\n📋 HELIUS WEBHOOKS")
    print("="*60)
    
    url = f"https://api.helius.xyz/v0/webhooks?api-key={HELIUS_KEY}"
    
    try:
        response = requests.get(url, timeout=15)
        
        if response.status_code == 200:
            webhooks = response.json()
            print(f"Found {len(webhooks)} webhook(s)\n")
            
            for i, wh in enumerate(webhooks):
                webhook_id = wh.get('webhookID', 'unknown')
                addresses = wh.get('accountAddresses', [])
                wh_url = wh.get('webhookURL', 'unknown')
                wh_type = wh.get('webhookType', 'unknown')
                
                print(f"Webhook {i+1}:")
                print(f"  ID: {webhook_id}")
                print(f"  Addresses: {len(addresses)}/25")
                print(f"  URL: {wh_url}")
                print(f"  Type: {wh_type}")
                print()
            
            return webhooks
        else:
            print(f"❌ Error: {response.status_code}")
            print(response.text[:300])
            return []
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return []


def list_db_webhooks(db: DatabaseV2):
    """List webhooks registered in database"""
    print("\n📋 DATABASE WEBHOOKS")
    print("="*60)
    
    with db.connection() as conn:
        rows = conn.execute("""
            SELECT webhook_id, webhook_url, wallet_count, status, created_at
            FROM webhook_registry
            ORDER BY created_at
        """).fetchall()
        
        print(f"Found {len(rows)} webhook(s) in database\n")
        
        for row in rows:
            print(f"  ID: {row['webhook_id']}")
            print(f"  Wallets: {row['wallet_count']}/25")
            print(f"  Status: {row['status']}")
            print(f"  Created: {row['created_at']}")
            print()
        
        return rows


def register_webhook(db: DatabaseV2, webhook_id: str):
    """Register a webhook in the database"""
    print(f"\n📝 Registering webhook: {webhook_id}")
    
    # Verify it exists in Helius
    url = f"https://api.helius.xyz/v0/webhooks/{webhook_id}?api-key={HELIUS_KEY}"
    
    try:
        response = requests.get(url, timeout=15)
        
        if response.status_code != 200:
            print(f"❌ Webhook not found in Helius: {response.status_code}")
            return False
        
        wh = response.json()
        addresses = wh.get('accountAddresses', [])
        wh_url = wh.get('webhookURL', '')
        
        print(f"  Found in Helius:")
        print(f"    Addresses: {len(addresses)}")
        print(f"    URL: {wh_url}")
        
        # Add to database
        with db.connection() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO webhook_registry
                (webhook_id, webhook_url, created_at, wallet_count, status, last_updated)
                VALUES (?, ?, ?, ?, 'active', ?)
            """, (webhook_id, wh_url, datetime.now(), len(addresses), datetime.now()))
            
            # Also register existing addresses
            for addr in addresses:
                conn.execute("""
                    INSERT OR IGNORE INTO wallet_webhook_assignments
                    (wallet_address, webhook_id, assigned_at)
                    VALUES (?, ?, ?)
                """, (addr, webhook_id, datetime.now()))
        
        print(f"✅ Webhook registered in database!")
        print(f"   {len(addresses)} address(es) also registered")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def sync_all_webhooks(db: DatabaseV2):
    """Sync all Helius webhooks with database"""
    print("\n🔄 SYNCING ALL WEBHOOKS")
    print("="*60)
    
    # Get all Helius webhooks
    url = f"https://api.helius.xyz/v0/webhooks?api-key={HELIUS_KEY}"
    
    try:
        response = requests.get(url, timeout=15)
        
        if response.status_code != 200:
            print(f"❌ Error getting webhooks: {response.status_code}")
            return
        
        webhooks = response.json()
        print(f"Found {len(webhooks)} webhook(s) in Helius")
        
        registered = 0
        for wh in webhooks:
            webhook_id = wh.get('webhookID')
            wh_url = wh.get('webhookURL', '')
            addresses = wh.get('accountAddresses', [])
            
            # Only register webhooks that match our URL (if set)
            if WEBHOOK_URL and WEBHOOK_URL not in wh_url:
                print(f"  ⏭️  Skipping {webhook_id[:12]}... (different URL)")
                continue
            
            with db.connection() as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO webhook_registry
                    (webhook_id, webhook_url, created_at, wallet_count, status, last_updated)
                    VALUES (?, ?, COALESCE(
                        (SELECT created_at FROM webhook_registry WHERE webhook_id = ?),
                        ?
                    ), ?, 'active', ?)
                """, (webhook_id, wh_url, webhook_id, datetime.now(), 
                      len(addresses), datetime.now()))
                
                # Register addresses
                for addr in addresses:
                    conn.execute("""
                        INSERT OR IGNORE INTO wallet_webhook_assignments
                        (wallet_address, webhook_id, assigned_at)
                        VALUES (?, ?, ?)
                    """, (addr, webhook_id, datetime.now()))
            
            print(f"  ✅ Synced {webhook_id[:12]}... ({len(addresses)} addresses)")
            registered += 1
        
        print(f"\n✅ Synced {registered} webhook(s) to database")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


def add_wallets_to_webhook(db: DatabaseV2, webhook_id: str, addresses: list):
    """Add wallet addresses to a specific webhook"""
    print(f"\n📝 Adding {len(addresses)} wallet(s) to webhook {webhook_id[:12]}...")
    
    # Get current webhook state
    url = f"https://api.helius.xyz/v0/webhooks/{webhook_id}?api-key={HELIUS_KEY}"
    
    try:
        response = requests.get(url, timeout=15)
        
        if response.status_code != 200:
            print(f"❌ Webhook not found: {response.status_code}")
            return False
        
        wh = response.json()
        existing = set(wh.get('accountAddresses', []))
        
        # Filter new addresses
        to_add = [a for a in addresses if a not in existing]
        
        if not to_add:
            print("ℹ️  All addresses already in webhook")
            return True
        
        if len(existing) + len(to_add) > 25:
            print(f"⚠️  Would exceed 25 address limit!")
            print(f"   Current: {len(existing)}, Adding: {len(to_add)}")
            max_can_add = 25 - len(existing)
            to_add = to_add[:max_can_add]
            print(f"   Only adding {max_can_add}")
        
        # Update webhook
        all_addresses = list(existing) + to_add
        
        update_url = f"https://api.helius.xyz/v0/webhooks/{webhook_id}?api-key={HELIUS_KEY}"
        payload = {
            "webhookURL": wh.get('webhookURL'),
            "accountAddresses": all_addresses,
            "webhookType": wh.get('webhookType', 'enhanced'),
            "transactionTypes": wh.get('transactionTypes', ['SWAP'])
        }
        
        response = requests.put(update_url, json=payload, timeout=15)
        
        if response.status_code == 200:
            # Update database
            with db.connection() as conn:
                for addr in to_add:
                    conn.execute("""
                        INSERT OR IGNORE INTO wallet_webhook_assignments
                        (wallet_address, webhook_id, assigned_at)
                        VALUES (?, ?, ?)
                    """, (addr, webhook_id, datetime.now()))
                
                conn.execute("""
                    UPDATE webhook_registry 
                    SET wallet_count = ?, last_updated = ?
                    WHERE webhook_id = ?
                """, (len(all_addresses), datetime.now(), webhook_id))
            
            print(f"✅ Added {len(to_add)} wallet(s)")
            print(f"   Webhook now has {len(all_addresses)}/25 addresses")
            return True
        else:
            print(f"❌ Update failed: {response.status_code}")
            print(response.text[:200])
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


if __name__ == "__main__":
    if not HELIUS_KEY:
        print("❌ HELIUS_KEY not set")
        sys.exit(1)
    
    db = DatabaseV2()
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python setup/register_webhook.py list       - List all webhooks")
        print("  python setup/register_webhook.py sync       - Sync Helius → Database")
        print("  python setup/register_webhook.py <id>       - Register specific webhook")
        print("  python setup/register_webhook.py add <id> <addr1> [addr2...]  - Add wallets to webhook")
        sys.exit(0)
    
    command = sys.argv[1].lower()
    
    if command == 'list':
        list_helius_webhooks()
        list_db_webhooks(db)
    
    elif command == 'sync':
        sync_all_webhooks(db)
    
    elif command == 'add' and len(sys.argv) >= 4:
        webhook_id = sys.argv[2]
        addresses = sys.argv[3:]
        add_wallets_to_webhook(db, webhook_id, addresses)
    
    else:
        # Assume it's a webhook ID to register
        webhook_id = sys.argv[1]
        register_webhook(db, webhook_id)
