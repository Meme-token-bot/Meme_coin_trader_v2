"""
WEBHOOK FIX SCRIPT
==================

Fixes the issue where webhooks exist but have no addresses.

Your situation:
- 3 webhooks with 0 addresses each
- 53 wallets in database not being tracked

This script will:
1. Delete empty/unused webhooks
2. Add your verified wallets to remaining webhooks
"""

import os
import sys
import requests
import time
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()

from database_v2 import DatabaseV2

HELIUS_KEY = os.getenv('HELIUS_KEY')
WEBHOOK_URL = os.getenv('WEBHOOK_URL')
MAX_PER_WEBHOOK = 25


def list_webhooks():
    """List all Helius webhooks"""
    url = f"https://api.helius.xyz/v0/webhooks?api-key={HELIUS_KEY}"
    response = requests.get(url, timeout=15)
    
    if response.status_code == 200:
        return response.json()
    return []


def delete_webhook(webhook_id: str) -> bool:
    """Delete a webhook"""
    url = f"https://api.helius.xyz/v0/webhooks/{webhook_id}?api-key={HELIUS_KEY}"
    response = requests.delete(url, timeout=15)
    return response.status_code == 200


def update_webhook_addresses(webhook_id: str, addresses: list) -> bool:
    """Update a webhook with new addresses"""
    # First get current state
    url = f"https://api.helius.xyz/v0/webhooks/{webhook_id}?api-key={HELIUS_KEY}"
    response = requests.get(url, timeout=15)
    
    if response.status_code != 200:
        return False
    
    wh = response.json()
    
    # Update with new addresses
    payload = {
        "webhookURL": wh.get('webhookURL', WEBHOOK_URL),
        "accountAddresses": addresses,
        "webhookType": wh.get('webhookType', 'enhanced'),
        "transactionTypes": wh.get('transactionTypes', ['SWAP'])
    }
    
    update_url = f"https://api.helius.xyz/v0/webhooks/{webhook_id}?api-key={HELIUS_KEY}"
    response = requests.put(update_url, json=payload, timeout=15)
    
    return response.status_code == 200


def create_webhook_with_addresses(addresses: list) -> str:
    """Create a new webhook with addresses"""
    url = f"https://api.helius.xyz/v0/webhooks?api-key={HELIUS_KEY}"
    
    payload = {
        "webhookURL": WEBHOOK_URL,
        "accountAddresses": addresses,
        "webhookType": "enhanced",
        "transactionTypes": ["SWAP"]
    }
    
    response = requests.post(url, json=payload, timeout=15)
    
    if response.status_code in [200, 201]:
        result = response.json()
        return result.get('webhookID')
    
    print(f"   ❌ Failed to create webhook: {response.status_code}")
    print(f"   {response.text[:200]}")
    return None


def fix_webhooks():
    """Main fix function"""
    print("\n" + "="*60)
    print("🔧 WEBHOOK FIX SCRIPT")
    print("="*60)
    
    if not HELIUS_KEY:
        print("❌ HELIUS_KEY not set")
        return
    
    if not WEBHOOK_URL:
        print("❌ WEBHOOK_URL not set")
        return
    
    # Step 1: Get current state
    print("\n📋 Current webhooks:")
    webhooks = list_webhooks()
    
    our_webhooks = []
    other_webhooks = []
    
    for wh in webhooks:
        webhook_id = wh.get('webhookID')
        addresses = wh.get('accountAddresses', [])
        wh_url = wh.get('webhookURL', '')
        
        print(f"   {webhook_id[:16]}... | {len(addresses)} addresses")
        print(f"      URL: {wh_url[:50]}...")
        
        # Check if it's our webhook
        if WEBHOOK_URL and WEBHOOK_URL in wh_url:
            our_webhooks.append(wh)
        else:
            other_webhooks.append(wh)
    
    print(f"\n   Our webhooks: {len(our_webhooks)}")
    print(f"   Other webhooks: {len(other_webhooks)}")
    
    # Step 2: Get verified wallets from database
    db = DatabaseV2()
    wallets = db.get_all_verified_wallets()
    wallet_addresses = [w['address'] for w in wallets]
    
    print(f"\n📊 Database:")
    print(f"   Verified wallets: {len(wallet_addresses)}")
    
    # Step 3: Check what's already tracked
    already_tracked = set()
    for wh in our_webhooks:
        for addr in wh.get('accountAddresses', []):
            already_tracked.add(addr)
    
    not_tracked = [a for a in wallet_addresses if a not in already_tracked]
    
    print(f"   Already tracked: {len(already_tracked)}")
    print(f"   Not tracked: {len(not_tracked)}")
    
    if not not_tracked:
        print("\n✅ All wallets are already tracked!")
        return
    
    # Step 4: Delete empty webhooks (keep at least 1)
    print(f"\n🗑️ Cleaning up empty webhooks...")
    
    empty_webhooks = [wh for wh in our_webhooks if len(wh.get('accountAddresses', [])) == 0]
    
    # Keep the first one, delete the rest
    if len(empty_webhooks) > 1:
        for wh in empty_webhooks[1:]:
            webhook_id = wh.get('webhookID')
            print(f"   Deleting empty webhook: {webhook_id[:16]}...")
            if delete_webhook(webhook_id):
                print(f"   ✅ Deleted")
                our_webhooks.remove(wh)
            else:
                print(f"   ❌ Failed to delete")
            time.sleep(0.5)
    
    # Step 5: Add wallets to webhooks
    print(f"\n📦 Adding {len(not_tracked)} wallet(s) to webhooks...")
    
    # Batch wallets into groups of 25
    batches = [not_tracked[i:i+MAX_PER_WEBHOOK] for i in range(0, len(not_tracked), MAX_PER_WEBHOOK)]
    
    print(f"   Need {len(batches)} webhook(s) to hold all wallets")
    
    # Find webhooks with capacity
    for batch_num, batch in enumerate(batches):
        print(f"\n   Batch {batch_num + 1}: {len(batch)} wallets")
        
        # Find a webhook with capacity
        webhook_to_use = None
        for wh in our_webhooks:
            current_count = len(wh.get('accountAddresses', []))
            available = MAX_PER_WEBHOOK - current_count
            
            if available >= len(batch):
                webhook_to_use = wh
                break
        
        if webhook_to_use:
            webhook_id = webhook_to_use.get('webhookID')
            current_addresses = webhook_to_use.get('accountAddresses', [])
            new_addresses = current_addresses + batch
            
            print(f"      Using existing webhook: {webhook_id[:16]}...")
            
            if update_webhook_addresses(webhook_id, new_addresses):
                print(f"      ✅ Added {len(batch)} wallets ({len(new_addresses)}/25)")
                webhook_to_use['accountAddresses'] = new_addresses
            else:
                print(f"      ❌ Failed to update webhook")
        else:
            # Need to create a new webhook
            print(f"      Creating new webhook...")
            
            new_webhook_id = create_webhook_with_addresses(batch)
            
            if new_webhook_id:
                print(f"      ✅ Created webhook: {new_webhook_id[:16]}...")
                our_webhooks.append({
                    'webhookID': new_webhook_id,
                    'accountAddresses': batch
                })
            else:
                print(f"      ❌ Failed to create webhook")
                print(f"      ⚠️ You may have hit your webhook limit!")
                print(f"      💡 Consider upgrading your Helius plan")
        
        time.sleep(0.5)
    
    # Step 6: Update database registry
    print(f"\n📝 Updating database registry...")
    
    with db.connection() as conn:
        # Clear old registry
        conn.execute("DELETE FROM webhook_registry")
        conn.execute("DELETE FROM wallet_webhook_assignments")
        
        # Add current webhooks
        for wh in our_webhooks:
            webhook_id = wh.get('webhookID')
            addresses = wh.get('accountAddresses', [])
            
            conn.execute("""
                INSERT INTO webhook_registry 
                (webhook_id, webhook_url, created_at, wallet_count, status, last_updated)
                VALUES (?, ?, ?, ?, 'active', ?)
            """, (webhook_id, WEBHOOK_URL, datetime.now(), len(addresses), datetime.now()))
            
            for addr in addresses:
                conn.execute("""
                    INSERT INTO wallet_webhook_assignments
                    (wallet_address, webhook_id, assigned_at)
                    VALUES (?, ?, ?)
                """, (addr, webhook_id, datetime.now()))
    
    print(f"   ✅ Database updated")
    
    # Step 7: Final status
    print(f"\n" + "="*60)
    print(f"✅ WEBHOOK FIX COMPLETE")
    print(f"="*60)
    
    total_tracked = sum(len(wh.get('accountAddresses', [])) for wh in our_webhooks)
    print(f"   Webhooks: {len(our_webhooks)}")
    print(f"   Total wallets tracked: {total_tracked}")
    print(f"   Database wallets: {len(wallet_addresses)}")
    
    if total_tracked < len(wallet_addresses):
        print(f"\n   ⚠️ {len(wallet_addresses) - total_tracked} wallet(s) still not tracked")
        print(f"   💡 You may need to upgrade your Helius plan for more webhooks")


if __name__ == "__main__":
    fix_webhooks()
