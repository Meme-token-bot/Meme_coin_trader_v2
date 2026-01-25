"""
Import Existing Helius Webhooks
Connects your existing Helius webhooks to the multi-webhook manager database
"""

import os
import requests
from datetime import datetime
from dotenv import load_dotenv
from database_v2 import DatabaseV2

load_dotenv()

def import_existing_webhooks():
    """Import webhooks from Helius into local database"""
    
    print("\n" + "="*60)
    print("📥 IMPORTING EXISTING HELIUS WEBHOOKS")
    print("="*60)
    
    # Get API credentials
    api_key = os.getenv('HELIUS_KEY')
    webhook_url = os.getenv('WEBHOOK_URL')
    
    if not api_key:
        print("❌ HELIUS_KEY not found in .env")
        return
    
    if not webhook_url:
        print("❌ WEBHOOK_URL not found in .env")
        return
    
    print(f"\n✅ Using webhook URL: {webhook_url}")
    
    # Connect to database
    db = DatabaseV2()
    
    # Get webhooks from Helius API
    print(f"\n🔍 Fetching webhooks from Helius...")
    
    try:
        url = f"https://api.helius.xyz/v0/webhooks?api-key={api_key}"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        webhooks = response.json()
        
        if not webhooks:
            print("❌ No webhooks found in your Helius account")
            print("\nCreate webhooks at: https://dashboard.helius.dev")
            return
        
        print(f"✅ Found {len(webhooks)} webhook(s) in Helius\n")
        
        # Import each webhook
        imported_count = 0
        wallet_count = 0
        
        for webhook in webhooks:
            webhook_id = webhook.get('webhookID')
            webhook_type = webhook.get('webhookType')
            addresses = webhook.get('accountAddresses', [])
            
            print(f"\n📌 Webhook: {webhook_id}")
            print(f"   Type: {webhook_type}")
            print(f"   URL: {webhook.get('webhookURL')}")
            print(f"   Addresses: {len(addresses)}")
            
            # Save to database
            with db.connection() as conn:
                # Register webhook
                conn.execute("""
                    INSERT OR REPLACE INTO webhook_registry 
                    (webhook_id, webhook_url, created_at, wallet_count, status, last_updated)
                    VALUES (?, ?, ?, ?, 'active', ?)
                """, (
                    webhook_id,
                    webhook_url,
                    datetime.now(),
                    len(addresses),
                    datetime.now()
                ))
                
                # Register wallet assignments
                for address in addresses:
                    conn.execute("""
                        INSERT OR REPLACE INTO wallet_webhook_assignments 
                        (wallet_address, webhook_id, assigned_at)
                        VALUES (?, ?, ?)
                    """, (address, webhook_id, datetime.now()))
                    
                    wallet_count += 1
            
            imported_count += 1
            print(f"   ✅ Imported with {len(addresses)} wallet(s)")
        
        print(f"\n{'='*60}")
        print("✅ IMPORT COMPLETE")
        print(f"{'='*60}")
        print(f"Webhooks imported: {imported_count}")
        print(f"Total wallets registered: {wallet_count}")
        print(f"{'='*60}\n")
        
        # Show current status
        print("\n📊 Current System Status:")
        with db.connection() as conn:
            stats = conn.execute("""
                SELECT 
                    COUNT(DISTINCT webhook_id) as webhooks,
                    COUNT(*) as total_wallets
                FROM wallet_webhook_assignments
            """).fetchone()
            
            print(f"   Active webhooks: {stats['webhooks']}")
            print(f"   Total wallets tracked: {stats['total_wallets']}")
        
        # Now sync any missing wallets from database
        print("\n🔄 Checking for wallets not yet in webhooks...")
        
        all_wallets = db.get_all_verified_wallets()
        
        with db.connection() as conn:
            assigned = conn.execute("""
                SELECT wallet_address FROM wallet_webhook_assignments
            """).fetchall()
            assigned_set = {row['wallet_address'] for row in assigned}
        
        missing = [w['address'] for w in all_wallets if w['address'] not in assigned_set]
        
        if missing:
            print(f"   Found {len(missing)} wallet(s) in database not yet in webhooks")
            print(f"   Run: python multi_webhook_manager.py sync")
            print(f"   (This will add them to your existing webhooks)")
        else:
            print(f"   ✅ All {len(all_wallets)} database wallets are already in webhooks!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import_existing_webhooks()
