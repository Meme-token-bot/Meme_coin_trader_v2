"""
Diagnostic Script for Birdeye API and Helius Webhook Issues
"""

import os
import requests
import json
from dotenv import load_dotenv
load_dotenv()

BIRDEYE_KEY = os.getenv('BIRDEYE_API_KEY')
HELIUS_KEY = os.getenv('HELIUS_KEY')
WEBHOOK_URL = os.getenv('WEBHOOK_URL', '')


def test_birdeye_trending():
    """Test Birdeye trending tokens API"""
    print("\n" + "="*60)
    print("TEST 1: BIRDEYE TRENDING TOKENS")
    print("="*60)
    
    if not BIRDEYE_KEY:
        print("❌ BIRDEYE_API_KEY not set")
        return None
    
    url = "https://public-api.birdeye.so/defi/token_trending"
    headers = {
        "X-API-KEY": BIRDEYE_KEY,
        "x-chain": "solana"
    }
    params = {
        "sort_by": "rank",
        "sort_type": "asc",
        "offset": 0,
        "limit": 5
    }
    
    print(f"URL: {url}")
    
    try:
        response = requests.get(url, headers=headers, params=params, timeout=15)
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"Response keys: {list(data.keys())}")
            
            # Extract tokens
            tokens = []
            if isinstance(data, dict):
                if data.get('success') and isinstance(data.get('data'), dict):
                    tokens = data['data'].get('tokens', [])
                elif isinstance(data.get('data'), dict):
                    tokens = data['data'].get('tokens', [])
                elif isinstance(data.get('data'), list):
                    tokens = data['data']
                elif 'tokens' in data:
                    tokens = data['tokens']
            
            print(f"Found {len(tokens)} tokens")
            
            if tokens:
                print("\nFirst token:")
                first = tokens[0]
                print(f"  Keys: {list(first.keys())}")
                print(f"  Address: {first.get('address', 'N/A')}")
                print(f"  Symbol: {first.get('symbol', 'N/A')}")
                return tokens
        else:
            print(f"Error: {response.text[:300]}")
            
    except Exception as e:
        print(f"❌ Error: {e}")
    
    return None


def test_birdeye_top_traders(token_address: str):
    """Test Birdeye top traders for a specific token"""
    print("\n" + "="*60)
    print("TEST 2: BIRDEYE TOP TRADERS")
    print("="*60)
    
    if not BIRDEYE_KEY:
        print("❌ BIRDEYE_API_KEY not set")
        return
    
    url = "https://public-api.birdeye.so/defi/v2/tokens/top_traders"
    headers = {
        "X-API-KEY": BIRDEYE_KEY,
        "x-chain": "solana"
    }
    params = {
        "address": token_address,
        "time_frame": "24h",
        "sort_by": "volume",  # Valid: volume (not pnl!)
        "sort_type": "desc",
        "offset": 0,
        "limit": 10
    }
    
    print(f"URL: {url}")
    print(f"Token: {token_address}")
    
    try:
        response = requests.get(url, headers=headers, params=params, timeout=15)
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"Response keys: {list(data.keys())}")
            
            # Show raw response structure
            if 'data' in data:
                if isinstance(data['data'], dict):
                    print(f"data keys: {list(data['data'].keys())}")
                    if 'items' in data['data']:
                        print(f"data.items count: {len(data['data']['items'])}")
                elif isinstance(data['data'], list):
                    print(f"data is list with {len(data['data'])} items")
            
            # Try to extract items
            items = []
            if isinstance(data, dict):
                if data.get('success') and isinstance(data.get('data'), dict):
                    items = data['data'].get('items', data['data'].get('traders', []))
                elif isinstance(data.get('data'), dict):
                    items = data['data'].get('items', data['data'].get('traders', []))
                elif isinstance(data.get('data'), list):
                    items = data['data']
                elif 'items' in data:
                    items = data['items']
            
            print(f"\nExtracted {len(items)} traders")
            
            if items and len(items) > 0:
                print("\nFirst trader structure:")
                first = items[0]
                print(f"  Keys: {list(first.keys())}")
                
                # Show all fields
                for key, value in first.items():
                    print(f"  {key}: {value}")
                
                # Check which address field works
                print(f"\nAddress field check:")
                print(f"  address: {first.get('address')}")
                print(f"  owner: {first.get('owner')}")
                print(f"  wallet: {first.get('wallet')}")
                print(f"  trader: {first.get('trader')}")
                
                # Count profitable
                profitable = 0
                for item in items:
                    pnl = item.get('pnl') or item.get('realizedPnl') or 0
                    try:
                        if float(pnl) > 0:
                            profitable += 1
                    except:
                        pass
                
                print(f"\nProfitable traders (PnL > 0): {profitable}/{len(items)}")
            else:
                print("\n⚠️ No traders found - checking raw response...")
                print(json.dumps(data, indent=2, default=str)[:1000])
        else:
            print(f"Error: {response.text[:500]}")
            
    except Exception as e:
        print(f"❌ Error: {e}")


def test_helius_webhooks():
    """Test Helius webhook status and creation"""
    print("\n" + "="*60)
    print("TEST 3: HELIUS WEBHOOKS")
    print("="*60)
    
    if not HELIUS_KEY:
        print("❌ HELIUS_KEY not set")
        return
    
    # List existing webhooks
    list_url = f"https://api.helius.xyz/v0/webhooks?api-key={HELIUS_KEY}"
    
    print(f"Listing existing webhooks...")
    
    try:
        response = requests.get(list_url, timeout=10)
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            webhooks = response.json()
            print(f"\n✅ Found {len(webhooks)} existing webhook(s)")
            
            print(f"\nHelius webhook limits by plan:")
            print(f"  - Free: 2 webhooks")
            print(f"  - Developer: 5 webhooks")
            print(f"  - Business: 10 webhooks")
            
            if len(webhooks) >= 2:
                print(f"\n⚠️ You have {len(webhooks)} webhooks!")
                print(f"   If you're on Free plan, you've hit the limit!")
            
            total_addresses = 0
            for i, wh in enumerate(webhooks):
                webhook_id = wh.get('webhookID', 'unknown')
                addresses = wh.get('accountAddresses', [])
                wh_url = wh.get('webhookURL', 'unknown')
                total_addresses += len(addresses)
                print(f"\n   Webhook {i+1}: {webhook_id[:16]}...")
                print(f"      Addresses: {len(addresses)}/25")
                print(f"      URL: {wh_url[:50]}...")
            
            print(f"\n   Total addresses tracked: {total_addresses}")
        else:
            print(f"Error: {response.text[:300]}")
            
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Test webhook creation
    print(f"\n--- Testing webhook creation ---")
    
    if not WEBHOOK_URL:
        print("⚠️ WEBHOOK_URL not set - skipping creation test")
        return
    
    create_url = f"https://api.helius.xyz/v0/webhooks?api-key={HELIUS_KEY}"
    
    # Test 1: Empty addresses (current approach)
    print(f"\nTest A: Create with EMPTY addresses...")
    payload_empty = {
        "webhookURL": WEBHOOK_URL,
        "accountAddresses": [],
        "webhookType": "enhanced",
        "transactionTypes": ["SWAP"]
    }
    
    try:
        response = requests.post(create_url, json=payload_empty, timeout=10)
        print(f"Status: {response.status_code}")
        if response.status_code != 200:
            print(f"Error: {response.text[:300]}")
            print(f"\n⚠️ Empty addresses might not be allowed!")
    except Exception as e:
        print(f"Error: {e}")
    
    # Test 2: With one dummy address
    print(f"\nTest B: Create with ONE address (WSOL)...")
    test_address = "So11111111111111111111111111111111111111112"  # WSOL
    payload_with_addr = {
        "webhookURL": WEBHOOK_URL,
        "accountAddresses": [test_address],
        "webhookType": "enhanced",
        "transactionTypes": ["SWAP"]
    }
    
    try:
        response = requests.post(create_url, json=payload_with_addr, timeout=10)
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            webhook_id = result.get('webhookID')
            print(f"✅ SUCCESS! Created webhook: {webhook_id}")
            
            # Clean up
            print(f"Cleaning up test webhook...")
            delete_url = f"https://api.helius.xyz/v0/webhooks/{webhook_id}?api-key={HELIUS_KEY}"
            del_response = requests.delete(delete_url, timeout=10)
            print(f"Delete status: {del_response.status_code}")
        else:
            error_text = response.text
            print(f"Error: {error_text[:300]}")
            
            if 'limit' in error_text.lower() or 'maximum' in error_text.lower() or 'exceed' in error_text.lower():
                print(f"\n❌ WEBHOOK LIMIT REACHED!")
                print(f"   Your Helius plan doesn't allow more webhooks.")
                print(f"   Solutions:")
                print(f"   1. Upgrade Helius plan")
                print(f"   2. Delete existing webhooks")
                print(f"   3. Consolidate wallets into fewer webhooks")
            
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("🔍 DIAGNOSTIC SCRIPT - BIRDEYE & HELIUS")
    print("="*60)
    
    # Test 1: Birdeye trending tokens
    tokens = test_birdeye_trending()
    
    # Test 2: Birdeye top traders
    if tokens and len(tokens) > 0:
        first_token = tokens[0].get('address')
        if first_token:
            test_birdeye_top_traders(first_token)
        else:
            print("\n⚠️ Could not get token address from trending response")
    
    # Test 3: Helius webhooks
    test_helius_webhooks()
    
    print("\n" + "="*60)
    print("✅ DIAGNOSTICS COMPLETE")
    print("="*60)
