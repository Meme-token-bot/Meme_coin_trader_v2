"""
DexScreener API Endpoint Discovery
Find which endpoints actually work
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # Go up ONE level
import requests

def test_endpoint(name, url):
    """Test an endpoint and show results"""
    print(f"\n   {name}")
    print(f"   URL: {url}")
    
    try:
        res = requests.get(url, timeout=15)
        print(f"   Status: {res.status_code}")
        
        if res.status_code == 200:
            data = res.json()
            
            # Check what we got
            if isinstance(data, dict):
                print(f"   Keys: {list(data.keys())}")
                
                # Look for pairs
                pairs = data.get('pairs')
                if pairs is None:
                    print(f"   pairs: None ❌")
                elif isinstance(pairs, list):
                    print(f"   pairs: {len(pairs)} items ✅")
                    if pairs:
                        sample = pairs[0]
                        if isinstance(sample, dict):
                            base = sample.get('baseToken', {})
                            print(f"   Sample: ${base.get('symbol', '?')}")
                        return True, pairs
                else:
                    print(f"   pairs type: {type(pairs)}")
            elif isinstance(data, list):
                print(f"   Response is list with {len(data)} items")
                if data:
                    return True, data
            
            return False, None
        else:
            return False, None
            
    except Exception as e:
        print(f"   Error: {e}")
        return False, None


def main():
    print("\n" + "="*70)
    print("🔍 DEXSCREENER ENDPOINT DISCOVERY")
    print("="*70)
    
    # List of endpoints to test
    endpoints = [
        ("Trending tokens", "https://api.dexscreener.com/latest/dex/tokens/trending"),
        ("Search 'solana'", "https://api.dexscreener.com/latest/dex/search?q=solana"),
        ("Search 'pump'", "https://api.dexscreener.com/latest/dex/search?q=pump"),
        ("Search 'sol'", "https://api.dexscreener.com/latest/dex/search?q=sol"),
        ("Search 'meme'", "https://api.dexscreener.com/latest/dex/search?q=meme"),
        ("Token boosts", "https://api.dexscreener.com/token-boosts/latest/v1"),
        ("Token profiles", "https://api.dexscreener.com/token-profiles/latest/v1"),
        ("Pairs by chain", "https://api.dexscreener.com/latest/dex/pairs/solana"),
        ("Specific token (SOL)", "https://api.dexscreener.com/latest/dex/tokens/So11111111111111111111111111111111111111112"),
    ]
    
    working_endpoints = []
    
    for name, url in endpoints:
        success, data = test_endpoint(name, url)
        if success:
            working_endpoints.append((name, url, len(data) if data else 0))
    
    print("\n" + "="*70)
    print("📋 SUMMARY")
    print("="*70)
    
    if working_endpoints:
        print("\n   ✅ Working endpoints:")
        for name, url, count in working_endpoints:
            print(f"      {name}: {count} results")
            print(f"         {url}")
    else:
        print("\n   ❌ No working endpoints found!")
        print("      DexScreener may have changed their API")
    
    print("\n" + "="*70 + "\n")
    
    return working_endpoints


if __name__ == "__main__":
    main()
