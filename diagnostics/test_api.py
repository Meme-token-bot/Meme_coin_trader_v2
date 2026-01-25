"""
Simple API Test - Check if DexScreener is accessible
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # Go up ONE level
import requests

def test_dexscreener():
    print("\n" + "="*60)
    print("🧪 DEXSCREENER API TEST")
    print("="*60)
    
    # Test the correct endpoint
    url = "https://api.dexscreener.com/latest/dex/pairs/solana"
    
    print(f"\n   Testing: {url}")
    
    try:
        res = requests.get(url, timeout=15)
        print(f"   Status code: {res.status_code}")
        
        if res.status_code == 200:
            data = res.json()
            print(f"   Response type: {type(data)}")
            print(f"   Response keys: {list(data.keys()) if isinstance(data, dict) else 'N/A'}")
            
            pairs = data.get('pairs', [])
            print(f"   Pairs found: {len(pairs)}")
            
            if pairs:
                print(f"\n   ✅ API is working! Sample tokens:")
                for pair in pairs[:5]:
                    base = pair.get('baseToken', {})
                    symbol = base.get('symbol', '?')
                    liq = pair.get('liquidity', {})
                    liq_usd = liq.get('usd', 0) if isinstance(liq, dict) else 0
                    print(f"      ${symbol}: Liq ${liq_usd:,.0f}")
                return True
            else:
                print(f"\n   ⚠️  API returned empty pairs list")
                return False
        else:
            print(f"\n   ❌ API returned error status: {res.status_code}")
            print(f"   Response: {res.text[:200]}")
            return False
            
    except requests.exceptions.Timeout:
        print(f"\n   ❌ Request timed out - network issue?")
        return False
    except requests.exceptions.RequestException as e:
        print(f"\n   ❌ Network error: {e}")
        return False
    except Exception as e:
        print(f"\n   ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_old_endpoint():
    """Test the old endpoint that the original code uses"""
    print("\n" + "="*60)
    print("🧪 TESTING OLD ENDPOINT (used by original code)")
    print("="*60)
    
    url = "https://api.dexscreener.com/latest/dex/tokens/trending"
    
    print(f"\n   Testing: {url}")
    
    try:
        res = requests.get(url, timeout=15)
        print(f"   Status code: {res.status_code}")
        
        if res.status_code == 200:
            data = res.json()
            print(f"   Response: {str(data)[:200]}...")
            
            if data and 'pairs' in str(data):
                print(f"\n   ✅ Old endpoint works")
            else:
                print(f"\n   ⚠️  Old endpoint returns unexpected format")
        else:
            print(f"\n   ❌ Old endpoint returned: {res.status_code}")
            
    except Exception as e:
        print(f"\n   ❌ Error: {e}")


if __name__ == "__main__":
    # Test the correct endpoint
    working = test_dexscreener()
    
    # Also test the old one to see what's happening
    test_old_endpoint()
    
    print("\n" + "="*60)
    if working:
        print("✅ DexScreener API is accessible")
        print("   The issue is in how hybrid_discovery.py calls it")
        print("\n   Make sure you replaced hybrid_discovery.py with")
        print("   the fixed_hybrid_discovery_v2.py file!")
    else:
        print("❌ DexScreener API is not accessible")
        print("   Check your internet connection")
    print("="*60 + "\n")
