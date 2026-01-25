"""
Test Discovery System
Quick test to verify the hybrid discovery system works
"""

import os
from dotenv import load_dotenv
load_dotenv()

from core.database_v2 import DatabaseV2
from historian import Historian

def test_discovery():
    print("\n" + "="*70)
    print("🧪 TESTING HYBRID DISCOVERY SYSTEM")
    print("="*70)
    
    # Check API keys
    helius_key = os.getenv('HELIUS_KEY')
    birdeye_key = os.getenv('BIRDEYE_API_KEY')
    
    if not helius_key:
        print("❌ ERROR: HELIUS_KEY not found in environment")
        return
    
    print(f"\n✅ Helius API key found")
    if birdeye_key:
        print(f"✅ Birdeye API key found")
    else:
        print(f"⚠️  Birdeye API key not found (optional - will use Helius only)")
    
    # Initialize database
    print(f"\n📦 Initializing database...")
    db = DatabaseV2('test_discovery.db')
    
    # Initialize historian
    print(f"\n📚 Initializing Historian...")
    historian = Historian(db, helius_key)
    
    # Run discovery with small budget for testing
    print(f"\n🎯 Running discovery (test mode - limited budget)...")
    print(f"   API Budget: 100 credits")
    print(f"   Max Wallets: 3")
    
    stats = historian.run_discovery(
        api_budget=100,  # Small budget for testing
        max_wallets=3    # Just find a few wallets
    )
    
    # Print results
    print(f"\n{'='*70}")
    print("📊 DISCOVERY TEST RESULTS")
    print(f"{'='*70}")
    print(f"   Tokens discovered: {stats.get('tokens_discovered', 0)}")
    print(f"   Wallet candidates: {stats.get('wallet_candidates_found', 0)}")
    print(f"   Wallets profiled: {stats.get('wallets_profiled', 0)}")
    print(f"   Wallets verified: {stats.get('wallets_verified', 0)}")
    print(f"   Helius API calls used: {stats.get('helius_api_calls', 0)}/100")
    
    # Verify data was saved
    wallet_count = db.get_wallet_count()
    print(f"\n   Total wallets in database: {wallet_count}")
    
    if wallet_count > 0:
        print(f"\n✅ SUCCESS! Discovery system is working")
        print(f"\nVerified wallets:")
        wallets = db.get_all_verified_wallets()
        for wallet in wallets:
            print(f"   {wallet['address'][:8]}... | WR: {wallet['win_rate']:.1%} | PnL: {wallet['pnl_7d']:.2f} SOL")
    else:
        print(f"\n⚠️  No wallets verified (try increasing budget or lowering thresholds)")
    
    print(f"\n{'='*70}\n")
    
    # Cleanup test database
    import os as os_mod
    try:
        os_mod.remove('test_discovery.db')
        print("✅ Test database cleaned up")
    except:
        pass

if __name__ == "__main__":
    try:
        test_discovery()
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted")
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
