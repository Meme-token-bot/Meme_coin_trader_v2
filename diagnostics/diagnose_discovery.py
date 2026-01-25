"""
Discovery Diagnostic Tool
Run this to understand why wallets aren't being discovered
"""

import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # Go up ONE level
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

def check_environment():
    """Check all required environment variables"""
    print("\n" + "="*70)
    print("1. ENVIRONMENT CHECK")
    print("="*70)
    
    required = {
        'HELIUS_KEY': os.getenv('HELIUS_KEY'),
        'HELIUS_DISCOVERY_KEY': os.getenv('HELIUS_DISCOVERY_KEY'),
        'BIRDEYE_API_KEY': os.getenv('BIRDEYE_API_KEY'),
        'WEBHOOK_URL': os.getenv('WEBHOOK_URL'),
    }
    
    issues = []
    for key, value in required.items():
        if value:
            masked = value[:8] + "..." if len(value) > 8 else value
            print(f"   ✅ {key}: {masked}")
        else:
            print(f"   ❌ {key}: NOT SET")
            if key == 'HELIUS_KEY':
                issues.append("HELIUS_KEY is required for discovery")
    
    return len(issues) == 0, issues


def check_database():
    """Check database state"""
    print("\n" + "="*70)
    print("2. DATABASE STATE")
    print("="*70)
    
    try:
        # Import here to avoid issues if DB doesn't exist
        sys.path.insert(0, '/mnt/project')
        from core.database_v2 import DatabaseV2
        
        db = DatabaseV2()
        
        # Count wallets
        wallet_count = db.get_wallet_count()
        print(f"   Total tracked wallets: {wallet_count}")
        
        # Check wallet cap
        from core.discovery_config import config
        remaining = config.max_total_wallets - wallet_count
        print(f"   Wallet cap: {config.max_total_wallets}")
        print(f"   Slots remaining: {remaining}")
        
        if remaining <= 0:
            print(f"   ⚠️  AT CAPACITY - Discovery will be skipped!")
            return False, ["Wallet capacity reached"]
        
        # Check recent discoveries
        with db.connection() as conn:
            recent = conn.execute("""
                SELECT COUNT(*) FROM verified_wallets
                WHERE discovered_at >= datetime('now', '-7 days')
            """).fetchone()[0]
            
            print(f"   Wallets discovered (last 7 days): {recent}")
            
            # Check discovery cycles if table exists
            try:
                cycles = conn.execute("""
                    SELECT cycle_start, wallets_verified, helius_calls_used
                    FROM discovery_cycles
                    ORDER BY cycle_start DESC
                    LIMIT 5
                """).fetchall()
                
                if cycles:
                    print(f"\n   Recent discovery cycles:")
                    for cycle in cycles:
                        print(f"      {cycle[0]} - Found: {cycle[1]} wallets, API: {cycle[2]} calls")
                else:
                    print(f"\n   ⚠️  No discovery cycle history found")
            except:
                print(f"\n   ⚠️  Discovery cycle tracking table doesn't exist")
        
        return True, []
    
    except Exception as e:
        print(f"   ❌ Database error: {e}")
        return False, [str(e)]


def test_token_discovery():
    """Test the token discovery APIs (FREE)"""
    print("\n" + "="*70)
    print("3. TOKEN DISCOVERY TEST (DexScreener - FREE)")
    print("="*70)
    
    try:
        sys.path.insert(0, '/mnt/project')
        from hybrid_discovery import TokenDiscoveryEngine
        
        engine = TokenDiscoveryEngine()
        
        # Test pumping tokens
        print("\n   Testing pumping token discovery...")
        pumping = engine.find_pumping_tokens(min_gain=50, limit=5)  # Lower threshold for test
        print(f"   Found {len(pumping)} pumping tokens")
        
        if pumping:
            for t in pumping[:3]:
                print(f"      ${t['symbol']}: +{t['price_change_24h']:.0f}% | Liq: ${t['liquidity']:,.0f}")
        else:
            print("   ⚠️  No pumping tokens found - this is unusual!")
        
        # Test trending tokens
        print("\n   Testing trending token discovery...")
        trending = engine.find_trending_tokens(min_volume=25000, limit=5)
        print(f"   Found {len(trending)} trending tokens")
        
        if trending:
            for t in trending[:3]:
                print(f"      ${t['symbol']}: Vol ${t['volume_24h']:,.0f} | Liq: ${t['liquidity']:,.0f}")
        
        total_tokens = len(set([t['address'] for t in pumping + trending]))
        print(f"\n   ✅ Total unique tokens available: {total_tokens}")
        
        return total_tokens > 0, pumping + trending
    
    except Exception as e:
        print(f"   ❌ Token discovery error: {e}")
        import traceback
        traceback.print_exc()
        return False, []


def test_wallet_extraction(tokens):
    """Test wallet extraction from tokens (USES HELIUS CREDITS)"""
    print("\n" + "="*70)
    print("4. WALLET EXTRACTION TEST (Helius - ~6 credits per token)")
    print("="*70)
    
    helius_key = os.getenv('HELIUS_KEY')
    if not helius_key:
        print("   ❌ Cannot test - HELIUS_KEY not set")
        return False, []
    
    if not tokens:
        print("   ❌ Cannot test - no tokens from previous step")
        return False, []
    
    try:
        sys.path.insert(0, '/mnt/project')
        from hybrid_discovery import WalletDiscoveryEngine
        
        engine = WalletDiscoveryEngine(helius_key)
        
        # Test on just ONE token to save credits
        test_token = tokens[0]
        print(f"\n   Testing holder extraction for ${test_token['symbol']}...")
        print(f"   Token address: {test_token['address']}")
        
        holders = engine.get_top_holders(test_token['address'], limit=3)
        
        print(f"   Found {len(holders)} holder wallet(s)")
        
        if holders:
            for h in holders:
                print(f"      {h[:16]}...")
            return True, holders
        else:
            print("   ⚠️  No holders found - check Helius API access")
            return False, []
    
    except Exception as e:
        print(f"   ❌ Wallet extraction error: {e}")
        import traceback
        traceback.print_exc()
        return False, []


def test_wallet_profiling(wallets):
    """Test wallet profiling (USES HELIUS CREDITS)"""
    print("\n" + "="*70)
    print("5. WALLET PROFILING TEST (Helius - ~25 credits per wallet)")
    print("="*70)
    
    helius_key = os.getenv('HELIUS_KEY')
    if not helius_key:
        print("   ❌ Cannot test - HELIUS_KEY not set")
        return False
    
    if not wallets:
        print("   ❌ Cannot test - no wallets from previous step")
        return False
    
    try:
        sys.path.insert(0, '/mnt/project')
        from core.database_v2 import DatabaseV2
        from historian import TokenScanner, WalletProfiler
        
        db = DatabaseV2()
        scanner = TokenScanner(db, helius_key)
        profiler = WalletProfiler(scanner)
        
        # Test on ONE wallet to save credits
        test_wallet = wallets[0]
        print(f"\n   Profiling wallet: {test_wallet[:16]}...")
        
        performance = profiler.calculate_performance(test_wallet, days=7)
        
        print(f"\n   Results:")
        print(f"      Win rate: {performance['win_rate']:.1%}")
        print(f"      PnL (7d): {performance['pnl']:.2f} SOL")
        print(f"      Completed swings: {performance['completed_swings']}")
        print(f"      Avg hold: {performance['avg_hold_hours']:.1f}h")
        
        # Check against thresholds
        print(f"\n   Verification check:")
        print(f"      Win rate >= 55%: {'✅' if performance['win_rate'] >= 0.55 else '❌'} ({performance['win_rate']:.1%})")
        print(f"      PnL >= 3 SOL: {'✅' if performance['pnl'] >= 3.0 else '❌'} ({performance['pnl']:.2f})")
        print(f"      Swings >= 5: {'✅' if performance['completed_swings'] >= 5 else '❌'} ({performance['completed_swings']})")
        
        would_verify = (
            performance['win_rate'] >= 0.55 and
            performance['pnl'] >= 3.0 and
            performance['completed_swings'] >= 5
        )
        
        if would_verify:
            print(f"\n   ✅ This wallet WOULD be verified!")
        else:
            print(f"\n   ❌ This wallet would NOT pass verification")
            print(f"      (This is normal - most wallets don't qualify)")
        
        return True
    
    except Exception as e:
        print(f"   ❌ Profiling error: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_discovery_config():
    """Check discovery configuration"""
    print("\n" + "="*70)
    print("6. DISCOVERY CONFIGURATION")
    print("="*70)
    
    try:
        sys.path.insert(0, '/mnt/project')
        from core.discovery_config import config
        
        print(f"\n   Schedule:")
        print(f"      Discovery interval: {config.discovery_interval_hours} hours")
        
        print(f"\n   Per-cycle limits:")
        print(f"      Max tokens to scan: {config.max_tokens_to_scan}")
        print(f"      Max candidates to profile: {config.max_candidates_to_profile}")
        print(f"      Max API calls: {config.max_api_calls_per_discovery}")
        
        print(f"\n   Verification thresholds (from config):")
        print(f"      Min win rate: {config.min_win_rate:.0%}")
        print(f"      Min PnL: {config.min_pnl} SOL")
        print(f"      Min swings: {config.min_completed_swings}")
        
        print(f"\n   ⚠️  NOTE: hybrid_discovery.py uses STRICTER thresholds:")
        print(f"      Min win rate: 55% (config says {config.min_win_rate:.0%})")
        print(f"      Min PnL: 3 SOL (config says {config.min_pnl})")
        print(f"      Min swings: 5 (config says {config.min_completed_swings})")
        
        print(f"\n   Daily limits:")
        print(f"      Max new wallets per day: {config.max_new_wallets_per_day}")
        print(f"      Max total wallets: {config.max_total_wallets}")
        
        return True
    
    except Exception as e:
        print(f"   ❌ Config error: {e}")
        return False


def run_full_diagnostic():
    """Run complete diagnostic"""
    print("\n" + "="*70)
    print("🔍 DISCOVERY SYSTEM DIAGNOSTIC")
    print("="*70)
    print(f"   Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Run all checks
    env_ok, env_issues = check_environment()
    db_ok, db_issues = check_database()
    check_discovery_config()
    
    tokens_ok, tokens = test_token_discovery()
    
    # Only test Helius calls if user confirms (costs credits)
    print("\n" + "="*70)
    print("⚠️  HELIUS API TESTS")
    print("="*70)
    print("   The next tests will use Helius API credits (~31 credits total)")
    response = input("   Run Helius tests? (y/n): ").strip().lower()
    
    wallets = []
    if response == 'y':
        wallets_ok, wallets = test_wallet_extraction(tokens)
        if wallets:
            test_wallet_profiling(wallets)
    else:
        print("   Skipping Helius tests")
    
    # Summary
    print("\n" + "="*70)
    print("📋 DIAGNOSTIC SUMMARY")
    print("="*70)
    
    issues = []
    
    if not env_ok:
        issues.extend(env_issues)
    
    if not db_ok:
        issues.extend(db_issues)
    
    if not tokens_ok:
        issues.append("Token discovery is failing - check internet connection")
    
    if issues:
        print("\n   ❌ ISSUES FOUND:")
        for issue in issues:
            print(f"      • {issue}")
    else:
        print("\n   ✅ No critical issues found!")
    
    print("\n   💡 LIKELY REASONS FOR LOW DISCOVERY:")
    print("      1. Strict thresholds (55% WR, 3 SOL PnL, 5 swings)")
    print("      2. Only runs every 24 hours")
    print("      3. Only scans 8 tokens × 5 holders = 40 candidates max")
    print("      4. Most wallets don't meet quality thresholds")
    
    print("\n   🔧 SUGGESTED FIXES:")
    print("      1. Lower verification thresholds in hybrid_discovery.py")
    print("      2. Scan more tokens per cycle (increase limit)")
    print("      3. Run discovery more frequently")
    print("      4. Add manual seeding of known good wallets")
    
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    run_full_diagnostic()
