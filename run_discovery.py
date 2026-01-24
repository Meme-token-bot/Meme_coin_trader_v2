"""
Manual Discovery Runner v2
Run discovery manually with optimized budget and detailed output

USAGE:
  python run_discovery.py              - Run discovery with default budget (5000)
  python run_discovery.py 10000        - Run with custom budget
  python run_discovery.py history      - Show discovery history  
  python run_discovery.py candidates   - Show candidates from last run
  python run_discovery.py budget       - Show budget calculator
  python run_discovery.py help         - Show help
"""

import os
import sys
import json
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# Add project path
sys.path.insert(0, os.getcwd())

# Global to store profiled results during discovery
_profiled_results = []

def run_manual_discovery(api_budget: int = 5000, max_wallets: int = 15):
    """Run discovery with specified budget"""
    global _profiled_results
    _profiled_results = []
    
    print("\n" + "="*70)
    print("ðŸŽ¯ MANUAL DISCOVERY RUN v2")
    print(f"   Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   Budget: {api_budget:,} credits")
    print(f"   Target: {max_wallets} wallets")
    print("="*70)
    
    # Check environment
    helius_key = os.getenv('HELIUS_KEY')
    if not helius_key:
        print("\nâŒ ERROR: HELIUS_KEY not found in environment")
        print("   Add it to your .env file")
        return
    
    # Initialize components
    print("\nðŸ“¦ Initializing components...")
    
    from database_v2 import DatabaseV2
    from historian import Historian
    
    db = DatabaseV2()
    
    # Show current state
    wallet_count = db.get_wallet_count()
    print(f"   Current wallet count: {wallet_count}")
    
    # Check capacity
    try:
        from discovery_config import config
        max_total = config.max_total_wallets
    except:
        max_total = 100
    
    remaining_slots = max_total - wallet_count
    print(f"   Max wallet limit: {max_total}")
    print(f"   Remaining slots: {remaining_slots}")
    
    if wallet_count >= max_total:
        print(f"\nâš ï¸  WARNING: At wallet capacity!")
        print(f"   Either increase max_total_wallets in discovery_config.py")
        print(f"   Or remove some inactive wallets")
        return
    
    # Adjust max_wallets if near capacity
    max_wallets = min(max_wallets, remaining_slots)
    
    # Get discovery key
    discovery_key = os.getenv('HELIUS_DISCOVERY_KEY')
    
    # Initialize multi-webhook manager for auto-add
    webhook_url = os.getenv('WEBHOOK_URL')
    multi_webhook_manager = None
    
    if webhook_url:
        try:
            from multi_webhook_manager import MultiWebhookManager
            multi_webhook_manager = MultiWebhookManager(helius_key, webhook_url, db)
            print(f"   ✅ Webhook auto-add enabled: {webhook_url[:50]}...")
        except Exception as e:
            print(f"   ⚠️  Webhook auto-add disabled: {e}")
    else:
        print("   ⚠️  WEBHOOK_URL not set - auto-add disabled")
    
    # Initialize historian WITH webhook manager
    print("\n📚 Initializing Historian...")
    historian = Historian(db, helius_key, discovery_key, multi_webhook_manager=multi_webhook_manager)
    
    # Patch discovery for debugging
    patch_discovery_for_debug(historian)
    
    # Run discovery with specified budget
    print(f"\nðŸš€ Starting discovery (budget: {api_budget:,} credits)...")
    print("   (This may take a few minutes)\n")
    
    try:
        stats = historian.run_discovery(
            api_budget=api_budget,
            max_wallets=max_wallets
        )
        
        # Save debug data
        save_debug_data(stats)
        
        # Summary
        print("\n" + "="*70)
        print("ðŸ“Š DISCOVERY RESULTS")
        print("="*70)
        
        print(f"\n   Tokens scanned: {stats.get('tokens_discovered', 0)}")
        print(f"   Total candidates: {stats.get('wallet_candidates_found', 0)}")
        
        # Pre-filter stats
        prefiltered = stats.get('wallets_prefiltered_out', 0)
        saved = stats.get('credits_saved_by_prefilter', 0)
        if prefiltered > 0:
            print(f"   Pre-filtered out: {prefiltered} (saved ~{saved:,} credits)")
        
        print(f"   Wallets profiled: {stats.get('wallets_profiled', 0)}")
        print(f"   Wallets VERIFIED: {stats.get('wallets_verified', 0)} âœ…")
        print(f"   API calls used: {stats.get('helius_api_calls', 0):,}/{api_budget:,}")
        
        # Show verification rate
        profiled = stats.get('wallets_profiled', 0)
        verified = stats.get('wallets_verified', 0)
        if profiled > 0:
            rate = 100 * verified / profiled
            print(f"   Verification rate: {rate:.0f}%")
        
        # Show new wallets with full addresses
        new_count = db.get_wallet_count()
        added = new_count - wallet_count
        
        if added > 0:
            print(f"\n   âœ… Added {added} new wallet(s)!")
            
            with db.connection() as conn:
                new_wallets = conn.execute("""
                    SELECT address, win_rate, pnl_7d, completed_swings
                    FROM verified_wallets
                    ORDER BY discovered_at DESC
                    LIMIT ?
                """, (added,)).fetchall()
                
                print("\n   NEW VERIFIED WALLETS (full addresses):")
                print("   " + "-"*60)
                for w in new_wallets:
                    print(f"\n   {w[0]}")
                    print(f"      WR: {w[1]:.1%} | PnL: {w[2]:.2f} SOL | Swings: {w[3]}")
        else:
            print(f"\n   âš ï¸  No new wallets added")
            
            # Show rejection breakdown
            rejection = stats.get('rejection_reasons', {})
            prefilter = stats.get('prefilter_reasons', {})
            
            if prefilter:
                print(f"\n   ðŸ“‹ Pre-filter breakdown:")
                for reason, count in sorted(prefilter.items(), key=lambda x: -x[1]):
                    print(f"      {reason}: {count}")
            
            if rejection:
                print(f"\n   ðŸ“‹ Profile rejection breakdown:")
                for reason, count in sorted(rejection.items(), key=lambda x: -x[1]):
                    print(f"      {reason}: {count}")
            
            print(f"\n   ðŸ’¡ Debug tips:")
            print(f"      1. View candidates: python run_discovery.py candidates")
            print(f"      2. Debug a wallet: python debug_profiler.py <address>")
            print(f"      3. Full data saved to: discovery_debug.json")
        
        print("\n" + "="*70 + "\n")
        
    except Exception as e:
        print(f"\nâŒ Discovery error: {e}")
        import traceback
        traceback.print_exc()


def patch_discovery_for_debug(historian):
    """Patch discovery system to capture wallet data for debugging"""
    global _profiled_results
    
    if not hasattr(historian, 'hybrid_discovery'):
        return
    
    hd = historian.hybrid_discovery
    
    if not hasattr(hd, '_profile_wallet'):
        return
    
    original_profile = hd._profile_wallet
    
    def patched_profile(wallet):
        result = original_profile(wallet)
        _profiled_results.append({
            'address': wallet,
            'win_rate': result.get('win_rate', 0),
            'pnl': result.get('pnl', 0),
            'completed_swings': result.get('completed_swings', 0),
            'avg_hold_hours': result.get('avg_hold_hours', 0),
        })
        return result
    
    hd._profile_wallet = patched_profile


def save_debug_data(stats):
    """Save discovery debug data to file"""
    global _profiled_results
    
    debug_data = {
        'timestamp': datetime.now().isoformat(),
        'stats': {
            'tokens_discovered': stats.get('tokens_discovered', 0),
            'wallet_candidates_found': stats.get('wallet_candidates_found', 0),
            'wallets_prefiltered_out': stats.get('wallets_prefiltered_out', 0),
            'credits_saved_by_prefilter': stats.get('credits_saved_by_prefilter', 0),
            'wallets_profiled': stats.get('wallets_profiled', 0),
            'wallets_verified': stats.get('wallets_verified', 0),
            'helius_api_calls': stats.get('helius_api_calls', 0),
        },
        'prefilter_reasons': stats.get('prefilter_reasons', {}),
        'rejection_reasons': stats.get('rejection_reasons', {}),
        'verified_wallets': stats.get('verified_wallets', []),
        'profiled_wallets': _profiled_results,
    }
    
    with open('discovery_debug.json', 'w') as f:
        json.dump(debug_data, f, indent=2, default=str)
    
    print(f"\n   ðŸ“ Debug data saved to: discovery_debug.json")


def show_discovery_history():
    """Show recent discovery activity"""
    
    print("\n" + "="*70)
    print("ðŸ“œ DISCOVERY HISTORY")
    print("="*70)
    
    from database_v2 import DatabaseV2
    db = DatabaseV2()
    
    with db.connection() as conn:
        wallets = conn.execute("""
            SELECT 
                DATE(discovered_at) as date,
                COUNT(*) as count
            FROM verified_wallets
            WHERE discovered_at IS NOT NULL
            GROUP BY DATE(discovered_at)
            ORDER BY date DESC
            LIMIT 14
        """).fetchall()
        
        if wallets:
            print("\n   Wallets discovered by date:")
            total_recent = 0
            for row in wallets:
                print(f"      {row[0]}: {row[1]} wallet(s)")
                total_recent += row[1]
            print(f"      â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€")
            print(f"      Total (14 days): {total_recent}")
        else:
            print("\n   No discovery history found")
        
        total = conn.execute("SELECT COUNT(*) FROM verified_wallets").fetchone()[0]
        active = conn.execute("SELECT COUNT(*) FROM verified_wallets WHERE is_active = 1").fetchone()[0]
        
        print(f"\n   Total wallets: {total}")
        print(f"   Active wallets: {active}")
        
        # Show top performers
        top = conn.execute("""
            SELECT address, win_rate, pnl_7d, completed_swings
            FROM verified_wallets
            WHERE is_active = 1
            ORDER BY pnl_7d DESC
            LIMIT 5
        """).fetchall()
        
        if top:
            print(f"\n   ðŸ† Top performing wallets:")
            for w in top:
                print(f"      {w[0][:12]}... | WR: {w[1]:.1%} | PnL: {w[2]:.2f} SOL")
    
    print("\n" + "="*70 + "\n")


def show_candidates():
    """Show candidates from last discovery"""
    
    print("\n" + "="*70)
    print("ðŸ“‹ DISCOVERY CANDIDATES (from last run)")
    print("="*70)
    
    try:
        with open('discovery_debug.json', 'r') as f:
            data = json.load(f)
        
        print(f"\n   Discovery time: {data.get('timestamp', 'Unknown')}")
        stats = data.get('stats', {})
        
        print(f"\n   ðŸ“Š Summary:")
        print(f"      Candidates found: {stats.get('wallet_candidates_found', 0)}")
        print(f"      Pre-filtered out: {stats.get('wallets_prefiltered_out', 0)}")
        print(f"      Profiled: {stats.get('wallets_profiled', 0)}")
        print(f"      Verified: {stats.get('wallets_verified', 0)}")
        
        # Pre-filter breakdown
        prefilter = data.get('prefilter_reasons', {})
        if prefilter:
            print(f"\n   ðŸš« Pre-filter reasons:")
            for reason, count in sorted(prefilter.items(), key=lambda x: -x[1]):
                print(f"      {reason}: {count}")
        
        # Profiled wallets
        profiled = data.get('profiled_wallets', [])
        if profiled:
            print(f"\n   ðŸ“ PROFILED WALLETS ({len(profiled)} total):")
            print("   " + "-"*60)
            
            # Check thresholds
            try:
                from discovery_config import config
                min_wr = config.min_win_rate
                min_pnl = config.min_pnl
                min_swings = config.min_completed_swings
            except:
                min_wr, min_pnl, min_swings = 0.50, 2.0, 3
            
            for w in profiled:
                addr = w.get('address', 'Unknown')
                wr = w.get('win_rate', 0)
                pnl = w.get('pnl', 0)
                swings = w.get('completed_swings', 0)
                
                # Check each criterion
                wr_ok = wr >= min_wr
                pnl_ok = pnl >= min_pnl
                swings_ok = swings >= min_swings
                
                passed = wr_ok and pnl_ok and swings_ok
                status = 'âœ…' if passed else 'âŒ'
                
                print(f"\n   {status} {addr}")
                print(f"      WR: {wr:.1%} {'âœ“' if wr_ok else 'âœ—'} | "
                      f"PnL: {pnl:.2f} {'âœ“' if pnl_ok else 'âœ—'} | "
                      f"Swings: {swings} {'âœ“' if swings_ok else 'âœ—'}")
        
        # Verified wallets
        verified = data.get('verified_wallets', [])
        if verified:
            print(f"\n   âœ… VERIFIED WALLETS ({len(verified)}):")
            print("   " + "-"*60)
            for w in verified:
                addr = w.get('address', 'Unknown')
                print(f"   {addr}")
                print(f"      WR: {w.get('win_rate', 0):.1%} | "
                      f"PnL: {w.get('pnl', 0):.2f} SOL | "
                      f"Swings: {w.get('swings', 0)}")
        
        print(f"\n   ðŸ’¡ To debug a specific wallet:")
        print(f"      python debug_profiler.py <wallet_address>")
        
    except FileNotFoundError:
        print("\n   âŒ No discovery_debug.json found")
        print("   Run discovery first: python run_discovery.py")
    except Exception as e:
        print(f"\n   âŒ Error: {e}")
    
    print("\n" + "="*70 + "\n")


def show_budget_calculator():
    """Show budget allocation and recommendations"""
    
    try:
        from discovery_config import calculate_monthly_budget
        calculate_monthly_budget()
    except ImportError:
        print("\n" + "="*60)
        print("ðŸ“Š BUDGET CALCULATOR")
        print("="*60)
        
        monthly = 1_000_000
        daily = monthly // 30
        
        print(f"\nðŸ’° Monthly credits: {monthly:,}")
        print(f"ðŸ“… Daily budget: {daily:,}")
        
        print(f"\nðŸ“ˆ Recommended per-run budget:")
        print(f"   Conservative: 3,000 credits")
        print(f"   Standard: 5,000 credits")
        print(f"   Aggressive: 10,000 credits")
        
        print(f"\nðŸ”„ With 2 runs/day at 5,000 each:")
        print(f"   Discovery: ~10,000/day")
        print(f"   Webhook monitoring: ~5,000/day")
        print(f"   Total: ~15,000/day = 450,000/month (45%)")
        
        print("\n" + "="*60 + "\n")


def show_help():
    """Show usage help"""
    print("""
â•”â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•—
â•‘                   DISCOVERY RUNNER v2 - HELP                     â•‘
â• â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•£
â•‘                                                                  â•‘
â•‘  USAGE:                                                          â•‘
â•‘    python run_discovery.py [budget]    Run with budget (5000)    â•‘
â•‘    python run_discovery.py history     Show discovery history    â•‘
â•‘    python run_discovery.py candidates  Show last run candidates  â•‘
â•‘    python run_discovery.py budget      Show budget calculator    â•‘
â•‘    python run_discovery.py help        Show this help            â•‘
â•‘                                                                  â•‘
â•‘  EXAMPLES:                                                       â•‘
â•‘    python run_discovery.py             # Default 5000 credits    â•‘
â•‘    python run_discovery.py 10000       # Use 10000 credits       â•‘
â•‘    python run_discovery.py 3000        # Conservative run        â•‘
â•‘                                                                  â•‘
â•‘  DEBUG:                                                          â•‘
â•‘    python debug_profiler.py <address>  Debug specific wallet     â•‘
â•‘                                                                  â•‘
â•‘  BUDGET RECOMMENDATIONS (1M credits/month):                      â•‘
â•‘    Conservative: 3,000 per run (can run 5+ times/day)           â•‘
â•‘    Standard:     5,000 per run (can run 3 times/day)            â•‘
â•‘    Aggressive:  10,000 per run (can run 2 times/day)            â•‘
â•‘                                                                  â•‘
â•šâ•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
""")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        cmd = sys.argv[1].lower()
        
        if cmd == 'history':
            show_discovery_history()
        elif cmd == 'candidates':
            show_candidates()
        elif cmd == 'budget':
            show_budget_calculator()
        elif cmd == 'help' or cmd == '-h' or cmd == '--help':
            show_help()
        elif cmd.isdigit():
            # Custom budget
            budget = int(cmd)
            if budget < 100:
                print(f"âš ï¸  Budget too low ({budget}). Minimum recommended: 1000")
            elif budget > 20000:
                print(f"âš ï¸  Budget very high ({budget}). Are you sure? Max recommended: 10000")
                confirm = input("Continue? (y/n): ").strip().lower()
                if confirm == 'y':
                    run_manual_discovery(api_budget=budget)
            else:
                run_manual_discovery(api_budget=budget)
        else:
            print(f"Unknown command: {cmd}")
            show_help()
    else:
        # Default: run with standard budget
        run_manual_discovery(api_budget=5000)
