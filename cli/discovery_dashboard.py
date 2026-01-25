"""
Discovery Performance Monitor
Track how well your hybrid discovery system is performing
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # Go up ONE level
from core.database_v2 import DatabaseV2
from datetime import datetime, timedelta
import json


class DiscoveryMonitor:
    """Monitor and analyze discovery performance"""
    
    def __init__(self, db_path: str = "swing_traders.db"):
        self.db = DatabaseV2(db_path)
    
    def print_summary(self, days: int = 7):
        """Print comprehensive discovery summary"""
        print("\n" + "="*70)
        print(f"📊 DISCOVERY PERFORMANCE - Last {days} Days")
        print("="*70)
        
        # Wallet discovery stats
        self._print_wallet_stats(days)
        
        # Token discovery stats
        self._print_token_stats(days)
        
        # Best discoveries
        self._print_best_discoveries(days)
        
        # API usage estimate
        self._print_api_usage(days)
        
        print("="*70 + "\n")
    
    def _print_wallet_stats(self, days: int):
        """Print wallet discovery statistics"""
        print(f"\n🔍 WALLET DISCOVERY")
        print("-" * 70)
        
        with self.db.connection() as conn:
            # Total wallets discovered
            cutoff = datetime.now() - timedelta(days=days)
            
            total = conn.execute("""
                SELECT COUNT(*) FROM verified_wallets
                WHERE discovered_at >= ?
            """, (cutoff.isoformat(),)).fetchone()[0]
            
            # Average performance
            avg_stats = conn.execute("""
                SELECT 
                    AVG(win_rate) as avg_wr,
                    AVG(pnl_7d) as avg_pnl,
                    AVG(completed_swings) as avg_swings
                FROM verified_wallets
                WHERE discovered_at >= ?
            """, (cutoff.isoformat(),)).fetchone()
            
            # Best performers
            best = conn.execute("""
                SELECT address, win_rate, pnl_7d
                FROM verified_wallets
                WHERE discovered_at >= ?
                ORDER BY pnl_7d DESC
                LIMIT 3
            """, (cutoff.isoformat(),)).fetchall()
            
            print(f"  Total wallets discovered: {total}")
            
            if avg_stats[0]:
                print(f"  Average win rate: {avg_stats[0]:.1%}")
                print(f"  Average PnL: {avg_stats[1]:.2f} SOL")
                print(f"  Average swings: {avg_stats[2]:.1f}")
            
            if best:
                print(f"\n  Top 3 discoveries:")
                for i, wallet in enumerate(best, 1):
                    print(f"    #{i}: {wallet[0][:8]}... | WR: {wallet[1]:.1%} | PnL: {wallet[2]:.2f} SOL")
    
    def _print_token_stats(self, days: int):
        """Print token discovery statistics"""
        print(f"\n📈 TOKEN DISCOVERY")
        print("-" * 70)
        
        with self.db.connection() as conn:
            # Check if table exists
            tables = conn.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name='discovered_tokens'
            """).fetchone()
            
            if not tables:
                print("  No token data yet (table not created)")
                return
            
            cutoff = datetime.now() - timedelta(days=days)
            
            # Tokens by source
            by_source = conn.execute("""
                SELECT source, COUNT(*) as count
                FROM discovered_tokens
                WHERE discovered_at >= ?
                GROUP BY source
            """, (cutoff.isoformat(),)).fetchall()
            
            # Best pumpers
            pumpers = conn.execute("""
                SELECT symbol, price_change_24h, liquidity
                FROM discovered_tokens
                WHERE discovered_at >= ?
                AND price_change_24h > 0
                ORDER BY price_change_24h DESC
                LIMIT 5
            """, (cutoff.isoformat(),)).fetchall()
            
            total_tokens = sum(row[1] for row in by_source)
            print(f"  Total tokens scanned: {total_tokens}")
            
            if by_source:
                print(f"  By source:")
                for source, count in by_source:
                    print(f"    {source}: {count} tokens")
            
            if pumpers:
                print(f"\n  Top pumpers discovered:")
                for symbol, change, liq in pumpers:
                    print(f"    ${symbol}: +{change:.0f}% | Liq: ${liq:,.0f}")
    
    def _print_best_discoveries(self, days: int):
        """Print wallets that led to most discoveries"""
        print(f"\n🎯 DISCOVERY SOURCES")
        print("-" * 70)
        
        with self.db.connection() as conn:
            # Check if table exists
            tables = conn.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name='wallet_discovery_metadata'
            """).fetchone()
            
            if not tables:
                print("  No discovery metadata yet")
                return
            
            cutoff = datetime.now() - timedelta(days=days)
            
            # Most common tokens
            all_tokens = conn.execute("""
                SELECT seen_in_symbols FROM wallet_discovery_metadata
                WHERE discovered_at >= ?
            """, (cutoff.isoformat(),)).fetchall()
            
            if not all_tokens:
                print("  No discovery data yet")
                return
            
            # Count token appearances
            token_counts = {}
            for row in all_tokens:
                try:
                    symbols = json.loads(row[0])
                    for symbol in symbols:
                        token_counts[symbol] = token_counts.get(symbol, 0) + 1
                except:
                    pass
            
            if token_counts:
                top_tokens = sorted(token_counts.items(), key=lambda x: x[1], reverse=True)[:5]
                print(f"  Tokens that led to most wallet discoveries:")
                for symbol, count in top_tokens:
                    print(f"    ${symbol}: {count} wallets discovered")
    
    def _print_api_usage(self, days: int):
        """Estimate API usage"""
        print(f"\n💰 ESTIMATED API USAGE")
        print("-" * 70)
        
        with self.db.connection() as conn:
            cutoff = datetime.now() - timedelta(days=days)
            
            # Wallets profiled (estimate 25 credits each)
            wallets = conn.execute("""
                SELECT COUNT(*) FROM verified_wallets
                WHERE discovered_at >= ?
            """, (cutoff.isoformat(),)).fetchone()[0]
            
            # Assume 2x profiling rate (50% rejection)
            estimated_profiled = wallets * 2
            estimated_credits = estimated_profiled * 25
            
            daily_avg = estimated_credits / max(days, 1)
            monthly_projection = daily_avg * 30
            
            print(f"  Wallets verified: {wallets}")
            print(f"  Estimated wallets profiled: {estimated_profiled}")
            print(f"  Estimated Helius credits used: {estimated_credits:,.0f}")
            print(f"  Daily average: {daily_avg:,.0f} credits")
            print(f"  Monthly projection: {monthly_projection:,.0f} credits ({monthly_projection/1000000*100:.1f}% of 1M budget)")
            
            if monthly_projection < 100000:
                print(f"  ✅ Well within budget!")
            elif monthly_projection < 500000:
                print(f"  ⚠️ Moderate usage - monitor closely")
            else:
                print(f"  🚨 High usage - consider reducing frequency")
    
    def export_discovery_report(self, filename: str = "discovery_report.txt"):
        """Export detailed discovery report"""
        with open(filename, 'w') as f:
            f.write(f"DISCOVERY REPORT - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*70 + "\n\n")
            
            # All discovered wallets
            with self.db.connection() as conn:
                wallets = conn.execute("""
                    SELECT 
                        address, discovered_at, win_rate, pnl_7d, 
                        completed_swings, avg_hold_hours
                    FROM verified_wallets
                    ORDER BY discovered_at DESC
                """).fetchall()
                
                f.write(f"DISCOVERED WALLETS ({len(wallets)} total)\n")
                f.write("-"*70 + "\n")
                
                for wallet in wallets:
                    f.write(f"\nWallet: {wallet[0]}\n")
                    f.write(f"  Discovered: {wallet[1]}\n")
                    f.write(f"  Win Rate: {wallet[2]:.1%}\n")
                    f.write(f"  PnL: {wallet[3]:.2f} SOL\n")
                    f.write(f"  Swings: {wallet[4]}\n")
                    f.write(f"  Avg Hold: {wallet[5]:.1f}h\n")
        
        print(f"✅ Report exported to {filename}")


def main():
    """Main entry point"""
    import sys
    
    monitor = DiscoveryMonitor()
    
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == 'summary':
            days = int(sys.argv[2]) if len(sys.argv) > 2 else 7
            monitor.print_summary(days)
        
        elif command == 'export':
            monitor.export_discovery_report()
        
        elif command == 'help':
            print("""
Discovery Monitor

Usage:
  python discovery_monitor.py summary [days]  - Show summary (default: 7 days)
  python discovery_monitor.py export          - Export detailed report
  python discovery_monitor.py help            - Show this help

Examples:
  python discovery_monitor.py summary         - Last 7 days
  python discovery_monitor.py summary 30      - Last 30 days
  python discovery_monitor.py export          - Create discovery_report.txt
            """)
        else:
            print(f"Unknown command: {command}")
            print("Use 'help' for usage info")
    else:
        # Default: show 7-day summary
        monitor.print_summary(7)


if __name__ == "__main__":
    main()
