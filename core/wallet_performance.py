"""
WALLET PERFORMANCE ANALYZER
============================

Tracks wallet performance from paper trades and removes underperformers.

Key features:
1. Per-wallet win rate and PnL tracking
2. Automatic identification of poor performers
3. Removal from webhooks to free capacity
4. Integration with strategy learner

Author: Claude
"""

import sqlite3
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from dataclasses import dataclass
from contextlib import contextmanager


@dataclass
class WalletPerformance:
    """Performance metrics for a single wallet"""
    address: str
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    total_pnl_sol: float = 0.0
    avg_pnl_sol: float = 0.0
    win_rate: float = 0.0
    best_trade_pct: float = 0.0
    worst_trade_pct: float = 0.0
    last_trade_time: Optional[str] = None
    status: str = "active"  # active, probation, removed
    
    def to_dict(self) -> Dict:
        return {
            'address': self.address,
            'total_trades': self.total_trades,
            'wins': self.wins,
            'losses': self.losses,
            'win_rate': self.win_rate,
            'total_pnl_sol': self.total_pnl_sol,
            'avg_pnl_sol': self.avg_pnl_sol,
            'best_trade_pct': self.best_trade_pct,
            'worst_trade_pct': self.worst_trade_pct,
            'last_trade_time': self.last_trade_time,
            'status': self.status
        }


class WalletPerformanceAnalyzer:
    """
    Analyzes wallet performance from paper trades.
    
    Identifies underperformers for removal.
    """
    
    def __init__(self, 
                 paper_db_path: str = "robust_paper_trades_v6.db",
                 main_db_path: str = "swing_traders.db"):
        self.paper_db_path = paper_db_path
        self.main_db_path = main_db_path
        
        # Thresholds for removal
        self.min_trades_for_evaluation = 5  # Need at least 5 trades to judge
        self.min_win_rate = 0.35  # Below 35% = poor performer
        self.min_pnl_sol = -0.5   # Losing more than 0.5 SOL = poor
        self.probation_trades = 10  # Trades before final judgment
        
        print(f"📊 Wallet Performance Analyzer initialized")
        print(f"   Min trades for eval: {self.min_trades_for_evaluation}")
        print(f"   Min win rate: {self.min_win_rate:.0%}")
        print(f"   Min PnL: {self.min_pnl_sol} SOL")
    
    @contextmanager
    def _get_paper_connection(self):
        conn = sqlite3.connect(self.paper_db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()
    
    @contextmanager
    def _get_main_connection(self):
        conn = sqlite3.connect(self.main_db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()
    
    def analyze_all_wallets(self, days: int = 14) -> Dict[str, WalletPerformance]:
        """Analyze performance of all wallets from paper trades"""
        cutoff = datetime.utcnow() - timedelta(days=days)
        
        wallet_stats = defaultdict(lambda: {
            'trades': [],
            'wins': 0,
            'pnl': 0.0,
            'pnl_pcts': []
        })
        
        with self._get_paper_connection() as conn:
            # Get all closed trades
            trades = conn.execute("""
                SELECT * FROM paper_positions_v6
                WHERE status = 'closed'
                AND exit_time > ?
            """, (cutoff,)).fetchall()
            
            for trade in trades:
                # Extract wallet from entry context
                try:
                    context = json.loads(trade['entry_context_json'] or '{}')
                    wallet = context.get('wallet_address') or context.get('wallet', '')
                except:
                    wallet = ''
                
                if not wallet:
                    continue
                
                pnl_sol = trade['pnl_sol'] or 0
                pnl_pct = trade['pnl_pct'] or 0
                
                wallet_stats[wallet]['trades'].append(trade)
                wallet_stats[wallet]['pnl'] += pnl_sol
                wallet_stats[wallet]['pnl_pcts'].append(pnl_pct)
                
                if pnl_sol > 0:
                    wallet_stats[wallet]['wins'] += 1
        
        # Build performance objects
        performances = {}
        
        for wallet, stats in wallet_stats.items():
            total = len(stats['trades'])
            if total == 0:
                continue
            
            perf = WalletPerformance(
                address=wallet,
                total_trades=total,
                wins=stats['wins'],
                losses=total - stats['wins'],
                total_pnl_sol=stats['pnl'],
                avg_pnl_sol=stats['pnl'] / total,
                win_rate=stats['wins'] / total,
                best_trade_pct=max(stats['pnl_pcts']) if stats['pnl_pcts'] else 0,
                worst_trade_pct=min(stats['pnl_pcts']) if stats['pnl_pcts'] else 0,
                last_trade_time=stats['trades'][-1]['exit_time'] if stats['trades'] else None
            )
            
            # Determine status
            if total >= self.min_trades_for_evaluation:
                if perf.win_rate < self.min_win_rate or perf.total_pnl_sol < self.min_pnl_sol:
                    if total >= self.probation_trades:
                        perf.status = "remove"  # Enough data, consistently bad
                    else:
                        perf.status = "probation"  # Bad but need more data
                else:
                    perf.status = "active"
            else:
                perf.status = "evaluating"  # Not enough trades yet
            
            performances[wallet] = perf
        
        return performances
    
    def get_wallets_to_remove(self, days: int = 14) -> List[WalletPerformance]:
        """Get list of wallets that should be removed"""
        performances = self.analyze_all_wallets(days)
        
        to_remove = [
            perf for perf in performances.values()
            if perf.status == "remove"
        ]
        
        # Sort by worst performers first
        to_remove.sort(key=lambda p: p.win_rate)
        
        return to_remove
    
    def get_top_performers(self, days: int = 14, min_trades: int = 5) -> List[WalletPerformance]:
        """Get list of top performing wallets"""
        performances = self.analyze_all_wallets(days)
        
        top = [
            perf for perf in performances.values()
            if perf.total_trades >= min_trades and perf.win_rate >= 0.50
        ]
        
        # Sort by win rate * trade count (favor consistent + active)
        top.sort(key=lambda p: p.win_rate * min(p.total_trades, 20), reverse=True)
        
        return top
    
    def remove_wallet_from_tracking(self, wallet_address: str, webhook_manager=None, db=None) -> bool:
        """
        Remove a wallet from tracking.
        
        Args:
            wallet_address: The wallet to remove
            webhook_manager: MultiWebhookManager instance
            db: Database instance
        
        Returns:
            True if removed successfully
        """
        removed = False
        
        # Remove from webhooks
        if webhook_manager:
            try:
                # The webhook manager should have a method to remove wallets
                if hasattr(webhook_manager, 'remove_wallet'):
                    webhook_manager.remove_wallet(wallet_address)
                    removed = True
                    print(f"  ✅ Removed {wallet_address[:8]}... from webhooks")
            except Exception as e:
                print(f"  ⚠️ Failed to remove from webhooks: {e}")
        
        # Mark as inactive in database
        if db:
            try:
                if hasattr(db, 'deactivate_wallet'):
                    db.deactivate_wallet(wallet_address)
                else:
                    # Direct SQL if method doesn't exist
                    with self._get_main_connection() as conn:
                        conn.execute("""
                            UPDATE wallets SET status = 'removed', 
                            removed_at = ?, removal_reason = 'poor_performance'
                            WHERE address = ?
                        """, (datetime.utcnow().isoformat(), wallet_address))
                        conn.commit()
                removed = True
                print(f"  ✅ Marked {wallet_address[:8]}... as removed in database")
            except Exception as e:
                print(f"  ⚠️ Failed to update database: {e}")
        
        return removed
    
    def run_cleanup(self, webhook_manager=None, db=None, dry_run: bool = True) -> Dict:
        """
        Run wallet cleanup - identify and remove poor performers.
        
        Args:
            webhook_manager: MultiWebhookManager instance
            db: Database instance  
            dry_run: If True, only report what would be removed
        
        Returns:
            Cleanup results
        """
        print("\n" + "=" * 70)
        print("🧹 WALLET PERFORMANCE CLEANUP")
        print("=" * 70)
        
        results = {
            'timestamp': datetime.utcnow().isoformat(),
            'dry_run': dry_run,
            'wallets_analyzed': 0,
            'wallets_to_remove': [],
            'wallets_removed': [],
            'top_performers': []
        }
        
        # Analyze all wallets
        performances = self.analyze_all_wallets(days=14)
        results['wallets_analyzed'] = len(performances)
        
        print(f"\n📊 Analyzed {len(performances)} wallets")
        
        # Find wallets to remove
        to_remove = self.get_wallets_to_remove(days=14)
        results['wallets_to_remove'] = [p.to_dict() for p in to_remove]
        
        if to_remove:
            print(f"\n❌ Wallets to Remove ({len(to_remove)}):")
            for perf in to_remove[:10]:  # Show top 10
                print(f"   {perf.address[:12]}... | "
                      f"{perf.total_trades} trades | "
                      f"{perf.win_rate:.0%} WR | "
                      f"{perf.total_pnl_sol:+.4f} SOL")
            
            if not dry_run:
                print(f"\n🗑️ Removing {len(to_remove)} wallets...")
                for perf in to_remove:
                    if self.remove_wallet_from_tracking(perf.address, webhook_manager, db):
                        results['wallets_removed'].append(perf.address)
                print(f"   Removed {len(results['wallets_removed'])} wallets")
            else:
                print(f"\n   (Dry run - no wallets actually removed)")
                print(f"   Run with dry_run=False to remove")
        else:
            print(f"\n✅ No wallets need removal")
        
        # Show top performers
        top = self.get_top_performers(days=14, min_trades=5)[:10]
        results['top_performers'] = [p.to_dict() for p in top]
        
        if top:
            print(f"\n🏆 Top Performers ({len(top)}):")
            for perf in top[:5]:
                print(f"   {perf.address[:12]}... | "
                      f"{perf.total_trades} trades | "
                      f"{perf.win_rate:.0%} WR | "
                      f"{perf.total_pnl_sol:+.4f} SOL")
        
        # Summary by status
        status_counts = defaultdict(int)
        for perf in performances.values():
            status_counts[perf.status] += 1
        
        print(f"\n📈 Wallet Status Summary:")
        for status, count in sorted(status_counts.items()):
            print(f"   {status}: {count}")
        
        print("\n" + "=" * 70)
        
        return results
    
    def print_performance_report(self, days: int = 14):
        """Print detailed performance report"""
        performances = self.analyze_all_wallets(days)
        
        print("\n" + "=" * 70)
        print(f"📊 WALLET PERFORMANCE REPORT (Last {days} Days)")
        print("=" * 70)
        
        if not performances:
            print("No wallet performance data available")
            return
        
        # Sort by trade count
        sorted_perfs = sorted(performances.values(), 
                             key=lambda p: p.total_trades, reverse=True)
        
        print(f"\n{'Wallet':<14} | {'Trades':>6} | {'WR':>6} | {'PnL':>10} | {'Status':<10}")
        print("-" * 60)
        
        for perf in sorted_perfs[:20]:
            status_icon = {
                'active': '✅',
                'probation': '⚠️',
                'remove': '❌',
                'evaluating': '🔄'
            }.get(perf.status, '❓')
            
            print(f"{perf.address[:12]}... | "
                  f"{perf.total_trades:>6} | "
                  f"{perf.win_rate:>5.0%} | "
                  f"{perf.total_pnl_sol:>+9.4f} | "
                  f"{status_icon} {perf.status}")
        
        if len(sorted_perfs) > 20:
            print(f"... and {len(sorted_perfs) - 20} more wallets")
        
        print("\n" + "=" * 70)


def main():
    """CLI for wallet performance analyzer"""
    import sys
    
    analyzer = WalletPerformanceAnalyzer()
    
    if len(sys.argv) < 2:
        print("\nUsage: python wallet_performance.py <command>")
        print("Commands:")
        print("  report     - Print performance report")
        print("  cleanup    - Dry run of cleanup (shows what would be removed)")
        print("  remove     - Actually remove poor performers")
        print("  top        - Show top performers")
        return
    
    command = sys.argv[1].lower()
    
    if command == 'report':
        analyzer.print_performance_report()
    
    elif command == 'cleanup':
        analyzer.run_cleanup(dry_run=True)
    
    elif command == 'remove':
        confirm = input("This will remove poor performing wallets. Type 'YES' to confirm: ")
        if confirm == 'YES':
            analyzer.run_cleanup(dry_run=False)
        else:
            print("Cancelled")
    
    elif command == 'top':
        top = analyzer.get_top_performers()
        print("\n🏆 Top Performing Wallets:")
        for i, perf in enumerate(top[:10], 1):
            print(f"  {i}. {perf.address[:12]}... | "
                  f"{perf.total_trades} trades | "
                  f"{perf.win_rate:.0%} WR | "
                  f"{perf.total_pnl_sol:+.4f} SOL")
    
    else:
        print(f"Unknown command: {command}")


if __name__ == "__main__":
    main()
