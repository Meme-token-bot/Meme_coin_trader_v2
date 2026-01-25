"""
SYSTEM INTERROGATION & PROFITABILITY ANALYSIS
==============================================

Purpose: Validate data quality and model path to $200 NZD/day profit

Sections:
1. DATABASE VALIDATION - Are tracked wallets actually profitable?
2. PAPER TRADING ANALYSIS - What does the data show?
3. PATH TO $200 NZD/DAY - What's required to hit the goal?
4. RECOMMENDATIONS - What needs to change?
"""

import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # Go up ONE level
import json
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from collections import defaultdict
from dotenv import load_dotenv
load_dotenv()

from core.database_v2 import DatabaseV2

HELIUS_KEY = os.getenv('HELIUS_KEY')
NZD_USD_RATE = 0.58  # Approximate - 1 NZD = 0.58 USD
TARGET_DAILY_NZD = 200
TARGET_DAILY_USD = TARGET_DAILY_NZD * NZD_USD_RATE  # ~$116 USD


class SystemInterrogator:
    """Comprehensive system analysis"""
    
    def __init__(self):
        self.db = DatabaseV2()
        self.helius_url = f"https://mainnet.helius-rpc.com/?api-key={HELIUS_KEY}"
        self.helius_api = f"https://api.helius.xyz/v0"
    
    # =========================================================================
    # SECTION 1: DATABASE VALIDATION
    # =========================================================================
    
    def validate_wallet_data(self, sample_size: int = 10) -> Dict:
        """
        Validate that stored wallet data reflects actual on-chain activity.
        
        Compares database stats with fresh data from Helius.
        """
        print("\n" + "="*70)
        print("📊 SECTION 1: DATABASE VALIDATION")
        print("="*70)
        
        wallets = self.db.get_all_verified_wallets()
        print(f"\nTotal verified wallets in database: {len(wallets)}")
        
        if not wallets:
            print("❌ No wallets to validate!")
            return {'valid': False, 'reason': 'No wallets in database'}
        
        # Sample wallets for validation
        sample = wallets[:sample_size]
        
        validation_results = []
        
        print(f"\nValidating {len(sample)} wallet(s) against on-chain data...\n")
        
        for wallet in sample:
            address = wallet['address']
            db_stats = {
                'win_rate': wallet.get('win_rate', 0),
                'pnl': wallet.get('pnl_7d', 0),
                'completed_swings': wallet.get('completed_swings', 0)
            }
            
            # Get fresh data from Helius
            fresh_stats = self._get_fresh_wallet_stats(address)
            
            if fresh_stats:
                # Compare
                wr_diff = abs(db_stats['win_rate'] - fresh_stats['win_rate'])
                pnl_match = (db_stats['pnl'] > 0) == (fresh_stats['pnl'] > 0)  # Same direction
                
                status = "✅" if wr_diff < 0.2 and pnl_match else "⚠️"
                
                validation_results.append({
                    'address': address,
                    'db_wr': db_stats['win_rate'],
                    'fresh_wr': fresh_stats['win_rate'],
                    'db_pnl': db_stats['pnl'],
                    'fresh_pnl': fresh_stats['pnl'],
                    'valid': wr_diff < 0.2 and pnl_match
                })
                
                print(f"{status} {address[:12]}...")
                print(f"   DB:    WR={db_stats['win_rate']:.1%} | PnL={db_stats['pnl']:.2f} SOL")
                print(f"   Fresh: WR={fresh_stats['win_rate']:.1%} | PnL={fresh_stats['pnl']:.2f} SOL")
                print(f"   Trades: {fresh_stats['total_trades']} ({fresh_stats['wins']}W/{fresh_stats['losses']}L)")
            else:
                print(f"⏭️  {address[:12]}... (couldn't fetch fresh data)")
        
        # Summary
        valid_count = sum(1 for r in validation_results if r.get('valid', False))
        total_checked = len(validation_results)
        
        print(f"\n📋 Validation Summary:")
        print(f"   Checked: {total_checked}")
        print(f"   Valid: {valid_count} ({100*valid_count/max(1,total_checked):.0f}%)")
        
        return {
            'total_wallets': len(wallets),
            'sample_size': total_checked,
            'valid_count': valid_count,
            'validation_rate': valid_count / max(1, total_checked),
            'results': validation_results
        }
    
    def _get_fresh_wallet_stats(self, address: str) -> Optional[Dict]:
        """Get fresh trading stats for a wallet from Helius"""
        try:
            # Get recent transactions
            url = f"{self.helius_api}/addresses/{address}/transactions?api-key={HELIUS_KEY}&limit=100"
            response = requests.get(url, timeout=15)
            
            if response.status_code != 200:
                return None
            
            txs = response.json()
            
            # Analyze swaps
            swaps = [tx for tx in txs if tx.get('type') == 'SWAP']
            
            if not swaps:
                return {'win_rate': 0, 'pnl': 0, 'total_trades': 0, 'wins': 0, 'losses': 0}
            
            # Track token positions
            positions = defaultdict(lambda: {'bought': 0, 'sold': 0, 'buy_value': 0, 'sell_value': 0})
            
            for tx in swaps:
                token_transfers = tx.get('tokenTransfers', [])
                native_transfers = tx.get('nativeTransfers', [])
                
                for transfer in token_transfers:
                    mint = transfer.get('mint', '')
                    amount = float(transfer.get('tokenAmount', 0) or 0)
                    
                    if transfer.get('toUserAccount') == address:
                        positions[mint]['bought'] += amount
                    elif transfer.get('fromUserAccount') == address:
                        positions[mint]['sold'] += amount
                
                # Track SOL spent/received
                for transfer in native_transfers:
                    sol_amount = float(transfer.get('amount', 0) or 0) / 1e9
                    
                    if transfer.get('fromUserAccount') == address:
                        # Spent SOL (buying)
                        for t in token_transfers:
                            if t.get('toUserAccount') == address:
                                positions[t['mint']]['buy_value'] += sol_amount
                    elif transfer.get('toUserAccount') == address:
                        # Received SOL (selling)
                        for t in token_transfers:
                            if t.get('fromUserAccount') == address:
                                positions[t['mint']]['sell_value'] += sol_amount
            
            # Calculate PnL
            total_pnl = 0
            wins = 0
            losses = 0
            completed = 0
            
            for mint, pos in positions.items():
                if pos['bought'] > 0 and pos['sold'] > 0:
                    # Completed trade
                    completed += 1
                    pnl = pos['sell_value'] - pos['buy_value']
                    total_pnl += pnl
                    
                    if pnl > 0:
                        wins += 1
                    else:
                        losses += 1
            
            win_rate = wins / max(1, completed)
            
            return {
                'win_rate': win_rate,
                'pnl': total_pnl,
                'total_trades': completed,
                'wins': wins,
                'losses': losses
            }
            
        except Exception as e:
            print(f"      Error: {e}")
            return None
    
    def analyze_wallet_quality(self) -> Dict:
        """Analyze the quality distribution of tracked wallets"""
        print("\n" + "="*70)
        print("📊 WALLET QUALITY ANALYSIS")
        print("="*70)
        
        wallets = self.db.get_all_verified_wallets()
        
        if not wallets:
            print("❌ No wallets to analyze!")
            return {}
        
        # Categorize wallets
        tiers = {
            'elite': [],      # WR >= 70%, PnL >= 5 SOL
            'strong': [],     # WR >= 60%, PnL >= 2 SOL
            'moderate': [],   # WR >= 50%, PnL >= 0.5 SOL
            'weak': [],       # Below moderate thresholds
            'negative': []    # PnL < 0
        }
        
        for w in wallets:
            wr = w.get('win_rate', 0)
            pnl = w.get('pnl_7d', 0)
            
            if pnl < 0:
                tiers['negative'].append(w)
            elif wr >= 0.70 and pnl >= 5:
                tiers['elite'].append(w)
            elif wr >= 0.60 and pnl >= 2:
                tiers['strong'].append(w)
            elif wr >= 0.50 and pnl >= 0.5:
                tiers['moderate'].append(w)
            else:
                tiers['weak'].append(w)
        
        print(f"\nWallet Quality Distribution (n={len(wallets)}):")
        print(f"  🏆 Elite (WR≥70%, PnL≥5):    {len(tiers['elite'])} ({100*len(tiers['elite'])/len(wallets):.1f}%)")
        print(f"  💪 Strong (WR≥60%, PnL≥2):   {len(tiers['strong'])} ({100*len(tiers['strong'])/len(wallets):.1f}%)")
        print(f"  📊 Moderate (WR≥50%, PnL≥0.5): {len(tiers['moderate'])} ({100*len(tiers['moderate'])/len(wallets):.1f}%)")
        print(f"  📉 Weak:                      {len(tiers['weak'])} ({100*len(tiers['weak'])/len(wallets):.1f}%)")
        print(f"  ❌ Negative PnL:              {len(tiers['negative'])} ({100*len(tiers['negative'])/len(wallets):.1f}%)")
        
        # Top performers
        print(f"\n🏆 Top 5 Performers:")
        top = sorted(wallets, key=lambda x: (x.get('win_rate', 0) * x.get('pnl_7d', 0)), reverse=True)[:5]
        for i, w in enumerate(top):
            print(f"   {i+1}. {w['address'][:16]}... | WR: {w.get('win_rate', 0):.1%} | PnL: {w.get('pnl_7d', 0):.2f} SOL")
        
        # Worst performers (should consider removing)
        print(f"\n⚠️ Bottom 5 (Consider Removing):")
        bottom = sorted(wallets, key=lambda x: x.get('pnl_7d', 0))[:5]
        for i, w in enumerate(bottom):
            print(f"   {i+1}. {w['address'][:16]}... | WR: {w.get('win_rate', 0):.1%} | PnL: {w.get('pnl_7d', 0):.2f} SOL")
        
        return {
            'total': len(wallets),
            'tiers': {k: len(v) for k, v in tiers.items()},
            'top_performers': top[:5],
            'bottom_performers': bottom[:5]
        }
    
    # =========================================================================
    # SECTION 2: PAPER TRADING ANALYSIS
    # =========================================================================
    
    def analyze_paper_trading(self) -> Dict:
        """Analyze paper trading results"""
        print("\n" + "="*70)
        print("📊 SECTION 2: PAPER TRADING ANALYSIS")
        print("="*70)
        
        # Check for paper trading data in database
        with self.db.connection() as conn:
            # Check if paper_trades table exists
            tables = conn.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name LIKE '%paper%'
            """).fetchall()
            
            print(f"\nPaper trading tables found: {[t['name'] for t in tables]}")
            
            # Try to get paper trade data
            try:
                trades = conn.execute("""
                    SELECT * FROM paper_trades 
                    ORDER BY timestamp DESC
                """).fetchall()
                
                if not trades:
                    print("\n⚠️ No paper trades recorded")
                    return self._no_paper_data_analysis()
                
                print(f"\nFound {len(trades)} paper trade(s)")
                
                return self._analyze_paper_trades(trades)
                
            except Exception as e:
                print(f"\n⚠️ Could not access paper trades: {e}")
                return self._no_paper_data_analysis()
    
    def _no_paper_data_analysis(self) -> Dict:
        """Handle case where no paper trading data exists"""
        print("""
╔══════════════════════════════════════════════════════════════════════╗
║  ⚠️  CRITICAL: NO CONTINUOUS PAPER TRADING DATA                      ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  Without continuous paper trading data, we cannot:                   ║
║                                                                      ║
║  • Validate strategy performance                                     ║
║  • Understand win rates in practice                                  ║
║  • Calculate actual PnL from copy trading                           ║
║  • Identify which wallets to follow                                  ║
║  • Model drawdowns and risk                                          ║
║                                                                      ║
║  RECOMMENDATION: Run the system continuously for 7+ days             ║
║  before making any decisions about real money trading.               ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
""")
        return {
            'has_data': False,
            'recommendation': 'Run system continuously for 7+ days'
        }
    
    def _analyze_paper_trades(self, trades: List) -> Dict:
        """Analyze actual paper trading results"""
        # Calculate statistics
        total_trades = len(trades)
        
        wins = [t for t in trades if t.get('pnl', 0) > 0]
        losses = [t for t in trades if t.get('pnl', 0) <= 0]
        
        total_pnl = sum(t.get('pnl', 0) for t in trades)
        win_rate = len(wins) / max(1, total_trades)
        
        avg_win = sum(t.get('pnl', 0) for t in wins) / max(1, len(wins))
        avg_loss = sum(t.get('pnl', 0) for t in losses) / max(1, len(losses))
        
        # Time span
        if trades:
            first_trade = min(t.get('timestamp', datetime.now()) for t in trades)
            last_trade = max(t.get('timestamp', datetime.now()) for t in trades)
            days_active = (last_trade - first_trade).days + 1
        else:
            days_active = 0
        
        print(f"\n📈 Paper Trading Results:")
        print(f"   Period: {days_active} day(s)")
        print(f"   Total trades: {total_trades}")
        print(f"   Win rate: {win_rate:.1%}")
        print(f"   Total PnL: {total_pnl:.4f} SOL")
        print(f"   Avg win: {avg_win:.4f} SOL")
        print(f"   Avg loss: {avg_loss:.4f} SOL")
        
        if avg_loss != 0:
            risk_reward = abs(avg_win / avg_loss)
            print(f"   Risk/Reward: {risk_reward:.2f}")
        
        # Daily breakdown
        daily_pnl = defaultdict(float)
        for t in trades:
            day = t.get('timestamp', datetime.now()).date()
            daily_pnl[day] += t.get('pnl', 0)
        
        if daily_pnl:
            print(f"\n📅 Daily PnL:")
            for day in sorted(daily_pnl.keys())[-7:]:  # Last 7 days
                pnl = daily_pnl[day]
                print(f"   {day}: {pnl:+.4f} SOL")
        
        return {
            'has_data': True,
            'total_trades': total_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'days_active': days_active
        }
    
    # =========================================================================
    # SECTION 3: PATH TO $200 NZD/DAY
    # =========================================================================
    
    def model_profitability(self) -> Dict:
        """Model what's needed to achieve $200 NZD/day"""
        print("\n" + "="*70)
        print("📊 SECTION 3: PATH TO $200 NZD/DAY")
        print("="*70)
        
        # Get current SOL price
        sol_price = self._get_sol_price()
        print(f"\nCurrent SOL price: ${sol_price:.2f} USD")
        
        # Target calculation
        target_usd = TARGET_DAILY_USD
        target_sol = target_usd / sol_price
        
        print(f"\n🎯 TARGET: ${TARGET_DAILY_NZD} NZD/day")
        print(f"   = ${target_usd:.2f} USD/day")
        print(f"   = {target_sol:.2f} SOL/day (at current price)")
        
        # Model different scenarios
        print(f"\n📊 SCENARIOS TO HIT TARGET:")
        
        scenarios = [
            {'capital': 10, 'trades': 10, 'win_rate': 0.60, 'avg_win_pct': 0.15, 'avg_loss_pct': 0.08},
            {'capital': 25, 'trades': 8, 'win_rate': 0.55, 'avg_win_pct': 0.12, 'avg_loss_pct': 0.06},
            {'capital': 50, 'trades': 5, 'win_rate': 0.50, 'avg_win_pct': 0.10, 'avg_loss_pct': 0.05},
            {'capital': 100, 'trades': 4, 'win_rate': 0.50, 'avg_win_pct': 0.08, 'avg_loss_pct': 0.04},
        ]
        
        for i, s in enumerate(scenarios):
            capital = s['capital']
            trades = s['trades']
            wr = s['win_rate']
            avg_win = s['avg_win_pct']
            avg_loss = s['avg_loss_pct']
            
            # Position size per trade (equal allocation)
            pos_size = capital / 3  # Max 3 concurrent positions
            
            # Expected daily PnL
            wins = trades * wr
            losses = trades * (1 - wr)
            
            win_pnl = wins * pos_size * avg_win
            loss_pnl = losses * pos_size * avg_loss
            
            daily_pnl_sol = win_pnl - loss_pnl
            daily_pnl_usd = daily_pnl_sol * sol_price
            daily_pnl_nzd = daily_pnl_usd / NZD_USD_RATE
            
            meets_target = "✅" if daily_pnl_nzd >= TARGET_DAILY_NZD else "❌"
            
            print(f"\n   Scenario {i+1}: {meets_target}")
            print(f"      Capital: {capital} SOL (${capital * sol_price:.0f} USD)")
            print(f"      Trades/day: {trades}")
            print(f"      Win rate: {wr:.0%}")
            print(f"      Avg win: {avg_win:.0%} | Avg loss: {avg_loss:.0%}")
            print(f"      Expected daily: {daily_pnl_sol:.2f} SOL = ${daily_pnl_nzd:.0f} NZD")
        
        # What the current system needs
        print(f"\n" + "="*70)
        print(f"📋 WHAT YOUR SYSTEM NEEDS:")
        print(f"="*70)
        
        # Based on wallet quality
        wallets = self.db.get_all_verified_wallets()
        avg_wr = sum(w.get('win_rate', 0) for w in wallets) / max(1, len(wallets))
        
        print(f"""
Based on your current wallet quality:
   • Average wallet win rate: {avg_wr:.1%}
   • Tracked wallets: {len(wallets)}

To hit ${TARGET_DAILY_NZD} NZD/day, you likely need:

   CONSERVATIVE PATH (Lower Risk):
   ├── Capital: 50-100 SOL (${50*sol_price:.0f}-${100*sol_price:.0f} USD)
   ├── Trades/day: 4-6
   ├── Win rate: ≥55%
   ├── Avg position: 10-20 SOL
   └── Copy only ELITE wallets (WR≥70%)

   AGGRESSIVE PATH (Higher Risk):
   ├── Capital: 20-30 SOL (${20*sol_price:.0f}-${30*sol_price:.0f} USD)
   ├── Trades/day: 8-12
   ├── Win rate: ≥60%
   ├── Avg position: 5-10 SOL
   └── Copy all STRONG+ wallets (WR≥60%)
""")
        
        return {
            'target_nzd': TARGET_DAILY_NZD,
            'target_usd': target_usd,
            'target_sol': target_sol,
            'sol_price': sol_price,
            'avg_wallet_wr': avg_wr
        }
    
    def _get_sol_price(self) -> float:
        """Get current SOL price in USD"""
        try:
            response = requests.get(
                "https://api.coingecko.com/api/v3/simple/price",
                params={"ids": "solana", "vs_currencies": "usd"},
                timeout=10
            )
            return response.json()['solana']['usd']
        except:
            return 100.0  # Fallback estimate
    
    # =========================================================================
    # SECTION 4: RECOMMENDATIONS
    # =========================================================================
    
    def generate_recommendations(self, validation: Dict, quality: Dict, paper: Dict, model: Dict) -> None:
        """Generate actionable recommendations"""
        print("\n" + "="*70)
        print("📊 SECTION 4: RECOMMENDATIONS")
        print("="*70)
        
        recommendations = []
        
        # Based on validation
        if validation.get('validation_rate', 0) < 0.7:
            recommendations.append({
                'priority': 'HIGH',
                'area': 'Data Quality',
                'issue': 'Wallet data may not accurately reflect on-chain activity',
                'action': 'Re-profile wallets with fresh data, remove invalid ones'
            })
        
        # Based on quality
        elite_pct = quality.get('tiers', {}).get('elite', 0) / max(1, quality.get('total', 1))
        negative_pct = quality.get('tiers', {}).get('negative', 0) / max(1, quality.get('total', 1))
        
        if elite_pct < 0.1:
            recommendations.append({
                'priority': 'HIGH',
                'area': 'Wallet Selection',
                'issue': f'Only {elite_pct:.0%} of wallets are elite performers',
                'action': 'Tighten discovery thresholds, focus on finding WR≥70% wallets'
            })
        
        if negative_pct > 0.2:
            recommendations.append({
                'priority': 'MEDIUM',
                'area': 'Wallet Cleanup',
                'issue': f'{negative_pct:.0%} of wallets have negative PnL',
                'action': 'Remove wallets with negative 7-day PnL'
            })
        
        # Based on paper trading
        if not paper.get('has_data'):
            recommendations.append({
                'priority': 'CRITICAL',
                'area': 'Data Collection',
                'issue': 'No continuous paper trading data',
                'action': 'Run system continuously for 7+ days without resets'
            })
        elif paper.get('total_trades', 0) < 50:
            recommendations.append({
                'priority': 'HIGH',
                'area': 'Statistical Significance',
                'issue': f"Only {paper.get('total_trades', 0)} trades - need 50+ for reliable stats",
                'action': 'Continue paper trading until 50+ trades completed'
            })
        
        # Print recommendations
        for r in sorted(recommendations, key=lambda x: {'CRITICAL': 0, 'HIGH': 1, 'MEDIUM': 2, 'LOW': 3}[x['priority']]):
            emoji = {'CRITICAL': '🚨', 'HIGH': '⚠️', 'MEDIUM': '📋', 'LOW': 'ℹ️'}[r['priority']]
            print(f"\n{emoji} [{r['priority']}] {r['area']}")
            print(f"   Issue: {r['issue']}")
            print(f"   Action: {r['action']}")
        
        # Final summary
        print(f"""
╔══════════════════════════════════════════════════════════════════════╗
║                      🎯 PATH TO ${TARGET_DAILY_NZD} NZD/DAY                        ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  PHASE 1: VALIDATION (Current)                                       ║
║  ├── Run system continuously for 7+ days                             ║
║  ├── Collect 50+ paper trades                                        ║
║  ├── Validate wallet data accuracy                                   ║
║  └── Remove underperforming wallets                                  ║
║                                                                      ║
║  PHASE 2: OPTIMIZATION (Next)                                        ║
║  ├── Analyze which wallets generate profitable signals               ║
║  ├── Tune entry/exit timing                                          ║
║  ├── Implement position sizing based on wallet quality               ║
║  └── Target: 55%+ win rate on paper trades                           ║
║                                                                      ║
║  PHASE 3: SMALL LIVE TESTING                                         ║
║  ├── Start with 5 SOL capital                                        ║
║  ├── Max position: 1 SOL                                             ║
║  ├── Validate paper vs live performance                              ║
║  └── Target: Consistent daily profit for 14+ days                    ║
║                                                                      ║
║  PHASE 4: SCALE TO TARGET                                            ║
║  ├── Increase capital to 50-100 SOL                                  ║
║  ├── Position size: 10-20 SOL                                        ║
║  ├── 4-8 trades/day                                                  ║
║  └── Target: ${TARGET_DAILY_NZD} NZD/day = ~{model.get('target_sol', 1):.1f} SOL/day               ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
""")
    
    def run_full_analysis(self):
        """Run complete system analysis"""
        print("\n" + "="*70)
        print("🔍 SYSTEM INTERROGATION & PROFITABILITY ANALYSIS")
        print("="*70)
        print(f"   Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"   Target: ${TARGET_DAILY_NZD} NZD/day")
        
        # Run all analyses
        validation = self.validate_wallet_data(sample_size=5)
        quality = self.analyze_wallet_quality()
        paper = self.analyze_paper_trading()
        model = self.model_profitability()
        
        # Generate recommendations
        self.generate_recommendations(validation, quality, paper, model)
        
        return {
            'validation': validation,
            'quality': quality,
            'paper_trading': paper,
            'profitability_model': model
        }


if __name__ == "__main__":
    interrogator = SystemInterrogator()
    results = interrogator.run_full_analysis()
