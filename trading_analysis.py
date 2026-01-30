"""
PAPER TRADING ANALYSIS - Go/No-Go Assessment
=============================================

Comprehensive analysis of paper trading results to determine
if the strategy is ready for live trading.

Run: python trading_analysis.py
"""

import sqlite3
import json
import statistics
from datetime import datetime, timedelta
from collections import defaultdict
from contextlib import contextmanager
from typing import Dict, List, Tuple
import os


class TradingAnalyzer:
    """Comprehensive analysis of paper trading performance"""
    
    def __init__(self, db_path: str = "robust_paper_trades_v6.db"):
        self.db_path = db_path
        
        if not os.path.exists(db_path):
            print(f"❌ Database not found: {db_path}")
            print("   Try: robust_paper_trades_v6.db or paper_trades.db")
            return
        
        print(f"📊 Loading data from: {db_path}")
    
    @contextmanager
    def _get_connection(self):
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()
    
    def get_all_trades(self) -> List[Dict]:
        """Get all closed trades"""
        with self._get_connection() as conn:
            # Try V6 table first
            try:
                rows = conn.execute("""
                    SELECT * FROM paper_positions_v6
                    WHERE status = 'closed'
                    ORDER BY exit_time ASC
                """).fetchall()
            except:
                # Fallback to other table names
                rows = conn.execute("""
                    SELECT * FROM paper_positions_v5
                    WHERE status = 'closed'
                    ORDER BY exit_time ASC
                """).fetchall()
            
            return [dict(r) for r in rows]
    
    def get_account_state(self) -> Dict:
        """Get current account state"""
        with self._get_connection() as conn:
            try:
                row = conn.execute("SELECT * FROM paper_account_v6 WHERE id = 1").fetchone()
            except:
                row = conn.execute("SELECT * FROM paper_account_v5 WHERE id = 1").fetchone()
            
            return dict(row) if row else {}
    
    def run_full_analysis(self):
        """Run complete analysis and print report"""
        trades = self.get_all_trades()
        account = self.get_account_state()
        
        if not trades:
            print("❌ No closed trades found!")
            return
        
        print("\n" + "=" * 80)
        print("📊 PAPER TRADING PERFORMANCE ANALYSIS")
        print("=" * 80)
        
        # 1. Overall Summary
        self._print_overall_summary(trades, account)
        
        # 2. Win/Loss Analysis
        self._print_win_loss_analysis(trades)
        
        # 3. PnL Distribution
        self._print_pnl_distribution(trades)
        
        # 4. Performance by Exit Reason
        self._print_exit_analysis(trades)
        
        # 5. Performance by Hour
        self._print_hourly_analysis(trades)
        
        # 6. Performance by Wallet
        self._print_wallet_analysis(trades)
        
        # 7. Performance Over Time
        self._print_time_analysis(trades)
        
        # 8. Risk Analysis
        self._print_risk_analysis(trades, account)
        
        # 9. Go/No-Go Assessment
        self._print_go_nogo_assessment(trades, account)
        
        # 10. Recommendations
        self._print_recommendations(trades)
    
    def _print_overall_summary(self, trades: List[Dict], account: Dict):
        """Print overall performance summary"""
        print("\n" + "-" * 80)
        print("📈 OVERALL SUMMARY")
        print("-" * 80)
        
        total = len(trades)
        wins = sum(1 for t in trades if (t.get('pnl_sol') or 0) > 0)
        losses = total - wins
        
        total_pnl = sum(t.get('pnl_sol', 0) or 0 for t in trades)
        gross_profit = sum(t.get('pnl_sol', 0) for t in trades if (t.get('pnl_sol') or 0) > 0)
        gross_loss = sum(t.get('pnl_sol', 0) for t in trades if (t.get('pnl_sol') or 0) < 0)
        
        win_rate = wins / total if total > 0 else 0
        
        # Calculate profit factor
        profit_factor = abs(gross_profit / gross_loss) if gross_loss != 0 else float('inf')
        
        # Time span
        first_trade = trades[0].get('entry_time', '')
        last_trade = trades[-1].get('exit_time', '')
        
        starting_bal = account.get('starting_balance', 10)
        current_bal = account.get('current_balance', starting_bal + total_pnl)
        return_pct = ((current_bal / starting_bal) - 1) * 100 if starting_bal > 0 else 0
        
        print(f"\n  📅 Period: {first_trade[:10] if first_trade else 'N/A'} to {last_trade[:10] if last_trade else 'N/A'}")
        print(f"\n  💰 BALANCE:")
        print(f"     Starting:    {starting_bal:.4f} SOL")
        print(f"     Current:     {current_bal:.4f} SOL")
        print(f"     Total PnL:   {total_pnl:+.4f} SOL")
        print(f"     Return:      {return_pct:+.1f}%")
        
        print(f"\n  📊 TRADES:")
        print(f"     Total:       {total}")
        print(f"     Wins:        {wins} ({win_rate:.1%})")
        print(f"     Losses:      {losses} ({1-win_rate:.1%})")
        
        print(f"\n  💵 PROFIT/LOSS:")
        print(f"     Gross Profit: {gross_profit:+.4f} SOL")
        print(f"     Gross Loss:   {gross_loss:+.4f} SOL")
        print(f"     Profit Factor: {profit_factor:.2f}")
        
        # Expected value per trade
        ev_per_trade = total_pnl / total if total > 0 else 0
        print(f"\n  📐 EXPECTED VALUE:")
        print(f"     Per Trade:    {ev_per_trade:+.4f} SOL")
        print(f"     Per 100 Trades: {ev_per_trade * 100:+.4f} SOL")
    
    def _print_win_loss_analysis(self, trades: List[Dict]):
        """Analyze win/loss characteristics"""
        print("\n" + "-" * 80)
        print("🎯 WIN/LOSS ANALYSIS")
        print("-" * 80)
        
        winners = [t for t in trades if (t.get('pnl_sol') or 0) > 0]
        losers = [t for t in trades if (t.get('pnl_sol') or 0) < 0]
        
        if winners:
            win_pnls = [t['pnl_sol'] for t in winners]
            win_pcts = [t.get('pnl_pct', 0) or 0 for t in winners]
            
            print(f"\n  ✅ WINNERS ({len(winners)}):")
            print(f"     Avg Win:      {statistics.mean(win_pnls):+.4f} SOL ({statistics.mean(win_pcts):+.1f}%)")
            print(f"     Median Win:   {statistics.median(win_pnls):+.4f} SOL ({statistics.median(win_pcts):+.1f}%)")
            print(f"     Best Win:     {max(win_pnls):+.4f} SOL ({max(win_pcts):+.1f}%)")
            print(f"     Smallest Win: {min(win_pnls):+.4f} SOL ({min(win_pcts):+.1f}%)")
        
        if losers:
            loss_pnls = [t['pnl_sol'] for t in losers]
            loss_pcts = [t.get('pnl_pct', 0) or 0 for t in losers]
            
            print(f"\n  ❌ LOSERS ({len(losers)}):")
            print(f"     Avg Loss:     {statistics.mean(loss_pnls):+.4f} SOL ({statistics.mean(loss_pcts):+.1f}%)")
            print(f"     Median Loss:  {statistics.median(loss_pnls):+.4f} SOL ({statistics.median(loss_pcts):+.1f}%)")
            print(f"     Worst Loss:   {min(loss_pnls):+.4f} SOL ({min(loss_pcts):+.1f}%)")
        
        if winners and losers:
            avg_win = statistics.mean(win_pnls)
            avg_loss = abs(statistics.mean(loss_pnls))
            risk_reward = avg_win / avg_loss if avg_loss > 0 else float('inf')
            print(f"\n  📐 Risk/Reward Ratio: {risk_reward:.2f}:1")
            
            # Required win rate for breakeven with this R:R
            breakeven_wr = 1 / (1 + risk_reward)
            print(f"     Breakeven WR needed: {breakeven_wr:.1%}")
    
    def _print_pnl_distribution(self, trades: List[Dict]):
        """Show PnL distribution"""
        print("\n" + "-" * 80)
        print("📊 PNL DISTRIBUTION")
        print("-" * 80)
        
        pnl_pcts = [t.get('pnl_pct', 0) or 0 for t in trades]
        
        # Define buckets
        buckets = {
            'Big Loss (<-20%)': 0,
            'Medium Loss (-20% to -10%)': 0,
            'Small Loss (-10% to 0%)': 0,
            'Small Win (0% to 15%)': 0,
            'Medium Win (15% to 30%)': 0,
            'Big Win (30% to 100%)': 0,
            'Huge Win (>100%)': 0
        }
        
        for pct in pnl_pcts:
            if pct < -20:
                buckets['Big Loss (<-20%)'] += 1
            elif pct < -10:
                buckets['Medium Loss (-20% to -10%)'] += 1
            elif pct < 0:
                buckets['Small Loss (-10% to 0%)'] += 1
            elif pct < 15:
                buckets['Small Win (0% to 15%)'] += 1
            elif pct < 30:
                buckets['Medium Win (15% to 30%)'] += 1
            elif pct < 100:
                buckets['Big Win (30% to 100%)'] += 1
            else:
                buckets['Huge Win (>100%)'] += 1
        
        total = len(trades)
        print()
        for bucket, count in buckets.items():
            pct = count / total * 100 if total > 0 else 0
            bar = '█' * int(pct / 2)
            print(f"  {bucket:<30} {count:>4} ({pct:>5.1f}%) {bar}")
    
    def _print_exit_analysis(self, trades: List[Dict]):
        """Analyze by exit reason"""
        print("\n" + "-" * 80)
        print("🚪 EXIT REASON ANALYSIS")
        print("-" * 80)
        
        by_exit = defaultdict(lambda: {'count': 0, 'wins': 0, 'pnl': 0, 'pnl_pcts': []})
        
        for t in trades:
            reason = t.get('exit_reason', 'UNKNOWN')
            by_exit[reason]['count'] += 1
            by_exit[reason]['pnl'] += t.get('pnl_sol', 0) or 0
            by_exit[reason]['pnl_pcts'].append(t.get('pnl_pct', 0) or 0)
            if (t.get('pnl_sol') or 0) > 0:
                by_exit[reason]['wins'] += 1
        
        print(f"\n  {'Exit Reason':<20} {'Count':>6} {'Win Rate':>10} {'Total PnL':>12} {'Avg PnL%':>10}")
        print("  " + "-" * 60)
        
        for reason, data in sorted(by_exit.items(), key=lambda x: x[1]['count'], reverse=True):
            wr = data['wins'] / data['count'] if data['count'] > 0 else 0
            avg_pct = statistics.mean(data['pnl_pcts']) if data['pnl_pcts'] else 0
            print(f"  {reason:<20} {data['count']:>6} {wr:>9.1%} {data['pnl']:>+11.4f} {avg_pct:>+9.1f}%")
    
    def _print_hourly_analysis(self, trades: List[Dict]):
        """Analyze by hour of day"""
        print("\n" + "-" * 80)
        print("⏰ HOURLY ANALYSIS (UTC)")
        print("-" * 80)
        
        by_hour = defaultdict(lambda: {'count': 0, 'wins': 0, 'pnl': 0})
        
        for t in trades:
            entry_time = t.get('entry_time', '')
            if entry_time:
                try:
                    if isinstance(entry_time, str):
                        dt = datetime.fromisoformat(entry_time.replace('Z', ''))
                    else:
                        dt = entry_time
                    hour = dt.hour
                    
                    by_hour[hour]['count'] += 1
                    by_hour[hour]['pnl'] += t.get('pnl_sol', 0) or 0
                    if (t.get('pnl_sol') or 0) > 0:
                        by_hour[hour]['wins'] += 1
                except:
                    pass
        
        # Find best and worst hours
        valid_hours = [(h, d) for h, d in by_hour.items() if d['count'] >= 5]
        
        if valid_hours:
            best = sorted(valid_hours, key=lambda x: x[1]['wins']/x[1]['count'], reverse=True)[:5]
            worst = sorted(valid_hours, key=lambda x: x[1]['wins']/x[1]['count'])[:5]
            
            print(f"\n  🏆 BEST HOURS:")
            for hour, data in best:
                wr = data['wins'] / data['count']
                print(f"     {hour:02d}:00 UTC: {data['count']:>3} trades, {wr:>5.1%} WR, {data['pnl']:>+.4f} SOL")
            
            print(f"\n  ⚠️ WORST HOURS:")
            for hour, data in worst:
                wr = data['wins'] / data['count']
                print(f"     {hour:02d}:00 UTC: {data['count']:>3} trades, {wr:>5.1%} WR, {data['pnl']:>+.4f} SOL")
    
    def _print_wallet_analysis(self, trades: List[Dict]):
        """Analyze by wallet"""
        print("\n" + "-" * 80)
        print("👛 WALLET ANALYSIS")
        print("-" * 80)
        
        by_wallet = defaultdict(lambda: {'count': 0, 'wins': 0, 'pnl': 0})
        
        for t in trades:
            try:
                context = json.loads(t.get('entry_context_json', '{}'))
                wallet = context.get('wallet_address', context.get('wallet', 'UNKNOWN'))
            except:
                wallet = 'UNKNOWN'
            
            if wallet:
                by_wallet[wallet]['count'] += 1
                by_wallet[wallet]['pnl'] += t.get('pnl_sol', 0) or 0
                if (t.get('pnl_sol') or 0) > 0:
                    by_wallet[wallet]['wins'] += 1
        
        # Sort by trade count
        sorted_wallets = sorted(by_wallet.items(), key=lambda x: x[1]['count'], reverse=True)
        
        # Top performers (min 5 trades)
        with_enough_trades = [(w, d) for w, d in sorted_wallets if d['count'] >= 5]
        
        if with_enough_trades:
            top_by_wr = sorted(with_enough_trades, key=lambda x: x[1]['wins']/x[1]['count'], reverse=True)[:5]
            top_by_pnl = sorted(with_enough_trades, key=lambda x: x[1]['pnl'], reverse=True)[:5]
            worst_by_wr = sorted(with_enough_trades, key=lambda x: x[1]['wins']/x[1]['count'])[:5]
            
            print(f"\n  🏆 TOP BY WIN RATE (min 5 trades):")
            for wallet, data in top_by_wr:
                wr = data['wins'] / data['count']
                print(f"     {wallet[:12]}... {data['count']:>3} trades, {wr:>5.1%} WR, {data['pnl']:>+.4f} SOL")
            
            print(f"\n  💰 TOP BY PNL:")
            for wallet, data in top_by_pnl:
                wr = data['wins'] / data['count']
                print(f"     {wallet[:12]}... {data['count']:>3} trades, {wr:>5.1%} WR, {data['pnl']:>+.4f} SOL")
            
            print(f"\n  ❌ WORST PERFORMERS:")
            for wallet, data in worst_by_wr:
                wr = data['wins'] / data['count']
                print(f"     {wallet[:12]}... {data['count']:>3} trades, {wr:>5.1%} WR, {data['pnl']:>+.4f} SOL")
        
        print(f"\n  📊 WALLET SUMMARY:")
        print(f"     Total wallets: {len(by_wallet)}")
        profitable = sum(1 for w, d in by_wallet.items() if d['pnl'] > 0)
        print(f"     Profitable wallets: {profitable} ({profitable/len(by_wallet)*100:.0f}%)")
    
    def _print_time_analysis(self, trades: List[Dict]):
        """Analyze performance over time"""
        print("\n" + "-" * 80)
        print("📈 PERFORMANCE OVER TIME")
        print("-" * 80)
        
        # Group by day
        by_day = defaultdict(lambda: {'count': 0, 'wins': 0, 'pnl': 0})
        
        for t in trades:
            exit_time = t.get('exit_time', '')
            if exit_time:
                try:
                    if isinstance(exit_time, str):
                        dt = datetime.fromisoformat(exit_time.replace('Z', ''))
                    else:
                        dt = exit_time
                    day = dt.strftime('%Y-%m-%d')
                    
                    by_day[day]['count'] += 1
                    by_day[day]['pnl'] += t.get('pnl_sol', 0) or 0
                    if (t.get('pnl_sol') or 0) > 0:
                        by_day[day]['wins'] += 1
                except:
                    pass
        
        print(f"\n  {'Date':<12} {'Trades':>7} {'Win Rate':>10} {'PnL':>12} {'Cumulative':>12}")
        print("  " + "-" * 55)
        
        cumulative = 0
        for day in sorted(by_day.keys()):
            data = by_day[day]
            wr = data['wins'] / data['count'] if data['count'] > 0 else 0
            cumulative += data['pnl']
            print(f"  {day:<12} {data['count']:>7} {wr:>9.1%} {data['pnl']:>+11.4f} {cumulative:>+11.4f}")
    
    def _print_risk_analysis(self, trades: List[Dict], account: Dict):
        """Analyze risk metrics"""
        print("\n" + "-" * 80)
        print("⚠️ RISK ANALYSIS")
        print("-" * 80)
        
        # Calculate drawdown
        starting_bal = account.get('starting_balance', 10)
        running_bal = starting_bal
        peak_bal = starting_bal
        max_drawdown = 0
        max_drawdown_pct = 0
        
        daily_returns = []
        current_day_pnl = 0
        current_day = None
        
        for t in trades:
            pnl = t.get('pnl_sol', 0) or 0
            running_bal += pnl
            
            if running_bal > peak_bal:
                peak_bal = running_bal
            
            drawdown = peak_bal - running_bal
            drawdown_pct = drawdown / peak_bal if peak_bal > 0 else 0
            
            if drawdown > max_drawdown:
                max_drawdown = drawdown
                max_drawdown_pct = drawdown_pct
            
            # Track daily returns
            exit_time = t.get('exit_time', '')
            if exit_time:
                try:
                    if isinstance(exit_time, str):
                        dt = datetime.fromisoformat(exit_time.replace('Z', ''))
                    else:
                        dt = exit_time
                    day = dt.strftime('%Y-%m-%d')
                    
                    if current_day != day:
                        if current_day is not None:
                            daily_returns.append(current_day_pnl)
                        current_day = day
                        current_day_pnl = pnl
                    else:
                        current_day_pnl += pnl
                except:
                    pass
        
        if current_day_pnl:
            daily_returns.append(current_day_pnl)
        
        # Consecutive losses
        max_consec_losses = 0
        current_consec = 0
        for t in trades:
            if (t.get('pnl_sol') or 0) < 0:
                current_consec += 1
                max_consec_losses = max(max_consec_losses, current_consec)
            else:
                current_consec = 0
        
        print(f"\n  💸 DRAWDOWN:")
        print(f"     Max Drawdown: {max_drawdown:.4f} SOL ({max_drawdown_pct:.1%})")
        print(f"     Peak Balance: {peak_bal:.4f} SOL")
        
        print(f"\n  📉 LOSING STREAKS:")
        print(f"     Max Consecutive Losses: {max_consec_losses}")
        
        if daily_returns:
            print(f"\n  📆 DAILY RETURNS:")
            print(f"     Best Day:  {max(daily_returns):+.4f} SOL")
            print(f"     Worst Day: {min(daily_returns):+.4f} SOL")
            print(f"     Avg Day:   {statistics.mean(daily_returns):+.4f} SOL")
            
            winning_days = sum(1 for r in daily_returns if r > 0)
            print(f"     Winning Days: {winning_days}/{len(daily_returns)} ({winning_days/len(daily_returns)*100:.0f}%)")
    
    def _print_go_nogo_assessment(self, trades: List[Dict], account: Dict):
        """Print go/no-go assessment for live trading"""
        print("\n" + "-" * 80)
        print("🚦 GO / NO-GO ASSESSMENT")
        print("-" * 80)
        
        total = len(trades)
        wins = sum(1 for t in trades if (t.get('pnl_sol') or 0) > 0)
        win_rate = wins / total if total > 0 else 0
        
        total_pnl = sum(t.get('pnl_sol', 0) or 0 for t in trades)
        starting_bal = account.get('starting_balance', 10)
        return_pct = (total_pnl / starting_bal) * 100 if starting_bal > 0 else 0
        
        # Calculate metrics
        winners = [t for t in trades if (t.get('pnl_sol') or 0) > 0]
        losers = [t for t in trades if (t.get('pnl_sol') or 0) < 0]
        
        avg_win = statistics.mean([t['pnl_sol'] for t in winners]) if winners else 0
        avg_loss = abs(statistics.mean([t['pnl_sol'] for t in losers])) if losers else 0
        risk_reward = avg_win / avg_loss if avg_loss > 0 else 0
        
        gross_profit = sum(t.get('pnl_sol', 0) for t in winners)
        gross_loss = abs(sum(t.get('pnl_sol', 0) for t in losers))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Criteria checks
        criteria = []
        
        # 1. Sample size
        if total >= 100:
            criteria.append(('✅', f'Sample size: {total} trades (need 100+)'))
        else:
            criteria.append(('⚠️', f'Sample size: {total} trades (need 100+)'))
        
        # 2. Profitability
        if total_pnl > 0:
            criteria.append(('✅', f'Profitable: {total_pnl:+.4f} SOL ({return_pct:+.1f}%)'))
        else:
            criteria.append(('❌', f'Not profitable: {total_pnl:+.4f} SOL ({return_pct:+.1f}%)'))
        
        # 3. Win rate with R:R context
        breakeven_wr = 1 / (1 + risk_reward) if risk_reward > 0 else 0.5
        if win_rate > breakeven_wr:
            criteria.append(('✅', f'Win rate {win_rate:.1%} > breakeven {breakeven_wr:.1%}'))
        else:
            criteria.append(('❌', f'Win rate {win_rate:.1%} < breakeven {breakeven_wr:.1%}'))
        
        # 4. Profit factor
        if profit_factor >= 1.5:
            criteria.append(('✅', f'Profit factor: {profit_factor:.2f} (good if >1.5)'))
        elif profit_factor >= 1.0:
            criteria.append(('⚠️', f'Profit factor: {profit_factor:.2f} (marginal, want >1.5)'))
        else:
            criteria.append(('❌', f'Profit factor: {profit_factor:.2f} (losing)'))
        
        # 5. Consistency (positive days)
        by_day = defaultdict(float)
        for t in trades:
            exit_time = t.get('exit_time', '')
            if exit_time:
                try:
                    dt = datetime.fromisoformat(exit_time.replace('Z', ''))
                    day = dt.strftime('%Y-%m-%d')
                    by_day[day] += t.get('pnl_sol', 0) or 0
                except:
                    pass
        
        if by_day:
            positive_days = sum(1 for pnl in by_day.values() if pnl > 0)
            day_win_rate = positive_days / len(by_day)
            if day_win_rate >= 0.6:
                criteria.append(('✅', f'Consistent: {positive_days}/{len(by_day)} profitable days ({day_win_rate:.0%})'))
            elif day_win_rate >= 0.4:
                criteria.append(('⚠️', f'Mixed: {positive_days}/{len(by_day)} profitable days ({day_win_rate:.0%})'))
            else:
                criteria.append(('❌', f'Inconsistent: {positive_days}/{len(by_day)} profitable days ({day_win_rate:.0%})'))
        
        print("\n  CRITERIA CHECK:")
        for status, msg in criteria:
            print(f"  {status} {msg}")
        
        # Overall assessment
        greens = sum(1 for s, _ in criteria if s == '✅')
        reds = sum(1 for s, _ in criteria if s == '❌')
        
        print("\n  " + "=" * 50)
        if reds == 0 and greens >= 4:
            print("  🟢 GO FOR LIVE TRADING")
            print("     Strategy shows consistent profitability.")
            print("     Start with small position sizes (25-50% of paper).")
        elif reds == 0:
            print("  🟡 CAUTIOUS GO")
            print("     Strategy is promising but needs more data.")
            print("     Consider extended paper trading or very small live.")
        elif total_pnl > 0:
            print("  🟡 NOT YET - But Promising")
            print("     Profitable but some concerns remain.")
            print("     Address issues before going live.")
        else:
            print("  🔴 NO-GO")
            print("     Strategy is not ready for live trading.")
            print("     Continue optimizing in paper mode.")
        print("  " + "=" * 50)
    
    def _print_recommendations(self, trades: List[Dict]):
        """Print specific recommendations"""
        print("\n" + "-" * 80)
        print("💡 RECOMMENDATIONS")
        print("-" * 80)
        
        recommendations = []
        
        # Analyze exit reasons
        by_exit = defaultdict(lambda: {'count': 0, 'wins': 0})
        for t in trades:
            reason = t.get('exit_reason', 'UNKNOWN')
            by_exit[reason]['count'] += 1
            if (t.get('pnl_sol') or 0) > 0:
                by_exit[reason]['wins'] += 1
        
        # Check stop loss rate
        sl_data = by_exit.get('STOP_LOSS', {'count': 0, 'wins': 0})
        if sl_data['count'] > 0:
            sl_pct = sl_data['count'] / len(trades)
            if sl_pct > 0.50:
                recommendations.append(
                    f"⚠️ {sl_pct:.0%} of trades hit stop loss. Consider widening stops or improving entry timing."
                )
        
        # Check time stops
        ts_data = by_exit.get('TIME_STOP', {'count': 0, 'wins': 0})
        if ts_data['count'] >= 10:
            ts_wr = ts_data['wins'] / ts_data['count']
            if ts_wr < 0.30:
                recommendations.append(
                    f"⚠️ Time stops have {ts_wr:.0%} WR. Consider reducing max hold time."
                )
        
        # Check for wallet concentration
        by_wallet = defaultdict(int)
        for t in trades:
            try:
                context = json.loads(t.get('entry_context_json', '{}'))
                wallet = context.get('wallet_address', 'UNKNOWN')
                by_wallet[wallet] += 1
            except:
                pass
        
        if by_wallet:
            top_wallet_trades = max(by_wallet.values())
            if top_wallet_trades > len(trades) * 0.3:
                recommendations.append(
                    f"⚠️ One wallet accounts for {top_wallet_trades/len(trades):.0%} of trades. Diversify wallet sources."
                )
        
        # Win rate suggestions
        total = len(trades)
        wins = sum(1 for t in trades if (t.get('pnl_sol') or 0) > 0)
        win_rate = wins / total if total > 0 else 0
        
        if win_rate < 0.40:
            recommendations.append(
                "📈 Win rate is below 40%. Focus on better entry criteria and wallet selection."
            )
        
        # Positive recommendations
        total_pnl = sum(t.get('pnl_sol', 0) or 0 for t in trades)
        if total_pnl > 0:
            recommendations.append(
                "✅ Strategy is profitable! Focus on consistency and risk management for live trading."
            )
        
        if win_rate >= 0.40:
            recommendations.append(
                "✅ Win rate is healthy for asymmetric R:R strategy."
            )
        
        print()
        for rec in recommendations:
            print(f"  {rec}")
        
        print("\n  📋 BEFORE GOING LIVE:")
        print("     1. Start with 25-50% of paper trading position sizes")
        print("     2. Set strict daily loss limits (e.g., 10% of capital)")
        print("     3. Monitor first 20 live trades closely")
        print("     4. Have a kill switch ready if things go wrong")
        print("     5. Never risk more than you can afford to lose")


def main():
    """Run the analysis"""
    import sys
    
    # Try to find the database
    db_paths = [
        "robust_paper_trades_v6.db",
        "paper_trades.db",
        "learning_paper_trades.db"
    ]
    
    db_path = None
    for path in db_paths:
        if os.path.exists(path):
            db_path = path
            break
    
    if len(sys.argv) > 1:
        db_path = sys.argv[1]
    
    if not db_path:
        print("❌ No database found!")
        print("   Usage: python trading_analysis.py [database_path]")
        return
    
    analyzer = TradingAnalyzer(db_path)
    analyzer.run_full_analysis()


if __name__ == "__main__":
    main()
