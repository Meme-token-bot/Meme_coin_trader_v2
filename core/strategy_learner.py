"""
STRATEGY LEARNER V1
===================

This is the ACTUAL learning system that evolves trading strategy over time.

It analyzes closed trades to learn:
1. Which wallet characteristics predict wins
2. Which token characteristics predict wins
3. Optimal entry timing (hour of day)
4. Optimal exit parameters (SL/TP)
5. Which signals to filter out
6. Which wallets to stop tracking (poor performers)

The goal: Evolve from ~43% WR to 55%+ WR by learning what works.

Author: Claude
"""

import sqlite3
import json
import statistics
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from contextlib import contextmanager
import os

# Import wallet performance analyzer
try:
    from core.wallet_performance import WalletPerformanceAnalyzer
    WALLET_ANALYZER_AVAILABLE = True
except ImportError:
    try:
        from wallet_performance import WalletPerformanceAnalyzer
        WALLET_ANALYZER_AVAILABLE = True
    except ImportError:
        WALLET_ANALYZER_AVAILABLE = False


@dataclass
class StrategyConfig:
    """Current strategy configuration - evolves over time"""
    # Entry filters
    min_wallet_wr: float = 0.30          # Minimum wallet win rate
    min_conviction: int = 30             # Minimum conviction score
    min_liquidity: float = 5000          # Minimum liquidity USD
    
    # Time filters (learned)
    blocked_hours_utc: List[int] = field(default_factory=list)
    preferred_hours_utc: List[int] = field(default_factory=list)
    
    # Exit parameters
    stop_loss_pct: float = -15.0
    take_profit_pct: float = 30.0
    trailing_stop_pct: float = 10.0
    max_hold_hours: int = 12
    
    # Learning state
    iteration: int = 0
    phase: str = "exploration"  # exploration -> refinement -> optimization
    last_updated: str = ""
    
    # Performance tracking
    baseline_wr: float = 0.0             # WR before any filtering
    current_wr: float = 0.0              # WR with current filters
    target_wr: float = 0.55              # Goal win rate


class StrategyLearner:
    """
    Learns from trade outcomes to improve strategy.
    
    Run every 6 hours (or after N trades) to:
    1. Analyze what's working
    2. Update filters
    3. Track improvement
    4. Remove poor performing wallets
    """
    
    def __init__(self, db_path: str = "robust_paper_trades_v6.db", main_db_path: str = "swing_traders.db"):
        self.db_path = db_path
        self.main_db_path = main_db_path
        self.config_path = "strategy_config.json"
        self.config = self._load_config()
        
        # Initialize wallet performance analyzer
        if WALLET_ANALYZER_AVAILABLE:
            self.wallet_analyzer = WalletPerformanceAnalyzer(
                paper_db_path=db_path,
                main_db_path=main_db_path
            )
        else:
            self.wallet_analyzer = None
            print("  ⚠️ Wallet analyzer not available")
        
        print(f"📚 Strategy Learner initialized")
        print(f"   Phase: {self.config.phase}")
        print(f"   Iteration: {self.config.iteration}")
        print(f"   Current WR target: {self.config.target_wr:.0%}")
    
    def _load_config(self) -> StrategyConfig:
        """Load strategy config from file"""
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    data = json.load(f)
                    return StrategyConfig(**data)
            except:
                pass
        return StrategyConfig()
    
    def _save_config(self):
        """Save strategy config to file"""
        self.config.last_updated = datetime.utcnow().isoformat()
        with open(self.config_path, 'w') as f:
            json.dump(asdict(self.config), f, indent=2)
    
    @contextmanager
    def _get_connection(self):
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()
    
    def get_closed_trades(self, days: int = 7) -> List[Dict]:
        """Get closed trades for analysis"""
        cutoff = datetime.utcnow() - timedelta(days=days)
        
        with self._get_connection() as conn:
            rows = conn.execute("""
                SELECT * FROM paper_positions_v6
                WHERE status = 'closed'
                AND exit_time > ?
                ORDER BY exit_time DESC
            """, (cutoff,)).fetchall()
            
            return [dict(r) for r in rows]
    
    def run_learning_iteration(self) -> Dict:
        """
        Run a full learning iteration.
        
        Returns dict with findings and actions taken.
        """
        print("\n" + "=" * 70)
        print("🧠 RUNNING STRATEGY LEARNING ITERATION")
        print("=" * 70)
        
        self.config.iteration += 1
        results = {
            'iteration': self.config.iteration,
            'timestamp': datetime.utcnow().isoformat(),
            'findings': [],
            'actions': [],
            'recommendations': []
        }
        
        # Get trade data
        trades = self.get_closed_trades(days=7)
        
        if len(trades) < 20:
            print(f"  ⏳ Not enough trades yet ({len(trades)}/20 minimum)")
            results['status'] = 'insufficient_data'
            return results
        
        print(f"  📊 Analyzing {len(trades)} trades from last 7 days")
        
        # 1. Overall performance
        overall = self._analyze_overall(trades)
        results['overall'] = overall
        print(f"\n  📈 Overall Performance:")
        print(f"     Win Rate: {overall['win_rate']:.1%}")
        print(f"     Total PnL: {overall['total_pnl']:+.4f} SOL")
        print(f"     Avg PnL per trade: {overall['avg_pnl']:+.4f} SOL")
        
        # 2. Analysis by wallet win rate
        wallet_analysis = self._analyze_by_wallet_wr(trades)
        results['wallet_analysis'] = wallet_analysis
        self._print_wallet_analysis(wallet_analysis, results)
        
        # 3. Analysis by conviction score
        conviction_analysis = self._analyze_by_conviction(trades)
        results['conviction_analysis'] = conviction_analysis
        self._print_conviction_analysis(conviction_analysis, results)
        
        # 4. Analysis by hour of day
        hourly_analysis = self._analyze_by_hour(trades)
        results['hourly_analysis'] = hourly_analysis
        self._print_hourly_analysis(hourly_analysis, results)
        
        # 5. Analysis by exit reason
        exit_analysis = self._analyze_by_exit_reason(trades)
        results['exit_analysis'] = exit_analysis
        self._print_exit_analysis(exit_analysis, results)
        
        # 6. Analysis by token characteristics
        token_analysis = self._analyze_by_token_chars(trades)
        results['token_analysis'] = token_analysis
        
        # 7. Generate and apply recommendations
        self._generate_recommendations(results)
        self._apply_recommendations(results)
        
        # 8. Wallet cleanup - identify poor performers
        if self.wallet_analyzer:
            print("\n  🧹 Analyzing wallet performance...")
            wallet_cleanup = self._analyze_wallet_performance()
            results['wallet_cleanup'] = wallet_cleanup
        
        # Save updated config
        self.config.current_wr = overall['win_rate']
        self._save_config()
        
        # Print summary
        self._print_summary(results)
        
        results['status'] = 'completed'
        return results
    
    def _analyze_wallet_performance(self) -> Dict:
        """Analyze and report on wallet performance"""
        if not self.wallet_analyzer:
            return {'status': 'analyzer_not_available'}
        
        to_remove = self.wallet_analyzer.get_wallets_to_remove(days=14)
        top = self.wallet_analyzer.get_top_performers(days=14, min_trades=5)[:5]
        
        result = {
            'wallets_to_remove': len(to_remove),
            'remove_list': [p.to_dict() for p in to_remove[:10]],
            'top_performers': [p.to_dict() for p in top]
        }
        
        if to_remove:
            print(f"     ❌ {len(to_remove)} wallets flagged for removal")
            for perf in to_remove[:3]:
                print(f"        {perf.address[:12]}... | {perf.win_rate:.0%} WR | {perf.total_pnl_sol:+.4f} SOL")
        
        if top:
            print(f"     🏆 Top performers:")
            for perf in top[:3]:
                print(f"        {perf.address[:12]}... | {perf.win_rate:.0%} WR | {perf.total_pnl_sol:+.4f} SOL")
        
        return result
    
    def run_wallet_cleanup(self, webhook_manager=None, db=None, dry_run: bool = True) -> Dict:
        """
        Run wallet cleanup to remove poor performers.
        
        Should be called periodically (e.g., daily) with dry_run=False to actually remove wallets.
        """
        if not self.wallet_analyzer:
            return {'status': 'analyzer_not_available'}
        
        return self.wallet_analyzer.run_cleanup(
            webhook_manager=webhook_manager,
            db=db,
            dry_run=dry_run
        )
    
    def _analyze_overall(self, trades: List[Dict]) -> Dict:
        """Calculate overall performance metrics"""
        wins = sum(1 for t in trades if (t.get('pnl_sol') or 0) > 0)
        total_pnl = sum(t.get('pnl_sol', 0) or 0 for t in trades)
        
        return {
            'total_trades': len(trades),
            'wins': wins,
            'losses': len(trades) - wins,
            'win_rate': wins / len(trades) if trades else 0,
            'total_pnl': total_pnl,
            'avg_pnl': total_pnl / len(trades) if trades else 0,
        }
    
    def _analyze_by_wallet_wr(self, trades: List[Dict]) -> Dict:
        """Analyze performance by wallet win rate buckets"""
        buckets = {
            '30-40%': {'trades': [], 'range': (0.30, 0.40)},
            '40-50%': {'trades': [], 'range': (0.40, 0.50)},
            '50-60%': {'trades': [], 'range': (0.50, 0.60)},
            '60-70%': {'trades': [], 'range': (0.60, 0.70)},
            '70%+': {'trades': [], 'range': (0.70, 1.0)},
        }
        
        for trade in trades:
            # Get wallet WR from entry context
            try:
                context = json.loads(trade.get('entry_context_json', '{}'))
                wallet_wr = context.get('wallet_win_rate', 0)
                if wallet_wr > 1:
                    wallet_wr = wallet_wr / 100
            except:
                wallet_wr = 0.5
            
            for bucket_name, bucket_data in buckets.items():
                low, high = bucket_data['range']
                if low <= wallet_wr < high:
                    bucket_data['trades'].append(trade)
                    break
        
        # Calculate stats for each bucket
        result = {}
        for bucket_name, bucket_data in buckets.items():
            trades_in_bucket = bucket_data['trades']
            if trades_in_bucket:
                wins = sum(1 for t in trades_in_bucket if (t.get('pnl_sol') or 0) > 0)
                pnl = sum(t.get('pnl_sol', 0) or 0 for t in trades_in_bucket)
                result[bucket_name] = {
                    'count': len(trades_in_bucket),
                    'wins': wins,
                    'win_rate': wins / len(trades_in_bucket),
                    'total_pnl': pnl,
                    'avg_pnl': pnl / len(trades_in_bucket)
                }
            else:
                result[bucket_name] = {'count': 0, 'wins': 0, 'win_rate': 0, 'total_pnl': 0, 'avg_pnl': 0}
        
        return result
    
    def _print_wallet_analysis(self, analysis: Dict, results: Dict):
        """Print wallet WR analysis"""
        print(f"\n  👛 Performance by Wallet Win Rate:")
        
        for bucket, data in analysis.items():
            if data['count'] >= 5:
                status = "✅" if data['win_rate'] >= 0.50 else "❌"
                print(f"     {bucket}: {data['count']} trades, "
                      f"{data['win_rate']:.0%} WR, {data['total_pnl']:+.4f} SOL {status}")
                
                # Generate finding
                if data['win_rate'] < 0.40 and data['count'] >= 10:
                    results['findings'].append(
                        f"Wallet WR {bucket} underperforms ({data['win_rate']:.0%} WR)"
                    )
                elif data['win_rate'] >= 0.55 and data['count'] >= 10:
                    results['findings'].append(
                        f"Wallet WR {bucket} outperforms ({data['win_rate']:.0%} WR)"
                    )
    
    def _analyze_by_conviction(self, trades: List[Dict]) -> Dict:
        """Analyze performance by conviction score buckets"""
        buckets = defaultdict(lambda: {'trades': [], 'wins': 0, 'pnl': 0})
        
        for trade in trades:
            conviction = trade.get('conviction_score', 50) or 50
            
            if conviction < 40:
                bucket = '<40'
            elif conviction < 50:
                bucket = '40-50'
            elif conviction < 60:
                bucket = '50-60'
            elif conviction < 70:
                bucket = '60-70'
            else:
                bucket = '70+'
            
            buckets[bucket]['trades'].append(trade)
            if (trade.get('pnl_sol') or 0) > 0:
                buckets[bucket]['wins'] += 1
            buckets[bucket]['pnl'] += trade.get('pnl_sol', 0) or 0
        
        result = {}
        for bucket, data in buckets.items():
            count = len(data['trades'])
            result[bucket] = {
                'count': count,
                'wins': data['wins'],
                'win_rate': data['wins'] / count if count > 0 else 0,
                'total_pnl': data['pnl'],
                'avg_pnl': data['pnl'] / count if count > 0 else 0
            }
        
        return result
    
    def _print_conviction_analysis(self, analysis: Dict, results: Dict):
        """Print conviction analysis"""
        print(f"\n  🎯 Performance by Conviction Score:")
        
        for bucket in ['<40', '40-50', '50-60', '60-70', '70+']:
            data = analysis.get(bucket, {})
            if data.get('count', 0) >= 5:
                status = "✅" if data['win_rate'] >= 0.50 else "❌"
                print(f"     Conv {bucket}: {data['count']} trades, "
                      f"{data['win_rate']:.0%} WR, {data['total_pnl']:+.4f} SOL {status}")
    
    def _analyze_by_hour(self, trades: List[Dict]) -> Dict:
        """Analyze performance by entry hour (UTC)"""
        by_hour = defaultdict(lambda: {'count': 0, 'wins': 0, 'pnl': 0})
        
        for trade in trades:
            entry_time = trade.get('entry_time')
            if entry_time:
                try:
                    if isinstance(entry_time, str):
                        entry_time = datetime.fromisoformat(entry_time.replace('Z', ''))
                    hour = entry_time.hour
                    
                    by_hour[hour]['count'] += 1
                    if (trade.get('pnl_sol') or 0) > 0:
                        by_hour[hour]['wins'] += 1
                    by_hour[hour]['pnl'] += trade.get('pnl_sol', 0) or 0
                except:
                    pass
        
        result = {}
        for hour in range(24):
            data = by_hour[hour]
            result[hour] = {
                'count': data['count'],
                'wins': data['wins'],
                'win_rate': data['wins'] / data['count'] if data['count'] > 0 else 0,
                'total_pnl': data['pnl'],
                'avg_pnl': data['pnl'] / data['count'] if data['count'] > 0 else 0
            }
        
        return result
    
    def _print_hourly_analysis(self, analysis: Dict, results: Dict):
        """Print hourly analysis"""
        print(f"\n  ⏰ Performance by Hour (UTC):")
        
        # Find best and worst hours
        valid_hours = [(h, d) for h, d in analysis.items() if d['count'] >= 5]
        
        if valid_hours:
            best_hours = sorted(valid_hours, key=lambda x: x[1]['win_rate'], reverse=True)[:3]
            worst_hours = sorted(valid_hours, key=lambda x: x[1]['win_rate'])[:3]
            
            print(f"     Best hours:")
            for hour, data in best_hours:
                print(f"       {hour:02d}:00 UTC: {data['count']} trades, "
                      f"{data['win_rate']:.0%} WR, {data['total_pnl']:+.4f} SOL ✅")
                if data['win_rate'] >= 0.55:
                    results['findings'].append(f"Hour {hour:02d}:00 UTC is a high WR time ({data['win_rate']:.0%})")
            
            print(f"     Worst hours:")
            for hour, data in worst_hours:
                print(f"       {hour:02d}:00 UTC: {data['count']} trades, "
                      f"{data['win_rate']:.0%} WR, {data['total_pnl']:+.4f} SOL ❌")
                if data['win_rate'] <= 0.35 and data['count'] >= 10:
                    results['findings'].append(f"Hour {hour:02d}:00 UTC underperforms ({data['win_rate']:.0%} WR)")
    
    def _analyze_by_exit_reason(self, trades: List[Dict]) -> Dict:
        """Analyze performance by exit reason"""
        by_reason = defaultdict(lambda: {'count': 0, 'wins': 0, 'pnl': 0, 'hold_times': []})
        
        for trade in trades:
            reason = trade.get('exit_reason', 'UNKNOWN')
            
            by_reason[reason]['count'] += 1
            if (trade.get('pnl_sol') or 0) > 0:
                by_reason[reason]['wins'] += 1
            by_reason[reason]['pnl'] += trade.get('pnl_sol', 0) or 0
            
            hold_mins = trade.get('hold_duration_minutes', 0) or 0
            by_reason[reason]['hold_times'].append(hold_mins)
        
        result = {}
        for reason, data in by_reason.items():
            result[reason] = {
                'count': data['count'],
                'wins': data['wins'],
                'win_rate': data['wins'] / data['count'] if data['count'] > 0 else 0,
                'total_pnl': data['pnl'],
                'avg_hold_mins': statistics.mean(data['hold_times']) if data['hold_times'] else 0
            }
        
        return result
    
    def _print_exit_analysis(self, analysis: Dict, results: Dict):
        """Print exit reason analysis"""
        print(f"\n  🚪 Performance by Exit Reason:")
        
        for reason, data in sorted(analysis.items(), key=lambda x: x[1]['count'], reverse=True):
            if data['count'] >= 3:
                pct_of_total = data['count'] / sum(d['count'] for d in analysis.values()) * 100
                print(f"     {reason}: {data['count']} ({pct_of_total:.0f}%), "
                      f"{data['total_pnl']:+.4f} SOL, avg hold: {data['avg_hold_mins']:.0f}m")
    
    def _analyze_by_token_chars(self, trades: List[Dict]) -> Dict:
        """Analyze by token characteristics"""
        by_liquidity = defaultdict(lambda: {'count': 0, 'wins': 0, 'pnl': 0})
        by_cluster = defaultdict(lambda: {'count': 0, 'wins': 0, 'pnl': 0})
        
        for trade in trades:
            try:
                context = json.loads(trade.get('entry_context_json', '{}'))
                
                # Liquidity buckets
                liq = context.get('liquidity', 0)
                if liq < 10000:
                    liq_bucket = '<10k'
                elif liq < 50000:
                    liq_bucket = '10k-50k'
                elif liq < 100000:
                    liq_bucket = '50k-100k'
                else:
                    liq_bucket = '100k+'
                
                by_liquidity[liq_bucket]['count'] += 1
                if (trade.get('pnl_sol') or 0) > 0:
                    by_liquidity[liq_bucket]['wins'] += 1
                by_liquidity[liq_bucket]['pnl'] += trade.get('pnl_sol', 0) or 0
                
                # Cluster signals
                is_cluster = trade.get('is_cluster_signal', False)
                cluster_key = 'cluster' if is_cluster else 'single'
                by_cluster[cluster_key]['count'] += 1
                if (trade.get('pnl_sol') or 0) > 0:
                    by_cluster[cluster_key]['wins'] += 1
                by_cluster[cluster_key]['pnl'] += trade.get('pnl_sol', 0) or 0
            except:
                pass
        
        return {
            'by_liquidity': {k: {
                'count': v['count'],
                'win_rate': v['wins'] / v['count'] if v['count'] > 0 else 0,
                'total_pnl': v['pnl']
            } for k, v in by_liquidity.items()},
            'by_cluster': {k: {
                'count': v['count'],
                'win_rate': v['wins'] / v['count'] if v['count'] > 0 else 0,
                'total_pnl': v['pnl']
            } for k, v in by_cluster.items()}
        }
    
    def _generate_recommendations(self, results: Dict):
        """Generate strategy recommendations based on analysis"""
        findings = results['findings']
        recommendations = []
        
        # Wallet WR recommendations
        wallet_data = results.get('wallet_analysis', {})
        
        # Check if low WR wallets are dragging down performance
        low_wr_buckets = ['30-40%', '40-50%']
        low_wr_trades = sum(wallet_data.get(b, {}).get('count', 0) for b in low_wr_buckets)
        low_wr_winrate = 0
        total_low_trades = 0
        for b in low_wr_buckets:
            d = wallet_data.get(b, {})
            if d.get('count', 0) > 0:
                low_wr_winrate += d.get('wins', 0)
                total_low_trades += d.get('count', 0)
        
        if total_low_trades > 0:
            low_wr_winrate = low_wr_winrate / total_low_trades
            if low_wr_winrate < 0.40 and total_low_trades >= 20:
                recommendations.append({
                    'type': 'min_wallet_wr',
                    'current': self.config.min_wallet_wr,
                    'recommended': 0.50,
                    'reason': f"Low WR wallets (<50%) have {low_wr_winrate:.0%} WR - filter them out",
                    'impact': f"Would filter {total_low_trades} trades"
                })
        
        # Hourly recommendations
        hourly_data = results.get('hourly_analysis', {})
        bad_hours = []
        good_hours = []
        
        for hour, data in hourly_data.items():
            if data['count'] >= 10:
                if data['win_rate'] <= 0.35:
                    bad_hours.append(hour)
                elif data['win_rate'] >= 0.55:
                    good_hours.append(hour)
        
        if bad_hours:
            recommendations.append({
                'type': 'blocked_hours',
                'current': self.config.blocked_hours_utc,
                'recommended': bad_hours,
                'reason': f"Hours {bad_hours} have <35% WR",
                'impact': "Block trading during these hours"
            })
        
        if good_hours:
            recommendations.append({
                'type': 'preferred_hours',
                'current': self.config.preferred_hours_utc,
                'recommended': good_hours,
                'reason': f"Hours {good_hours} have >55% WR",
                'impact': "Prioritize trading during these hours"
            })
        
        # Exit parameter recommendations
        exit_data = results.get('exit_analysis', {})
        
        sl_data = exit_data.get('STOP_LOSS', {})
        tp_data = exit_data.get('TAKE_PROFIT', {})
        time_data = exit_data.get('TIME_STOP', {})
        
        # If too many stop losses, consider wider stops
        if sl_data.get('count', 0) > 0:
            sl_pct = sl_data['count'] / sum(d['count'] for d in exit_data.values() if d['count'] > 0)
            if sl_pct > 0.50:  # More than 50% hit stop loss
                recommendations.append({
                    'type': 'stop_loss',
                    'current': self.config.stop_loss_pct,
                    'recommended': self.config.stop_loss_pct * 1.2,  # 20% wider
                    'reason': f"{sl_pct:.0%} of exits are stop losses - may be too tight",
                    'impact': "Wider stops may let winners recover"
                })
        
        # If time stops are mostly losers, reduce hold time
        if time_data.get('count', 0) >= 10 and time_data.get('win_rate', 0) < 0.30:
            recommendations.append({
                'type': 'max_hold_hours',
                'current': self.config.max_hold_hours,
                'recommended': 8,
                'reason': f"Time stops have {time_data['win_rate']:.0%} WR - reduce hold time",
                'impact': "Exit faster before positions decay"
            })
        
        results['recommendations'] = recommendations
    
    def _apply_recommendations(self, results: Dict):
        """Apply recommendations (conservatively)"""
        actions = []
        
        for rec in results['recommendations']:
            rec_type = rec['type']
            
            # Only apply in refinement phase or later
            if self.config.phase == 'exploration' and self.config.iteration < 5:
                continue
            
            if rec_type == 'min_wallet_wr':
                if rec['recommended'] > self.config.min_wallet_wr:
                    old = self.config.min_wallet_wr
                    self.config.min_wallet_wr = rec['recommended']
                    actions.append(f"Increased min_wallet_wr: {old:.0%} → {rec['recommended']:.0%}")
            
            elif rec_type == 'blocked_hours':
                if rec['recommended']:
                    self.config.blocked_hours_utc = rec['recommended']
                    actions.append(f"Set blocked hours: {rec['recommended']}")
            
            elif rec_type == 'preferred_hours':
                if rec['recommended']:
                    self.config.preferred_hours_utc = rec['recommended']
                    actions.append(f"Set preferred hours: {rec['recommended']}")
            
            elif rec_type == 'stop_loss':
                # Apply conservatively
                new_sl = rec['recommended']
                if -25 < new_sl < -10:  # Keep within reasonable bounds
                    old = self.config.stop_loss_pct
                    self.config.stop_loss_pct = new_sl
                    actions.append(f"Adjusted stop_loss: {old}% → {new_sl:.1f}%")
            
            elif rec_type == 'max_hold_hours':
                old = self.config.max_hold_hours
                self.config.max_hold_hours = rec['recommended']
                actions.append(f"Adjusted max_hold: {old}h → {rec['recommended']}h")
        
        # Update phase based on progress
        if self.config.iteration >= 3 and self.config.phase == 'exploration':
            self.config.phase = 'refinement'
            actions.append("Phase: exploration → refinement")
        
        if self.config.current_wr >= 0.50 and self.config.phase == 'refinement':
            self.config.phase = 'optimization'
            actions.append("Phase: refinement → optimization")
        
        results['actions'] = actions
    
    def _print_summary(self, results: Dict):
        """Print learning iteration summary"""
        print("\n" + "=" * 70)
        print("📋 LEARNING ITERATION SUMMARY")
        print("=" * 70)
        
        print(f"\n  Iteration: {self.config.iteration}")
        print(f"  Phase: {self.config.phase}")
        print(f"  Current WR: {self.config.current_wr:.1%}")
        print(f"  Target WR: {self.config.target_wr:.0%}")
        
        if results['findings']:
            print(f"\n  📊 Key Findings:")
            for finding in results['findings'][:5]:
                print(f"     • {finding}")
        
        if results['recommendations']:
            print(f"\n  💡 Recommendations:")
            for rec in results['recommendations']:
                print(f"     • {rec['type']}: {rec['reason']}")
        
        if results['actions']:
            print(f"\n  ✅ Actions Taken:")
            for action in results['actions']:
                print(f"     • {action}")
        
        print(f"\n  📐 Current Strategy Config:")
        print(f"     Min Wallet WR: {self.config.min_wallet_wr:.0%}")
        print(f"     Min Conviction: {self.config.min_conviction}")
        print(f"     Stop Loss: {self.config.stop_loss_pct}%")
        print(f"     Take Profit: {self.config.take_profit_pct}%")
        print(f"     Max Hold: {self.config.max_hold_hours}h")
        print(f"     Blocked Hours: {self.config.blocked_hours_utc or 'None'}")
        print(f"     Preferred Hours: {self.config.preferred_hours_utc or 'None'}")
        
        print("\n" + "=" * 70)
    
    def get_current_filters(self) -> Dict:
        """Get current filter settings for use in trading"""
        return {
            'min_wallet_wr': self.config.min_wallet_wr,
            'min_conviction': self.config.min_conviction,
            'min_liquidity': self.config.min_liquidity,
            'blocked_hours_utc': self.config.blocked_hours_utc,
            'preferred_hours_utc': self.config.preferred_hours_utc,
            'stop_loss_pct': self.config.stop_loss_pct,
            'take_profit_pct': self.config.take_profit_pct,
            'trailing_stop_pct': self.config.trailing_stop_pct,
            'max_hold_hours': self.config.max_hold_hours,
        }
    
    def should_take_trade(self, signal: Dict, wallet_data: Dict, current_hour_utc: int = None) -> Tuple[bool, str]:
        """
        Check if a trade passes current learned filters.
        
        Returns (should_trade, reason)
        """
        # Check blocked hours
        if current_hour_utc is None:
            current_hour_utc = datetime.utcnow().hour
        
        if current_hour_utc in self.config.blocked_hours_utc:
            return False, f"Hour {current_hour_utc} is blocked"
        
        # Check wallet WR
        wallet_wr = wallet_data.get('win_rate', 0)
        if wallet_wr > 1:
            wallet_wr = wallet_wr / 100
        
        if wallet_wr < self.config.min_wallet_wr:
            return False, f"Wallet WR {wallet_wr:.0%} < {self.config.min_wallet_wr:.0%}"
        
        # Check liquidity
        liquidity = signal.get('liquidity', 0)
        if liquidity < self.config.min_liquidity:
            return False, f"Liquidity ${liquidity:,.0f} < ${self.config.min_liquidity:,.0f}"
        
        return True, "Passed all filters"
    
    def print_status(self):
        """Print current learner status"""
        print("\n" + "=" * 70)
        print("📚 STRATEGY LEARNER STATUS")
        print("=" * 70)
        
        print(f"\n  Phase: {self.config.phase}")
        print(f"  Iteration: {self.config.iteration}")
        print(f"  Last Updated: {self.config.last_updated or 'Never'}")
        
        print(f"\n  📐 Current Filters:")
        print(f"     Min Wallet WR: {self.config.min_wallet_wr:.0%}")
        print(f"     Min Conviction: {self.config.min_conviction}")
        print(f"     Stop Loss: {self.config.stop_loss_pct}%")
        print(f"     Take Profit: {self.config.take_profit_pct}%")
        print(f"     Max Hold: {self.config.max_hold_hours}h")
        
        if self.config.blocked_hours_utc:
            print(f"     Blocked Hours (UTC): {self.config.blocked_hours_utc}")
        if self.config.preferred_hours_utc:
            print(f"     Preferred Hours (UTC): {self.config.preferred_hours_utc}")
        
        print(f"\n  📈 Performance:")
        print(f"     Baseline WR: {self.config.baseline_wr:.1%}")
        print(f"     Current WR: {self.config.current_wr:.1%}")
        print(f"     Target WR: {self.config.target_wr:.0%}")
        
        print("\n" + "=" * 70)


def main():
    """CLI for strategy learner"""
    import sys
    
    learner = StrategyLearner()
    
    if len(sys.argv) < 2:
        learner.print_status()
        print("\nUsage: python strategy_learner.py <command>")
        print("Commands: status, learn, filters")
        return
    
    command = sys.argv[1].lower()
    
    if command == 'status':
        learner.print_status()
    
    elif command == 'learn':
        results = learner.run_learning_iteration()
        print(f"\nLearning complete: {results['status']}")
    
    elif command == 'filters':
        filters = learner.get_current_filters()
        print("\nCurrent Filters:")
        for k, v in filters.items():
            print(f"  {k}: {v}")
    
    else:
        print(f"Unknown command: {command}")


if __name__ == "__main__":
    main()
