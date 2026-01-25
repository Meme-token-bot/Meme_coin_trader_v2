"""
MASTER V2 - ROBUST PAPER TRADING PLATFORM INTEGRATION
======================================================

Full integration with all 6 improvements:
1. Exit Monitoring Reliability (Watchdog)
2. Baseline Comparison
3. A/B Testing Framework
4. Historical Backtesting
5. Signal Quality Metrics
6. Dynamic Exit Parameters

COMPLETE VERSION - Ready to run
"""

import os
import sys
import time
import json
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass, field, asdict
from collections import deque

from flask import Flask, request, jsonify
from dotenv import load_dotenv

load_dotenv()

from core.database_v2 import DatabaseV2
from core.discovery_integration import HistorianV8 as Historian
from core.strategist_v2 import Strategist
from core.discovery_config import config as discovery_config
from infrastructure.helius_webhook_manager import HeliusWebhookManager

# ============================================================================
# IMPORT ROBUST PAPER TRADING PLATFORM
# ============================================================================
from core.paper_trading_platform import (
    RobustPaperTrader,
    SignalQualityAnalyzer,
    SignalQualityMetrics,
    DynamicExitCalculator,
    DynamicExitParams,
    BaselineTracker,
    ABTestingFramework,
    HistoricalDataStore,
    ExitMonitorWatchdog,
    ExitReason
)


@dataclass
class MasterConfig:
    """Master configuration"""
    webhook_host: str = '0.0.0.0'
    webhook_port: int = 5000
    position_check_interval: int = 300
    discovery_enabled: bool = True
    max_token_lookups_per_minute: int = 20
    max_api_calls_per_hour: int = 500
    paper_trading_enabled: bool = True
    paper_starting_balance: float = 10.0
    use_llm: bool = True
    max_open_positions: int = 999  # Unlimited for learning mode
    max_position_size_sol: float = 1.0
    
    # Robust platform features
    enable_watchdog: bool = True
    enable_baseline_tracking: bool = True
    enable_historical_storage: bool = True
    enable_ab_testing: bool = True
    
    @property
    def discovery_interval_hours(self) -> int:
        return discovery_config.discovery_interval_hours
    
    @property
    def discovery_api_budget(self) -> int:
        return discovery_config.max_api_calls_per_discovery


CONFIG = MasterConfig()


class RateLimiter:
    def __init__(self, max_per_minute: int = 20, max_per_hour: int = 500):
        self.max_per_minute = max_per_minute
        self.max_per_hour = max_per_hour
        self.minute_calls = deque()
        self.hour_calls = deque()
    
    def can_call(self) -> bool:
        now = datetime.now()
        minute_ago = now - timedelta(minutes=1)
        hour_ago = now - timedelta(hours=1)
        
        while self.minute_calls and self.minute_calls[0] < minute_ago:
            self.minute_calls.popleft()
        while self.hour_calls and self.hour_calls[0] < hour_ago:
            self.hour_calls.popleft()
        
        if len(self.minute_calls) >= self.max_per_minute:
            return False
        if len(self.hour_calls) >= self.max_per_hour:
            return False
        return True
    
    def record_call(self):
        now = datetime.now()
        self.minute_calls.append(now)
        self.hour_calls.append(now)
    
    def get_stats(self) -> Dict:
        return {
            'calls_last_minute': len(self.minute_calls),
            'calls_last_hour': len(self.hour_calls),
            'limit_per_minute': self.max_per_minute,
            'limit_per_hour': self.max_per_hour
        }


@dataclass
class DiagnosticsTracker:
    start_time: datetime = field(default_factory=datetime.now)
    last_webhook_received: Optional[datetime] = None
    webhooks_received: int = 0
    webhooks_processed: int = 0
    webhooks_skipped: int = 0
    api_calls_made: int = 0
    api_errors: int = 0
    positions_opened: int = 0
    positions_closed: int = 0
    llm_calls: int = 0
    discoveries_run: int = 0
    wallets_discovered: int = 0
    learning_iterations: int = 0
    # Debug counters
    buy_signals_detected: int = 0
    sell_signals_detected: int = 0
    untracked_wallet_skips: int = 0
    duplicate_sig_skips: int = 0
    non_swap_skips: int = 0
    parse_failures: int = 0
    token_info_failures: int = 0
    learning_filter_passes: int = 0
    learning_filter_fails: int = 0
    # New: Quality metrics tracking
    cluster_signals_detected: int = 0
    high_conviction_signals: int = 0
    baseline_signals_recorded: int = 0
    recent_events: deque = field(default_factory=lambda: deque(maxlen=50))
    
    def log_event(self, event_type: str, details: str = ""):
        self.recent_events.append({
            'time': datetime.now().isoformat(),
            'type': event_type,
            'details': details
        })
    
    def to_dict(self) -> Dict:
        uptime = datetime.now() - self.start_time
        minutes_since_webhook = None
        if self.last_webhook_received:
            minutes_since_webhook = (datetime.now() - self.last_webhook_received).total_seconds() / 60
        
        return {
            'uptime_hours': uptime.total_seconds() / 3600,
            'last_webhook_received': self.last_webhook_received.isoformat() if self.last_webhook_received else None,
            'minutes_since_last_webhook': minutes_since_webhook,
            'webhooks': {
                'received': self.webhooks_received, 
                'processed': self.webhooks_processed, 
                'skipped': self.webhooks_skipped,
                'skip_reasons': {
                    'untracked_wallet': self.untracked_wallet_skips,
                    'duplicate_sig': self.duplicate_sig_skips,
                    'non_swap': self.non_swap_skips,
                    'parse_failures': self.parse_failures,
                }
            },
            'signals': {
                'buy_detected': self.buy_signals_detected,
                'sell_detected': self.sell_signals_detected,
                'cluster_signals': self.cluster_signals_detected,
                'high_conviction': self.high_conviction_signals,
            },
            'api': {'calls': self.api_calls_made, 'errors': self.api_errors, 'token_info_failures': self.token_info_failures},
            'positions': {'opened': self.positions_opened, 'closed': self.positions_closed},
            'discovery': {'runs': self.discoveries_run, 'wallets_found': self.wallets_discovered},
            'learning_iterations': self.learning_iterations,
            'baseline_signals_recorded': self.baseline_signals_recorded,
            'llm_calls': self.llm_calls,
            'recent_events': list(self.recent_events)[-10:]
        }


class Notifier:
    """Telegram notifier with enhanced alerts"""
    
    def __init__(self, token: str = None, chat_id: str = None):
        self.token = token or os.getenv('TELEGRAM_BOT_TOKEN')
        self.chat_id = chat_id or os.getenv('TELEGRAM_CHAT_ID')
        self.enabled = bool(self.token and self.chat_id)
        self._last_status_sent = None
    
    def send(self, message: str):
        if not self.enabled:
            return
        try:
            import requests
            url = f"https://api.telegram.org/bot{self.token}/sendMessage"
            requests.post(url, json={'chat_id': self.chat_id, 'text': message, 'parse_mode': 'HTML'}, timeout=10)
        except Exception as e:
            print(f"  ⚠️ Telegram error: {e}")
    
    def send_entry_alert(self, signal: Dict, decision: Dict, quality: SignalQualityMetrics = None):
        """Send entry notification with quality metrics"""
        current_hour = datetime.utcnow().hour
        
        cluster_str = "🔥 CLUSTER" if (quality and quality.is_cluster_signal) else ""
        quality_score = quality.calculate_composite_score() if quality else 50
        
        msg = f"""🎯 <b>ENTRY SIGNAL</b> {cluster_str}

Token: ${signal.get('token_symbol', 'UNKNOWN')}
Conviction: {decision.get('conviction_score', 0):.0f}/100
Quality: {quality_score:.0f}/100
Wallets: {decision.get('wallet_count', 1)}
Regime: {decision.get('regime', 'UNKNOWN')}
Position: {decision.get('position_size_sol', 0):.3f} SOL
Stop: {decision.get('stop_loss_pct', -15):.0f}%
Target: {decision.get('take_profit_pct', 30):.0f}%
Hour (UTC): {current_hour:02d}:00"""
        self.send(msg)
    
    def send_exit_alert(self, position: Dict, reason: str, pnl_pct: float, result: Dict = None):
        """Send exit notification with hold time"""
        emoji = "🟢" if pnl_pct > 0 else "🔴"
        
        hold_mins = self._get_hold_time(position, result)
        
        if hold_mins >= 60:
            hold_str = f"{hold_mins/60:.1f}h ({hold_mins:.0f}m)"
        else:
            hold_str = f"{hold_mins:.0f} min"
        
        pnl_sol = result.get('pnl_sol', 0) if result else 0
        
        msg = f"""{emoji} <b>EXIT</b>

Token: ${position.get('token_symbol', 'UNKNOWN')}
Reason: {reason}
P&L: {pnl_pct:+.1f}% ({pnl_sol:+.4f} SOL)
Hold: {hold_str}"""
        self.send(msg)
    
    def _get_hold_time(self, position: Dict, result: Dict = None) -> float:
        """Extract hold time from various sources"""
        if result:
            if 'hold_minutes' in result:
                return result['hold_minutes']
            if 'hold_duration_minutes' in result:
                return result['hold_duration_minutes']
        
        if 'hold_duration_minutes' in position:
            return position['hold_duration_minutes']
        
        entry_time = position.get('entry_time')
        if entry_time:
            if isinstance(entry_time, str):
                entry_time = datetime.fromisoformat(entry_time.replace('Z', ''))
            return (datetime.utcnow() - entry_time).total_seconds() / 60
        
        return 0
    
    def send_hourly_status(self, stats: Dict, diag: Dict, learning_stats: Dict = None):
        """Send hourly status with platform metrics"""
        uptime_hours = diag.get('uptime_hours', 0)
        
        msg = f"""📊 <b>Hourly Status</b>

Uptime: {uptime_hours:.1f}h
Webhooks: {diag['webhooks']['received']} received
Last webhook: {diag.get('minutes_since_last_webhook', 0):.0f}m ago
Paper: {stats.get('balance', 0):.2f} SOL ({stats.get('return_pct', 0):+.1f}%)
Open positions: {stats.get('open_positions', 0)}
Win rate: {stats.get('win_rate', 0):.0%}

📈 <b>Webhook Breakdown:</b>
• Processed: {diag['webhooks']['processed']}
• BUY signals: {diag['signals']['buy_detected']}
• SELL signals: {diag['signals']['sell_detected']}
• Positions opened: {diag['positions']['opened']}
• Skip: {diag['webhooks']['skip_reasons']['untracked_wallet']} untracked, {diag['webhooks']['skip_reasons']['non_swap']} non-swap

🔥 <b>Quality Metrics:</b>
• Cluster signals: {diag['signals'].get('cluster_signals', 0)}
• High conviction: {diag['signals'].get('high_conviction', 0)}
• Baseline recorded: {diag.get('baseline_signals_recorded', 0)}"""

        if learning_stats:
            msg += f"""

🧪 <b>Learning:</b>
Phase: {learning_stats.get('phase', 'exploration')}
Iteration: {learning_stats.get('iteration', 0)}
Blocked hours: {learning_stats.get('blocked_hours', 'None') or 'None'}"""
        
        self.send(msg)
    
    def send_cluster_alert(self, token_symbol: str, wallet_count: int, wallets: List[str]):
        """Send alert when cluster signal detected"""
        wallet_preview = ", ".join(w[:8] + "..." for w in wallets[:3])
        msg = f"""🔥 <b>CLUSTER SIGNAL DETECTED</b>

Token: ${token_symbol}
Wallets buying: {wallet_count}
Wallets: {wallet_preview}

Multiple smart wallets buying simultaneously!"""
        self.send(msg)
    
    def send_baseline_report(self, report: Dict):
        """Send baseline comparison report"""
        msg = "📊 <b>BASELINE COMPARISON (7 days)</b>\n\n"
        
        for baseline, data in report.items():
            if data['entered'] > 0:
                msg += f"<b>{baseline.upper()}</b>\n"
                msg += f"  Entered: {data['entered']} | WR: {data['win_rate']:.0%} | PnL: {data['total_pnl_pct']:+.1f}%\n\n"
        
        # Find best
        best = max(report.items(), key=lambda x: x[1].get('total_pnl_pct', 0))
        msg += f"🏆 Best: {best[0].upper()} with {best[1]['total_pnl_pct']:+.1f}% PnL"
        
        self.send(msg)


# ============================================================================
# ROBUST PAPER TRADING ENGINE WRAPPER
# ============================================================================
class RobustPaperTradingEngine:
    """
    Full integration wrapper for RobustPaperTrader with all features.
    
    Features:
    1. ✅ Exit Monitoring Watchdog
    2. ✅ Baseline Comparison
    3. ✅ A/B Testing
    4. ✅ Historical Backtesting
    5. ✅ Signal Quality Metrics
    6. ✅ Dynamic Exit Parameters
    """
    
    def __init__(self, db, starting_balance: float = 10.0, max_positions: int = None):
        self.db = db
        
        # Initialize the robust trader with all features
        self._trader = RobustPaperTrader(
            db_path="robust_paper_trades.db",
            starting_balance=starting_balance,
            enable_watchdog=CONFIG.enable_watchdog,
            enable_baseline_tracking=CONFIG.enable_baseline_tracking,
            enable_historical_storage=CONFIG.enable_historical_storage
        )
        
        # Store reference to components for direct access
        self.quality_analyzer = self._trader.quality_analyzer
        self.exit_calculator = self._trader.exit_calculator
        self.baseline_tracker = self._trader.baseline_tracker
        self.historical_store = self._trader.historical_store
        self.ab_testing = self._trader.ab_testing
        self.watchdog = self._trader.watchdog
        
        # Track top wallets for baseline comparison
        self._top_wallets: List[str] = []
        self._last_top_wallets_update = datetime.utcnow() - timedelta(hours=1)
        
        # Learning timing
        self._last_learning = datetime.utcnow() - timedelta(hours=5)
        
        print(f"🚀 ROBUST PAPER TRADING ENGINE initialized")
        print(f"   Balance: {self._trader.balance:.4f} SOL")
        print(f"   Positions: {self._trader.open_position_count}")
        print(f"   Watchdog: {'ENABLED' if CONFIG.enable_watchdog else 'DISABLED'}")
        print(f"   Baseline Tracking: {'ENABLED' if CONFIG.enable_baseline_tracking else 'DISABLED'}")
        print(f"   Historical Storage: {'ENABLED' if CONFIG.enable_historical_storage else 'DISABLED'}")
    
    @property
    def balance(self) -> float:
        return self._trader.balance
    
    @property
    def available_balance(self) -> float:
        return self._trader.balance - self._trader.reserved_balance
    
    def update_top_wallets(self, db):
        """Update list of top wallets for baseline comparison"""
        if (datetime.utcnow() - self._last_top_wallets_update).total_seconds() < 3600:
            return  # Only update hourly
        
        try:
            # Get top 20 wallets by win rate
            wallets = db.get_wallets(limit=100)
            sorted_wallets = sorted(wallets, key=lambda w: w.get('win_rate', 0), reverse=True)
            self._top_wallets = [w['address'] for w in sorted_wallets[:20] if w.get('win_rate', 0) >= 0.5]
            self._last_top_wallets_update = datetime.utcnow()
            print(f"  📊 Updated top wallets: {len(self._top_wallets)} wallets with WR >= 50%")
        except Exception as e:
            print(f"  ⚠️ Failed to update top wallets: {e}")
    
    def process_signal(self, signal: Dict, wallet_data: Dict = None) -> Dict:
        """
        Process a trading signal through the full pipeline.
        
        This is the main entry point that:
        1. Analyzes signal quality (clusters, timing, sizing)
        2. Calculates dynamic exit parameters
        3. Records for baseline comparison
        4. Stores for historical backtesting
        5. Opens position if appropriate
        
        Returns dict with decision info and position_id if opened
        """
        return self._trader.process_signal(
            signal=signal,
            wallet_data=wallet_data,
            top_wallets=self._top_wallets
        )
    
    def open_position(self, signal: Dict, decision: Dict, price: float) -> Optional[int]:
        """
        Legacy interface for opening positions.
        Converts to process_signal() internally.
        """
        signal['price'] = price
        
        # Build wallet_data from signal
        wallet_data = {
            'address': signal.get('wallet', signal.get('wallet_address', '')),
            'win_rate': signal.get('wallet_win_rate', 0.5),
            'cluster': signal.get('wallet_cluster', 'UNKNOWN'),
        }
        
        result = self.process_signal(signal, wallet_data)
        return result.get('position_id')
    
    def close_position(self, position_id: int, exit_reason: str, exit_price: float) -> Optional[Dict]:
        """Close a paper position"""
        reason_map = {
            'STOP_LOSS': ExitReason.STOP_LOSS,
            'TAKE_PROFIT': ExitReason.TAKE_PROFIT,
            'TRAILING_STOP': ExitReason.TRAILING_STOP,
            'TIME_STOP': ExitReason.TIME_STOP,
            'SMART_EXIT': ExitReason.SMART_EXIT,
            'MANUAL': ExitReason.MANUAL,
            'WATCHDOG': ExitReason.WATCHDOG,
        }
        
        clean_reason = exit_reason.split(':')[0] if ':' in exit_reason else exit_reason
        reason_enum = reason_map.get(clean_reason, ExitReason.MANUAL)
        
        result = self._trader.close_position(position_id, exit_price, reason_enum)
        
        if result:
            return {
                'pnl_pct': result['pnl_pct'],
                'pnl_sol': result['pnl_sol'],
                'hold_minutes': result['hold_minutes'],
                'hold_duration_minutes': result['hold_minutes'],
                'exit_reason': result['exit_reason']
            }
        return None
    
    def get_open_positions(self) -> List[Dict]:
        return self._trader.get_open_positions()
    
    def get_stats(self) -> Dict:
        """Get comprehensive trading statistics"""
        summary = self._trader.get_performance_summary()
        return {
            'balance': summary.get('balance', 0),
            'starting_balance': summary.get('starting_balance', 0),
            'total_pnl': summary.get('total_pnl_sol', 0),
            'return_pct': summary.get('return_pct', 0),
            'open_positions': summary.get('open_positions', 0),
            'total_trades': summary.get('total_trades', 0),
            'win_rate': summary.get('win_rate', 0),
            'winning_trades': summary.get('winning_trades', 0),
            # Learning-related (for backwards compatibility)
            'phase': 'learning',
            'iteration': 0,
            'blocked_hours': [],
            'preferred_hours': [],
        }
    
    def get_baseline_comparison(self, days: int = 7) -> Dict:
        """Get baseline comparison report"""
        if self.baseline_tracker:
            return self.baseline_tracker.get_comparison_report(days)
        return {}
    
    def print_baseline_comparison(self, days: int = 7):
        """Print baseline comparison report"""
        if self.baseline_tracker:
            self.baseline_tracker.print_comparison(days)
    
    def create_ab_test(self, name: str, variant_a: Dict, variant_b: Dict,
                      min_samples: int = 30) -> str:
        """Create a new A/B test"""
        if self.ab_testing:
            return self.ab_testing.create_test(name, variant_a, variant_b, min_samples=min_samples)
        return ""
    
    def get_ab_test_status(self, test_id: str) -> Dict:
        """Get A/B test status"""
        if self.ab_testing:
            return self.ab_testing.evaluate_test(test_id)
        return {}
    
    def get_active_ab_tests(self) -> List[str]:
        """Get list of active A/B test IDs"""
        if self.ab_testing:
            return list(self.ab_testing._active_tests.keys())
        return []
    
    def run_backtest(self, strategy_func, days: int = 7) -> Dict:
        """Run backtest on historical data"""
        if self.historical_store:
            from datetime import datetime, timedelta
            return self.historical_store.backtest(
                strategy_func,
                start_date=datetime.utcnow() - timedelta(days=days)
            )
        return {}
    
    def should_run_learning(self) -> bool:
        """Check if it's time to run learning analysis"""
        hours_since = (datetime.utcnow() - self._last_learning).total_seconds() / 3600
        return hours_since >= 6.0
    
    def run_learning(self, force: bool = False, notifier=None) -> Dict:
        """Run learning analysis (baseline comparison + insights)"""
        if not force and not self.should_run_learning():
            return {'status': 'skipped', 'reason': 'not time yet'}
        
        self._last_learning = datetime.utcnow()
        
        results = {
            'status': 'completed',
            'timestamp': datetime.utcnow().isoformat(),
        }
        
        # Get baseline comparison
        if self.baseline_tracker:
            results['baseline_comparison'] = self.get_baseline_comparison()
            
            # Send report via Telegram
            if notifier and hasattr(notifier, 'send_baseline_report'):
                try:
                    notifier.send_baseline_report(results['baseline_comparison'])
                except Exception as e:
                    print(f"  ⚠️ Baseline notification error: {e}")
        
        # Check A/B tests
        if self.ab_testing:
            results['ab_tests'] = {}
            for test_id in self.get_active_ab_tests():
                results['ab_tests'][test_id] = self.get_ab_test_status(test_id)
        
        return results
    
    def get_diurnal_report(self) -> Dict:
        """Get time-of-day performance report (placeholder for compatibility)"""
        return {}
    
    def get_strategy_feedback(self) -> Dict:
        """Get detailed feedback for strategy improvement"""
        return {
            'summary': self.get_stats(),
            'baseline': self.get_baseline_comparison() if self.baseline_tracker else {},
            'ab_tests': {tid: self.get_ab_test_status(tid) for tid in self.get_active_ab_tests()}
        }
    
    def print_status(self):
        self._trader.print_status()
    
    def stop(self):
        self._trader.stop()


# ============================================================================
# TRADING SYSTEM
# ============================================================================
class TradingSystem:
    def __init__(self):
        print("\n" + "="*70)
        print("🚀 TRADING SYSTEM V2 - ROBUST PAPER TRADING PLATFORM")
        print("="*70)
        
        self.helius_key = os.getenv('HELIUS_KEY')
        self.anthropic_key = os.getenv('ANTHROPIC_API_KEY')
        self.webhook_id = os.getenv('HELIUS_WEBHOOK_ID')
        
        if not self.helius_key:
            raise ValueError("HELIUS_KEY not set!")
        
        print("\n📦 Loading Components...")
        
        self.db = DatabaseV2()
        
        self.strategist = Strategist(self.db, self.anthropic_key)
        
        self.notifier = Notifier()
        print(f"  {'✅' if self.notifier.enabled else '⚠️'} Telegram: {'enabled' if self.notifier.enabled else 'disabled'}")
        
        # Initialize Robust Paper Trading Engine
        if CONFIG.paper_trading_enabled:
            self.paper_engine = RobustPaperTradingEngine(
                self.db, 
                starting_balance=CONFIG.paper_starting_balance,
                max_positions=CONFIG.max_open_positions
            )
            print(f"  ✅ Robust Paper Trading (Balance: {self.paper_engine.balance:.2f} SOL)")
        else:
            self.paper_engine = None
        
        self.rate_limiter = RateLimiter(CONFIG.max_token_lookups_per_minute, CONFIG.max_api_calls_per_hour)
        print(f"  ✅ Rate Limiter ({CONFIG.max_token_lookups_per_minute}/min, {CONFIG.max_api_calls_per_hour}/hr)")
        
        self.diagnostics = DiagnosticsTracker()
        print("  ✅ Diagnostics Tracker")
        
        webhook_url = os.getenv('WEBHOOK_URL', f'http://0.0.0.0:{CONFIG.webhook_port}/webhook/helius')

        try:
            from infrastructure.multi_webhook_manager import MultiWebhookManager
            self.multi_webhook_manager = MultiWebhookManager(self.helius_key, webhook_url, self.db)
            print(f"  ✅ Multi-Webhook Manager (unlimited scaling)")
        except Exception as e:
            self.multi_webhook_manager = None
            print(f"  ⚠️  Multi-Webhook disabled: {e}")

        if CONFIG.discovery_enabled:
            discovery_key = os.getenv('HELIUS_DISCOVERY_KEY')
            self.historian = Historian(
                self.db, 
                self.helius_key, 
                discovery_key,
                multi_webhook_manager=self.multi_webhook_manager
            )
            print(f"  ✅ Discovery (every {CONFIG.discovery_interval_hours}h, budget: {CONFIG.discovery_api_budget:,})")
        else:
            self.historian = None
            print("  ⚠️ Discovery disabled")
        
        self.start_time = datetime.now()
        
        print("\n" + "="*70)
        print("✅ ROBUST PAPER TRADING PLATFORM READY")
        print("="*70)
        self._print_status()
    
    def _print_status(self):
        status = self.strategist.get_status()
        llm_cost = self.strategist.get_llm_cost_today()
        
        print(f"\n📊 Status:")
        print(f"  Wallets tracked: {self.db.get_wallet_count()}")
        print(f"  Regime: {status['regime']} ({status['confidence']:.0%})")
        print(f"  Champion: {status['champion']}")
        print(f"  LLM: {'✅' if status['llm_enabled'] else '❌'} | Today: {llm_cost['calls']} calls (${llm_cost['cost_usd']:.4f})")
        
        if self.paper_engine:
            stats = self.paper_engine.get_stats()
            print(f"  Paper: {stats['balance']:.2f} SOL ({stats['return_pct']:+.1f}%) | {stats['open_positions']} open")
            print(f"  Features: Watchdog={'✅' if CONFIG.enable_watchdog else '❌'} | "
                  f"Baseline={'✅' if CONFIG.enable_baseline_tracking else '❌'} | "
                  f"Historical={'✅' if CONFIG.enable_historical_storage else '❌'}")
        
        rate_stats = self.rate_limiter.get_stats()
        print(f"  Rate limit: {rate_stats['calls_last_hour']}/{rate_stats['limit_per_hour']} calls/hr")
        
        d = self.diagnostics
        print(f"\n📈 Webhook Breakdown:")
        print(f"  Total received: {d.webhooks_received}")
        print(f"  Processed: {d.webhooks_processed} | Skipped: {d.webhooks_skipped}")
        print(f"  Skip reasons:")
        print(f"    - Untracked wallet: {d.untracked_wallet_skips}")
        print(f"    - Duplicate sig: {d.duplicate_sig_skips}")
        print(f"    - Non-swap: {d.non_swap_skips}")
        print(f"    - Parse failures: {d.parse_failures}")
        print(f"  Signals: {d.buy_signals_detected} BUY | {d.sell_signals_detected} SELL")
        print(f"  Token info failures: {d.token_info_failures}")
        print(f"  Positions opened: {d.positions_opened}")
        print(f"  Cluster signals: {d.cluster_signals_detected}")
    
    def process_webhook(self, data: Dict) -> Dict:
        """Process incoming Helius webhook"""
        self.diagnostics.webhooks_received += 1
        self.diagnostics.last_webhook_received = datetime.now()
        
        result = {'processed': False, 'reason': ''}
        
        try:
            txs = data if isinstance(data, list) else [data]
            
            for tx in txs:
                try:
                    self._process_transaction(tx)
                    self.diagnostics.webhooks_processed += 1
                except Exception as e:
                    self.diagnostics.parse_failures += 1
                    self.diagnostics.log_event('PARSE_ERROR', str(e)[:100])
            
            result['processed'] = True
            
        except Exception as e:
            result['reason'] = str(e)
            self.diagnostics.log_event('WEBHOOK_ERROR', str(e)[:100])
        
        return result
    
    def _process_transaction(self, tx: Dict):
        """Process a single transaction"""
        fee_payer = tx.get('feePayer', '')
        signature = tx.get('signature', '')
        
        if not fee_payer or not signature:
            return
        
        # Check if tracked wallet
        wallet_data = self.db.get_wallet(fee_payer)
        if not wallet_data:
            self.diagnostics.untracked_wallet_skips += 1
            return
        
        # Check for duplicate
        if self.db.is_signature_processed(signature):
            self.diagnostics.duplicate_sig_skips += 1
            return
        
        # Parse as swap
        trade = self._parse_swap(tx)
        if not trade:
            self.diagnostics.non_swap_skips += 1
            return
        
        self.db.mark_signature_processed(signature)
        
        token_addr = trade.get('token_out') if trade['type'] == 'BUY' else trade.get('token_in')
        if not token_addr:
            return
        
        self.diagnostics.log_event('TRADE_DETECTED', f"{trade['type']} from {fee_payer[:8]}...")
        
        if trade['type'] == 'BUY':
            self.diagnostics.buy_signals_detected += 1
            return self._process_buy(trade, wallet_data, token_addr, signature)
        elif trade['type'] == 'SELL':
            self.diagnostics.sell_signals_detected += 1
            return self._process_sell(trade, wallet_data, token_addr, fee_payer)
    
    def _parse_swap(self, tx: Dict) -> Optional[Dict]:
        """Parse transaction as a swap"""
        instructions = tx.get('instructions', [])
        token_transfers = tx.get('tokenTransfers', [])
        
        if not token_transfers:
            return None
        
        is_swap = any('swap' in str(instr).lower() for instr in instructions)
        
        if not is_swap and len(token_transfers) < 2:
            return None
        
        sol_mint = "So11111111111111111111111111111111111111112"
        fee_payer = tx.get('feePayer', '')
        
        sol_in = None
        sol_out = None
        token_in = None
        token_out = None
        
        for transfer in token_transfers:
            mint = transfer.get('mint', '')
            from_addr = transfer.get('fromUserAccount', '')
            to_addr = transfer.get('toUserAccount', '')
            amount = float(transfer.get('tokenAmount', 0))
            
            if amount <= 0:
                continue
            
            if mint == sol_mint:
                if from_addr == fee_payer:
                    sol_out = amount
                elif to_addr == fee_payer:
                    sol_in = amount
            else:
                if from_addr == fee_payer:
                    token_in = mint
                elif to_addr == fee_payer:
                    token_out = mint
        
        if sol_out and token_out:
            return {'type': 'BUY', 'token_out': token_out, 'sol_spent': sol_out}
        elif sol_in and token_in:
            return {'type': 'SELL', 'token_in': token_in, 'sol_received': sol_in}
        
        return None
    
    def _process_buy(self, trade: Dict, wallet_data: Dict, token_addr: str, signature: str) -> Dict:
        """
        Process a BUY signal through the full robust pipeline.
        
        Uses:
        - Signal Quality Analysis (cluster detection, timing, sizing)
        - Dynamic Exit Parameters
        - Baseline Comparison Recording
        - Historical Storage
        """
        result = {'processed': True, 'action': 'BUY_SIGNAL', 'reason': ''}
        
        if self.db.is_position_tracked(wallet_data['address'], token_addr):
            result['reason'] = 'Position already tracked'
            return result
        
        if not self.rate_limiter.can_call():
            self.diagnostics.log_event('RATE_LIMITED', 'Token lookup skipped')
            result['reason'] = 'Rate limited'
            return result
        
        self.rate_limiter.record_call()
        self.diagnostics.api_calls_made += 1
        
        token_info = self.historian.scanner.get_token_info(token_addr)
        
        if not token_info:
            self.diagnostics.api_errors += 1
            self.diagnostics.token_info_failures += 1
            result['reason'] = 'Could not get token info'
            return result
        
        price = token_info.get('price_usd', 0)
        if price <= 0:
            result['reason'] = 'Invalid price'
            return result
        
        # Normalize wallet win rate
        wallet_wr = wallet_data.get('win_rate', 0.5)
        if wallet_wr > 1:
            wallet_wr = wallet_wr / 100.0
        
        # Handle liquidity field
        liquidity = token_info.get('liquidity', 0)
        if isinstance(liquidity, dict):
            liquidity = liquidity.get('usd', 0)
        liquidity = float(liquidity or 0)
        
        # Build signal data
        signal_data = {
            'token_address': token_addr,
            'token_symbol': token_info.get('symbol', 'UNKNOWN'),
            'price': price,
            'liquidity': liquidity,
            'volume_24h': token_info.get('volume_24h', 0),
            'market_cap': token_info.get('market_cap', 0),
            'token_age_hours': token_info.get('age_hours', 0),
            'holder_count': token_info.get('holder_count', 0),
            'wallet': wallet_data['address'],
            'wallet_address': wallet_data['address'],
            'wallet_win_rate': wallet_wr,
            'wallet_cluster': wallet_data.get('cluster', 'UNKNOWN')
        }
        
        # Build wallet data for quality analysis
        wallet_info = {
            'address': wallet_data['address'],
            'win_rate': wallet_wr,
            'cluster': wallet_data.get('cluster', 'UNKNOWN'),
            'roi_7d': wallet_data.get('roi_7d', 0),
            'trades_count': wallet_data.get('total_trades', 0),
            'avg_position_size_sol': wallet_data.get('avg_position_size', 0),
        }
        
        # Update top wallets periodically
        if self.paper_engine:
            self.paper_engine.update_top_wallets(self.db)
        
        # ====================================================================
        # PROCESS THROUGH ROBUST PIPELINE
        # ====================================================================
        if self.paper_engine:
            # Check minimum wallet WR (very permissive for learning)
            if wallet_wr < 0.30:
                print(f"  ⏭️ SKIP: Low WR ({wallet_wr:.0%} < 30%)")
                result['reason'] = f"Low wallet WR: {wallet_wr:.0%}"
                return result
            
            # Process through full pipeline
            process_result = self.paper_engine.process_signal(signal_data, wallet_info)
            
            # Track quality metrics in diagnostics
            quality_metrics = process_result.get('quality_metrics', {})
            if quality_metrics.get('is_cluster_signal'):
                self.diagnostics.cluster_signals_detected += 1
                # Send cluster alert
                if self.notifier:
                    self.notifier.send_cluster_alert(
                        signal_data['token_symbol'],
                        quality_metrics.get('concurrent_wallet_count', 1),
                        quality_metrics.get('cluster_wallets', [])
                    )
            
            conviction = process_result.get('conviction', 50)
            if conviction >= 70:
                self.diagnostics.high_conviction_signals += 1
            
            if self.paper_engine.baseline_tracker:
                self.diagnostics.baseline_signals_recorded += 1
            
            # Log the processing
            print(f"\n  🎯 SIGNAL: ${signal_data['token_symbol']}")
            print(f"     Price: ${price:.8f} | Liquidity: ${liquidity:,.0f}")
            print(f"     Wallet WR: {wallet_wr:.0%} | Conviction: {conviction:.0f}")
            print(f"     Quality: {process_result.get('quality_score', 50):.0f} | "
                  f"Cluster: {'🔥 YES' if quality_metrics.get('is_cluster_signal') else 'No'}")
            
            # Check if position was opened
            if process_result.get('position_id'):
                pos_id = process_result['position_id']
                self.diagnostics.positions_opened += 1
                
                # Get exit params for display
                exit_params = process_result.get('exit_params', {})
                print(f"     ✅ POSITION OPENED (ID: {pos_id})")
                print(f"     Stop: {exit_params.get('stop_loss_pct', -15):.0f}% | "
                      f"TP: {exit_params.get('take_profit_pct', 30):.0f}%")
                
                # Send entry notification
                if self.notifier:
                    decision = {
                        'conviction_score': conviction,
                        'wallet_count': quality_metrics.get('concurrent_wallet_count', 1),
                        'regime': self.strategist.regime_detector.current_regime,
                        'position_size_sol': 0.3,
                        'stop_loss_pct': exit_params.get('stop_loss_pct', -15),
                        'take_profit_pct': exit_params.get('take_profit_pct', 30),
                    }
                    
                    # Create quality metrics object for notification
                    quality_obj = SignalQualityMetrics(
                        is_cluster_signal=quality_metrics.get('is_cluster_signal', False),
                        concurrent_wallet_count=quality_metrics.get('concurrent_wallet_count', 1)
                    )
                    
                    self.notifier.send_entry_alert(signal_data, decision, quality_obj)
                
                result['action'] = 'POSITION_OPENED'
                result['position_id'] = pos_id
                result['quality'] = process_result.get('quality_score', 50)
                result['conviction'] = conviction
                
            else:
                print(f"     ⚠️ Position not opened")
                result['reason'] = 'Position open criteria not met'
        
        return result
    
    def _process_sell(self, trade: Dict, wallet_data: Dict, token_addr: str, wallet_addr: str) -> Dict:
        """Process a sell signal"""
        result = {'processed': True, 'action': 'SELL_SIGNAL'}
        # Sells are tracked but don't trigger paper trades
        return result
    
    def run_discovery(self) -> Dict:
        """Run discovery cycle"""
        if not self.historian:
            return {'status': 'disabled'}
        
        try:
            self.diagnostics.discoveries_run += 1
            result = self.historian.run_discovery()
            
            new_wallets = result.get('wallets_added', 0)
            self.diagnostics.wallets_discovered += new_wallets
            
            return result
        except Exception as e:
            return {'status': 'error', 'error': str(e)}


# ============================================================================
# FLASK APP
# ============================================================================
app = Flask(__name__)
trading_system: TradingSystem = None


@app.route('/webhook/helius', methods=['POST'])
def helius_webhook():
    global trading_system
    if not trading_system:
        return jsonify({'error': 'System not initialized'}), 503
    
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No data'}), 400
    
    result = trading_system.process_webhook(data)
    return jsonify(result)


@app.route('/status', methods=['GET'])
def status():
    global trading_system
    if not trading_system:
        return jsonify({'error': 'System not initialized'}), 503
    
    stats = {}
    if trading_system.paper_engine:
        stats = trading_system.paper_engine.get_stats()
    
    regime = trading_system.strategist.get_status()
    
    return jsonify({
        'status': 'running',
        'uptime_hours': (datetime.now() - trading_system.start_time).total_seconds() / 3600,
        'wallets_tracked': trading_system.db.get_wallet_count(),
        'regime': regime,
        'paper_trading': stats,
        'features': {
            'watchdog': CONFIG.enable_watchdog,
            'baseline_tracking': CONFIG.enable_baseline_tracking,
            'historical_storage': CONFIG.enable_historical_storage,
        }
    })


@app.route('/diagnostics', methods=['GET'])
def diagnostics():
    global trading_system
    if not trading_system:
        return jsonify({'error': 'System not initialized'}), 503
    
    return jsonify(trading_system.diagnostics.to_dict())


@app.route('/positions', methods=['GET'])
def positions():
    global trading_system
    if not trading_system or not trading_system.paper_engine:
        return jsonify({'error': 'Paper trading not enabled'}), 503
    
    positions = trading_system.paper_engine.get_open_positions()
    return jsonify({
        'count': len(positions),
        'positions': positions
    })


@app.route('/new_wallets', methods=['GET'])
def new_wallets():
    global trading_system
    if not trading_system:
        return jsonify({'error': 'System not initialized'}), 503
    
    hours = request.args.get('hours', 24, type=int)
    wallets = trading_system.db.get_recent_wallets(hours=hours)
    return jsonify({'count': len(wallets), 'wallets': wallets})


@app.route('/discovery/run', methods=['POST'])
def run_discovery():
    global trading_system
    if not trading_system:
        return jsonify({'error': 'System not initialized'}), 503
    
    result = trading_system.run_discovery()
    return jsonify(result)


@app.route('/learning/run', methods=['POST'])
def run_learning():
    global trading_system
    if not trading_system or not trading_system.paper_engine:
        return jsonify({'error': 'Paper trading not enabled'}), 503
    
    force = request.args.get('force', 'false').lower() == 'true'
    result = trading_system.paper_engine.run_learning(force=force, notifier=trading_system.notifier)
    return jsonify(result)


@app.route('/learning/insights', methods=['GET'])
def learning_insights():
    global trading_system
    if not trading_system or not trading_system.paper_engine:
        return jsonify({'error': 'Paper trading not enabled'}), 503
    
    return jsonify(trading_system.paper_engine.get_strategy_feedback())


# ============================================================================
# NEW ENDPOINTS FOR ROBUST PLATFORM
# ============================================================================

@app.route('/baseline', methods=['GET'])
def baseline_comparison():
    """Get baseline comparison report"""
    global trading_system
    if not trading_system or not trading_system.paper_engine:
        return jsonify({'error': 'Paper trading not enabled'}), 503
    
    days = request.args.get('days', 7, type=int)
    report = trading_system.paper_engine.get_baseline_comparison(days)
    return jsonify(report)


@app.route('/ab_test', methods=['GET', 'POST'])
def ab_testing():
    """A/B testing endpoint"""
    global trading_system
    if not trading_system or not trading_system.paper_engine:
        return jsonify({'error': 'Paper trading not enabled'}), 503
    
    if request.method == 'POST':
        # Create new A/B test
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        test_id = trading_system.paper_engine.create_ab_test(
            name=data.get('name', 'Unnamed Test'),
            variant_a=data.get('variant_a', {}),
            variant_b=data.get('variant_b', {}),
            min_samples=data.get('min_samples', 30)
        )
        return jsonify({'test_id': test_id, 'status': 'created'})
    
    else:
        # Get A/B test status
        test_id = request.args.get('test_id')
        if test_id:
            status = trading_system.paper_engine.get_ab_test_status(test_id)
            return jsonify(status)
        else:
            # Return all active tests
            tests = trading_system.paper_engine.get_active_ab_tests()
            return jsonify({'active_tests': tests})


@app.route('/backtest', methods=['POST'])
def run_backtest():
    """Run backtest with custom strategy"""
    global trading_system
    if not trading_system or not trading_system.paper_engine:
        return jsonify({'error': 'Paper trading not enabled'}), 503
    
    data = request.get_json() or {}
    days = data.get('days', 7)
    
    # Default strategy: enter if wallet WR > 50%
    def default_strategy(signal, quality):
        wallet_wr = signal.get('wallet_win_rate', 0)
        if wallet_wr > 1:
            wallet_wr /= 100
        return {
            'enter': wallet_wr >= 0.5,
            'exit_params': {
                'stop_loss_pct': -15,
                'take_profit_pct': 30,
                'max_hold_hours': 12
            }
        }
    
    result = trading_system.paper_engine.run_backtest(default_strategy, days=days)
    return jsonify(result)


@app.route('/test', methods=['GET'])
def test():
    return jsonify({
        'status': 'ok',
        'message': 'Robust Paper Trading Platform is running!',
        'features': {
            'watchdog': CONFIG.enable_watchdog,
            'baseline_tracking': CONFIG.enable_baseline_tracking,
            'historical_storage': CONFIG.enable_historical_storage,
            'ab_testing': CONFIG.enable_ab_testing,
        }
    })


# ============================================================================
# BACKGROUND TASKS
# ============================================================================
def background_tasks():
    """Background task loop"""
    global trading_system
    
    last_discovery = datetime.now()
    last_status = datetime.now()
    last_learning = datetime.now()
    
    while True:
        try:
            time.sleep(60)
            
            if not trading_system:
                continue
            
            now = datetime.now()
            
            # Hourly status
            if (now - last_status).total_seconds() >= 3600:
                last_status = now
                
                if trading_system.paper_engine and trading_system.notifier:
                    stats = trading_system.paper_engine.get_stats()
                    diag = trading_system.diagnostics.to_dict()
                    trading_system.notifier.send_hourly_status(stats, diag, stats)
            
            # Discovery
            if CONFIG.discovery_enabled:
                hours_since = (now - last_discovery).total_seconds() / 3600
                if hours_since >= CONFIG.discovery_interval_hours:
                    last_discovery = now
                    print(f"\n🔍 Running scheduled discovery...")
                    trading_system.run_discovery()
            
            # Learning analysis (every 6 hours)
            if trading_system.paper_engine:
                if (now - last_learning).total_seconds() >= 6 * 3600:
                    last_learning = now
                    print(f"\n🧪 Running learning analysis...")
                    trading_system.paper_engine.run_learning(
                        force=True, 
                        notifier=trading_system.notifier
                    )
        
        except Exception as e:
            print(f"Background task error: {e}")


# ============================================================================
# STARTUP
# ============================================================================
def setup_environment():
    """Check environment and set up venv if needed"""
    venv_path = os.path.join(os.getcwd(), 'venv')
    venv_python = os.path.join(venv_path, 'bin', 'python')
    
    if os.path.exists(venv_python):
        if sys.executable != venv_python:
            print(f"  ⚠️  Running outside venv. Use: source venv/bin/activate")
    else:
        print(f"  ⚠️  No venv found at {venv_path}")


def start_ngrok(port: int = 5000) -> Optional[str]:
    """Start ngrok tunnel in a new terminal"""
    import subprocess
    import shutil
    
    ngrok_path = shutil.which('ngrok')
    if not ngrok_path:
        print("  ⚠️  ngrok not found in PATH")
        return None
    
    try:
        result = subprocess.run(['pgrep', '-f', f'ngrok.*{port}'], capture_output=True)
        if result.returncode == 0:
            print(f"  ℹ️  ngrok already running for port {port}")
        else:
            print(f"  🚀 Starting ngrok in new terminal...")
            
            terminal_cmd = None
            if shutil.which('gnome-terminal'):
                terminal_cmd = ['gnome-terminal', '--', 'ngrok', 'http', str(port)]
            elif shutil.which('xterm'):
                terminal_cmd = ['xterm', '-e', f'ngrok http {port}']
            elif shutil.which('konsole'):
                terminal_cmd = ['konsole', '-e', 'ngrok', 'http', str(port)]
            
            if terminal_cmd:
                subprocess.Popen(terminal_cmd, start_new_session=True)
                time.sleep(3)
            else:
                print("  ⚠️  No terminal emulator found")
                return None
        
        # Get ngrok URL
        time.sleep(2)
        import requests
        try:
            resp = requests.get('http://127.0.0.1:4040/api/tunnels', timeout=5)
            if resp.status_code == 200:
                tunnels = resp.json().get('tunnels', [])
                for tunnel in tunnels:
                    if tunnel.get('proto') == 'https':
                        url = tunnel.get('public_url')
                        print(f"  ✅ ngrok URL: {url}")
                        return url
        except:
            pass
        
        print("  ⚠️  Could not get ngrok URL")
        return None
        
    except Exception as e:
        print(f"  ⚠️  ngrok error: {e}")
        return None


def main():
    global trading_system
    
    print("\n" + "="*70)
    print("🔧 ENVIRONMENT SETUP")
    print("="*70)
    
    setup_environment()
    
    ngrok_url = start_ngrok(port=CONFIG.webhook_port)
    
    if ngrok_url:
        webhook_url = f"{ngrok_url}/webhook/helius"
        print(f"\n   💡 TIP: If your Helius webhook URL is different, update it to:")
        print(f"      {webhook_url}")
    else:
        print("\n   ⚠️  ngrok failed to start!")
        existing_url = os.getenv('WEBHOOK_URL')
        if existing_url:
            print(f"   Using fallback: {existing_url}")
    
    try:
        trading_system = TradingSystem()
        
        bg_thread = threading.Thread(target=background_tasks, daemon=True)
        bg_thread.start()
        print(f"\n🔄 Background tasks running")
        
        print(f"\n🎧 WEBHOOK SERVER STARTING (ROBUST PLATFORM)")
        print(f"   Webhook:     http://{CONFIG.webhook_host}:{CONFIG.webhook_port}/webhook/helius")
        print(f"   Status:      http://{CONFIG.webhook_host}:{CONFIG.webhook_port}/status")
        print(f"   Diagnostics: http://{CONFIG.webhook_host}:{CONFIG.webhook_port}/diagnostics")
        print(f"   Positions:   http://{CONFIG.webhook_host}:{CONFIG.webhook_port}/positions")
        print(f"   Baseline:    http://{CONFIG.webhook_host}:{CONFIG.webhook_port}/baseline")
        print(f"   A/B Tests:   http://{CONFIG.webhook_host}:{CONFIG.webhook_port}/ab_test")
        print(f"   Backtest:    POST http://{CONFIG.webhook_host}:{CONFIG.webhook_port}/backtest")
        print(f"   Discovery:   POST http://{CONFIG.webhook_host}:{CONFIG.webhook_port}/discovery/run")
        print(f"   Learning:    POST http://{CONFIG.webhook_host}:{CONFIG.webhook_port}/learning/run")
        print(f"   Test:        http://{CONFIG.webhook_host}:{CONFIG.webhook_port}/test")
        print(f"\n   Press Ctrl+C to stop\n")
        
        app.run(host=CONFIG.webhook_host, port=CONFIG.webhook_port, debug=False, use_reloader=False)
        
    except KeyboardInterrupt:
        print("\n\n👋 Shutting down...")
        if trading_system:
            trading_system._print_status()
            if trading_system.paper_engine:
                trading_system.paper_engine.stop()
    except Exception as e:
        print(f"\n💥 FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
