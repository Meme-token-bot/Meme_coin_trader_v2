"""
MASTER V2 - FIXED PAPER TRADING INTEGRATION
============================================

Uses the fixed V6 paper trading platform with:
1. ✅ Correct balance calculations
2. ✅ Enforced position limits
3. ✅ Reliable exit monitoring

CHANGES FROM ORIGINAL:
- Imports fixed V6 platform instead of buggy V5
- Uses configurable max_open_positions (default 5)
- Better error handling and logging

Run migration first:
  python migrate_database.py fresh
"""

import os
import sys
import time
import json
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
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
# IMPORT FIXED V6 PAPER TRADING PLATFORM
# ============================================================================
from core.paper_trading_platform_v6_fixed import (
    RobustPaperTrader,
    SignalQualityAnalyzer,
    SignalQualityMetrics,
    DynamicExitCalculator,
    DynamicExitParams,
    BaselineTracker,
    ABTestingFramework,
    HistoricalDataStore,
    ExitMonitorWatchdog,
    ExitReason,
    set_platform_verbosity
)

# Import strategy learner for actual learning
from core.strategy_learner import StrategyLearner


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
    
    # Position limit - adjust based on phase:
    # Learning mode: 100+ (gather lots of data)
    # Refinement mode: 20-50 (testing stricter criteria)
    # Production mode: 5-10 (only high conviction trades)
    max_open_positions: int = 100
    
    max_position_size_sol: float = 0.3
    
    # Platform features
    enable_watchdog: bool = True
    enable_baseline_tracking: bool = True
    enable_historical_storage: bool = True
    enable_ab_testing: bool = True
    
    # Logging verbosity (0=quiet, 1=normal, 2=verbose, 3=debug)
    verbosity: int = 1
    
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
        
        return (len(self.minute_calls) < self.max_per_minute and 
                len(self.hour_calls) < self.max_per_hour)
    
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
    buy_signals_detected: int = 0
    sell_signals_detected: int = 0
    untracked_wallet_skips: int = 0
    duplicate_sig_skips: int = 0
    non_swap_skips: int = 0
    parse_failures: int = 0
    token_info_failures: int = 0
    position_limit_skips: int = 0  # NEW: Track limit hits
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
                    'position_limit': self.position_limit_skips,
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
    """Minimal Telegram notifications"""
    
    def __init__(self, token: str = None, chat_id: str = None):
        self.token = token or os.getenv('TELEGRAM_BOT_TOKEN')
        self.chat_id = chat_id or os.getenv('TELEGRAM_CHAT_ID')
        self.enabled = bool(self.token and self.chat_id)
        self._last_30min_update = datetime.now() - timedelta(minutes=30)
        self._last_stats = {}
    
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
        pass  # Disabled - too noisy
    
    def send_exit_alert(self, position: Dict, reason: str, pnl_pct: float, result: Dict = None):
        pass  # Disabled - too noisy
    
    def send_cluster_alert(self, token_symbol: str, wallet_count: int, wallets: List[str]):
        pass  # Disabled - logged to console
    
    def send_critical_alert(self, message: str):
        self.send(f"🚨 <b>CRITICAL</b>\n\n{message}")
    
    def send_30min_update(self, stats: Dict, diag: Dict):
        now = datetime.now()
        if (now - self._last_30min_update).total_seconds() < 1800:
            return
        
        self._last_30min_update = now
        
        prev = self._last_stats
        new_positions = diag['positions']['opened'] - prev.get('opened', 0)
        new_closes = (stats.get('total_trades', 0) - stats.get('open_positions', 0)) - prev.get('closed', 0)
        pnl_delta = stats.get('total_pnl', 0) - prev.get('pnl', 0)
        
        self._last_stats = {
            'opened': diag['positions']['opened'],
            'closed': stats.get('total_trades', 0) - stats.get('open_positions', 0),
            'pnl': stats.get('total_pnl', 0),
            'balance': stats.get('balance', 0),
        }
        
        emoji = "📈" if pnl_delta >= 0 else "📉"
        
        msg = f"""{emoji} <b>30-Min Update (V6 Fixed)</b>

<b>Account:</b>
Balance: {stats.get('balance', 0):.2f} SOL ({stats.get('return_pct', 0):+.1f}%)
Open: {stats.get('open_positions', 0)}/{stats.get('max_positions', 5)} positions

<b>Last 30 min:</b>
Opened: {new_positions} | Closed: {new_closes}
PnL: {pnl_delta:+.4f} SOL

<b>Session:</b>
Win rate: {stats.get('win_rate', 0):.0%}
Total PnL: {stats.get('total_pnl', 0):+.4f} SOL
Webhooks: {diag['webhooks']['received']}"""
        
        self.send(msg)
    
    def send_baseline_report(self, report: Dict):
        if not report:
            return
        best = max(report.items(), key=lambda x: x[1].get('total_pnl_pct', 0))
        if best[1].get('entered', 0) < 20:
            return
        
        msg = f"""📊 <b>Strategy Insight</b>

Best: <b>{best[0].upper()}</b>
PnL: {best[1]['total_pnl_pct']:+.1f}% | WR: {best[1]['win_rate']:.0%}"""
        
        self.send(msg)


# ============================================================================
# FIXED PAPER TRADING ENGINE WRAPPER
# ============================================================================
class FixedPaperTradingEngine:
    """
    Wrapper for the fixed V6 RobustPaperTrader.
    
    FIXES:
    1. ✅ Correct balance calculations (pnl_sol not exit_value)
    2. ✅ Enforced position limits (atomic check-and-insert)
    3. ✅ Reliable exit monitoring with watchdog
    """
    
    def __init__(self, db, starting_balance: float = 10.0, max_positions: int = 100):
        self.db = db
        self._notifier = None
        
        # Initialize the FIXED V6 trader
        self._trader = RobustPaperTrader(
            db_path="robust_paper_trades_v6.db",
            starting_balance=starting_balance,
            max_open_positions=max_positions,  # NOW ENFORCED!
            enable_watchdog=CONFIG.enable_watchdog,
            enable_baseline_tracking=CONFIG.enable_baseline_tracking,
            enable_historical_storage=CONFIG.enable_historical_storage
        )
        
        # Initialize STRATEGY LEARNER - this is where the magic happens!
        self.strategy_learner = StrategyLearner(db_path="robust_paper_trades_v6.db")
        
        # Store references
        self.quality_analyzer = self._trader.quality_analyzer
        self.exit_calculator = self._trader.exit_calculator
        self.baseline_tracker = self._trader.baseline_tracker
        self.historical_store = self._trader.historical_store
        self.ab_testing = self._trader.ab_testing
        self.watchdog = self._trader.watchdog
        
        self._top_wallets: List[str] = []
        self._last_top_wallets_update = datetime.utcnow() - timedelta(hours=1)
        self._last_learning = datetime.utcnow() - timedelta(hours=5)
        
        print(f"🚀 FIXED V6 PAPER TRADING ENGINE initialized")
        print(f"   Balance: {self._trader.balance:.4f} SOL")
        print(f"   Positions: {self._trader.open_position_count}/{self._trader.max_open_positions}")
        print(f"   Strategy Phase: {self.strategy_learner.config.phase}")
        print(f"   Learning Iteration: {self.strategy_learner.config.iteration}")
    
    def set_notifier(self, notifier):
        self._notifier = notifier
        self._trader.on_position_closed = self._on_exit_callback
    
    def _on_exit_callback(self, result: Dict):
        if self._notifier and hasattr(self._notifier, 'send_exit_alert'):
            self._notifier.send_exit_alert(
                {'token_symbol': result.get('token_symbol', 'UNKNOWN')},
                result.get('exit_reason', 'UNKNOWN'),
                result.get('pnl_pct', 0),
                result
            )
    
    @property
    def balance(self) -> float:
        return self._trader.balance
    
    @property
    def available_balance(self) -> float:
        return self._trader.balance - self._trader.reserved_balance
    
    def update_top_wallets(self, db):
        if (datetime.utcnow() - self._last_top_wallets_update).total_seconds() < 3600:
            return
        
        try:
            wallets = []
            if hasattr(db, 'get_all_wallets'):
                wallets = db.get_all_wallets()
            elif hasattr(db, 'get_tracked_wallets'):
                wallets = db.get_tracked_wallets()
            
            if wallets:
                sorted_wallets = sorted(wallets, key=lambda w: w.get('win_rate', 0), reverse=True)
                self._top_wallets = [w['address'] for w in sorted_wallets[:20] if w.get('win_rate', 0) >= 0.5]
            
            self._last_top_wallets_update = datetime.utcnow()
        except Exception as e:
            self._last_top_wallets_update = datetime.utcnow()
    
    def process_signal(self, signal: Dict, wallet_data: Dict = None) -> Dict:
        """Process a trading signal through the fixed pipeline."""
        return self._trader.process_signal(
            signal=signal,
            wallet_data=wallet_data,
            top_wallets=self._top_wallets
        )
    
    def get_open_positions(self) -> List[Dict]:
        return self._trader.get_open_positions()
    
    def get_stats(self) -> Dict:
        summary = self._trader.get_performance_summary()
        return {
            'balance': summary.get('balance', 0),
            'starting_balance': summary.get('starting_balance', 0),
            'total_pnl': summary.get('total_pnl_sol', 0),
            'return_pct': summary.get('return_pct', 0),
            'open_positions': summary.get('open_positions', 0),
            'max_positions': summary.get('max_positions', 5),
            'total_trades': summary.get('total_trades', 0),
            'win_rate': summary.get('win_rate', 0),
            'winning_trades': summary.get('winning_trades', 0),
            'phase': 'learning',
        }
    
    def get_baseline_comparison(self, days: int = 7) -> Dict:
        if self.baseline_tracker:
            return self.baseline_tracker.get_comparison_report(days)
        return {}
    
    def print_baseline_comparison(self, days: int = 7):
        if self.baseline_tracker:
            self.baseline_tracker.print_comparison(days)
    
    def run_learning(self, force: bool = False, notifier=None) -> Dict:
        """
        Run the ACTUAL learning iteration.
        
        This analyzes closed trades to learn:
        - Which wallet characteristics predict wins
        - Which hours perform best
        - Optimal exit parameters
        
        And updates strategy filters accordingly.
        """
        if not force and (datetime.utcnow() - self._last_learning).total_seconds() < 6 * 3600:
            return {'status': 'skipped', 'reason': 'Not time yet'}
        
        self._last_learning = datetime.utcnow()
        
        # Run the actual learning iteration
        results = self.strategy_learner.run_learning_iteration()
        
        # Add baseline comparison
        if self.baseline_tracker:
            results['baseline_comparison'] = self.get_baseline_comparison()
        
        # Send summary via Telegram
        if notifier and results.get('status') == 'completed':
            try:
                self._send_learning_summary(notifier, results)
            except Exception as e:
                print(f"  ⚠️ Failed to send learning summary: {e}")
        
        return results
    
    def _send_learning_summary(self, notifier, results: Dict):
        """Send learning results via Telegram"""
        overall = results.get('overall', {})
        actions = results.get('actions', [])
        
        msg = f"""🧠 <b>Learning Iteration #{self.strategy_learner.config.iteration}</b>

<b>Performance:</b>
Win Rate: {overall.get('win_rate', 0):.1%}
Total PnL: {overall.get('total_pnl', 0):+.4f} SOL

<b>Phase:</b> {self.strategy_learner.config.phase}

<b>Current Filters:</b>
Min Wallet WR: {self.strategy_learner.config.min_wallet_wr:.0%}
Stop Loss: {self.strategy_learner.config.stop_loss_pct}%"""
        
        if actions:
            msg += "\n\n<b>Changes Made:</b>"
            for action in actions[:3]:
                msg += f"\n• {action}"
        
        notifier.send(msg)
    
    def get_learned_filters(self) -> Dict:
        """Get current learned filter settings"""
        return self.strategy_learner.get_current_filters()
    
    def should_take_trade(self, signal: Dict, wallet_data: Dict) -> Tuple[bool, str]:
        """Check if trade passes learned filters"""
        return self.strategy_learner.should_take_trade(signal, wallet_data)
    
    def get_strategy_feedback(self) -> Dict:
        return {
            'summary': self.get_stats(),
            'baseline': self.get_baseline_comparison() if self.baseline_tracker else {},
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
        print("🚀 TRADING SYSTEM V2 - FIXED V6 PAPER TRADING")
        print("="*70)
        
        set_platform_verbosity(CONFIG.verbosity)
        
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
        
        # Initialize FIXED Paper Trading Engine
        if CONFIG.paper_trading_enabled:
            self.paper_engine = FixedPaperTradingEngine(
                self.db,
                starting_balance=CONFIG.paper_starting_balance,
                max_positions=CONFIG.max_open_positions
            )
            self.paper_engine.set_notifier(self.notifier)
            print(f"  ✅ Fixed V6 Paper Trading (Max: {CONFIG.max_open_positions} positions)")
        else:
            self.paper_engine = None
        
        self.rate_limiter = RateLimiter(CONFIG.max_token_lookups_per_minute, CONFIG.max_api_calls_per_hour)
        self.diagnostics = DiagnosticsTracker()
        
        webhook_url = os.getenv('WEBHOOK_URL', f'http://0.0.0.0:{CONFIG.webhook_port}/webhook/helius')

        try:
            from infrastructure.multi_webhook_manager import MultiWebhookManager
            self.multi_webhook_manager = MultiWebhookManager(self.helius_key, webhook_url, self.db)
            print(f"  ✅ Multi-Webhook Manager")
        except Exception as e:
            self.multi_webhook_manager = None

        if CONFIG.discovery_enabled:
            discovery_key = os.getenv('HELIUS_DISCOVERY_KEY')
            self.historian = Historian(
                self.db, 
                self.helius_key, 
                discovery_key,
                multi_webhook_manager=self.multi_webhook_manager
            )
            print(f"  ✅ Discovery")
        else:
            self.historian = None
        
        self.start_time = datetime.now()
        
        print("\n" + "="*70)
        print("✅ FIXED V6 PAPER TRADING PLATFORM READY")
        print("="*70)
        self._print_status()
    
    def _print_status(self):
        status = self.strategist.get_status()
        
        print(f"\n📊 Status:")
        print(f"  Wallets tracked: {self.db.get_wallet_count()}")
        print(f"  Regime: {status['regime']} ({status['confidence']:.0%})")
        
        if self.paper_engine:
            stats = self.paper_engine.get_stats()
            print(f"  Paper: {stats['balance']:.4f} SOL ({stats['return_pct']:+.1f}%)")
            print(f"  Positions: {stats['open_positions']}/{stats['max_positions']}")
            print(f"  Win Rate: {stats['win_rate']:.1%} ({stats['total_trades']} trades)")
    
    def process_webhook(self, data: Dict) -> Dict:
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
            
            result['processed'] = True
            
        except Exception as e:
            result['reason'] = str(e)
        
        return result
    
    def _process_transaction(self, tx: Dict):
        fee_payer = tx.get('feePayer', '')
        signature = tx.get('signature', '')
        
        if not fee_payer or not signature:
            return
        
        wallet_data = self.db.get_wallet(fee_payer)
        if not wallet_data:
            self.diagnostics.untracked_wallet_skips += 1
            return
        
        if self.db.is_signature_processed(signature):
            self.diagnostics.duplicate_sig_skips += 1
            return
        
        trade = self._parse_swap(tx)
        if not trade:
            self.diagnostics.non_swap_skips += 1
            return
        
        self.db.mark_signature_processed(signature)
        
        token_addr = trade.get('token_out') if trade['type'] == 'BUY' else trade.get('token_in')
        if not token_addr:
            return
        
        if trade['type'] == 'BUY':
            self.diagnostics.buy_signals_detected += 1
            return self._process_buy(trade, wallet_data, token_addr, signature)
        elif trade['type'] == 'SELL':
            self.diagnostics.sell_signals_detected += 1
    
    def _parse_swap(self, tx: Dict) -> Optional[Dict]:
        instructions = tx.get('instructions', [])
        token_transfers = tx.get('tokenTransfers', [])
        
        if not token_transfers:
            return None
        
        is_swap = any('swap' in str(instr).lower() for instr in instructions)
        
        if not is_swap and len(token_transfers) < 2:
            return None
        
        sol_mint = "So11111111111111111111111111111111111111112"
        fee_payer = tx.get('feePayer', '')
        
        sol_in = sol_out = token_in = token_out = None
        
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
        result = {'processed': True, 'action': 'BUY_SIGNAL', 'reason': ''}
        
        if self.db.is_position_tracked(wallet_data['address'], token_addr):
            result['reason'] = 'Position already tracked'
            return result
        
        if not self.rate_limiter.can_call():
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
        
        wallet_wr = wallet_data.get('win_rate', 0.5)
        if wallet_wr > 1:
            wallet_wr = wallet_wr / 100.0
        
        liquidity = token_info.get('liquidity', 0)
        if isinstance(liquidity, dict):
            liquidity = liquidity.get('usd', 0)
        liquidity = float(liquidity or 0)
        
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
        
        wallet_info = {
            'address': wallet_data['address'],
            'win_rate': wallet_wr,
            'cluster': wallet_data.get('cluster', 'UNKNOWN'),
        }
        
        if self.paper_engine:
            self.paper_engine.update_top_wallets(self.db)
            
            # Check LEARNED filters first (from strategy learner)
            should_trade, filter_reason = self.paper_engine.should_take_trade(signal_data, wallet_info)
            if not should_trade:
                self.diagnostics.webhooks_skipped += 1
                result['reason'] = f"Learned filter: {filter_reason}"
                return result
            
            # Basic wallet WR check (fallback if learner hasn't run yet)
            if wallet_wr < 0.30:
                result['reason'] = f"Low wallet WR: {wallet_wr:.0%}"
                return result
            
            process_result = self.paper_engine.process_signal(signal_data, wallet_info)
            
            quality_metrics = process_result.get('quality_metrics', {})
            if quality_metrics.get('is_cluster_signal'):
                self.diagnostics.cluster_signals_detected += 1
            
            conviction = process_result.get('conviction', 50)
            if conviction >= 70:
                self.diagnostics.high_conviction_signals += 1
            
            if self.paper_engine.baseline_tracker:
                self.diagnostics.baseline_signals_recorded += 1
            
            if process_result.get('position_id'):
                self.diagnostics.positions_opened += 1
                result['action'] = 'POSITION_OPENED'
                result['position_id'] = process_result['position_id']
            else:
                filter_reason = process_result.get('filter_reason', '')
                if 'limit' in filter_reason.lower():
                    self.diagnostics.position_limit_skips += 1
                result['reason'] = filter_reason or 'Position not opened'
        
        return result
    
    def run_discovery(self) -> Dict:
        if not self.historian:
            return {'status': 'disabled'}
        
        try:
            self.diagnostics.discoveries_run += 1
            result = self.historian.run_discovery()
            self.diagnostics.wallets_discovered += result.get('wallets_added', 0)
            return result
        except Exception as e:
            return {'status': 'error', 'error': str(e)}


# ============================================================================
# FLASK APP
# ============================================================================
app = Flask(__name__)
trading_system: TradingSystem = None

import logging as _logging
if CONFIG.verbosity < 3:
    _logging.getLogger('werkzeug').setLevel(_logging.WARNING)


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
    
    return jsonify({
        'status': 'running',
        'version': 'V6_FIXED',
        'uptime_hours': (datetime.now() - trading_system.start_time).total_seconds() / 3600,
        'wallets_tracked': trading_system.db.get_wallet_count(),
        'paper_trading': stats,
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
    stats = trading_system.paper_engine.get_stats()
    return jsonify({
        'count': len(positions),
        'max': stats.get('max_positions', 5),
        'positions': positions
    })


@app.route('/baseline', methods=['GET'])
def baseline_comparison():
    global trading_system
    if not trading_system or not trading_system.paper_engine:
        return jsonify({'error': 'Paper trading not enabled'}), 503
    
    days = request.args.get('days', 7, type=int)
    report = trading_system.paper_engine.get_baseline_comparison(days)
    return jsonify(report)


@app.route('/reset', methods=['POST'])
def reset_database():
    """Reset the paper trading database"""
    global trading_system
    if not trading_system or not trading_system.paper_engine:
        return jsonify({'error': 'Paper trading not enabled'}), 503
    
    confirm = request.args.get('confirm', '')
    if confirm != 'YES':
        return jsonify({'error': 'Add ?confirm=YES to confirm reset'}), 400
    
    trading_system.paper_engine._trader.reset_database()
    return jsonify({'status': 'reset', 'balance': trading_system.paper_engine.balance})


@app.route('/strategy', methods=['GET'])
def get_strategy():
    """Get current learned strategy configuration"""
    global trading_system
    if not trading_system or not trading_system.paper_engine:
        return jsonify({'error': 'Paper trading not enabled'}), 503
    
    learner = trading_system.paper_engine.strategy_learner
    
    return jsonify({
        'phase': learner.config.phase,
        'iteration': learner.config.iteration,
        'current_wr': learner.config.current_wr,
        'target_wr': learner.config.target_wr,
        'filters': learner.get_current_filters(),
        'last_updated': learner.config.last_updated
    })


@app.route('/learning/run', methods=['POST'])
def run_learning():
    """Manually trigger a learning iteration"""
    global trading_system
    if not trading_system or not trading_system.paper_engine:
        return jsonify({'error': 'Paper trading not enabled'}), 503
    
    force = request.args.get('force', 'true').lower() == 'true'
    results = trading_system.paper_engine.run_learning(force=force, notifier=trading_system.notifier)
    
    return jsonify(results)


@app.route('/learning/status', methods=['GET'])
def learning_status():
    """Get learning system status"""
    global trading_system
    if not trading_system or not trading_system.paper_engine:
        return jsonify({'error': 'Paper trading not enabled'}), 503
    
    learner = trading_system.paper_engine.strategy_learner
    
    # Get recent trades summary
    trades = learner.get_closed_trades(days=7)
    
    return jsonify({
        'phase': learner.config.phase,
        'iteration': learner.config.iteration,
        'trades_last_7d': len(trades),
        'current_wr': learner.config.current_wr,
        'target_wr': learner.config.target_wr,
        'min_wallet_wr': learner.config.min_wallet_wr,
        'blocked_hours': learner.config.blocked_hours_utc,
        'preferred_hours': learner.config.preferred_hours_utc,
        'last_updated': learner.config.last_updated
    })


@app.route('/wallets/performance', methods=['GET'])
def wallet_performance():
    """Get wallet performance analysis"""
    global trading_system
    if not trading_system or not trading_system.paper_engine:
        return jsonify({'error': 'Paper trading not enabled'}), 503
    
    learner = trading_system.paper_engine.strategy_learner
    
    if not learner.wallet_analyzer:
        return jsonify({'error': 'Wallet analyzer not available'}), 503
    
    days = request.args.get('days', 14, type=int)
    performances = learner.wallet_analyzer.analyze_all_wallets(days)
    
    # Convert to list sorted by trade count
    perf_list = sorted(
        [p.to_dict() for p in performances.values()],
        key=lambda x: x['total_trades'],
        reverse=True
    )
    
    return jsonify({
        'wallets_analyzed': len(perf_list),
        'performances': perf_list[:50]  # Top 50
    })


@app.route('/wallets/cleanup', methods=['POST'])
def wallet_cleanup():
    """Run wallet cleanup to remove poor performers"""
    global trading_system
    if not trading_system or not trading_system.paper_engine:
        return jsonify({'error': 'Paper trading not enabled'}), 503
    
    learner = trading_system.paper_engine.strategy_learner
    
    if not learner.wallet_analyzer:
        return jsonify({'error': 'Wallet analyzer not available'}), 503
    
    # Check for confirmation
    dry_run = request.args.get('confirm', '') != 'YES'
    
    if dry_run:
        result = learner.run_wallet_cleanup(
            webhook_manager=trading_system.multi_webhook_manager,
            db=trading_system.db,
            dry_run=True
        )
        result['note'] = 'Dry run - add ?confirm=YES to actually remove wallets'
    else:
        result = learner.run_wallet_cleanup(
            webhook_manager=trading_system.multi_webhook_manager,
            db=trading_system.db,
            dry_run=False
        )
    
    return jsonify(result)

@app.route('/test', methods=['GET'])
def test():
    return jsonify({
        'status': 'ok',
        'version': 'V6_FIXED',
        'message': 'Fixed Paper Trading Platform is running!',
    })


# ============================================================================
# BACKGROUND TASKS
# ============================================================================
def background_tasks():
    global trading_system
    
    last_discovery = datetime.now()
    last_status = datetime.now()
    last_learning = datetime.now()
    last_summary = datetime.now()
    
    while True:
        try:
            time.sleep(60)
            
            if not trading_system:
                continue
            
            now = datetime.now()
            
            # Console summary every 5 minutes
            if CONFIG.verbosity >= 1 and (now - last_summary).total_seconds() >= 300:
                last_summary = now
                d = trading_system.diagnostics
                stats = trading_system.paper_engine.get_stats() if trading_system.paper_engine else {}
                
                mins_since = (now - d.last_webhook_received).total_seconds() / 60 if d.last_webhook_received else 999
                print(f"📡 {now.strftime('%H:%M')} | "
                      f"Webhooks: {d.webhooks_received} | "
                      f"Signals: {d.buy_signals_detected} | "
                      f"Opened: {d.positions_opened} | "
                      f"Open: {stats.get('open_positions', 0)}/{stats.get('max_positions', 5)} | "
                      f"Balance: {stats.get('balance', 0):.4f} SOL | "
                      f"WR: {stats.get('win_rate', 0):.0%}")
            
            # Telegram update every 30 min
            if (now - last_status).total_seconds() >= 1800:
                last_status = now
                if trading_system.paper_engine and trading_system.notifier:
                    stats = trading_system.paper_engine.get_stats()
                    diag = trading_system.diagnostics.to_dict()
                    trading_system.notifier.send_30min_update(stats, diag)
            
            # Discovery
            if CONFIG.discovery_enabled:
                hours_since = (now - last_discovery).total_seconds() / 3600
                if hours_since >= CONFIG.discovery_interval_hours:
                    last_discovery = now
                    print(f"\n🔍 Running discovery...")
                    trading_system.run_discovery()
            
            # Learning (every 6 hours)
            if trading_system.paper_engine:
                if (now - last_learning).total_seconds() >= 6 * 3600:
                    last_learning = now
                    print(f"\n🧠 Running learning iteration...")
                    learning_result = trading_system.paper_engine.run_learning(
                        force=True, 
                        notifier=trading_system.notifier
                    )
                    
                    # After learning, run wallet cleanup (actually remove poor performers)
                    if learning_result.get('status') == 'completed':
                        learner = trading_system.paper_engine.strategy_learner
                        if learner.wallet_analyzer:
                            to_remove = learner.wallet_analyzer.get_wallets_to_remove(days=14)
                            if to_remove:
                                print(f"\n🧹 Removing {len(to_remove)} poor performing wallets...")
                                cleanup_result = learner.run_wallet_cleanup(
                                    webhook_manager=trading_system.multi_webhook_manager,
                                    db=trading_system.db,
                                    dry_run=False  # Actually remove!
                                )
                                if trading_system.notifier:
                                    removed_count = len(cleanup_result.get('wallets_removed', []))
                                    if removed_count > 0:
                                        trading_system.notifier.send(
                                            f"🧹 Removed {removed_count} poor performing wallets"
                                        )
        
        except Exception as e:
            print(f"Background task error: {e}")


# ============================================================================
# STARTUP
# ============================================================================
def main():
    global trading_system
    
    print("\n" + "="*70)
    print("🔧 STARTING FIXED V6 PAPER TRADING SYSTEM")
    print("="*70)
    
    try:
        trading_system = TradingSystem()
        
        bg_thread = threading.Thread(target=background_tasks, daemon=True)
        bg_thread.start()
        
        print(f"\n🎧 WEBHOOK SERVER (V6 FIXED)")
        print(f"   Webhook:     http://{CONFIG.webhook_host}:{CONFIG.webhook_port}/webhook/helius")
        print(f"   Status:      http://{CONFIG.webhook_host}:{CONFIG.webhook_port}/status")
        print(f"   Positions:   http://{CONFIG.webhook_host}:{CONFIG.webhook_port}/positions")
        print(f"   Baseline:    http://{CONFIG.webhook_host}:{CONFIG.webhook_port}/baseline")
        print(f"   Reset:       POST /reset?confirm=YES")
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
