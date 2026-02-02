"""
MASTER V2 - PAPER + LIVE TRADING INTEGRATION
=============================================

Uses the fixed V6 paper trading platform with:
1. ✅ Correct balance calculations
2. ✅ Enforced position limits
3. ✅ Reliable exit monitoring
4. ✅ LIVE TRADING SUPPORT (Hybrid Engine)

LIVE TRADING FEATURES:
- Jito bundle integration for fast execution
- Parallel paper/live trading for comparison
- NZ tax compliant record keeping
- Kill switch for emergencies

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

# =============================================================================
# SECRETS MANAGEMENT (AWS Secrets Manager or .env fallback)
# =============================================================================
try:
    from core.secrets_manager import init_secrets, get_secret, get_secrets
    
    # Initialize secrets at startup - this loads from AWS Secrets Manager
    # and sets them as environment variables for backward compatibility
    print("🔐 Initializing secrets...")
    init_secrets()
    SECRETS_AVAILABLE = True
except ImportError:
    print("⚠️ Secrets manager not available, falling back to .env")
    from dotenv import load_dotenv
    load_dotenv()
    SECRETS_AVAILABLE = False
    
    def get_secret(key, default=None):
        import os
        return os.getenv(key, default)

from core.database_v2 import DatabaseV2
from core.discovery_integration import HistorianV8 as Historian
from core.strategist_v2 import Strategist
from core.discovery_config import config as discovery_config
from infrastructure.helius_webhook_manager import HeliusWebhookManager
from core.stealth_trading_coordinator import StealthTradingCoordinator

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

# ============================================================================
# LIVE TRADING INTEGRATION
# ============================================================================
try:
    from core.hybrid_trading_engine import (
        HybridTradingEngine, 
        HybridTradingConfig, 
        create_hybrid_engine,
        TradingMode
    )
    HYBRID_ENGINE_AVAILABLE = True
except ImportError:
    HYBRID_ENGINE_AVAILABLE = False
    print("⚠️ Hybrid trading engine not available - paper trading only")

try:
    from core.live_trading_engine import LiveTradingEngine, LiveTradingConfig
    LIVE_ENGINE_AVAILABLE = True
except ImportError:
    LIVE_ENGINE_AVAILABLE = False


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
    
    # ==========================================================================
    # LIVE TRADING SETTINGS (Based on paper trading analysis)
    # ==========================================================================
    enable_live_trading: bool = False          # Master switch - set via env var
    live_position_size_sol: float = 0.08       # Per trade (6.5% fee overhead)
    max_live_positions: int = 10               # Max concurrent live positions
    max_daily_loss_sol: float = 0.25           # Stop live trading if exceeded
    min_balance_sol: float = 1.50              # Emergency stop threshold
    blocked_hours_utc: List[int] = field(default_factory=lambda: [1, 3, 5, 19, 23])
    
    @property
    def discovery_interval_hours(self) -> int:
        return discovery_config.discovery_interval_hours
    
    @property
    def discovery_api_budget(self) -> int:
        return discovery_config.max_api_calls_per_discovery


CONFIG = MasterConfig()

# Override from secrets/environment
# Note: This happens after secrets are initialized in the imports
def _update_config_from_secrets():
    """Update CONFIG from secrets after they're loaded"""
    if get_secret('ENABLE_LIVE_TRADING', '').lower() == 'true':
        CONFIG.enable_live_trading = True
    if get_secret('POSITION_SIZE_SOL'):
        try:
            CONFIG.live_position_size_sol = float(get_secret('POSITION_SIZE_SOL'))
        except:
            pass
    if get_secret('MAX_OPEN_POSITIONS'):
        try:
            CONFIG.max_live_positions = int(get_secret('MAX_OPEN_POSITIONS'))
        except:
            pass


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
    position_limit_skips: int = 0
    cluster_signals_detected: int = 0
    high_conviction_signals: int = 0
    baseline_signals_recorded: int = 0
    # Live trading stats
    live_trades_opened: int = 0
    live_trades_closed: int = 0
    live_pnl_sol: float = 0.0
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
            'live_trading': {
                'opened': self.live_trades_opened,
                'closed': self.live_trades_closed,
                'pnl_sol': self.live_pnl_sol
            },
            'discovery': {'runs': self.discoveries_run, 'wallets_found': self.wallets_discovered},
            'learning_iterations': self.learning_iterations,
            'baseline_signals_recorded': self.baseline_signals_recorded,
            'llm_calls': self.llm_calls,
            'recent_events': list(self.recent_events)[-10:]
        }


class Notifier:
    """Minimal Telegram notifications"""
    
    def __init__(self, token: str = None, chat_id: str = None):
        self.token = token or get_secret('TELEGRAM_BOT_TOKEN')
        self.chat_id = chat_id or get_secret('TELEGRAM_CHAT_ID')
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
        
        # Add live trading info if available
        live_info = ""
        live_stats = diag.get('live_trading', {})
        if live_stats.get('opened', 0) > 0:
            live_info = f"\n\n<b>Live Trading:</b>\nOpened: {live_stats['opened']} | PnL: {live_stats['pnl_sol']:+.4f} SOL"
        
        msg = f"""{emoji} <b>30-Min Update (V6 + Live)</b>

<b>Paper Account:</b>
Balance: {stats.get('balance', 0):.2f} SOL ({stats.get('return_pct', 0):+.1f}%)
Open: {stats.get('open_positions', 0)}/{stats.get('max_positions', 5)} positions

<b>Last 30 min:</b>
Opened: {new_positions} | Closed: {new_closes}
PnL: {pnl_delta:+.4f} SOL

<b>Session:</b>
Win rate: {stats.get('win_rate', 0):.0%}
Total PnL: {stats.get('total_pnl', 0):+.4f} SOL
Webhooks: {diag['webhooks']['received']}{live_info}"""
        
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
        print("🚀 TRADING SYSTEM V2 - PAPER + LIVE TRADING")
        print("="*70)
        
        set_platform_verbosity(CONFIG.verbosity)
        
        # Load API keys from secrets (AWS Secrets Manager or .env)
        self.helius_key = get_secret('HELIUS_KEY')
        self.anthropic_key = get_secret('ANTHROPIC_API_KEY')
        self.webhook_id = get_secret('HELIUS_WEBHOOK_ID')
        
        if not self.helius_key:
            raise ValueError("HELIUS_KEY not found in secrets!")
        
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

        # Initialize stealth coordinator
        self.stealth = StealthTradingCoordinator(
            helius_key=self.helius_key,
            telegram_token=get_secret('TELEGRAM_BOT_TOKEN'),
            telegram_chat_id=get_secret('TELEGRAM_CHAT_ID')
        )

        # Load wallet keys
        wallet_secrets = {
            'HOT_WALLET_1': get_secret('HOT_WALLET_1'),
            'HOT_WALLET_2': get_secret('HOT_WALLET_2'),
            'HOT_WALLET_3': get_secret('HOT_WALLET_3'),
            'HOT_WALLET_4': get_secret('HOT_WALLET_4'),
            'HOT_WALLET_5': get_secret('HOT_WALLET_5'),
            'BURNER_ADDRESS_1': get_secret('BURNER_ADDRESS_1'),
            'BURNER_ADDRESS_2': get_secret('BURNER_ADDRESS_2'),
            'BURNER_ADDRESS_3': get_secret('BURNER_ADDRESS_3'),
            'BURNER_ADDRESS_4': get_secret('BURNER_ADDRESS_4'),
            'BURNER_ADDRESS_5': get_secret('BURNER_ADDRESS_5')
        }
        self.stealth.load_wallet_keys(wallet_secrets)

        # Start background tasks
        self.stealth.start_background_tasks()
        
        # ======================================================================
        # INITIALIZE HYBRID TRADING ENGINE (Paper + Live)
        # ======================================================================
        self.hybrid_engine = None
        if HYBRID_ENGINE_AVAILABLE:
            try:
                self.hybrid_engine = create_hybrid_engine(
                    paper_engine=self.paper_engine,
                    notifier=self.notifier
                )
                
                live_status = "ENABLED ⚡" if self.hybrid_engine.config.enable_live_trading else "disabled"
                print(f"  ✅ Hybrid Trading Engine (Live: {live_status})")
                
                # Start live position monitoring if enabled
                if self.hybrid_engine.config.enable_live_trading and self.hybrid_engine.live_engine:
                    self.hybrid_engine.live_engine.start_monitoring()
                    print(f"  🔴 LIVE TRADING ACTIVE - Position size: {self.hybrid_engine.config.position_size_sol} SOL")
            except Exception as e:
                print(f"  ⚠️ Hybrid engine init failed: {e}")
                self.hybrid_engine = None
        else:
            print(f"  ℹ️ Hybrid engine not available (paper trading only)")
        
        self.rate_limiter = RateLimiter(CONFIG.max_token_lookups_per_minute, CONFIG.max_api_calls_per_hour)
        self.diagnostics = DiagnosticsTracker()
        
        webhook_url = get_secret('WEBHOOK_URL') or f'http://0.0.0.0:{CONFIG.webhook_port}/webhook/helius'

        try:
            from infrastructure.multi_webhook_manager import MultiWebhookManager
            self.multi_webhook_manager = MultiWebhookManager(self.helius_key, webhook_url, self.db)
            print(f"  ✅ Multi-Webhook Manager")
        except Exception as e:
            self.multi_webhook_manager = None

        if CONFIG.discovery_enabled:
            discovery_key = get_secret('HELIUS_DISCOVERY_KEY')
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
        if self.hybrid_engine and self.hybrid_engine.config.enable_live_trading:
            print("🔴 LIVE TRADING SYSTEM READY - REAL MONEY AT RISK!")
        else:
            print("✅ PAPER TRADING SYSTEM READY")
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
        
        if self.hybrid_engine and self.hybrid_engine.config.enable_live_trading:
            print(f"  🔴 LIVE: {self.hybrid_engine.config.position_size_sol} SOL per trade")
    
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
        # Use stealth coordinator instead of single wallet
        result = self.stealth.execute_buy(signal)
        
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
            'price_usd': price,
            'liquidity': liquidity,
            'liquidity_usd': liquidity,
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
        
        # ======================================================================
        # HYBRID TRADING: Route through paper AND live if enabled
        # ======================================================================
        if self.hybrid_engine and self.hybrid_engine.config.enable_live_trading:
            # Process through hybrid engine (handles both paper and live)
            hybrid_result = self.hybrid_engine.process_signal(signal_data, wallet_info)
            
            # Track diagnostics
            if hybrid_result.get('filter_passed'):
                live_result = hybrid_result.get('live_result', {})
                paper_result = hybrid_result.get('paper_result', {})
                
                if live_result and live_result.get('success'):
                    self.diagnostics.live_trades_opened += 1
                    result['action'] = 'LIVE_POSITION_OPENED'
                    result['live_signature'] = live_result.get('signature', '')[:16] if live_result.get('signature') else ''
                    print(f"🔴 LIVE BUY: {signal_data.get('token_symbol')} | {self.hybrid_engine.config.position_size_sol} SOL")
                
                if paper_result and paper_result.get('position_id'):
                    self.diagnostics.positions_opened += 1
                    result['paper_position_id'] = paper_result.get('position_id')
                
                conviction = paper_result.get('conviction', 50) if paper_result else 50
                if conviction >= 70:
                    self.diagnostics.high_conviction_signals += 1
            else:
                result['reason'] = f"Filter: {hybrid_result.get('filter_reason', 'unknown')}"
            
            return result
        
        # ======================================================================
        # PAPER ONLY: Original paper trading logic (when live is disabled)
        # ======================================================================
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
    
    live_info = {}
    if trading_system.hybrid_engine:
        live_info = {
            'enabled': trading_system.hybrid_engine.config.enable_live_trading,
            'mode': trading_system.hybrid_engine.mode.value if hasattr(trading_system.hybrid_engine, 'mode') else 'unknown'
        }
    
    return jsonify({
        'status': 'running',
        'version': 'V6_LIVE',
        'uptime_hours': (datetime.now() - trading_system.start_time).total_seconds() / 3600,
        'wallets_tracked': trading_system.db.get_wallet_count(),
        'paper_trading': stats,
        'live_trading': live_info
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
def baseline():
    global trading_system
    if not trading_system or not trading_system.paper_engine:
        return jsonify({'error': 'Paper trading not enabled'}), 503
    
    days = request.args.get('days', 7, type=int)
    return jsonify(trading_system.paper_engine.get_baseline_comparison(days))


@app.route('/reset', methods=['POST'])
def reset():
    global trading_system
    
    confirm = request.args.get('confirm', '')
    if confirm != 'YES':
        return jsonify({'error': 'Add ?confirm=YES to reset'}), 400
    
    if not trading_system or not trading_system.paper_engine:
        return jsonify({'error': 'Paper trading not enabled'}), 503
    
    trading_system.paper_engine._trader.reset()
    trading_system.diagnostics = DiagnosticsTracker()
    
    return jsonify({
        'status': 'reset',
        'balance': trading_system.paper_engine.balance
    })


@app.route('/learning/run', methods=['POST'])
def run_learning():
    global trading_system
    if not trading_system or not trading_system.paper_engine:
        return jsonify({'error': 'Paper trading not enabled'}), 503
    
    result = trading_system.paper_engine.run_learning(
        force=True,
        notifier=trading_system.notifier
    )
    trading_system.diagnostics.learning_iterations += 1
    
    return jsonify(result)


@app.route('/learning/status', methods=['GET'])
def learning_status():
    global trading_system
    if not trading_system or not trading_system.paper_engine:
        return jsonify({'error': 'Paper trading not enabled'}), 503
    
    learner = trading_system.paper_engine.strategy_learner
    return jsonify(learner.get_current_filters())


@app.route('/strategy', methods=['GET'])
def strategy():
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
        'version': 'V6_LIVE',
        'message': 'Paper + Live Trading Platform is running!',
        'live_available': HYBRID_ENGINE_AVAILABLE
    })


# ============================================================================
# LIVE TRADING ENDPOINTS
# ============================================================================

@app.route('/live/status', methods=['GET'])
def live_status():
    """Get live trading status"""
    global trading_system
    
    if not trading_system:
        return jsonify({'error': 'System not initialized'}), 503
    
    if not trading_system.hybrid_engine:
        return jsonify({
            'live_trading_available': False,
            'reason': 'Hybrid engine not initialized - check hybrid_trading_engine.py is in core/'
        })
    
    return jsonify(trading_system.hybrid_engine.get_status())


@app.route('/live/enable', methods=['POST'])
def enable_live():
    """Enable live trading (requires confirmation)"""
    global trading_system
    
    confirm = request.args.get('confirm', '')
    if confirm != 'LIVE':
        return jsonify({
            'error': 'Confirmation required',
            'message': 'Add ?confirm=LIVE to enable live trading',
            'warning': '⚠️ This will trade REAL money!'
        }), 400
    
    if not trading_system or not trading_system.hybrid_engine:
        return jsonify({'error': 'Hybrid engine not available'}), 503
    
    if not trading_system.hybrid_engine.live_engine:
        return jsonify({'error': 'Live engine not configured (check SOLANA_PRIVATE_KEY)'}), 503
    
    trading_system.hybrid_engine.config.enable_live_trading = True
    trading_system.hybrid_engine.mode = TradingMode.HYBRID
    
    # Start monitoring
    if trading_system.hybrid_engine.live_engine:
        trading_system.hybrid_engine.live_engine.start_monitoring()
    
    if trading_system.notifier:
        trading_system.notifier.send("🔴 LIVE TRADING ENABLED - Real money at risk!")
    
    return jsonify({
        'success': True,
        'message': '🔴 LIVE TRADING ENABLED',
        'position_size': trading_system.hybrid_engine.config.position_size_sol,
        'max_positions': trading_system.hybrid_engine.config.max_open_positions,
        'status': trading_system.hybrid_engine.get_status()
    })


@app.route('/live/disable', methods=['POST'])
def disable_live():
    """Disable live trading (keeps paper trading running)"""
    global trading_system
    
    if not trading_system or not trading_system.hybrid_engine:
        return jsonify({'error': 'Hybrid engine not available'}), 503
    
    trading_system.hybrid_engine.config.enable_live_trading = False
    trading_system.hybrid_engine.mode = TradingMode.PAPER_ONLY
    
    if trading_system.notifier:
        trading_system.notifier.send("📝 Live trading disabled - paper trading continues")
    
    return jsonify({
        'success': True,
        'message': 'Live trading disabled',
        'mode': 'paper_only'
    })


@app.route('/live/kill', methods=['POST'])
def kill_switch():
    """Emergency kill switch - close all live positions immediately"""
    global trading_system
    
    confirm = request.args.get('confirm', '')
    if confirm != 'KILL':
        return jsonify({
            'error': 'Confirmation required',
            'message': 'Add ?confirm=KILL to activate kill switch',
            'warning': '⚠️ This will close ALL live positions at market price!'
        }), 400
    
    if not trading_system or not trading_system.hybrid_engine:
        return jsonify({'error': 'Hybrid engine not available'}), 503
    
    trading_system.hybrid_engine._activate_kill_switch('Manual API activation')
    
    return jsonify({
        'success': True,
        'message': '🚨 KILL SWITCH ACTIVATED',
        'action': 'All live positions closed, live trading disabled'
    })


@app.route('/live/positions', methods=['GET'])
def live_positions():
    """Get current live positions"""
    global trading_system
    
    if not trading_system or not trading_system.hybrid_engine:
        return jsonify({'error': 'Hybrid engine not available'}), 503
    
    if not trading_system.hybrid_engine.live_engine:
        return jsonify({'error': 'Live engine not available'}), 503
    
    positions = trading_system.hybrid_engine.live_engine.tax_db.get_positions()
    
    return jsonify({
        'count': len(positions),
        'max': trading_system.hybrid_engine.config.max_open_positions,
        'positions': positions
    })


@app.route('/live/daily', methods=['GET'])
def live_daily_stats():
    """Get today's live trading stats"""
    global trading_system
    
    if not trading_system or not trading_system.hybrid_engine:
        return jsonify({'error': 'Hybrid engine not available'}), 503
    
    return jsonify(trading_system.hybrid_engine.daily_stats.get_stats())


# ============================================================================
# BACKGROUND TASKS
# ============================================================================
def background_tasks():
    global trading_system
    
    last_discovery = datetime.now()
    last_status = datetime.now()
    last_learning = datetime.now()
    last_summary = datetime.now()
    last_live_check = datetime.now()
    
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
                
                # Add live indicator
                live_indicator = ""
                if trading_system.hybrid_engine and trading_system.hybrid_engine.config.enable_live_trading:
                    live_indicator = " | 🔴 LIVE"
                
                print(f"📡 {now.strftime('%H:%M')} | "
                      f"Webhooks: {d.webhooks_received} | "
                      f"Signals: {d.buy_signals_detected} | "
                      f"Opened: {d.positions_opened} | "
                      f"Open: {stats.get('open_positions', 0)}/{stats.get('max_positions', 5)} | "
                      f"Balance: {stats.get('balance', 0):.4f} SOL | "
                      f"WR: {stats.get('win_rate', 0):.0%}{live_indicator}")
            
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
            
            # Live position exit checks (every 30 seconds effectively via monitoring thread)
            # But we can also do a manual check here as backup
            if (now - last_live_check).total_seconds() >= 60:
                last_live_check = now
                if trading_system.hybrid_engine and trading_system.hybrid_engine.config.enable_live_trading:
                    if trading_system.hybrid_engine.live_engine:
                        try:
                            exits = trading_system.hybrid_engine.live_engine.check_exit_conditions()
                            for exit in exits:
                                if exit.get('success'):
                                    pnl = exit.get('pnl_sol', 0)
                                    is_win = pnl > 0
                                    trading_system.hybrid_engine.record_exit(True, pnl, is_win)
                                    trading_system.diagnostics.live_trades_closed += 1
                                    trading_system.diagnostics.live_pnl_sol += pnl
                                    
                                    if trading_system.notifier:
                                        emoji = "✅" if is_win else "❌"
                                        trading_system.notifier.send(
                                            f"{emoji} LIVE EXIT: {exit.get('token_symbol')}\n"
                                            f"Reason: {exit.get('exit_reason')}\n"
                                            f"PnL: {pnl:+.4f} SOL ({exit.get('pnl_pct', 0):+.1f}%)"
                                        )
                        except Exception as e:
                            print(f"Live exit check error: {e}")
        
        except Exception as e:
            print(f"Background task error: {e}")


# ============================================================================
# STARTUP
# ============================================================================
def main():
    global trading_system
    
    print("\n" + "="*70)
    print("🔧 STARTING PAPER + LIVE TRADING SYSTEM")
    print("="*70)
    
    # Update config from secrets (now that they're loaded)
    _update_config_from_secrets()
    
    try:
        trading_system = TradingSystem()
        
        bg_thread = threading.Thread(target=background_tasks, daemon=True)
        bg_thread.start()
        
        # Print endpoint info
        live_status = "🔴 LIVE ENABLED" if (trading_system.hybrid_engine and 
                                            trading_system.hybrid_engine.config.enable_live_trading) else "📝 Paper Only"
        
        print(f"\n🎧 TRADING SERVER ({live_status})")
        print(f"   Webhook:     http://{CONFIG.webhook_host}:{CONFIG.webhook_port}/webhook/helius")
        print(f"   Status:      http://{CONFIG.webhook_host}:{CONFIG.webhook_port}/status")
        print(f"   Positions:   http://{CONFIG.webhook_host}:{CONFIG.webhook_port}/positions")
        print(f"   Live Status: http://{CONFIG.webhook_host}:{CONFIG.webhook_port}/live/status")
        print(f"   Enable Live: POST /live/enable?confirm=LIVE")
        print(f"   Kill Switch: POST /live/kill?confirm=KILL")
        print(f"\n   Press Ctrl+C to stop\n")
        
        app.run(host=CONFIG.webhook_host, port=CONFIG.webhook_port, debug=False, use_reloader=False)
        
    except KeyboardInterrupt:
        print("\n\n👋 Shutting down...")
        if trading_system:
            trading_system._print_status()
            if trading_system.paper_engine:
                trading_system.paper_engine.stop()
            if trading_system.hybrid_engine and trading_system.hybrid_engine.live_engine:
                trading_system.hybrid_engine.live_engine.stop_monitoring()
    except Exception as e:
        print(f"\n💥 FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
