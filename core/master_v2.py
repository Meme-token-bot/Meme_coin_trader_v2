"""
MASTER V2 - PAPER + LIVE TRADING INTEGRATION
=============================================

PATCHED VERSION: Includes LiveTradingIntegration for automatic exit management.

Uses the fixed V6 paper trading platform with:
1. ✅ Correct balance calculations
2. ✅ Enforced position limits
3. ✅ Reliable exit monitoring
4. ✅ LIVE TRADING SUPPORT (Hybrid Engine)
5. ✅ AUTOMATIC EXIT MANAGEMENT (Stop Loss, Take Profit, Trailing Stop, Time Stop)

LIVE TRADING FEATURES:
- Jito bundle integration for fast execution
- Parallel paper/live trading for comparison
- NZ tax compliant record keeping
- Kill switch for emergencies
- AUTOMATIC EXIT MONITORING with configurable thresholds

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
from dataclasses import dataclass, field, asdict
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

# ============================================================================
# NEW: EXIT MANAGER INTEGRATION
# ============================================================================
try:
    from core.live_trading_integration import LiveTradingIntegration, IntegrationConfig
    EXIT_MANAGER_AVAILABLE = True
except ImportError:
    EXIT_MANAGER_AVAILABLE = False
    print("⚠️ Exit manager not available - exits will use hybrid engine only")


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
    use_llm: bool = False
    
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
    live_position_size_sol: float = 0.05       # Per trade (6.5% fee overhead)
    max_live_positions: int = 2               # Max concurrent live positions
    max_daily_loss_sol: float = 0.1           # Stop live trading if exceeded
    min_balance_sol: float = 2.2              # Emergency stop threshold
    blocked_hours_utc: List[int] = field(default_factory=lambda: [1, 3, 5, 19, 23])
    
    # ==========================================================================
    # NEW: EXIT MANAGER SETTINGS
    # ==========================================================================
    exit_stop_loss_pct: float = -15.0          # Exit if P&L <= -15%
    exit_take_profit_pct: float = 30.0         # Exit if P&L >= +30%
    exit_trailing_stop_pct: float = 10.0       # Exit if price drops 10% from peak
    exit_max_hold_hours: int = 12              # Exit after 12 hours
    exit_min_conviction: float = 60.0          # Filter: require conviction >= 60
    exit_min_wallet_wr: float = 0.4            # Filter: require wallet WR >= 40%
    exit_min_liquidity_usd: float = 20000.0,    # Filter: require liquidity >= $20k
    max_daily_loss_sol=0.1,    # Stop trading after 1 SOL daily loss
    
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
    # NEW: Exit manager settings from secrets
    if get_secret('EXIT_STOP_LOSS_PCT'):
        try:
            CONFIG.exit_stop_loss_pct = float(get_secret('EXIT_STOP_LOSS_PCT'))
        except:
            pass
    if get_secret('EXIT_TAKE_PROFIT_PCT'):
        try:
            CONFIG.exit_take_profit_pct = float(get_secret('EXIT_TAKE_PROFIT_PCT'))
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
        
        return len(self.minute_calls) < self.max_per_minute and len(self.hour_calls) < self.max_per_hour
    
    def record_call(self):
        now = datetime.now()
        self.minute_calls.append(now)
        self.hour_calls.append(now)


class DiagnosticsTracker:
    """Track system diagnostics"""
    def __init__(self):
        self.start_time = datetime.now()
        self.webhooks_received = 0
        self.webhooks_processed = 0
        self.webhooks_skipped = 0
        self.buy_signals_detected = 0
        self.sell_signals_detected = 0
        self.cluster_signals_detected = 0
        self.high_conviction_signals = 0
        self.api_calls_made = 0
        self.api_errors = 0
        self.token_info_failures = 0
        self.positions_opened = 0
        self.positions_closed = 0
        self.discoveries_run = 0
        self.wallets_discovered = 0
        self.learning_iterations = 0
        self.llm_calls = 0
        self.baseline_signals_recorded = 0
        self.recent_events = deque(maxlen=100)
        self.last_webhook_received = None
        
        # Skip tracking
        self.untracked_wallet_skips = 0
        self.duplicate_sig_skips = 0
        self.non_swap_skips = 0
        self.parse_failures = 0
        self.position_limit_skips = 0
        
        # Live trading
        self.live_trades_opened = 0
        self.live_trades_closed = 0
        self.live_pnl_sol = 0.0
    
    def log_event(self, event_type: str, details: str = ""):
        self.recent_events.append({
            'time': datetime.now().isoformat(),
            'type': event_type,
            'details': details
        })
    
    def to_dict(self) -> Dict:
        uptime = datetime.now() - self.start_time
        minutes_since_webhook = 999 if not self.last_webhook_received else (datetime.now() - self.last_webhook_received).total_seconds() / 60
        
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
    """Telegram notifications comparing Paper vs Live trading"""
    
    def __init__(self, token: str = None, chat_id: str = None):
        self.token = token or get_secret('TELEGRAM_BOT_TOKEN')
        self.chat_id = chat_id or get_secret('TELEGRAM_CHAT_ID')
        self.enabled = bool(self.token and self.chat_id)
        self._last_30min_update = datetime.now() - timedelta(minutes=30)
        self._last_stats = {'paper': {}, 'live': {}}
    
    def send(self, message: str):
        if not self.enabled:
            return
        try:
            import requests
            url = f"https://api.telegram.org/bot{self.token}/sendMessage"
            requests.post(url, json={'chat_id': self.chat_id, 'text': message, 'parse_mode': 'HTML'}, timeout=10)
        except Exception as e:
            print(f"  ⚠️ Telegram error: {e}")
    
    def send_critical_alert(self, message: str):
        self.send(f"🚨 <b>CRITICAL</b>\n\n{message}")
    
    def send_entry_alert(self, signal: Dict, decision: Dict, quality = None, is_live: bool = False):
        """Notify when a position is opened"""
        mode = "🔴 LIVE" if is_live else "📝 PAPER"
        token = signal.get('token_symbol', 'UNKNOWN')
        conviction = decision.get('conviction', signal.get('conviction_score', 0))
        
        msg = f"""{mode} <b>BUY</b>

Token: <b>{token}</b>
Conviction: {conviction}%"""
        
        self.send(msg)
    
    def send_live_entry(self, token_symbol: str, token_address: str, 
                        amount_sol: float, conviction: int, 
                        signature: str = None):
        """Notify live entry"""
        sig_display = signature[:16] + "..." if signature else "N/A"
        msg = f"""🔴 <b>LIVE BUY</b>

Token: <b>{token_symbol}</b>
Amount: {amount_sol:.4f} SOL
Conviction: {conviction}%
Sig: <code>{sig_display}</code>"""
        
        self.send(msg)


# ============================================================================
# FIXED PAPER TRADING ENGINE WRAPPER
# ============================================================================
class FixedPaperTradingEngine:
    """Wrapper around RobustPaperTrader for master_v2.py compatibility"""
    
    def __init__(self, db, starting_balance: float = 10.0, max_positions: int = 5):
        self.db = db
        
        # Initialize the V6 robust trader
        self._trader = RobustPaperTrader(
            db_path="robust_paper_trades_v6.db",
            starting_balance=starting_balance,
            max_open_positions=max_positions
            #enable_background_monitoring=True
        )
        
        # Quality analyzer for signal scoring
        self.quality_analyzer = SignalQualityAnalyzer()
        
        # Baseline tracker
        self.baseline_tracker = BaselineTracker("robust_paper_trades_v6.db") if CONFIG.enable_baseline_tracking else None
        
        print(f"📊 Fixed V6 Paper Trading Engine initialized")
        print(f"   Balance: {self._trader.balance:.4f} SOL")
        print(f"   Max positions: {max_positions}")
    
    def set_notifier(self, notifier):
        self._trader.set_notifier(notifier)
    
    def update_top_wallets(self, db):
        pass  # No longer needed
    
    @property
    def balance(self) -> float:
        return self._trader.balance
    
    def can_open_position(self, signal: Dict = None) -> Tuple[bool, str]:
        return self._trader.can_open_position(signal)
    
    def open_position(self, signal: Dict, decision: Dict, price: float) -> Optional[int]:
        """Open a paper position"""
        # Build entry context
        context = {
            'token_address': signal.get('token_address', signal.get('token_out', '')),
            'token_symbol': signal.get('token_symbol', 'UNKNOWN'),
            'entry_price_usd': price,
            'wallet_address': signal.get('wallet_address', signal.get('fee_payer', '')),
            'conviction_score': decision.get('conviction', signal.get('conviction_score', 50)),
            'signal_source': signal.get('signal_type', 'COPY'),
            'wallet_win_rate': signal.get('wallet_win_rate', 0.5),
            'liquidity_usd': signal.get('liquidity_usd', 0),
            'stop_loss_pct': decision.get('stop_loss', -12.0),
            'take_profit_pct': decision.get('take_profit', 30.0),
            'trailing_stop_pct': decision.get('trailing_stop', 8.0),
        }
        
        position_id = self._trader.open_position(context)
        return position_id
    
    def get_open_positions(self) -> List[Dict]:
        return self._trader.get_open_positions()
    
    def get_stats(self) -> Dict:
        if hasattr(self._trader, "get_stats"):
            return self._trader.get_stats()

        summary = (
            self._trader.get_performance_summary()
            if hasattr(self._trader, "get_performance_summary")
            else {}
        )
        return {
            'balance': summary.get('balance', getattr(self._trader, 'balance', 0)),
            'starting_balance': summary.get('starting_balance', getattr(self._trader, 'starting_balance', 0)),
            'total_pnl': summary.get('total_pnl_sol', getattr(self._trader, 'total_pnl', 0)),
            'return_pct': summary.get('return_pct', 0),
            'open_positions': summary.get('open_positions', getattr(self._trader, 'open_position_count', 0)),
            'max_positions': summary.get('max_positions', getattr(self._trader, 'max_open_positions', 0)),
            'total_trades': summary.get('total_trades', getattr(self._trader, 'total_trades', 0)),
            'win_rate': summary.get('win_rate', 0),
        }
    
    def get_full_status(self) -> Dict:
        return {
            'summary': self.get_stats(),
            'baseline': self.get_baseline_comparison() if self.baseline_tracker else {},
        }
    
    def get_baseline_comparison(self) -> Dict:
        if self.baseline_tracker:
            return self.baseline_tracker.get_comparison()
        return {}
    
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
        if not CONFIG.use_llm:
            self.strategist.reasoning_agent.enabled = False
        
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
        
        # ======================================================================
        # NEW: INITIALIZE EXIT MANAGER INTEGRATION
        # ======================================================================
        self.live_integration = None
        if EXIT_MANAGER_AVAILABLE and self.hybrid_engine and self.hybrid_engine.live_engine:
            try:
                integration_config = IntegrationConfig(
                    stop_loss_pct=CONFIG.exit_stop_loss_pct,
                    take_profit_pct=CONFIG.exit_take_profit_pct,
                    trailing_stop_pct=CONFIG.exit_trailing_stop_pct,
                    max_hold_hours=CONFIG.exit_max_hold_hours,
                    min_conviction_score=CONFIG.exit_min_conviction,
                    min_wallet_win_rate=CONFIG.exit_min_wallet_wr,
                    min_liquidity_usd=CONFIG.exit_min_liquidity_usd,
                    max_daily_loss_sol=CONFIG.max_daily_loss_sol,
                    enable_notifications=True
                )
                
                self.live_integration = LiveTradingIntegration(
                    engine=self.hybrid_engine.live_engine,
                    config=integration_config,
                    notifier=self.notifier
                )
                
                # Start background exit monitoring
                self.live_integration.start_exit_monitoring()
                
                print(f"  ✅ Exit Manager initialized")
                print(f"     Stop Loss: {CONFIG.exit_stop_loss_pct}%")
                print(f"     Take Profit: {CONFIG.exit_take_profit_pct}%")
                print(f"     Trailing: {CONFIG.exit_trailing_stop_pct}%")
                print(f"     Max Hold: {CONFIG.exit_max_hold_hours}h")
                
            except Exception as e:
                print(f"  ⚠️ Exit Manager init failed: {e}")
                import traceback
                traceback.print_exc()
                self.live_integration = None
        elif not EXIT_MANAGER_AVAILABLE:
            print(f"  ℹ️ Exit Manager not available")
        
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
        
        # NEW: Show exit manager status
        if self.live_integration:
            status = self.live_integration.get_status()
            exit_status = status.get('exit_manager', {})
            print(f"  🔄 Exit Monitor: {'RUNNING' if exit_status.get('monitor_running') else 'STOPPED'}")
            print(f"     Open positions: {exit_status.get('open_positions', 0)}")
    
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
        
        # Continue with signal processing...
        self._process_signal(trade, wallet_data)
    
    def _parse_swap(self, tx: Dict) -> Optional[Dict]:
        """Parse a transaction for swap info"""
        # Simplified - your actual implementation may differ
        try:
            token_transfers = tx.get('tokenTransfers', [])
            if not token_transfers:
                return None
            
            # Extract swap details
            return {
                'signature': tx.get('signature'),
                'fee_payer': tx.get('feePayer'),
                'token_transfers': token_transfers,
                'timestamp': tx.get('timestamp')
            }
        except:
            return None
    
    def _process_signal(self, trade: Dict, wallet_data: Dict):
        """Process a trading signal"""
        token_transfers = trade.get('token_transfers') or []
        if not token_transfers:
            self.diagnostics.parse_failures += 1
            return

        fee_payer = trade.get('fee_payer')
        chosen_transfer = None

        for transfer in token_transfers:
            mint = transfer.get('mint') or transfer.get('tokenAddress')
            if not mint:
                continue
            if self.historian and self.historian.scanner.is_ignored_token(mint):
                continue
            if transfer.get('toUserAccount') == fee_payer:
                chosen_transfer = transfer
                break

        if not chosen_transfer:
            for transfer in token_transfers:
                mint = transfer.get('mint') or transfer.get('tokenAddress')
                if mint and self.historian and self.historian.scanner.is_ignored_token(mint):
                    continue
                chosen_transfer = transfer
                break

        if not chosen_transfer:
            self.diagnostics.parse_failures += 1
            return

        token_address = chosen_transfer.get('mint') or chosen_transfer.get('tokenAddress')
        if not token_address:
            self.diagnostics.parse_failures += 1
            return

        token_info = None
        if self.historian:
            if self.rate_limiter.can_call():
                self.rate_limiter.record_call()
                self.diagnostics.api_calls_made += 1
                try:
                    token_info = self.historian.scanner.get_token_info(token_address)
                except Exception:
                    self.diagnostics.api_errors += 1
            else:
                self.diagnostics.webhooks_skipped += 1
                return

        if not token_info:
            self.diagnostics.token_info_failures += 1
            return

        token_symbol = (
            token_info.get('symbol')
            or chosen_transfer.get('tokenSymbol')
            or 'UNKNOWN'
        )
        price = token_info.get('price_usd', 0) or token_info.get('price_native', 0)
        if price <= 0:
            self.diagnostics.parse_failures += 1
            return

        signal = {
            'token_address': token_address,
            'token_symbol': token_symbol,
            'price': price,
            'price_usd': price,
            'liquidity': token_info.get('liquidity', 0),
            'volume_24h': token_info.get('volume_24h', 0),
            'token_age_hours': token_info.get('age_hours', 0),
            'wallet_address': wallet_data.get('address', fee_payer),
            'wallet_win_rate': wallet_data.get('win_rate', 0.5),
            'wallet_roi_7d': wallet_data.get('roi_7d', 0),
            'signal_type': 'COPY',
            'signature': trade.get('signature'),
            'timestamp': trade.get('timestamp'),
        }

        decision = self.strategist.analyze_signal(signal, wallet_data)
        if decision.get('llm_called'):
            self.diagnostics.llm_calls += 1

        self.diagnostics.buy_signals_detected += 1
        if decision.get('conviction_score', 0) >= 80:
            self.diagnostics.high_conviction_signals += 1

        if decision.get('should_enter'):
            result = None
            if self.hybrid_engine:
                result = self.hybrid_engine.process_signal(signal, wallet_data)
            elif self.paper_engine:
                result = self.paper_engine.process_signal(signal, wallet_data)

            paper_result = None
            if isinstance(result, dict) and 'paper_result' in result:
                paper_result = result.get('paper_result')
            elif isinstance(result, dict):
                paper_result = result

            if paper_result:
                if paper_result.get('position_id') or paper_result.get('success'):
                    self.diagnostics.positions_opened += 1
                if paper_result.get('filter_reason', '').startswith("Position"):
                    self.diagnostics.position_limit_skips += 1
                if paper_result.get('quality_metrics', {}).get('is_cluster_signal'):
                    self.diagnostics.cluster_signals_detected += 1

            live_result = result.get('live_result') if isinstance(result, dict) else None
            if live_result and live_result.get('success'):
                self.diagnostics.live_trades_opened += 1

        self.db.mark_signature_processed(
            trade.get('signature'),
            wallet=wallet_data.get('address', fee_payer),
            trade_type='BUY',
            token=token_address,
        )


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
    
    # NEW: Add exit manager info
    exit_info = {}
    if trading_system.live_integration:
        exit_status = trading_system.live_integration.get_status()
        exit_info = {
            'monitor_running': exit_status.get('exit_manager', {}).get('monitor_running', False),
            'open_positions': exit_status.get('exit_manager', {}).get('open_positions', 0)
        }
    
    return jsonify({
        'status': 'running',
        'version': 'V6_LIVE_EXIT_MANAGER',
        'uptime_hours': (datetime.now() - trading_system.start_time).total_seconds() / 3600,
        'wallets_tracked': trading_system.db.get_wallet_count(),
        'paper_trading': stats,
        'live_trading': live_info,
        'exit_manager': exit_info
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


# ============================================================================
# NEW: EXIT MANAGER ENDPOINTS
# ============================================================================

@app.route('/live/exits/status', methods=['GET'])
def exit_manager_status():
    """Get exit manager status"""
    global trading_system
    if not trading_system or not trading_system.live_integration:
        return jsonify({'error': 'Exit manager not available'}), 503
    
    return jsonify(trading_system.live_integration.get_status())


@app.route('/live/exits/positions', methods=['GET'])
def exit_manager_positions():
    """Get live positions with exit metrics"""
    global trading_system
    if not trading_system or not trading_system.live_integration:
        return jsonify({'error': 'Exit manager not available'}), 503
    
    positions = trading_system.live_integration.get_open_positions()
    return jsonify({'count': len(positions), 'positions': positions})


@app.route('/live/exits/force', methods=['POST'])
def force_exit():
    """Force exit a position"""
    global trading_system
    if not trading_system or not trading_system.live_integration:
        return jsonify({'error': 'Exit manager not available'}), 503
    
    data = request.get_json()
    token_address = data.get('token_address')
    reason = data.get('reason', 'MANUAL')
    
    if not token_address:
        return jsonify({'error': 'token_address required'}), 400
    
    result = trading_system.live_integration.force_exit(token_address, reason)
    return jsonify(result)


@app.route('/live/exits/all', methods=['POST'])
def emergency_exit_all():
    """Emergency exit all positions"""
    global trading_system
    if not trading_system or not trading_system.live_integration:
        return jsonify({'error': 'Exit manager not available'}), 503
    
    data = request.get_json() or {}
    if data.get('confirm') != 'EXIT_ALL':
        return jsonify({'error': 'Confirmation required: {"confirm": "EXIT_ALL"}'}), 400
    
    results = trading_system.live_integration.emergency_exit_all()
    return jsonify({'results': results})


@app.route('/live/kill', methods=['POST'])
def kill_switch():
    """Activate kill switch"""
    global trading_system
    
    data = request.get_json() or {}
    if data.get('confirm') != 'KILL':
        return jsonify({'error': 'Confirmation required: {"confirm": "KILL"}'}), 400
    
    # Exit all via integration if available
    if trading_system and trading_system.live_integration:
        results = trading_system.live_integration.activate_kill_switch()
        return jsonify({
            'success': True,
            'message': '🚨 KILL SWITCH ACTIVATED',
            'results': results
        })
    
    # Fallback to hybrid engine
    if trading_system and trading_system.hybrid_engine:
        trading_system.hybrid_engine._activate_kill_switch('Manual API activation')
        return jsonify({
            'success': True,
            'message': '🚨 KILL SWITCH ACTIVATED via hybrid engine'
        })
    
    return jsonify({'error': 'No live trading engine available'}), 503


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
    last_exit_check = datetime.now()  # NEW
    
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
                
                # NEW: Add exit manager indicator
                exit_indicator = ""
                if trading_system.live_integration:
                    exit_status = trading_system.live_integration.get_status()
                    open_pos = exit_status.get('exit_manager', {}).get('open_positions', 0)
                    exit_indicator = f" | 🔄 Exit({open_pos})"
                
                print(f"📡 {now.strftime('%H:%M')} | "
                      f"Webhooks: {d.webhooks_received} | "
                      f"Signals: {d.buy_signals_detected} | "
                      f"Opened: {d.positions_opened} | "
                      f"Open: {stats.get('open_positions', 0)}/{stats.get('max_positions', 5)} | "
                      f"Balance: {stats.get('balance', 0):.4f} SOL | "
                      f"WR: {stats.get('win_rate', 0):.0%}{live_indicator}{exit_indicator}")
            
            # ================================================================
            # NEW: BACKUP EXIT CHECK (every 60 seconds)
            # The exit manager runs its own background thread, but this is
            # a backup check in case the background thread fails.
            # ================================================================
            if trading_system.live_integration:
                if (now - last_exit_check).total_seconds() >= 60:
                    last_exit_check = now
                    
                    try:
                        exits = trading_system.live_integration.check_exits_now()
                        
                        for exit in exits:
                            if exit.get('success'):
                                pnl = exit.get('pnl_sol', 0)
                                is_win = pnl > 0
                                
                                trading_system.diagnostics.live_trades_closed += 1
                                trading_system.diagnostics.live_pnl_sol += pnl
                                
                                print(
                                    f"🚪 AUTO-EXIT: {exit.get('token_symbol', 'UNKNOWN')} | "
                                    f"Reason: {exit.get('exit_reason')} | "
                                    f"PnL: {pnl:+.4f} SOL ({exit.get('pnl_pct', 0):+.1f}%)"
                                )
                                
                                if trading_system.notifier:
                                    emoji = "✅" if is_win else "❌"
                                    trading_system.notifier.send(
                                        f"{emoji} LIVE EXIT: {exit.get('token_symbol')}\n"
                                        f"Reason: {exit.get('exit_reason')}\n"
                                        f"PnL: {pnl:+.4f} SOL ({exit.get('pnl_pct', 0):+.1f}%)"
                                    )
                    except Exception as e:
                        print(f"Exit check error: {e}")
            
            # Original live check (for hybrid engine without exit manager)
            elif trading_system.hybrid_engine and trading_system.hybrid_engine.live_engine:
                if (now - last_live_check).total_seconds() >= 60:
                    last_live_check = now
                    
                    try:
                        if hasattr(trading_system.hybrid_engine.live_engine, 'check_exits'):
                            exits = trading_system.hybrid_engine.live_engine.check_exits()
                            
                            for exit in exits:
                                if exit.get('success'):
                                    pnl = exit.get('pnl_sol', 0)
                                    is_win = pnl > 0
                                    
                                    trading_system.diagnostics.live_trades_closed += 1
                                    trading_system.diagnostics.live_pnl_sol += pnl
                                    
                                    print(
                                        f"🚪 LIVE EXIT: {exit.get('token_symbol', 'UNKNOWN')} | "
                                        f"Reason: {exit.get('exit_reason')} | "
                                        f"PnL: {pnl:+.4f} SOL ({exit.get('pnl_pct', 0):+.1f}%)"
                                    )
                                    
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
        
        exit_status = "✅ Exit Manager" if trading_system.live_integration else "❌ No Exit Manager"
        
        print(f"\n🎧 TRADING SERVER ({live_status} | {exit_status})")
        print(f"   Webhook:       http://{CONFIG.webhook_host}:{CONFIG.webhook_port}/webhook/helius")
        print(f"   Status:        http://{CONFIG.webhook_host}:{CONFIG.webhook_port}/status")
        print(f"   Positions:     http://{CONFIG.webhook_host}:{CONFIG.webhook_port}/positions")
        print(f"   Live Status:   http://{CONFIG.webhook_host}:{CONFIG.webhook_port}/live/exits/status")
        print(f"   Live Positions: http://{CONFIG.webhook_host}:{CONFIG.webhook_port}/live/exits/positions")
        print(f"   Force Exit:    POST /live/exits/force")
        print(f"   Exit All:      POST /live/exits/all")
        print(f"   Kill Switch:   POST /live/kill?confirm=KILL")
        print(f"\n   Press Ctrl+C to stop\n")
        
        app.run(host=CONFIG.webhook_host, port=CONFIG.webhook_port, debug=False, use_reloader=False)
        
    except KeyboardInterrupt:
        print("\n\n👋 Shutting down...")
        if trading_system:
            trading_system._print_status()
            if trading_system.paper_engine:
                trading_system.paper_engine.stop()
            # NEW: Stop exit monitoring
            if trading_system.live_integration:
                trading_system.live_integration.stop_exit_monitoring()
                print("✅ Exit monitoring stopped")
            if trading_system.hybrid_engine and trading_system.hybrid_engine.live_engine:
                trading_system.hybrid_engine.live_engine.stop_monitoring()
    except Exception as e:
        print(f"\n💥 FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
