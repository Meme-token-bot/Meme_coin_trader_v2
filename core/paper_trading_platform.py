"""
ROBUST PAPER TRADING PLATFORM V5
================================

A comprehensive paper trading system designed for strategy development.

FEATURES:
1. Exit Monitoring Reliability - Watchdog, heartbeats, alerts
2. Baseline Comparison - Track filtered vs unfiltered performance
3. Learning System Validation - A/B testing of parameters
4. Historical Backtesting - Store all signals, replay capability
5. Signal Quality Metrics - Cluster detection, sizing, token age
6. Dynamic Exit Parameters - Volatility-adjusted stops/TPs

Author: Claude
Version: 5.0
"""

import os
import json
import sqlite3
import threading
import time
import requests
import statistics
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field, asdict
from contextlib import contextmanager
from collections import defaultdict
from enum import Enum
from abc import ABC, abstractmethod
import logging
import hashlib
import random

# Configure logging - WARNING by default (quieter), can be changed via set_verbosity()
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger(__name__)


def set_platform_verbosity(level: int):
    """
    Set the verbosity level for the paper trading platform.
    
    0 = quiet (WARNING only)
    1 = normal (INFO)
    2 = verbose (DEBUG)
    """
    if level >= 2:
        logger.setLevel(logging.DEBUG)
    elif level >= 1:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.WARNING)


# =============================================================================
# ENUMS AND DATA CLASSES
# =============================================================================

class ExitReason(Enum):
    STOP_LOSS = "STOP_LOSS"
    TAKE_PROFIT = "TAKE_PROFIT"
    TRAILING_STOP = "TRAILING_STOP"
    TIME_STOP = "TIME_STOP"
    SMART_EXIT = "SMART_EXIT"
    MANUAL = "MANUAL"
    RUG_DETECTED = "RUG_DETECTED"
    WATCHDOG = "WATCHDOG"


class TradeStatus(Enum):
    OPEN = "open"
    CLOSED = "closed"
    STOPPED = "stopped"
    EXPIRED = "expired"


class LearningPhase(Enum):
    EXPLORATION = "exploration"
    INITIAL_FILTERING = "initial"
    REFINEMENT = "refinement"
    OPTIMIZATION = "optimization"
    PRODUCTION = "production"


class SignalType(Enum):
    """Types of signals for baseline comparison"""
    FILTERED = "filtered"      # Passed our strategy filters
    UNFILTERED = "unfiltered"  # Raw signal, no filtering
    BASELINE_A = "baseline_a"  # A/B test variant A
    BASELINE_B = "baseline_b"  # A/B test variant B


@dataclass
class SignalQualityMetrics:
    """
    Enhanced signal quality metrics (Improvement #5)
    """
    # Cluster detection
    concurrent_wallet_count: int = 1          # How many wallets buying simultaneously
    cluster_wallets: List[str] = field(default_factory=list)
    is_cluster_signal: bool = False           # 2+ wallets = cluster
    cluster_strength: float = 0.0             # Weighted by wallet quality
    
    # Position sizing intelligence
    wallet_typical_size_sol: float = 0.0      # This wallet's average position
    wallet_current_size_sol: float = 0.0      # Size of this specific buy
    size_multiplier: float = 1.0              # Current/typical ratio
    is_oversized_entry: bool = False          # Wallet betting big
    
    # Token timing
    token_age_minutes: float = 0.0            # How old when bought
    is_early_entry: bool = False              # < 30 min old
    is_late_entry: bool = False               # > 4 hours old
    
    # Wallet behavior
    wallet_avg_hold_minutes: float = 0.0      # How long wallet typically holds
    wallet_win_streak: int = 0                # Current winning streak
    wallet_recent_roi: float = 0.0            # Last 7 days ROI
    
    # Market context
    sol_price_1h_change: float = 0.0          # SOL momentum
    market_fear_greed: float = 50.0           # Market sentiment (0-100)
    
    def calculate_composite_score(self) -> float:
        """Calculate overall signal quality score (0-100)"""
        score = 50.0  # Base score
        
        # Cluster bonus (up to +20)
        if self.is_cluster_signal:
            score += min(20, self.concurrent_wallet_count * 5)
        
        # Size multiplier bonus (up to +15)
        if self.size_multiplier > 1.5:
            score += min(15, (self.size_multiplier - 1) * 10)
        
        # Early entry bonus (+10)
        if self.is_early_entry:
            score += 10
        elif self.is_late_entry:
            score -= 10
        
        # Wallet streak bonus (up to +10)
        if self.wallet_win_streak > 0:
            score += min(10, self.wallet_win_streak * 2)
        
        return max(0, min(100, score))


@dataclass
class DynamicExitParams:
    """
    Dynamic exit parameters based on signal quality (Improvement #6)
    """
    stop_loss_pct: float = -15.0
    take_profit_pct: float = 30.0
    trailing_stop_pct: float = 10.0
    max_hold_hours: int = 12
    
    # Trailing activation
    trailing_activation_pct: float = 15.0     # Only trail after this profit
    
    # Scaling parameters
    scale_out_enabled: bool = False
    scale_out_at_pct: float = 20.0            # Take partial at 20%
    scale_out_amount: float = 0.5             # Take 50% off
    
    @classmethod
    def from_signal_quality(cls, quality: SignalQualityMetrics, 
                           conviction: float = 50.0) -> 'DynamicExitParams':
        """
        Calculate dynamic exits based on signal quality.
        
        High conviction = tighter stops, higher targets
        Low conviction = wider stops, lower targets
        Cluster signals = more room to run
        """
        params = cls()
        
        # Base adjustments on conviction (0-100)
        conviction_factor = conviction / 50.0  # 1.0 at 50 conviction
        
        # High conviction: tighter stop, higher target
        if conviction >= 70:
            params.stop_loss_pct = -12.0
            params.take_profit_pct = 40.0
            params.trailing_stop_pct = 8.0
            params.max_hold_hours = 8
        elif conviction >= 50:
            params.stop_loss_pct = -15.0
            params.take_profit_pct = 30.0
            params.trailing_stop_pct = 10.0
            params.max_hold_hours = 12
        else:
            # Low conviction: wider stop, lower target, quick exit
            params.stop_loss_pct = -20.0
            params.take_profit_pct = 20.0
            params.trailing_stop_pct = 12.0
            params.max_hold_hours = 6
        
        # Cluster signal adjustment: let winners run
        if quality.is_cluster_signal:
            params.take_profit_pct *= 1.25
            params.trailing_activation_pct = 20.0
            params.scale_out_enabled = True
        
        # Early entry adjustment: more volatile, need wider stops
        if quality.is_early_entry:
            params.stop_loss_pct *= 1.2  # 20% wider stop
            params.take_profit_pct *= 1.3  # 30% higher target
        
        # Oversized wallet entry: they know something
        if quality.is_oversized_entry:
            params.take_profit_pct *= 1.2
            params.max_hold_hours = min(params.max_hold_hours + 4, 24)
        
        return params


@dataclass
class HistoricalSignal:
    """
    Stored signal for backtesting (Improvement #4)
    """
    id: str = ""
    timestamp: str = ""
    
    # Signal data
    token_address: str = ""
    token_symbol: str = ""
    price_at_signal: float = 0.0
    
    # Wallet data
    wallet_address: str = ""
    wallet_win_rate: float = 0.0
    wallet_cluster: str = "UNKNOWN"
    
    # Token metrics at signal time
    liquidity_usd: float = 0.0
    volume_24h: float = 0.0
    market_cap: float = 0.0
    token_age_minutes: float = 0.0
    holder_count: int = 0
    
    # Quality metrics
    quality_metrics: Dict = field(default_factory=dict)
    
    # Price history after signal (for backtesting)
    price_1m: float = 0.0
    price_5m: float = 0.0
    price_15m: float = 0.0
    price_30m: float = 0.0
    price_1h: float = 0.0
    price_2h: float = 0.0
    price_4h: float = 0.0
    price_12h: float = 0.0
    price_24h: float = 0.0
    
    # What our strategy decided
    strategy_decision: str = ""  # "ENTER", "SKIP", "FILTERED"
    filter_reason: str = ""
    
    # Actual outcome (if we entered)
    actual_entry: bool = False
    actual_exit_price: float = 0.0
    actual_exit_reason: str = ""
    actual_pnl_pct: float = 0.0


@dataclass
class ABTestConfig:
    """
    A/B testing configuration (Improvement #3)
    """
    test_id: str = ""
    test_name: str = ""
    started_at: str = ""
    
    # Variant A parameters (control)
    variant_a_params: Dict = field(default_factory=dict)
    
    # Variant B parameters (test)
    variant_b_params: Dict = field(default_factory=dict)
    
    # Traffic split (0.5 = 50/50)
    traffic_split: float = 0.5
    
    # Minimum samples before evaluating
    min_samples_per_variant: int = 30
    
    # Test status
    is_active: bool = True
    winner: str = ""  # "A", "B", or ""


# =============================================================================
# IMPROVEMENT #1: EXIT MONITORING RELIABILITY
# =============================================================================

class ExitMonitorWatchdog:
    """
    Watchdog that ensures exit monitoring is always running.
    
    Features:
    - Heartbeat logging every 5 minutes
    - Telegram alerts when positions exceed max hold
    - Automatic restart of monitor thread if it dies
    - Backup force-close of stale positions
    """
    
    def __init__(self, 
                 paper_trader: 'RobustPaperTrader',
                 telegram_token: str = None,
                 telegram_chat_id: str = None,
                 heartbeat_interval: int = 300,  # 5 minutes
                 stale_position_hours: float = 24.0):
        
        self.trader = paper_trader
        self.telegram_token = telegram_token or os.getenv('TELEGRAM_BOT_TOKEN')
        self.telegram_chat_id = telegram_chat_id or os.getenv('TELEGRAM_CHAT_ID')
        self.heartbeat_interval = heartbeat_interval
        self.stale_position_hours = stale_position_hours
        
        self._running = False
        self._thread = None
        self._last_heartbeat = datetime.utcnow()
        self._monitor_thread_healthy = True
        
        # Track alerts sent (avoid spam)
        self._alerts_sent: Dict[int, datetime] = {}
        self._alert_cooldown_minutes = 30
        
        logger.info("🐕 Watchdog initialized")
    
    def start(self):
        """Start the watchdog"""
        if self._running:
            return
        
        self._running = True
        self._thread = threading.Thread(target=self._watchdog_loop, daemon=True)
        self._thread.start()
        logger.info("🐕 Watchdog started - monitoring exit system health")
    
    def stop(self):
        """Stop the watchdog"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("🐕 Watchdog stopped")
    
    def _watchdog_loop(self):
        """Main watchdog loop"""
        while self._running:
            try:
                self._check_health()
                self._check_stale_positions()
                self._log_heartbeat()
            except Exception as e:
                logger.error(f"🐕 Watchdog error: {e}")
            
            time.sleep(60)  # Check every minute
    
    def _check_health(self):
        """Check if the main exit monitor is healthy"""
        # Check if monitor thread is alive
        if hasattr(self.trader, '_monitor_thread'):
            if self.trader._monitor_thread and not self.trader._monitor_thread.is_alive():
                logger.warning("🐕 Exit monitor thread DEAD - restarting!")
                self._send_alert("🚨 Exit monitor died - restarting...")
                self.trader.start_monitor()
                self._monitor_thread_healthy = False
            else:
                if not self._monitor_thread_healthy:
                    logger.info("🐕 Exit monitor recovered")
                    self._send_alert("✅ Exit monitor recovered")
                self._monitor_thread_healthy = True
    
    def _check_stale_positions(self):
        """Check for positions that have exceeded max hold time"""
        positions = self.trader.get_open_positions()
        now = datetime.utcnow()
        
        for pos in positions:
            position_id = pos['id']
            entry_time = pos.get('entry_time')
            max_hold = pos.get('max_hold_hours', 12)
            
            if entry_time:
                if isinstance(entry_time, str):
                    entry_time = datetime.fromisoformat(entry_time.replace('Z', ''))
                
                hold_hours = (now - entry_time).total_seconds() / 3600
                
                # Alert if over max hold
                if hold_hours > max_hold:
                    self._handle_stale_position(pos, hold_hours)
                
                # Force close if WAY over (2x max hold)
                if hold_hours > max_hold * 2:
                    self._force_close_position(pos, hold_hours)
    
    def _handle_stale_position(self, pos: Dict, hold_hours: float):
        """Handle a position that exceeded max hold time"""
        position_id = pos['id']
        symbol = pos.get('token_symbol', 'UNKNOWN')
        max_hold = pos.get('max_hold_hours', 12)
        
        # Check if we already alerted recently
        last_alert = self._alerts_sent.get(position_id)
        if last_alert:
            if (datetime.utcnow() - last_alert).seconds < self._alert_cooldown_minutes * 60:
                return
        
        msg = (f"⚠️ Position #{position_id} ({symbol}) exceeded max hold!\n"
               f"Hold time: {hold_hours:.1f}h (max: {max_hold}h)\n"
               f"Exit monitor may not be working!")
        
        logger.warning(f"🐕 {msg}")
        self._send_alert(msg)
        self._alerts_sent[position_id] = datetime.utcnow()
    
    def _force_close_position(self, pos: Dict, hold_hours: float):
        """Force close a position that's way overdue"""
        position_id = pos['id']
        symbol = pos.get('token_symbol', 'UNKNOWN')
        token_address = pos.get('token_address', '')
        
        logger.warning(f"🐕 FORCE CLOSING position #{position_id} ({symbol}) - {hold_hours:.1f}h old!")
        
        # Get current price
        try:
            current_price = self._get_token_price(token_address)
            if current_price > 0:
                self.trader.close_position(position_id, current_price, ExitReason.WATCHDOG)
                self._send_alert(f"🐕 Watchdog FORCE CLOSED #{position_id} ({symbol}) at {hold_hours:.1f}h")
        except Exception as e:
            logger.error(f"🐕 Failed to force close: {e}")
    
    def _log_heartbeat(self):
        """Log periodic heartbeat"""
        now = datetime.utcnow()
        if (now - self._last_heartbeat).seconds >= self.heartbeat_interval:
            open_count = len(self.trader.get_open_positions())
            stats = self.trader.get_performance_summary()
            
            logger.info(f"💓 HEARTBEAT | Open: {open_count} | "
                       f"Trades: {stats.get('total_trades', 0)} | "
                       f"WR: {stats.get('win_rate', 0):.0%} | "
                       f"PnL: {stats.get('total_pnl_sol', 0):+.4f} SOL")
            
            self._last_heartbeat = now
    
    def _get_token_price(self, token_address: str) -> float:
        """Get current token price"""
        try:
            url = f"https://api.dexscreener.com/latest/dex/tokens/{token_address}"
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                pairs = data.get('pairs', [])
                if pairs:
                    return float(pairs[0].get('priceUsd', 0))
        except:
            pass
        return 0
    
    def _send_alert(self, message: str):
        """Send Telegram alert"""
        if not self.telegram_token or not self.telegram_chat_id:
            return
        
        try:
            url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"
            requests.post(url, json={
                'chat_id': self.telegram_chat_id,
                'text': message,
                'parse_mode': 'HTML'
            }, timeout=10)
        except Exception as e:
            logger.error(f"Failed to send Telegram alert: {e}")


# =============================================================================
# IMPROVEMENT #2: BASELINE COMPARISON
# =============================================================================

class BaselineTracker:
    """
    Tracks performance of different strategies for comparison.
    
    Baselines:
    1. UNFILTERED: What if we entered EVERY signal?
    2. FILTERED: Our actual strategy
    3. TOP_WALLETS: Only follow top 10 wallets by WR
    4. CLUSTER_ONLY: Only enter on cluster signals
    """
    
    def __init__(self, db_path: str = "baseline_tracking.db"):
        self.db_path = db_path
        self._init_database()
        
        # In-memory tracking for quick comparisons
        self._baseline_positions: Dict[str, List[Dict]] = defaultdict(list)
        
        logger.info("📊 Baseline tracker initialized")
    
    def _init_database(self):
        """Initialize baseline tracking database"""
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS baseline_signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    signal_id TEXT UNIQUE,
                    timestamp TIMESTAMP,
                    token_address TEXT,
                    token_symbol TEXT,
                    entry_price REAL,
                    
                    -- Signal metadata
                    wallet_address TEXT,
                    wallet_win_rate REAL,
                    liquidity_usd REAL,
                    quality_score REAL,
                    is_cluster_signal BOOLEAN,
                    
                    -- What each baseline would do
                    unfiltered_action TEXT DEFAULT 'ENTER',
                    filtered_action TEXT,
                    top_wallets_action TEXT,
                    cluster_only_action TEXT,
                    
                    -- Outcomes (updated later)
                    price_peak REAL,
                    price_1h REAL,
                    price_12h REAL,
                    
                    -- Simulated results per baseline
                    unfiltered_pnl_pct REAL,
                    filtered_pnl_pct REAL,
                    top_wallets_pnl_pct REAL,
                    cluster_only_pnl_pct REAL,
                    
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS baseline_summary (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    baseline_name TEXT,
                    period TEXT,  -- 'daily', 'weekly', 'all_time'
                    period_start TIMESTAMP,
                    
                    total_signals INTEGER,
                    entered_count INTEGER,
                    win_count INTEGER,
                    total_pnl_pct REAL,
                    avg_pnl_pct REAL,
                    win_rate REAL,
                    
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(baseline_name, period, period_start)
                )
            """)
    
    @contextmanager
    def _get_connection(self):
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()
    
    def record_signal(self, signal: Dict, quality: SignalQualityMetrics,
                     strategy_decision: str, top_wallets: List[str] = None):
        """
        Record a signal and what each baseline would do.
        
        Args:
            signal: The trading signal
            quality: Quality metrics for the signal
            strategy_decision: What our strategy decided ("ENTER" or "SKIP")
            top_wallets: List of top wallet addresses (for top_wallets baseline)
        """
        signal_id = hashlib.md5(
            f"{signal.get('token_address', '')}{signal.get('wallet', '')}{datetime.utcnow().isoformat()}".encode()
        ).hexdigest()[:16]
        
        wallet = signal.get('wallet', signal.get('wallet_address', ''))
        
        # Determine what each baseline would do
        unfiltered_action = "ENTER"  # Always enters
        filtered_action = strategy_decision
        top_wallets_action = "ENTER" if wallet in (top_wallets or []) else "SKIP"
        cluster_only_action = "ENTER" if quality.is_cluster_signal else "SKIP"
        
        with self._get_connection() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO baseline_signals
                (signal_id, timestamp, token_address, token_symbol, entry_price,
                 wallet_address, wallet_win_rate, liquidity_usd, quality_score, is_cluster_signal,
                 unfiltered_action, filtered_action, top_wallets_action, cluster_only_action)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                signal_id, datetime.utcnow(), signal.get('token_address', ''),
                signal.get('token_symbol', 'UNKNOWN'), signal.get('price', 0),
                wallet, signal.get('wallet_win_rate', 0),
                signal.get('liquidity', 0), quality.calculate_composite_score(),
                quality.is_cluster_signal,
                unfiltered_action, filtered_action, top_wallets_action, cluster_only_action
            ))
        
        return signal_id
    
    def update_outcomes(self, signal_id: str, price_peak: float, 
                       price_1h: float, price_12h: float):
        """Update signal with actual price outcomes"""
        with self._get_connection() as conn:
            # Get original entry price
            row = conn.execute(
                "SELECT entry_price FROM baseline_signals WHERE signal_id = ?",
                (signal_id,)
            ).fetchone()
            
            if not row:
                return
            
            entry_price = row['entry_price']
            if entry_price <= 0:
                return
            
            # Calculate simulated PnL for each baseline
            # Assume: exit at 12h or if hit +30% TP or -15% SL
            
            def simulate_pnl(action: str) -> float:
                if action != "ENTER":
                    return 0.0
                
                peak_pnl = ((price_peak / entry_price) - 1) * 100
                final_pnl = ((price_12h / entry_price) - 1) * 100
                
                # Check if hit TP or SL
                if peak_pnl >= 30:
                    return 30.0  # Hit take profit
                elif final_pnl <= -15:
                    return -15.0  # Hit stop loss
                else:
                    return final_pnl  # Time exit
            
            # Get actions
            actions = conn.execute(
                """SELECT unfiltered_action, filtered_action, 
                          top_wallets_action, cluster_only_action 
                   FROM baseline_signals WHERE signal_id = ?""",
                (signal_id,)
            ).fetchone()
            
            if actions:
                conn.execute("""
                    UPDATE baseline_signals
                    SET price_peak = ?, price_1h = ?, price_12h = ?,
                        unfiltered_pnl_pct = ?, filtered_pnl_pct = ?,
                        top_wallets_pnl_pct = ?, cluster_only_pnl_pct = ?
                    WHERE signal_id = ?
                """, (
                    price_peak, price_1h, price_12h,
                    simulate_pnl(actions['unfiltered_action']),
                    simulate_pnl(actions['filtered_action']),
                    simulate_pnl(actions['top_wallets_action']),
                    simulate_pnl(actions['cluster_only_action']),
                    signal_id
                ))
    
    def get_comparison_report(self, days: int = 7) -> Dict:
        """Get comparison report across all baselines"""
        cutoff = datetime.utcnow() - timedelta(days=days)
        
        with self._get_connection() as conn:
            report = {}
            
            for baseline in ['unfiltered', 'filtered', 'top_wallets', 'cluster_only']:
                action_col = f"{baseline}_action"
                pnl_col = f"{baseline}_pnl_pct"
                
                row = conn.execute(f"""
                    SELECT 
                        COUNT(*) as total_signals,
                        SUM(CASE WHEN {action_col} = 'ENTER' THEN 1 ELSE 0 END) as entered,
                        SUM(CASE WHEN {action_col} = 'ENTER' AND {pnl_col} > 0 THEN 1 ELSE 0 END) as wins,
                        SUM(CASE WHEN {action_col} = 'ENTER' THEN {pnl_col} ELSE 0 END) as total_pnl,
                        AVG(CASE WHEN {action_col} = 'ENTER' THEN {pnl_col} END) as avg_pnl
                    FROM baseline_signals
                    WHERE timestamp > ? AND {pnl_col} IS NOT NULL
                """, (cutoff,)).fetchone()
                
                if row and row['entered']:
                    report[baseline] = {
                        'total_signals': row['total_signals'],
                        'entered': row['entered'],
                        'wins': row['wins'] or 0,
                        'win_rate': (row['wins'] or 0) / row['entered'] if row['entered'] > 0 else 0,
                        'total_pnl_pct': row['total_pnl'] or 0,
                        'avg_pnl_pct': row['avg_pnl'] or 0
                    }
                else:
                    report[baseline] = {
                        'total_signals': 0, 'entered': 0, 'wins': 0,
                        'win_rate': 0, 'total_pnl_pct': 0, 'avg_pnl_pct': 0
                    }
            
            return report
    
    def print_comparison(self, days: int = 7):
        """Print baseline comparison report"""
        report = self.get_comparison_report(days)
        
        print("\n" + "=" * 70)
        print(f"📊 BASELINE COMPARISON REPORT (Last {days} days)")
        print("=" * 70)
        
        # Header
        print(f"\n{'Baseline':<15} {'Signals':<10} {'Entered':<10} {'Win Rate':<10} {'Avg PnL':<10} {'Total PnL':<10}")
        print("-" * 65)
        
        for baseline, data in report.items():
            print(f"{baseline:<15} {data['total_signals']:<10} {data['entered']:<10} "
                  f"{data['win_rate']:.0%}       {data['avg_pnl_pct']:+.1f}%     {data['total_pnl_pct']:+.1f}%")
        
        # Find best baseline
        best = max(report.items(), key=lambda x: x[1].get('total_pnl_pct', 0))
        print(f"\n🏆 Best performing: {best[0].upper()} with {best[1]['total_pnl_pct']:+.1f}% total PnL")
        print("=" * 70)


# =============================================================================
# IMPROVEMENT #3: A/B TESTING FRAMEWORK
# =============================================================================

class ABTestingFramework:
    """
    A/B testing framework for validating strategy changes.
    
    Usage:
        ab = ABTestingFramework()
        test_id = ab.create_test(
            name="Liquidity Threshold Test",
            variant_a={'min_liquidity': 10000},
            variant_b={'min_liquidity': 5000}
        )
        
        # For each signal:
        variant = ab.get_variant(test_id, signal)
        # Use variant params...
        
        # Record outcome:
        ab.record_outcome(test_id, variant, pnl_pct)
        
        # Check results:
        ab.evaluate_test(test_id)
    """
    
    def __init__(self, db_path: str = "ab_testing.db"):
        self.db_path = db_path
        self._active_tests: Dict[str, ABTestConfig] = {}
        self._init_database()
        self._load_active_tests()
        
        logger.info("🧪 A/B Testing Framework initialized")
    
    def _init_database(self):
        """Initialize A/B testing database"""
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS ab_tests (
                    id TEXT PRIMARY KEY,
                    name TEXT,
                    started_at TIMESTAMP,
                    variant_a_params TEXT,
                    variant_b_params TEXT,
                    traffic_split REAL DEFAULT 0.5,
                    min_samples INTEGER DEFAULT 30,
                    is_active BOOLEAN DEFAULT TRUE,
                    winner TEXT,
                    ended_at TIMESTAMP
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS ab_outcomes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    test_id TEXT,
                    variant TEXT,
                    signal_id TEXT,
                    timestamp TIMESTAMP,
                    pnl_pct REAL,
                    hold_minutes REAL,
                    exit_reason TEXT,
                    FOREIGN KEY (test_id) REFERENCES ab_tests(id)
                )
            """)
    
    @contextmanager
    def _get_connection(self):
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()
    
    def _load_active_tests(self):
        """Load active tests from database"""
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM ab_tests WHERE is_active = TRUE"
            ).fetchall()
            
            for row in rows:
                test = ABTestConfig(
                    test_id=row['id'],
                    test_name=row['name'],
                    started_at=row['started_at'],
                    variant_a_params=json.loads(row['variant_a_params']),
                    variant_b_params=json.loads(row['variant_b_params']),
                    traffic_split=row['traffic_split'],
                    min_samples_per_variant=row['min_samples'],
                    is_active=True
                )
                self._active_tests[test.test_id] = test
    
    def create_test(self, name: str, variant_a: Dict, variant_b: Dict,
                   traffic_split: float = 0.5, min_samples: int = 30) -> str:
        """
        Create a new A/B test.
        
        Args:
            name: Human-readable test name
            variant_a: Parameters for variant A (control)
            variant_b: Parameters for variant B (test)
            traffic_split: Fraction of traffic to variant B (0.5 = 50/50)
            min_samples: Minimum samples before declaring winner
            
        Returns:
            test_id
        """
        test_id = hashlib.md5(f"{name}{datetime.utcnow().isoformat()}".encode()).hexdigest()[:12]
        
        test = ABTestConfig(
            test_id=test_id,
            test_name=name,
            started_at=datetime.utcnow().isoformat(),
            variant_a_params=variant_a,
            variant_b_params=variant_b,
            traffic_split=traffic_split,
            min_samples_per_variant=min_samples
        )
        
        with self._get_connection() as conn:
            conn.execute("""
                INSERT INTO ab_tests (id, name, started_at, variant_a_params, 
                                      variant_b_params, traffic_split, min_samples)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (test_id, name, test.started_at,
                  json.dumps(variant_a), json.dumps(variant_b),
                  traffic_split, min_samples))
        
        self._active_tests[test_id] = test
        
        logger.info(f"🧪 Created A/B test: {name} (ID: {test_id})")
        logger.info(f"   Variant A: {variant_a}")
        logger.info(f"   Variant B: {variant_b}")
        
        return test_id
    
    def get_variant(self, test_id: str, signal: Dict = None) -> Tuple[str, Dict]:
        """
        Get which variant to use for a signal.
        
        Uses deterministic assignment based on signal to ensure consistency.
        
        Returns:
            (variant_name, variant_params)
        """
        test = self._active_tests.get(test_id)
        if not test or not test.is_active:
            return "A", {}
        
        # Deterministic assignment based on signal hash
        if signal:
            signal_hash = hashlib.md5(
                f"{signal.get('token_address', '')}{signal.get('wallet', '')}".encode()
            ).hexdigest()
            assignment_value = int(signal_hash[:8], 16) / 0xFFFFFFFF
        else:
            assignment_value = random.random()
        
        if assignment_value < test.traffic_split:
            return "B", test.variant_b_params
        else:
            return "A", test.variant_a_params
    
    def record_outcome(self, test_id: str, variant: str, signal_id: str,
                      pnl_pct: float, hold_minutes: float = 0,
                      exit_reason: str = ""):
        """Record the outcome of a trade in an A/B test"""
        with self._get_connection() as conn:
            conn.execute("""
                INSERT INTO ab_outcomes (test_id, variant, signal_id, timestamp,
                                        pnl_pct, hold_minutes, exit_reason)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (test_id, variant, signal_id, datetime.utcnow(),
                  pnl_pct, hold_minutes, exit_reason))
    
    def evaluate_test(self, test_id: str) -> Dict:
        """
        Evaluate an A/B test and determine winner.
        
        Uses simple comparison of average PnL with sufficient samples.
        """
        with self._get_connection() as conn:
            results = {}
            
            for variant in ['A', 'B']:
                row = conn.execute("""
                    SELECT COUNT(*) as count,
                           SUM(CASE WHEN pnl_pct > 0 THEN 1 ELSE 0 END) as wins,
                           AVG(pnl_pct) as avg_pnl,
                           SUM(pnl_pct) as total_pnl
                    FROM ab_outcomes
                    WHERE test_id = ? AND variant = ?
                """, (test_id, variant)).fetchone()
                
                results[variant] = {
                    'count': row['count'] or 0,
                    'wins': row['wins'] or 0,
                    'win_rate': (row['wins'] or 0) / row['count'] if row['count'] else 0,
                    'avg_pnl': row['avg_pnl'] or 0,
                    'total_pnl': row['total_pnl'] or 0
                }
            
            # Check if we have enough samples
            test = self._active_tests.get(test_id)
            min_samples = test.min_samples_per_variant if test else 30
            
            has_enough_samples = (
                results['A']['count'] >= min_samples and
                results['B']['count'] >= min_samples
            )
            
            # Determine winner
            winner = ""
            confidence = "low"
            
            if has_enough_samples:
                diff = results['B']['avg_pnl'] - results['A']['avg_pnl']
                
                if abs(diff) > 2.0:  # >2% difference
                    winner = "B" if diff > 0 else "A"
                    confidence = "high" if abs(diff) > 5.0 else "medium"
            
            return {
                'test_id': test_id,
                'results': results,
                'has_enough_samples': has_enough_samples,
                'winner': winner,
                'confidence': confidence
            }
    
    def conclude_test(self, test_id: str, winner: str):
        """Mark a test as concluded with a winner"""
        with self._get_connection() as conn:
            conn.execute("""
                UPDATE ab_tests
                SET is_active = FALSE, winner = ?, ended_at = ?
                WHERE id = ?
            """, (winner, datetime.utcnow(), test_id))
        
        if test_id in self._active_tests:
            self._active_tests[test_id].is_active = False
            self._active_tests[test_id].winner = winner
        
        logger.info(f"🧪 A/B test {test_id} concluded - Winner: Variant {winner}")
    
    def print_test_status(self, test_id: str):
        """Print current test status"""
        evaluation = self.evaluate_test(test_id)
        test = self._active_tests.get(test_id)
        
        print(f"\n🧪 A/B TEST: {test.test_name if test else test_id}")
        print("=" * 50)
        print(f"Variant A: {test.variant_a_params if test else 'N/A'}")
        print(f"Variant B: {test.variant_b_params if test else 'N/A'}")
        print()
        
        for variant in ['A', 'B']:
            data = evaluation['results'][variant]
            print(f"Variant {variant}:")
            print(f"  Samples: {data['count']}")
            print(f"  Win Rate: {data['win_rate']:.0%}")
            print(f"  Avg PnL: {data['avg_pnl']:+.2f}%")
            print(f"  Total PnL: {data['total_pnl']:+.2f}%")
            print()
        
        if evaluation['winner']:
            print(f"🏆 Current Winner: Variant {evaluation['winner']} ({evaluation['confidence']} confidence)")
        else:
            print("⏳ Not enough data to determine winner yet")


# =============================================================================
# IMPROVEMENT #4: HISTORICAL BACKTESTING
# =============================================================================

class HistoricalDataStore:
    """
    Stores all signals and price history for backtesting.
    
    Every signal is stored with:
    - Full signal data at time of signal
    - Price snapshots at 1m, 5m, 15m, 30m, 1h, 2h, 4h, 12h, 24h
    - What our strategy decided
    - Actual outcome if we entered
    """
    
    def __init__(self, db_path: str = "historical_signals.db"):
        self.db_path = db_path
        self._price_update_queue: List[Dict] = []
        self._init_database()
        
        # Start background price updater
        self._updater_running = False
        self._updater_thread = None
        
        logger.info("📚 Historical data store initialized")
    
    def _init_database(self):
        """Initialize historical data database"""
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS historical_signals (
                    id TEXT PRIMARY KEY,
                    timestamp TIMESTAMP,
                    
                    -- Signal data
                    token_address TEXT,
                    token_symbol TEXT,
                    price_at_signal REAL,
                    
                    -- Wallet data
                    wallet_address TEXT,
                    wallet_win_rate REAL,
                    wallet_cluster TEXT,
                    
                    -- Token metrics
                    liquidity_usd REAL,
                    volume_24h REAL,
                    market_cap REAL,
                    token_age_minutes REAL,
                    holder_count INTEGER,
                    
                    -- Quality metrics (JSON)
                    quality_metrics TEXT,
                    
                    -- Price history (updated over time)
                    price_1m REAL,
                    price_5m REAL,
                    price_15m REAL,
                    price_30m REAL,
                    price_1h REAL,
                    price_2h REAL,
                    price_4h REAL,
                    price_12h REAL,
                    price_24h REAL,
                    price_peak REAL,
                    price_trough REAL,
                    
                    -- Strategy decision
                    strategy_decision TEXT,
                    filter_reason TEXT,
                    
                    -- Actual outcome
                    actual_entry BOOLEAN DEFAULT FALSE,
                    actual_exit_price REAL,
                    actual_exit_reason TEXT,
                    actual_pnl_pct REAL,
                    actual_hold_minutes REAL,
                    
                    -- Metadata
                    price_updates_complete BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_signals_timestamp 
                ON historical_signals(timestamp)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_signals_token 
                ON historical_signals(token_address)
            """)
    
    @contextmanager
    def _get_connection(self):
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()
    
    def store_signal(self, signal: Dict, quality: SignalQualityMetrics,
                    strategy_decision: str, filter_reason: str = "") -> str:
        """
        Store a signal for future backtesting.
        
        Returns signal_id for tracking.
        """
        signal_id = hashlib.md5(
            f"{signal.get('token_address', '')}{signal.get('wallet', '')}"
            f"{datetime.utcnow().isoformat()}".encode()
        ).hexdigest()[:16]
        
        with self._get_connection() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO historical_signals
                (id, timestamp, token_address, token_symbol, price_at_signal,
                 wallet_address, wallet_win_rate, wallet_cluster,
                 liquidity_usd, volume_24h, market_cap, token_age_minutes, holder_count,
                 quality_metrics, strategy_decision, filter_reason)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                signal_id, datetime.utcnow(),
                signal.get('token_address', ''), signal.get('token_symbol', 'UNKNOWN'),
                signal.get('price', 0),
                signal.get('wallet', signal.get('wallet_address', '')),
                signal.get('wallet_win_rate', 0), signal.get('wallet_cluster', 'UNKNOWN'),
                signal.get('liquidity', 0), signal.get('volume_24h', 0),
                signal.get('market_cap', 0), quality.token_age_minutes,
                signal.get('holder_count', 0),
                json.dumps(asdict(quality)), strategy_decision, filter_reason
            ))
        
        # Queue for price updates
        self._price_update_queue.append({
            'signal_id': signal_id,
            'token_address': signal.get('token_address', ''),
            'timestamp': datetime.utcnow(),
            'updates_needed': ['1m', '5m', '15m', '30m', '1h', '2h', '4h', '12h', '24h']
        })
        
        return signal_id
    
    def update_entry_outcome(self, signal_id: str, exit_price: float,
                            exit_reason: str, pnl_pct: float, hold_minutes: float):
        """Update a signal with actual trading outcome"""
        with self._get_connection() as conn:
            conn.execute("""
                UPDATE historical_signals
                SET actual_entry = TRUE, actual_exit_price = ?,
                    actual_exit_reason = ?, actual_pnl_pct = ?,
                    actual_hold_minutes = ?
                WHERE id = ?
            """, (exit_price, exit_reason, pnl_pct, hold_minutes, signal_id))
    
    def start_price_updater(self):
        """Start background thread to update price history"""
        if self._updater_running:
            return
        
        self._updater_running = True
        self._updater_thread = threading.Thread(target=self._price_update_loop, daemon=True)
        self._updater_thread.start()
        logger.info("📚 Price history updater started")
    
    def stop_price_updater(self):
        """Stop the price updater"""
        self._updater_running = False
        if self._updater_thread:
            self._updater_thread.join(timeout=5)
    
    def _price_update_loop(self):
        """Background loop to update price history"""
        while self._updater_running:
            try:
                self._process_price_updates()
            except Exception as e:
                logger.error(f"Price update error: {e}")
            time.sleep(60)  # Check every minute
    
    def _process_price_updates(self):
        """Process pending price updates"""
        now = datetime.utcnow()
        
        # Time intervals in minutes
        intervals = {
            '1m': 1, '5m': 5, '15m': 15, '30m': 30,
            '1h': 60, '2h': 120, '4h': 240, '12h': 720, '24h': 1440
        }
        
        for item in self._price_update_queue[:]:  # Copy to allow modification
            signal_time = item['timestamp']
            token_address = item['token_address']
            signal_id = item['signal_id']
            
            minutes_elapsed = (now - signal_time).total_seconds() / 60
            
            for interval_name in item['updates_needed'][:]:
                interval_minutes = intervals[interval_name]
                
                if minutes_elapsed >= interval_minutes:
                    # Time to update this interval
                    price = self._get_token_price(token_address)
                    
                    if price > 0:
                        column = f"price_{interval_name}"
                        with self._get_connection() as conn:
                            conn.execute(f"""
                                UPDATE historical_signals
                                SET {column} = ?,
                                    price_peak = MAX(COALESCE(price_peak, 0), ?),
                                    price_trough = CASE 
                                        WHEN price_trough IS NULL OR price_trough = 0 THEN ?
                                        ELSE MIN(price_trough, ?)
                                    END
                                WHERE id = ?
                            """, (price, price, price, price, signal_id))
                        
                        item['updates_needed'].remove(interval_name)
            
            # Remove from queue if all updates complete
            if not item['updates_needed']:
                self._price_update_queue.remove(item)
                with self._get_connection() as conn:
                    conn.execute("""
                        UPDATE historical_signals
                        SET price_updates_complete = TRUE
                        WHERE id = ?
                    """, (signal_id,))
    
    def _get_token_price(self, token_address: str) -> float:
        """Get current token price"""
        try:
            url = f"https://api.dexscreener.com/latest/dex/tokens/{token_address}"
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                pairs = data.get('pairs', [])
                if pairs:
                    return float(pairs[0].get('priceUsd', 0))
        except:
            pass
        return 0
    
    def backtest(self, strategy_func: Callable, 
                 start_date: datetime = None,
                 end_date: datetime = None,
                 initial_balance: float = 10.0) -> Dict:
        """
        Backtest a strategy against historical signals.
        
        Args:
            strategy_func: Function(signal_dict, quality_dict) -> {'enter': bool, 'exit_params': {...}}
            start_date: Start of backtest period
            end_date: End of backtest period
            initial_balance: Starting balance in SOL
            
        Returns:
            Backtest results
        """
        start_date = start_date or (datetime.utcnow() - timedelta(days=30))
        end_date = end_date or datetime.utcnow()
        
        with self._get_connection() as conn:
            signals = conn.execute("""
                SELECT * FROM historical_signals
                WHERE timestamp BETWEEN ? AND ?
                  AND price_updates_complete = TRUE
                ORDER BY timestamp
            """, (start_date, end_date)).fetchall()
        
        # Simulate trading
        balance = initial_balance
        trades = []
        wins = 0
        total_pnl = 0
        
        for signal_row in signals:
            signal = dict(signal_row)
            quality = SignalQualityMetrics(**json.loads(signal['quality_metrics']))
            
            # Ask strategy what to do
            decision = strategy_func(signal, asdict(quality))
            
            if decision.get('enter', False):
                # Simulate the trade
                entry_price = signal['price_at_signal']
                exit_params = decision.get('exit_params', {})
                
                stop_loss = exit_params.get('stop_loss_pct', -15)
                take_profit = exit_params.get('take_profit_pct', 30)
                max_hold_hours = exit_params.get('max_hold_hours', 12)
                
                # Determine exit based on price history
                pnl_pct = self._simulate_trade_outcome(
                    signal, entry_price, stop_loss, take_profit, max_hold_hours
                )
                
                position_size = min(0.3, balance * 0.1)
                pnl_sol = position_size * (pnl_pct / 100)
                balance += pnl_sol
                total_pnl += pnl_sol
                
                if pnl_pct > 0:
                    wins += 1
                
                trades.append({
                    'signal_id': signal['id'],
                    'token': signal['token_symbol'],
                    'entry_price': entry_price,
                    'pnl_pct': pnl_pct,
                    'pnl_sol': pnl_sol
                })
        
        return {
            'period': f"{start_date.date()} to {end_date.date()}",
            'signals_analyzed': len(signals),
            'trades_taken': len(trades),
            'wins': wins,
            'win_rate': wins / len(trades) if trades else 0,
            'total_pnl_sol': total_pnl,
            'final_balance': balance,
            'return_pct': ((balance / initial_balance) - 1) * 100,
            'trades': trades
        }
    
    def _simulate_trade_outcome(self, signal: Dict, entry_price: float,
                                stop_loss: float, take_profit: float,
                                max_hold_hours: int) -> float:
        """Simulate trade outcome based on price history"""
        # Check price at each interval
        price_intervals = [
            ('price_1m', 1), ('price_5m', 5), ('price_15m', 15),
            ('price_30m', 30), ('price_1h', 60), ('price_2h', 120),
            ('price_4h', 240), ('price_12h', 720), ('price_24h', 1440)
        ]
        
        for col, minutes in price_intervals:
            price = signal.get(col) or 0
            if price <= 0:
                continue
            
            pnl_pct = ((price / entry_price) - 1) * 100
            
            # Check exits
            if pnl_pct <= stop_loss:
                return stop_loss
            if pnl_pct >= take_profit:
                return take_profit
            
            # Check time stop
            if minutes >= max_hold_hours * 60:
                return pnl_pct
        
        # Default to 12h price
        final_price = signal.get('price_12h') or signal.get('price_24h') or entry_price
        return ((final_price / entry_price) - 1) * 100 if entry_price > 0 else 0


# =============================================================================
# IMPROVEMENT #5: SIGNAL QUALITY ANALYZER
# =============================================================================

class SignalQualityAnalyzer:
    """
    Analyzes signal quality with enhanced metrics.
    
    Features:
    - Cluster detection (multiple wallets buying same token)
    - Position size analysis (is wallet betting big?)
    - Token timing (early vs late entry)
    - Wallet behavior patterns
    """
    
    def __init__(self, db=None):
        self.db = db
        
        # Track recent signals for cluster detection
        self._recent_signals: Dict[str, List[Dict]] = defaultdict(list)
        self._signal_window_minutes = 5  # Signals within 5 min = cluster
        
        # Wallet history cache
        self._wallet_history: Dict[str, Dict] = {}
        
        logger.info("🔍 Signal Quality Analyzer initialized")
    
    def analyze_signal(self, signal: Dict, wallet_data: Dict = None) -> SignalQualityMetrics:
        """
        Analyze a signal and return quality metrics.
        
        Args:
            signal: The trading signal
            wallet_data: Historical data about the wallet
            
        Returns:
            SignalQualityMetrics with all calculated values
        """
        metrics = SignalQualityMetrics()
        
        token_address = signal.get('token_address', '')
        wallet = signal.get('wallet', signal.get('wallet_address', ''))
        now = datetime.utcnow()
        
        # 1. Cluster Detection
        metrics = self._detect_cluster(metrics, token_address, wallet, now)
        
        # 2. Position Size Analysis
        metrics = self._analyze_position_size(metrics, signal, wallet_data)
        
        # 3. Token Timing
        metrics = self._analyze_token_timing(metrics, signal)
        
        # 4. Wallet Behavior
        metrics = self._analyze_wallet_behavior(metrics, wallet_data)
        
        # 5. Market Context
        metrics = self._analyze_market_context(metrics)
        
        return metrics
    
    def _detect_cluster(self, metrics: SignalQualityMetrics, 
                       token_address: str, wallet: str, 
                       now: datetime) -> SignalQualityMetrics:
        """Detect if multiple wallets are buying the same token"""
        # Clean old signals
        cutoff = now - timedelta(minutes=self._signal_window_minutes)
        self._recent_signals[token_address] = [
            s for s in self._recent_signals[token_address]
            if s['timestamp'] > cutoff
        ]
        
        # Add current signal
        self._recent_signals[token_address].append({
            'wallet': wallet,
            'timestamp': now
        })
        
        # Count unique wallets
        unique_wallets = set(s['wallet'] for s in self._recent_signals[token_address])
        
        metrics.concurrent_wallet_count = len(unique_wallets)
        metrics.cluster_wallets = list(unique_wallets)
        metrics.is_cluster_signal = len(unique_wallets) >= 2
        
        # Calculate cluster strength (weighted by wallet quality if available)
        metrics.cluster_strength = min(1.0, len(unique_wallets) / 5.0)
        
        return metrics
    
    def _analyze_position_size(self, metrics: SignalQualityMetrics,
                               signal: Dict, wallet_data: Dict) -> SignalQualityMetrics:
        """Analyze if wallet is betting bigger than usual"""
        if not wallet_data:
            return metrics
        
        typical_size = wallet_data.get('avg_position_size_sol', 0)
        current_size = signal.get('position_size_sol', 0)
        
        if typical_size > 0 and current_size > 0:
            metrics.wallet_typical_size_sol = typical_size
            metrics.wallet_current_size_sol = current_size
            metrics.size_multiplier = current_size / typical_size
            metrics.is_oversized_entry = metrics.size_multiplier > 1.5
        
        return metrics
    
    def _analyze_token_timing(self, metrics: SignalQualityMetrics,
                              signal: Dict) -> SignalQualityMetrics:
        """Analyze token age at time of buy"""
        age_hours = signal.get('token_age_hours', 0)
        age_minutes = age_hours * 60 if age_hours else signal.get('token_age_minutes', 0)
        
        metrics.token_age_minutes = age_minutes
        metrics.is_early_entry = age_minutes < 30
        metrics.is_late_entry = age_minutes > 240  # 4 hours
        
        return metrics
    
    def _analyze_wallet_behavior(self, metrics: SignalQualityMetrics,
                                 wallet_data: Dict) -> SignalQualityMetrics:
        """Analyze wallet's historical behavior"""
        if not wallet_data:
            return metrics
        
        metrics.wallet_avg_hold_minutes = wallet_data.get('avg_hold_minutes', 0)
        metrics.wallet_win_streak = wallet_data.get('current_streak', 0)
        metrics.wallet_recent_roi = wallet_data.get('roi_7d', 0)
        
        return metrics
    
    def _analyze_market_context(self, metrics: SignalQualityMetrics) -> SignalQualityMetrics:
        """Analyze broader market context"""
        # TODO: Integrate with SOL price feed and fear/greed index
        metrics.sol_price_1h_change = 0.0
        metrics.market_fear_greed = 50.0
        
        return metrics
    
    def record_signal_for_cluster_tracking(self, token_address: str, wallet: str):
        """Record a signal for cluster detection"""
        self._recent_signals[token_address].append({
            'wallet': wallet,
            'timestamp': datetime.utcnow()
        })


# =============================================================================
# IMPROVEMENT #6: DYNAMIC EXIT CALCULATOR
# =============================================================================

class DynamicExitCalculator:
    """
    Calculates dynamic exit parameters based on signal quality.
    
    High conviction signals get:
    - Tighter stop losses (less room for error needed)
    - Higher take profit targets (let winners run)
    - Shorter max hold time (quick decisions)
    
    Low conviction signals get:
    - Wider stop losses (more room needed)
    - Lower take profit targets (take what you can)
    - Even shorter hold time (get out fast)
    
    Cluster signals get:
    - Special treatment: let them run!
    """
    
    def __init__(self):
        # Base parameters
        self.base_stop_loss = -15.0
        self.base_take_profit = 30.0
        self.base_trailing_stop = 10.0
        self.base_max_hold_hours = 12
        
        # Learning adjustments (updated by learning system)
        self.learned_adjustments: Dict[str, float] = {}
        
        logger.info("📐 Dynamic Exit Calculator initialized")
    
    def calculate_exits(self, quality: SignalQualityMetrics,
                       conviction: float = 50.0) -> DynamicExitParams:
        """
        Calculate dynamic exit parameters.
        
        Args:
            quality: Signal quality metrics
            conviction: Overall conviction score (0-100)
            
        Returns:
            DynamicExitParams with calculated values
        """
        params = DynamicExitParams.from_signal_quality(quality, conviction)
        
        # Apply any learned adjustments
        if self.learned_adjustments:
            if 'stop_loss_adjustment' in self.learned_adjustments:
                params.stop_loss_pct *= (1 + self.learned_adjustments['stop_loss_adjustment'])
            if 'take_profit_adjustment' in self.learned_adjustments:
                params.take_profit_pct *= (1 + self.learned_adjustments['take_profit_adjustment'])
        
        return params
    
    def update_from_learning(self, adjustments: Dict[str, float]):
        """Update calculator with learned adjustments"""
        self.learned_adjustments.update(adjustments)
        logger.info(f"📐 Updated exit parameters from learning: {adjustments}")


# =============================================================================
# MAIN: ROBUST PAPER TRADER
# =============================================================================

class RobustPaperTrader:
    """
    Main paper trading engine with all improvements integrated.
    
    Features:
    1. ✅ Exit Monitoring Reliability (Watchdog)
    2. ✅ Baseline Comparison
    3. ✅ A/B Testing Framework
    4. ✅ Historical Backtesting
    5. ✅ Signal Quality Metrics
    6. ✅ Dynamic Exit Parameters
    """
    
    def __init__(self, 
                 db_path: str = "robust_paper_trades.db",
                 starting_balance: float = 10.0,
                 enable_watchdog: bool = True,
                 enable_baseline_tracking: bool = True,
                 enable_historical_storage: bool = True):
        
        self.db_path = db_path
        self.starting_balance = starting_balance
        
        # Core state
        self.balance = starting_balance
        self.reserved_balance = 0.0
        self.total_trades = 0
        self.winning_trades = 0
        self.total_pnl = 0.0
        self.open_position_count = 0
        
        # Thread safety
        self._lock = threading.RLock()
        self._monitor_running = False
        self._monitor_thread = None
        
        # Initialize database
        self._init_database()
        self._load_state()
        
        # Initialize components
        self.quality_analyzer = SignalQualityAnalyzer()
        self.exit_calculator = DynamicExitCalculator()
        
        if enable_baseline_tracking:
            self.baseline_tracker = BaselineTracker()
        else:
            self.baseline_tracker = None
        
        if enable_historical_storage:
            self.historical_store = HistoricalDataStore()
            self.historical_store.start_price_updater()
        else:
            self.historical_store = None
        
        self.ab_testing = ABTestingFramework()
        
        # Start monitoring
        self.start_monitor()
        
        if enable_watchdog:
            self.watchdog = ExitMonitorWatchdog(self)
            self.watchdog.start()
        else:
            self.watchdog = None
        
        logger.info("=" * 60)
        logger.info("🚀 ROBUST PAPER TRADER V5 INITIALIZED")
        logger.info("=" * 60)
        logger.info(f"   Balance: {self.balance:.4f} SOL")
        logger.info(f"   Open Positions: {self.open_position_count}")
        logger.info(f"   Watchdog: {'ENABLED' if enable_watchdog else 'DISABLED'}")
        logger.info(f"   Baseline Tracking: {'ENABLED' if enable_baseline_tracking else 'DISABLED'}")
        logger.info(f"   Historical Storage: {'ENABLED' if enable_historical_storage else 'DISABLED'}")
        logger.info("=" * 60)
    
    def _init_database(self):
        """Initialize the database"""
        with self._get_connection() as conn:
            # Account state
            conn.execute("""
                CREATE TABLE IF NOT EXISTS paper_account_v5 (
                    id INTEGER PRIMARY KEY,
                    starting_balance REAL,
                    current_balance REAL,
                    reserved_balance REAL DEFAULT 0,
                    total_trades INTEGER DEFAULT 0,
                    winning_trades INTEGER DEFAULT 0,
                    total_pnl_sol REAL DEFAULT 0,
                    peak_balance REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Positions
            conn.execute("""
                CREATE TABLE IF NOT EXISTS paper_positions_v5 (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    token_address TEXT NOT NULL,
                    token_symbol TEXT,
                    
                    entry_price REAL NOT NULL,
                    entry_time TIMESTAMP NOT NULL,
                    size_sol REAL NOT NULL,
                    tokens_bought REAL NOT NULL,
                    
                    -- Dynamic exit parameters
                    stop_loss_pct REAL,
                    take_profit_pct REAL,
                    trailing_stop_pct REAL,
                    trailing_activation_pct REAL,
                    max_hold_hours INTEGER,
                    scale_out_enabled BOOLEAN DEFAULT FALSE,
                    
                    -- Tracking
                    current_price REAL,
                    peak_price REAL,
                    lowest_price REAL,
                    last_price_update TIMESTAMP,
                    
                    -- Status
                    status TEXT DEFAULT 'open',
                    exit_price REAL,
                    exit_time TIMESTAMP,
                    exit_reason TEXT,
                    pnl_sol REAL,
                    pnl_pct REAL,
                    hold_duration_minutes REAL,
                    
                    -- Quality metrics
                    conviction_score REAL,
                    quality_score REAL,
                    is_cluster_signal BOOLEAN,
                    
                    -- Context
                    entry_context_json TEXT,
                    signal_id TEXT,
                    ab_test_id TEXT,
                    ab_variant TEXT,
                    
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Initialize account
            account = conn.execute("SELECT * FROM paper_account_v5 WHERE id = 1").fetchone()
            if not account:
                conn.execute("""
                    INSERT INTO paper_account_v5 (id, starting_balance, current_balance, peak_balance)
                    VALUES (1, ?, ?, ?)
                """, (self.starting_balance, self.starting_balance, self.starting_balance))
    
    @contextmanager
    def _get_connection(self):
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()
    
    def _load_state(self):
        """Load state from database"""
        with self._get_connection() as conn:
            account = conn.execute("SELECT * FROM paper_account_v5 WHERE id = 1").fetchone()
            if account:
                self.balance = account['current_balance']
                self.reserved_balance = account['reserved_balance'] or 0
                self.total_trades = account['total_trades'] or 0
                self.winning_trades = account['winning_trades'] or 0
                self.total_pnl = account['total_pnl_sol'] or 0
            
            count = conn.execute(
                "SELECT COUNT(*) FROM paper_positions_v5 WHERE status = 'open'"
            ).fetchone()[0]
            self.open_position_count = count
    
    def process_signal(self, signal: Dict, wallet_data: Dict = None,
                      top_wallets: List[str] = None) -> Dict:
        """
        Process a trading signal through the full pipeline.
        
        This is the main entry point that:
        1. Analyzes signal quality
        2. Calculates dynamic exits
        3. Records for baseline comparison
        4. Stores for historical backtesting
        5. Applies A/B test variants
        6. Opens position if appropriate
        
        Returns:
            Dict with decision and position_id if opened
        """
        # 1. Analyze signal quality
        quality = self.quality_analyzer.analyze_signal(signal, wallet_data)
        quality_score = quality.calculate_composite_score()
        
        # 2. Calculate conviction
        wallet_wr = signal.get('wallet_win_rate', 0.5)
        if wallet_wr > 1:
            wallet_wr = wallet_wr / 100.0
        
        conviction = 50 + (wallet_wr * 30) + (quality_score - 50) * 0.4
        conviction = max(0, min(100, conviction))
        
        # 3. Calculate dynamic exit parameters
        exit_params = self.exit_calculator.calculate_exits(quality, conviction)
        
        # 4. Determine if we should enter
        should_enter = self._evaluate_entry(signal, quality, conviction)
        strategy_decision = "ENTER" if should_enter else "SKIP"
        filter_reason = "" if should_enter else "Did not meet entry criteria"
        
        # 5. Record for baseline comparison
        signal_id = None
        if self.baseline_tracker:
            signal_id = self.baseline_tracker.record_signal(
                signal, quality, strategy_decision, top_wallets
            )
        
        # 6. Store for historical backtesting
        if self.historical_store:
            signal_id = self.historical_store.store_signal(
                signal, quality, strategy_decision, filter_reason
            )
        
        # 7. Check A/B tests
        ab_test_id = None
        ab_variant = None
        for test_id, test in self.ab_testing._active_tests.items():
            if test.is_active:
                ab_variant, variant_params = self.ab_testing.get_variant(test_id, signal)
                ab_test_id = test_id
                # Apply variant parameters to exit params if relevant
                if 'stop_loss' in variant_params:
                    exit_params.stop_loss_pct = variant_params['stop_loss']
                if 'take_profit' in variant_params:
                    exit_params.take_profit_pct = variant_params['take_profit']
                break
        
        result = {
            'should_enter': should_enter,
            'conviction': conviction,
            'quality_score': quality_score,
            'quality_metrics': asdict(quality),
            'exit_params': asdict(exit_params),
            'signal_id': signal_id,
            'ab_test_id': ab_test_id,
            'ab_variant': ab_variant,
            'position_id': None
        }
        
        # 8. Open position if appropriate
        if should_enter:
            position_id = self.open_position(
                signal, exit_params, conviction, quality_score,
                quality.is_cluster_signal, signal_id, ab_test_id, ab_variant
            )
            result['position_id'] = position_id
        
        return result
    
    def _evaluate_entry(self, signal: Dict, quality: SignalQualityMetrics,
                       conviction: float) -> bool:
        """Evaluate if we should enter a position"""
        # Basic checks
        price = signal.get('price', 0)
        if price <= 0:
            return False
        
        # Balance check
        available = self.balance - self.reserved_balance
        if available < 0.1:
            return False
        
        # In learning/exploration mode, be very permissive
        # Just need valid price and reasonable wallet
        wallet_wr = signal.get('wallet_win_rate', 0)
        if wallet_wr > 1:
            wallet_wr = wallet_wr / 100.0
        
        return wallet_wr >= 0.30 and price > 0
    
    def open_position(self, signal: Dict, exit_params: DynamicExitParams,
                     conviction: float, quality_score: float,
                     is_cluster: bool, signal_id: str = None,
                     ab_test_id: str = None, ab_variant: str = None) -> Optional[int]:
        """Open a paper position"""
        price = signal.get('price', 0)
        if price <= 0:
            return None
        
        # Calculate position size
        available = self.balance - self.reserved_balance
        size_sol = min(0.3, available * 0.3)  # Max 30% of available
        
        if size_sol < 0.1:
            return None
        
        tokens_bought = size_sol / price
        
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.execute("""
                    INSERT INTO paper_positions_v5
                    (token_address, token_symbol, entry_price, entry_time,
                     size_sol, tokens_bought, stop_loss_pct, take_profit_pct,
                     trailing_stop_pct, trailing_activation_pct, max_hold_hours,
                     scale_out_enabled, peak_price, lowest_price,
                     conviction_score, quality_score, is_cluster_signal,
                     signal_id, ab_test_id, ab_variant, entry_context_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    signal.get('token_address', ''),
                    signal.get('token_symbol', 'UNKNOWN'),
                    price, datetime.utcnow(),
                    size_sol, tokens_bought,
                    exit_params.stop_loss_pct, exit_params.take_profit_pct,
                    exit_params.trailing_stop_pct, exit_params.trailing_activation_pct,
                    exit_params.max_hold_hours, exit_params.scale_out_enabled,
                    price, price,
                    conviction, quality_score, is_cluster,
                    signal_id, ab_test_id, ab_variant,
                    json.dumps(signal)
                ))
                
                position_id = cursor.lastrowid
                
                # Update account
                self.reserved_balance += size_sol
                conn.execute("""
                    UPDATE paper_account_v5
                    SET reserved_balance = ?, updated_at = ?
                    WHERE id = 1
                """, (self.reserved_balance, datetime.utcnow()))
            
            self.open_position_count += 1
            
            logger.info(f"📥 OPENED #{position_id}: {signal.get('token_symbol', 'UNKNOWN')}")
            logger.info(f"   Size: {size_sol:.4f} SOL @ ${price:.8f}")
            logger.info(f"   Conviction: {conviction:.0f} | Quality: {quality_score:.0f}")
            logger.info(f"   Stop: {exit_params.stop_loss_pct}% | TP: {exit_params.take_profit_pct}%")
            if is_cluster:
                logger.info(f"   🔥 CLUSTER SIGNAL!")
            
            return position_id
    
    def close_position(self, position_id: int, exit_price: float,
                      exit_reason: ExitReason) -> Optional[Dict]:
        """Close a position"""
        with self._lock:
            with self._get_connection() as conn:
                pos = conn.execute("""
                    SELECT * FROM paper_positions_v5 WHERE id = ? AND status = 'open'
                """, (position_id,)).fetchone()
                
                if not pos:
                    return None
                
                entry_price = pos['entry_price']
                size_sol = pos['size_sol']
                tokens = pos['tokens_bought']
                entry_time = datetime.fromisoformat(pos['entry_time'])
                
                exit_value = tokens * exit_price
                pnl_sol = exit_value - size_sol
                pnl_pct = ((exit_price / entry_price) - 1) * 100
                hold_minutes = (datetime.utcnow() - entry_time).total_seconds() / 60
                
                # Update position
                conn.execute("""
                    UPDATE paper_positions_v5
                    SET status = 'closed', exit_price = ?, exit_time = ?,
                        exit_reason = ?, pnl_sol = ?, pnl_pct = ?,
                        hold_duration_minutes = ?
                    WHERE id = ?
                """, (exit_price, datetime.utcnow(), exit_reason.value,
                      pnl_sol, pnl_pct, hold_minutes, position_id))
                
                # Update account
                self.balance += exit_value
                self.reserved_balance -= size_sol
                self.total_trades += 1
                self.total_pnl += pnl_sol
                
                is_win = pnl_sol > 0
                if is_win:
                    self.winning_trades += 1
                
                conn.execute("""
                    UPDATE paper_account_v5
                    SET current_balance = ?, reserved_balance = ?,
                        total_trades = ?, winning_trades = ?, total_pnl_sol = ?,
                        peak_balance = MAX(peak_balance, ?), updated_at = ?
                    WHERE id = 1
                """, (self.balance, self.reserved_balance, self.total_trades,
                      self.winning_trades, self.total_pnl, self.balance, datetime.utcnow()))
                
                # Record A/B test outcome
                if pos['ab_test_id'] and pos['ab_variant']:
                    self.ab_testing.record_outcome(
                        pos['ab_test_id'], pos['ab_variant'],
                        pos['signal_id'] or '', pnl_pct, hold_minutes, exit_reason.value
                    )
                
                # Update historical record
                if self.historical_store and pos['signal_id']:
                    self.historical_store.update_entry_outcome(
                        pos['signal_id'], exit_price, exit_reason.value,
                        pnl_pct, hold_minutes
                    )
            
            self.open_position_count -= 1
            
            emoji = "✅" if is_win else "❌"
            logger.info(f"📤 CLOSED #{position_id}: {pos['token_symbol']} ({exit_reason.value})")
            logger.info(f"   {emoji} PnL: {pnl_sol:+.4f} SOL ({pnl_pct:+.1f}%)")
            logger.info(f"   Hold time: {hold_minutes:.0f} min")
            
            return {
                'position_id': position_id,
                'pnl_sol': pnl_sol,
                'pnl_pct': pnl_pct,
                'hold_minutes': hold_minutes,
                'exit_reason': exit_reason.value
            }
    
    def get_open_positions(self) -> List[Dict]:
        """Get all open positions"""
        with self._get_connection() as conn:
            rows = conn.execute("""
                SELECT * FROM paper_positions_v5 WHERE status = 'open'
            """).fetchall()
            return [dict(r) for r in rows]
    
    def get_performance_summary(self) -> Dict:
        """Get performance summary"""
        win_rate = self.winning_trades / self.total_trades if self.total_trades > 0 else 0
        return_pct = ((self.balance / self.starting_balance) - 1) * 100
        
        return {
            'balance': self.balance,
            'starting_balance': self.starting_balance,
            'return_pct': return_pct,
            'total_pnl_sol': self.total_pnl,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'win_rate': win_rate,
            'open_positions': self.open_position_count
        }
    
    # =========================================================================
    # EXIT MONITORING
    # =========================================================================
    
    def start_monitor(self):
        """Start exit monitoring"""
        if self._monitor_running:
            return
        
        self._monitor_running = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        logger.info("🔄 Exit monitor started")
    
    def stop_monitor(self):
        """Stop exit monitoring"""
        self._monitor_running = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
    
    def _monitor_loop(self):
        """Background monitoring loop"""
        while self._monitor_running:
            try:
                self._check_all_exits()
            except Exception as e:
                logger.error(f"Monitor error: {e}")
            time.sleep(30)
    
    def _check_all_exits(self):
        """Check all positions for exit conditions"""
        positions = self.get_open_positions()
        for pos in positions:
            try:
                self._check_position_exit(pos)
            except Exception as e:
                logger.error(f"Error checking position {pos['id']}: {e}")
    
    def _check_position_exit(self, pos: Dict):
        """Check single position for exit"""
        token_address = pos['token_address']
        entry_price = pos['entry_price']
        entry_time = datetime.fromisoformat(pos['entry_time'])
        peak_price = pos.get('peak_price') or entry_price
        
        current_price = self._get_token_price(token_address)
        if current_price <= 0:
            return
        
        # Update tracking
        self._update_price_tracking(pos['id'], current_price, entry_price, peak_price)
        
        pnl_pct = ((current_price / entry_price) - 1) * 100
        from_peak_pct = ((current_price / peak_price) - 1) * 100 if peak_price > 0 else 0
        hold_hours = (datetime.utcnow() - entry_time).total_seconds() / 3600
        
        exit_reason = None
        
        # Check exits
        if pnl_pct <= pos['stop_loss_pct']:
            exit_reason = ExitReason.STOP_LOSS
        elif pnl_pct >= pos['take_profit_pct']:
            exit_reason = ExitReason.TAKE_PROFIT
        elif pnl_pct >= (pos.get('trailing_activation_pct') or 15):
            if from_peak_pct <= -pos['trailing_stop_pct']:
                exit_reason = ExitReason.TRAILING_STOP
        elif hold_hours >= pos['max_hold_hours']:
            exit_reason = ExitReason.TIME_STOP
        
        if exit_reason:
            self.close_position(pos['id'], current_price, exit_reason)
    
    def _update_price_tracking(self, position_id: int, current_price: float,
                               entry_price: float, old_peak: float):
        """Update price tracking"""
        new_peak = max(old_peak, current_price)
        with self._get_connection() as conn:
            conn.execute("""
                UPDATE paper_positions_v5
                SET current_price = ?, peak_price = ?,
                    lowest_price = MIN(COALESCE(lowest_price, ?), ?),
                    last_price_update = ?
                WHERE id = ?
            """, (current_price, new_peak, current_price, current_price,
                  datetime.utcnow(), position_id))
    
    def _get_token_price(self, token_address: str) -> float:
        """Get token price"""
        try:
            url = f"https://api.dexscreener.com/latest/dex/tokens/{token_address}"
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                pairs = data.get('pairs', [])
                if pairs:
                    return float(pairs[0].get('priceUsd', 0))
        except:
            pass
        return 0
    
    # =========================================================================
    # REPORTING
    # =========================================================================
    
    def print_status(self):
        """Print comprehensive status"""
        summary = self.get_performance_summary()
        
        print("\n" + "=" * 70)
        print("🚀 ROBUST PAPER TRADER V5 STATUS")
        print("=" * 70)
        
        print(f"\n📊 ACCOUNT:")
        print(f"   Balance: {summary['balance']:.4f} SOL")
        print(f"   Return: {summary['return_pct']:+.1f}%")
        print(f"   Total PnL: {summary['total_pnl_sol']:+.4f} SOL")
        
        print(f"\n📈 PERFORMANCE:")
        print(f"   Trades: {summary['total_trades']}")
        print(f"   Win Rate: {summary['win_rate']:.1%}")
        print(f"   Open Positions: {summary['open_positions']}")
        
        # Show baseline comparison if available
        if self.baseline_tracker:
            print(f"\n📊 BASELINE COMPARISON:")
            self.baseline_tracker.print_comparison(days=7)
        
        print("\n" + "=" * 70)
    
    def stop(self):
        """Stop all background processes"""
        self.stop_monitor()
        if self.watchdog:
            self.watchdog.stop()
        if self.historical_store:
            self.historical_store.stop_price_updater()
        logger.info("🛑 Robust Paper Trader stopped")


# =============================================================================
# DROP-IN REPLACEMENT FOR master_v2.py
# =============================================================================

class LearningPaperTradingEngine:
    """
    Drop-in replacement that integrates with master_v2.py
    """
    
    def __init__(self, db=None, starting_balance: float = 10.0, max_positions: int = 999):
        self._trader = RobustPaperTrader(
            starting_balance=starting_balance,
            enable_watchdog=True,
            enable_baseline_tracking=True,
            enable_historical_storage=True
        )
        
        self._top_wallets: List[str] = []
    
    @property
    def balance(self) -> float:
        return self._trader.balance
    
    @property
    def available_balance(self) -> float:
        return self._trader.balance - self._trader.reserved_balance
    
    def set_top_wallets(self, wallets: List[str]):
        """Set top wallets for baseline comparison"""
        self._top_wallets = wallets
    
    def open_position(self, signal: Dict, decision: Dict, price: float) -> Optional[int]:
        """Open position - compatible with master_v2.py"""
        signal['price'] = price
        result = self._trader.process_signal(signal, top_wallets=self._top_wallets)
        return result.get('position_id')
    
    def close_position(self, position_id: int, exit_reason: str, exit_price: float) -> Optional[Dict]:
        """Close position - compatible with master_v2.py"""
        reason_map = {
            'STOP_LOSS': ExitReason.STOP_LOSS,
            'TAKE_PROFIT': ExitReason.TAKE_PROFIT,
            'TRAILING_STOP': ExitReason.TRAILING_STOP,
            'TIME_STOP': ExitReason.TIME_STOP,
            'MANUAL': ExitReason.MANUAL
        }
        reason = reason_map.get(exit_reason, ExitReason.MANUAL)
        return self._trader.close_position(position_id, exit_price, reason)
    
    def get_open_positions(self) -> List[Dict]:
        return self._trader.get_open_positions()
    
    def get_stats(self) -> Dict:
        return self._trader.get_performance_summary()
    
    def print_status(self):
        self._trader.print_status()
    
    def stop(self):
        self._trader.stop()


# =============================================================================
# CLI
# =============================================================================

def main():
    import sys
    
    trader = RobustPaperTrader()
    
    if len(sys.argv) < 2:
        trader.print_status()
        return
    
    command = sys.argv[1].lower()
    
    if command == 'status':
        trader.print_status()
    
    elif command == 'baseline':
        if trader.baseline_tracker:
            trader.baseline_tracker.print_comparison()
    
    elif command == 'backtest':
        if trader.historical_store:
            # Simple example strategy
            def simple_strategy(signal, quality):
                return {
                    'enter': signal.get('wallet_win_rate', 0) > 0.5,
                    'exit_params': {'stop_loss_pct': -15, 'take_profit_pct': 30, 'max_hold_hours': 12}
                }
            
            results = trader.historical_store.backtest(simple_strategy)
            print(f"\n📊 BACKTEST RESULTS")
            print(f"   Period: {results['period']}")
            print(f"   Signals: {results['signals_analyzed']}")
            print(f"   Trades: {results['trades_taken']}")
            print(f"   Win Rate: {results['win_rate']:.0%}")
            print(f"   Return: {results['return_pct']:+.1f}%")
    
    elif command == 'stop':
        trader.stop()
    
    else:
        print(f"Unknown command: {command}")
        print("Commands: status, baseline, backtest, stop")


if __name__ == "__main__":
    main()
