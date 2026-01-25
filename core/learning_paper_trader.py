"""
LEARNING PAPER TRADER V4
========================

A comprehensive paper trading system designed for LEARNING:

KEY PHILOSOPHY:
- Open positions LIBERALLY to collect data (no position limits initially)
- Track EVERYTHING - entry context, time of day, hold duration, exit reasons
- MANDATORY stop losses on all positions
- ITERATIVE REFINEMENT - every 6 hours, analyze and tighten filters
- GOAL: After 7+ days, identify a profitable strategy with:
  - Positive PnL
  - Win rate >50%, preferably >60%
  - Path to 0.5-1 SOL/day profit

CRITICAL DATA TRACKED:
- Time of day (hour) for entry and exit - identify optimal trading windows
- Hold duration (accurate, in minutes) - understand time patterns
- All entry signals and context - learn what predicts success
- Exit reasons - understand why trades fail
- Diurnal patterns - avoid unprofitable trading periods

Author: Claude
Version: 4.0 (Learning Mode)
"""

import os
import json
import sqlite3
import threading
import time
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict
from contextlib import contextmanager
from collections import defaultdict
from enum import Enum
import logging
import statistics

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger(__name__)


class ExitReason(Enum):
    """Exit reasons for tracking"""
    STOP_LOSS = "STOP_LOSS"
    TAKE_PROFIT = "TAKE_PROFIT"
    TRAILING_STOP = "TRAILING_STOP"
    TIME_STOP = "TIME_STOP"
    SMART_EXIT = "SMART_EXIT"
    STRATEGIST_EXIT = "STRATEGIST_EXIT"
    MANUAL = "MANUAL"
    LIQUIDITY_DRIED = "LIQUIDITY_DRIED"
    RUG_DETECTED = "RUG_DETECTED"


class TradeStatus(Enum):
    """Position status"""
    OPEN = "open"
    CLOSED = "closed"
    STOPPED = "stopped"
    EXPIRED = "expired"


class LearningPhase(Enum):
    """Learning progression phases"""
    EXPLORATION = "exploration"      # Day 1-2: Open everything, collect data
    INITIAL_FILTERING = "initial"    # Day 2-4: Start basic filtering
    REFINEMENT = "refinement"        # Day 4-6: Refine based on patterns
    OPTIMIZATION = "optimization"    # Day 6+: Optimize for profitability
    PRODUCTION = "production"        # Day 7+: Ready for live testing


@dataclass
class LearningConfig:
    """
    Configuration that EVOLVES as we learn.
    Starts very permissive, gets stricter over time.
    """
    # Starting balance
    starting_balance_sol: float = 10.0
    
    # Position limits - START WITH NO LIMITS
    max_open_positions: int = 999  # Effectively unlimited initially
    max_position_size_sol: float = 0.5
    min_position_size_sol: float = 0.1
    default_position_size_sol: float = 0.2  # Smaller positions = more data points
    
    # Exit parameters - MANDATORY STOP LOSS
    default_stop_loss_pct: float = -15.0    # -15% stop loss (mandatory!)
    default_take_profit_pct: float = 30.0   # +30%
    default_trailing_stop_pct: float = 10.0 # 10% from peak
    max_hold_hours: int = 12
    
    # Monitoring
    price_check_interval_seconds: int = 30
    enable_auto_exits: bool = True
    
    # Learning parameters
    learning_interval_hours: float = 6.0    # Run learning every 6 hours
    min_trades_for_learning: int = 20       # Minimum trades before adjusting
    
    # Entry thresholds (EVOLVE OVER TIME)
    min_conviction_score: int = 30          # Start very low
    min_wallet_win_rate: float = 0.40       # Start very low
    min_liquidity_usd: float = 20000        # Start low
    
    # Time restrictions (learned over time)
    blocked_hours_utc: List[int] = field(default_factory=list)  # Hours to avoid
    preferred_hours_utc: List[int] = field(default_factory=list)  # Best hours
    
    # Current learning phase
    current_phase: str = "exploration"
    phase_started_at: str = ""
    iteration_count: int = 0


@dataclass 
class EntryContext:
    """Rich context captured at entry - CRITICAL FOR LEARNING"""
    # Wallet info
    wallet_address: str = ""
    wallet_cluster: str = "UNKNOWN"
    wallet_win_rate: float = 0.0
    wallet_roi_7d: float = 0.0
    wallet_trades_count: int = 0
    
    # Token metrics at entry
    liquidity_usd: float = 0.0
    volume_24h_usd: float = 0.0
    market_cap_usd: float = 0.0
    token_age_hours: float = 0.0
    holder_count: int = 0
    
    # Signal strength
    conviction_score: float = 50.0
    signal_wallets_count: int = 1
    clusters_detected: List[str] = field(default_factory=list)
    aggregated_signal: bool = False
    llm_called: bool = False
    llm_adjustment: float = 0.0
    
    # TIME DATA - CRITICAL FOR DIURNAL ANALYSIS
    entry_hour_utc: int = 0
    entry_day_of_week: int = 0  # 0=Monday, 6=Sunday
    entry_timestamp: str = ""
    
    # Market context
    sol_price_usd: float = 0.0
    market_regime: str = "NEUTRAL"
    
    # Strategy metadata
    strategy_name: str = "learning_v4"
    strategy_version: str = "4.0"
    entry_reason: str = ""
    learning_phase: str = "exploration"
    iteration_number: int = 0


class LearningPaperTrader:
    """
    Paper trading engine optimized for LEARNING profitable strategies.
    
    Key differences from previous versions:
    1. NO POSITION LIMITS initially - we want maximum data
    2. MANDATORY STOP LOSSES - prevent catastrophic losses
    3. COMPREHENSIVE TIME TRACKING - for diurnal analysis
    4. ITERATIVE LEARNING - gets smarter every 6 hours
    5. EVOLVING THRESHOLDS - start permissive, get strict
    """
    
    def __init__(self, 
                 db_path: str = "learning_paper_trades.db",
                 config: LearningConfig = None):
        self.db_path = db_path
        self.config = config or LearningConfig()
        
        # Thread safety
        self._lock = threading.RLock()
        self._monitor_running = False
        self._monitor_thread = None
        
        # Initialize database
        self._init_database()
        
        # Load state
        self._load_state()
        
        # Start background monitoring
        if self.config.enable_auto_exits:
            self.start_monitor()
        
        logger.info(f"🎓 LEARNING PAPER TRADER V4 initialized")
        logger.info(f"   Phase: {self.config.current_phase}")
        logger.info(f"   Iteration: {self.config.iteration_count}")
        logger.info(f"   Balance: {self.balance:.4f} SOL")
        logger.info(f"   Open positions: {self.open_position_count}")
    
    def _init_database(self):
        """Initialize database with comprehensive tracking schema"""
        with self._get_connection() as conn:
            # Learning config tracking
            conn.execute("""
                CREATE TABLE IF NOT EXISTS learning_config (
                    id INTEGER PRIMARY KEY,
                    config_json TEXT NOT NULL,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Account state
            conn.execute("""
                CREATE TABLE IF NOT EXISTS learning_account (
                    id INTEGER PRIMARY KEY,
                    starting_balance REAL NOT NULL,
                    current_balance REAL NOT NULL,
                    reserved_balance REAL DEFAULT 0,
                    total_trades INTEGER DEFAULT 0,
                    winning_trades INTEGER DEFAULT 0,
                    total_pnl_sol REAL DEFAULT 0,
                    current_streak INTEGER DEFAULT 0,
                    peak_balance REAL,
                    max_drawdown_pct REAL DEFAULT 0,
                    best_trade_pnl_pct REAL DEFAULT 0,
                    worst_trade_pnl_pct REAL DEFAULT 0,
                    last_trade_time TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Positions with COMPREHENSIVE tracking
            conn.execute("""
                CREATE TABLE IF NOT EXISTS learning_positions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    token_address TEXT NOT NULL,
                    token_symbol TEXT NOT NULL,
                    
                    -- Entry data
                    entry_price REAL NOT NULL,
                    entry_time TIMESTAMP NOT NULL,
                    size_sol REAL NOT NULL,
                    tokens_bought REAL NOT NULL,
                    
                    -- Exit parameters (MANDATORY STOP LOSS)
                    stop_loss_pct REAL NOT NULL,
                    take_profit_pct REAL NOT NULL,
                    trailing_stop_pct REAL NOT NULL,
                    max_hold_hours INTEGER NOT NULL,
                    
                    -- Live tracking
                    current_price REAL DEFAULT 0,
                    peak_price REAL DEFAULT 0,
                    peak_unrealized_pct REAL DEFAULT 0,
                    lowest_price REAL DEFAULT 0,
                    last_price_update TIMESTAMP,
                    
                    -- Status
                    status TEXT DEFAULT 'open',
                    exit_price REAL DEFAULT 0,
                    exit_time TIMESTAMP,
                    exit_reason TEXT,
                    
                    -- Final results
                    pnl_sol REAL DEFAULT 0,
                    pnl_pct REAL DEFAULT 0,
                    hold_duration_minutes REAL DEFAULT 0,
                    
                    -- TIME TRACKING (CRITICAL FOR DIURNAL ANALYSIS)
                    entry_hour_utc INTEGER,
                    entry_day_of_week INTEGER,
                    exit_hour_utc INTEGER,
                    exit_day_of_week INTEGER,
                    
                    -- Rich context (JSON)
                    entry_context_json TEXT,
                    notes TEXT,
                    
                    -- Learning metadata
                    learning_phase TEXT,
                    iteration_number INTEGER,
                    
                    -- Indexing
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    
                    UNIQUE(token_address, entry_time)
                )
            """)
            
            # Create indexes for time analysis
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_positions_entry_hour 
                ON learning_positions(entry_hour_utc)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_positions_status 
                ON learning_positions(status)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_positions_exit_reason 
                ON learning_positions(exit_reason)
            """)
            
            # Learning iterations log
            conn.execute("""
                CREATE TABLE IF NOT EXISTS learning_iterations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    iteration_number INTEGER NOT NULL,
                    timestamp TIMESTAMP NOT NULL,
                    phase TEXT NOT NULL,
                    
                    -- Metrics at this iteration
                    total_trades INTEGER,
                    win_rate REAL,
                    total_pnl_sol REAL,
                    avg_hold_minutes REAL,
                    
                    -- Learned parameters
                    new_min_conviction INTEGER,
                    new_min_wallet_wr REAL,
                    new_min_liquidity REAL,
                    blocked_hours TEXT,
                    preferred_hours TEXT,
                    
                    -- Analysis results
                    analysis_json TEXT,
                    recommendations_json TEXT,
                    
                    -- Phase transition
                    phase_changed BOOLEAN DEFAULT FALSE,
                    new_phase TEXT,
                    
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Hourly performance tracking (for diurnal analysis)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS hourly_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    hour_utc INTEGER NOT NULL,
                    day_of_week INTEGER,  -- NULL means aggregate for all days
                    
                    trades_count INTEGER DEFAULT 0,
                    wins INTEGER DEFAULT 0,
                    losses INTEGER DEFAULT 0,
                    total_pnl_sol REAL DEFAULT 0,
                    avg_pnl_pct REAL DEFAULT 0,
                    
                    -- Calculated
                    win_rate REAL DEFAULT 0,
                    is_profitable BOOLEAN DEFAULT FALSE,
                    
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    
                    UNIQUE(hour_utc, day_of_week)
                )
            """)
            
            # Trade journal for detailed analysis
            conn.execute("""
                CREATE TABLE IF NOT EXISTS learning_journal (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    position_id INTEGER,
                    event_type TEXT NOT NULL,
                    event_data TEXT,
                    notes TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Initialize account if needed
            account = conn.execute("SELECT * FROM learning_account WHERE id = 1").fetchone()
            if not account:
                conn.execute("""
                    INSERT INTO learning_account 
                    (id, starting_balance, current_balance, peak_balance)
                    VALUES (1, ?, ?, ?)
                """, (self.config.starting_balance_sol, 
                      self.config.starting_balance_sol,
                      self.config.starting_balance_sol))
            
            # Initialize config if needed
            cfg = conn.execute("SELECT * FROM learning_config WHERE id = 1").fetchone()
            if not cfg:
                self.config.phase_started_at = datetime.utcnow().isoformat()
                conn.execute("""
                    INSERT INTO learning_config (id, config_json)
                    VALUES (1, ?)
                """, (json.dumps(asdict(self.config)),))
    
    @contextmanager
    def _get_connection(self):
        """Thread-safe database connection"""
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()
    
    def _load_state(self):
        """Load current state from database"""
        with self._get_connection() as conn:
            # Load account
            account = conn.execute(
                "SELECT * FROM learning_account WHERE id = 1"
            ).fetchone()
            
            if account:
                self.balance = account['current_balance']
                self.reserved_balance = account['reserved_balance'] or 0
                self.total_trades = account['total_trades'] or 0
                self.winning_trades = account['winning_trades'] or 0
                self.total_pnl = account['total_pnl_sol'] or 0
                self.current_streak = account['current_streak'] or 0
                self.peak_balance = account['peak_balance'] or self.balance
            else:
                self.balance = self.config.starting_balance_sol
                self.reserved_balance = 0
                self.total_trades = 0
                self.winning_trades = 0
                self.total_pnl = 0
                self.current_streak = 0
                self.peak_balance = self.balance
            
            # Count open positions
            count = conn.execute(
                "SELECT COUNT(*) FROM learning_positions WHERE status = 'open'"
            ).fetchone()[0]
            self.open_position_count = count
            
            # Load config
            cfg = conn.execute(
                "SELECT config_json FROM learning_config WHERE id = 1"
            ).fetchone()
            if cfg:
                try:
                    saved_config = json.loads(cfg['config_json'])
                    # Update config from saved state
                    for key, value in saved_config.items():
                        if hasattr(self.config, key):
                            setattr(self.config, key, value)
                except:
                    pass
    
    def _save_config(self):
        """Save current config to database"""
        with self._get_connection() as conn:
            conn.execute("""
                UPDATE learning_config 
                SET config_json = ?, updated_at = ?
                WHERE id = 1
            """, (json.dumps(asdict(self.config)), datetime.utcnow()))
    
    # =========================================================================
    # POSITION MANAGEMENT
    # =========================================================================
    
    def can_open_position(self, signal: Dict = None) -> Tuple[bool, str]:
        """
        Check if we can open a position.
        In learning mode, we're very permissive but still check:
        1. Sufficient balance
        2. Time restrictions (if learned)
        3. Basic signal quality (conviction, wallet WR)
        """
        # Check balance
        available = self.balance - self.reserved_balance
        if available < self.config.min_position_size_sol:
            return False, "Insufficient balance"
        
        # Check if current hour is blocked
        current_hour = datetime.utcnow().hour
        if current_hour in self.config.blocked_hours_utc:
            return False, f"Hour {current_hour} is blocked (learned to be unprofitable)"
        
        # Check signal quality thresholds (these evolve over time)
        if signal:
            conviction = signal.get('conviction_score', 0)
            if conviction < self.config.min_conviction_score:
                return False, f"Conviction {conviction} < threshold {self.config.min_conviction_score}"
            
            wallet_wr = signal.get('wallet_win_rate', 0)
            if wallet_wr < self.config.min_wallet_win_rate:
                return False, f"Wallet WR {wallet_wr:.1%} < threshold {self.config.min_wallet_win_rate:.1%}"
            
            liquidity = signal.get('liquidity', 0)
            if liquidity < self.config.min_liquidity_usd:
                return False, f"Liquidity ${liquidity:,.0f} < threshold ${self.config.min_liquidity_usd:,.0f}"
        
        return True, "OK"
    
    def open_position(self,
                      token_address: str,
                      token_symbol: str,
                      entry_price: float,
                      signal: Dict = None,
                      wallet_data: Dict = None,
                      decision: Dict = None,
                      size_sol: float = None,
                      stop_loss: float = None,
                      take_profit: float = None,
                      trailing_stop: float = None,
                      max_hold_hours: int = None,
                      notes: str = "") -> Optional[int]:
        """
        Open a paper position with comprehensive tracking.
        
        STOP LOSS IS MANDATORY - will use default if not provided.
        """
        can_open, reason = self.can_open_position(signal)
        if not can_open:
            logger.info(f"❌ Cannot open position: {reason}")
            return None
        
        if entry_price <= 0:
            logger.error(f"Invalid entry price: {entry_price}")
            return None
        
        # Determine position size
        if size_sol is None:
            size_sol = self.config.default_position_size_sol
            # Adjust based on conviction if available
            if signal and signal.get('conviction_score', 0) >= 80:
                size_sol = min(self.config.max_position_size_sol, size_sol * 1.5)
        
        size_sol = max(self.config.min_position_size_sol, 
                       min(self.config.max_position_size_sol, size_sol))
        
        # Check balance
        available = self.balance - self.reserved_balance
        if size_sol > available:
            size_sol = available
        
        if size_sol < self.config.min_position_size_sol:
            logger.warning(f"Position size {size_sol} too small after adjustment")
            return None
        
        # MANDATORY STOP LOSS
        stop_loss = stop_loss or self.config.default_stop_loss_pct
        if stop_loss > 0:  # Ensure it's negative
            stop_loss = -abs(stop_loss)
        
        take_profit = take_profit or self.config.default_take_profit_pct
        trailing_stop = trailing_stop or self.config.default_trailing_stop_pct
        max_hold_hours = max_hold_hours or self.config.max_hold_hours
        
        tokens_bought = size_sol / entry_price
        
        # Build rich context
        now = datetime.utcnow()
        context = EntryContext(
            wallet_address=wallet_data.get('address', '') if wallet_data else '',
            wallet_cluster=wallet_data.get('cluster', 'UNKNOWN') if wallet_data else 'UNKNOWN',
            wallet_win_rate=wallet_data.get('win_rate', 0) if wallet_data else 0,
            wallet_roi_7d=wallet_data.get('roi_7d', 0) if wallet_data else 0,
            wallet_trades_count=wallet_data.get('trades_count', 0) if wallet_data else 0,
            
            liquidity_usd=signal.get('liquidity', 0) if signal else 0,
            volume_24h_usd=signal.get('volume_24h', 0) if signal else 0,
            market_cap_usd=signal.get('market_cap', 0) if signal else 0,
            token_age_hours=signal.get('token_age_hours', 0) if signal else 0,
            holder_count=signal.get('holder_count', 0) if signal else 0,
            
            conviction_score=decision.get('conviction_score', 50) if decision else 50,
            signal_wallets_count=decision.get('wallet_count', 1) if decision else 1,
            aggregated_signal=decision.get('aggregated', False) if decision else False,
            llm_called=decision.get('llm_called', False) if decision else False,
            llm_adjustment=decision.get('llm_adjustment', 0) if decision else 0,
            
            # TIME DATA
            entry_hour_utc=now.hour,
            entry_day_of_week=now.weekday(),
            entry_timestamp=now.isoformat(),
            
            sol_price_usd=signal.get('sol_price', 0) if signal else 0,
            market_regime=decision.get('regime', 'NEUTRAL') if decision else 'NEUTRAL',
            
            strategy_name=decision.get('strategy', 'learning_v4') if decision else 'learning_v4',
            entry_reason=decision.get('reason', '') if decision else '',
            learning_phase=self.config.current_phase,
            iteration_number=self.config.iteration_count
        )
        
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.execute("""
                    INSERT INTO learning_positions (
                        token_address, token_symbol, entry_price, entry_time,
                        size_sol, tokens_bought, stop_loss_pct, take_profit_pct,
                        trailing_stop_pct, max_hold_hours, peak_price, lowest_price,
                        entry_hour_utc, entry_day_of_week,
                        entry_context_json, notes, learning_phase, iteration_number
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    token_address, token_symbol, entry_price, now,
                    size_sol, tokens_bought, stop_loss, take_profit,
                    trailing_stop, max_hold_hours, entry_price, entry_price,
                    now.hour, now.weekday(),
                    json.dumps(asdict(context)), notes,
                    self.config.current_phase, self.config.iteration_count
                ))
                
                position_id = cursor.lastrowid
                
                # Update account
                self.reserved_balance += size_sol
                conn.execute("""
                    UPDATE learning_account
                    SET reserved_balance = ?, updated_at = ?
                    WHERE id = 1
                """, (self.reserved_balance, now))
                
                # Log to journal
                conn.execute("""
                    INSERT INTO learning_journal (position_id, event_type, event_data, notes)
                    VALUES (?, 'ENTRY', ?, ?)
                """, (position_id, json.dumps({
                    'price': entry_price,
                    'size_sol': size_sol,
                    'conviction': context.conviction_score,
                    'wallet_wr': context.wallet_win_rate,
                    'hour_utc': now.hour,
                    'phase': self.config.current_phase
                }), notes))
            
            self.open_position_count += 1
            
            logger.info(f"📥 OPENED: {token_symbol} (Learning Phase: {self.config.current_phase})")
            logger.info(f"   Size: {size_sol:.4f} SOL @ ${entry_price:.8f}")
            logger.info(f"   Stop: {stop_loss}% | TP: {take_profit}% | Trail: {trailing_stop}%")
            logger.info(f"   Hour (UTC): {now.hour} | Day: {now.strftime('%A')}")
            logger.info(f"   Conviction: {context.conviction_score:.0f} | Wallet WR: {context.wallet_win_rate:.1%}")
            
            return position_id
    
    def close_position(self,
                       position_id: int,
                       exit_price: float,
                       exit_reason: ExitReason = ExitReason.MANUAL,
                       notes: str = "") -> Optional[Dict]:
        """
        Close a position with comprehensive result tracking.
        
        Returns complete trade result including ACCURATE hold time.
        """
        with self._lock:
            with self._get_connection() as conn:
                pos = conn.execute("""
                    SELECT * FROM learning_positions WHERE id = ? AND status = 'open'
                """, (position_id,)).fetchone()
                
                if not pos:
                    logger.warning(f"Position {position_id} not found or already closed")
                    return None
                
                # Calculate PnL
                entry_price = pos['entry_price']
                size_sol = pos['size_sol']
                tokens = pos['tokens_bought']
                entry_time = datetime.fromisoformat(pos['entry_time'])
                now = datetime.utcnow()
                
                exit_value = tokens * exit_price
                pnl_sol = exit_value - size_sol
                pnl_pct = ((exit_price / entry_price) - 1) * 100 if entry_price > 0 else 0
                
                # ACCURATE hold time calculation
                hold_minutes = (now - entry_time).total_seconds() / 60
                
                # Exit time tracking
                exit_hour_utc = now.hour
                exit_day_of_week = now.weekday()
                
                # Determine status
                status = TradeStatus.CLOSED.value
                if exit_reason in [ExitReason.STOP_LOSS, ExitReason.TRAILING_STOP]:
                    status = TradeStatus.STOPPED.value
                elif exit_reason == ExitReason.TIME_STOP:
                    status = TradeStatus.EXPIRED.value
                
                # Update position
                conn.execute("""
                    UPDATE learning_positions
                    SET status = ?, exit_price = ?, exit_time = ?, exit_reason = ?,
                        pnl_sol = ?, pnl_pct = ?, hold_duration_minutes = ?,
                        exit_hour_utc = ?, exit_day_of_week = ?
                    WHERE id = ?
                """, (status, exit_price, now, exit_reason.value,
                      pnl_sol, pnl_pct, hold_minutes,
                      exit_hour_utc, exit_day_of_week, position_id))
                
                # Update account
                self.balance += exit_value
                self.reserved_balance -= size_sol
                self.total_trades += 1
                self.total_pnl += pnl_sol
                
                is_win = pnl_sol > 0
                if is_win:
                    self.winning_trades += 1
                    self.current_streak = max(1, self.current_streak + 1) if self.current_streak >= 0 else 1
                else:
                    self.current_streak = min(-1, self.current_streak - 1) if self.current_streak <= 0 else -1
                
                if self.balance > self.peak_balance:
                    self.peak_balance = self.balance
                drawdown = ((self.peak_balance - self.balance) / self.peak_balance) * 100
                
                conn.execute("""
                    UPDATE learning_account
                    SET current_balance = ?, reserved_balance = ?, total_trades = ?,
                        winning_trades = ?, total_pnl_sol = ?, current_streak = ?,
                        peak_balance = ?, max_drawdown_pct = MAX(max_drawdown_pct, ?),
                        best_trade_pnl_pct = MAX(best_trade_pnl_pct, ?),
                        worst_trade_pnl_pct = MIN(worst_trade_pnl_pct, ?),
                        last_trade_time = ?, updated_at = ?
                    WHERE id = 1
                """, (self.balance, self.reserved_balance, self.total_trades,
                      self.winning_trades, self.total_pnl, self.current_streak,
                      self.peak_balance, drawdown, pnl_pct, pnl_pct,
                      now, now))
                
                # Update hourly performance stats
                self._update_hourly_stats(conn, pos['entry_hour_utc'], 
                                         pos['entry_day_of_week'], pnl_sol, pnl_pct, is_win)
                
                # Log to journal
                conn.execute("""
                    INSERT INTO learning_journal (position_id, event_type, event_data, notes)
                    VALUES (?, 'EXIT', ?, ?)
                """, (position_id, json.dumps({
                    'price': exit_price,
                    'pnl_sol': pnl_sol,
                    'pnl_pct': pnl_pct,
                    'reason': exit_reason.value,
                    'hold_minutes': hold_minutes,
                    'entry_hour': pos['entry_hour_utc'],
                    'exit_hour': exit_hour_utc
                }), notes))
            
            self.open_position_count -= 1
            
            emoji = "✅" if is_win else "❌"
            logger.info(f"📤 CLOSED: {pos['token_symbol']} ({exit_reason.value})")
            logger.info(f"   {emoji} PnL: {pnl_sol:+.4f} SOL ({pnl_pct:+.1f}%)")
            logger.info(f"   Hold time: {hold_minutes:.1f} minutes ({hold_minutes/60:.2f} hours)")
            logger.info(f"   Entry hour: {pos['entry_hour_utc']} UTC | Exit hour: {exit_hour_utc} UTC")
            
            return {
                'position_id': position_id,
                'token_symbol': pos['token_symbol'],
                'token_address': pos['token_address'],
                'entry_price': entry_price,
                'exit_price': exit_price,
                'size_sol': size_sol,
                'pnl_sol': pnl_sol,
                'pnl_pct': pnl_pct,
                'exit_reason': exit_reason.value,
                'hold_minutes': hold_minutes,  # ACCURATE hold time
                'hold_duration_minutes': hold_minutes,  # Alias for compatibility
                'is_win': is_win,
                'entry_hour_utc': pos['entry_hour_utc'],
                'exit_hour_utc': exit_hour_utc
            }
    
    def _update_hourly_stats(self, conn, hour_utc: int, day_of_week: int,
                            pnl_sol: float, pnl_pct: float, is_win: bool):
        """Update hourly performance statistics for diurnal analysis"""
        # Update aggregate (all days)
        conn.execute("""
            INSERT INTO hourly_performance (hour_utc, day_of_week, trades_count, wins, losses, total_pnl_sol)
            VALUES (?, NULL, 1, ?, ?, ?)
            ON CONFLICT(hour_utc, day_of_week) DO UPDATE SET
                trades_count = trades_count + 1,
                wins = wins + ?,
                losses = losses + ?,
                total_pnl_sol = total_pnl_sol + ?,
                updated_at = CURRENT_TIMESTAMP
        """, (hour_utc, 1 if is_win else 0, 0 if is_win else 1, pnl_sol,
              1 if is_win else 0, 0 if is_win else 1, pnl_sol))
        
        # Update specific day
        conn.execute("""
            INSERT INTO hourly_performance (hour_utc, day_of_week, trades_count, wins, losses, total_pnl_sol)
            VALUES (?, ?, 1, ?, ?, ?)
            ON CONFLICT(hour_utc, day_of_week) DO UPDATE SET
                trades_count = trades_count + 1,
                wins = wins + ?,
                losses = losses + ?,
                total_pnl_sol = total_pnl_sol + ?,
                updated_at = CURRENT_TIMESTAMP
        """, (hour_utc, day_of_week, 1 if is_win else 0, 0 if is_win else 1, pnl_sol,
              1 if is_win else 0, 0 if is_win else 1, pnl_sol))
    
    def get_open_positions(self) -> List[Dict]:
        """Get all open positions"""
        with self._get_connection() as conn:
            rows = conn.execute("""
                SELECT * FROM learning_positions WHERE status = 'open'
                ORDER BY entry_time DESC
            """).fetchall()
            return [dict(r) for r in rows]
    
    def get_closed_positions(self, days: int = 7, limit: int = 500) -> List[Dict]:
        """Get closed positions for analysis"""
        cutoff = datetime.utcnow() - timedelta(days=days)
        with self._get_connection() as conn:
            rows = conn.execute("""
                SELECT * FROM learning_positions 
                WHERE status != 'open' AND exit_time > ?
                ORDER BY exit_time DESC
                LIMIT ?
            """, (cutoff, limit)).fetchall()
            return [dict(r) for r in rows]
    
    # =========================================================================
    # EXIT MONITORING
    # =========================================================================
    
    def start_monitor(self):
        """Start background exit monitoring"""
        if self._monitor_running:
            return
        
        self._monitor_running = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        logger.info("🔄 Exit monitor started")
    
    def stop_monitor(self):
        """Stop background monitoring"""
        self._monitor_running = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
        logger.info("⏹️ Exit monitor stopped")
    
    def _monitor_loop(self):
        """Background monitoring loop"""
        while self._monitor_running:
            try:
                self._check_all_exits()
            except Exception as e:
                logger.error(f"Monitor error: {e}")
            time.sleep(self.config.price_check_interval_seconds)
    
    def _check_all_exits(self):
        """Check all open positions for exit conditions"""
        positions = self.get_open_positions()
        for pos in positions:
            try:
                self._check_position_exit(pos)
            except Exception as e:
                logger.error(f"Error checking position {pos['id']}: {e}")
    
    def _check_position_exit(self, pos: Dict):
        """Check a single position for exit conditions"""
        position_id = pos['id']
        token_address = pos['token_address']
        entry_price = pos['entry_price']
        entry_time = datetime.fromisoformat(pos['entry_time'])
        peak_price = pos['peak_price'] or entry_price
        
        # Get current price
        current_price = self._get_token_price(token_address)
        if current_price <= 0:
            return
        
        # Update price tracking
        self._update_price_tracking(position_id, current_price, entry_price, peak_price)
        
        # Calculate metrics
        pnl_pct = ((current_price / entry_price) - 1) * 100 if entry_price > 0 else 0
        from_peak_pct = ((current_price / peak_price) - 1) * 100 if peak_price > 0 else 0
        hold_hours = (datetime.utcnow() - entry_time).total_seconds() / 3600
        
        exit_reason = None
        
        # 1. MANDATORY STOP LOSS
        if pnl_pct <= pos['stop_loss_pct']:
            exit_reason = ExitReason.STOP_LOSS
        
        # 2. Take profit
        elif pnl_pct >= pos['take_profit_pct']:
            exit_reason = ExitReason.TAKE_PROFIT
        
        # 3. Trailing stop (only if in profit)
        elif pnl_pct > 0 and from_peak_pct <= -pos['trailing_stop_pct']:
            exit_reason = ExitReason.TRAILING_STOP
        
        # 4. Time stop
        elif hold_hours >= pos['max_hold_hours']:
            exit_reason = ExitReason.TIME_STOP
        
        if exit_reason:
            logger.info(f"🚨 Exit triggered: {pos['token_symbol']} - {exit_reason.value}")
            self.close_position(position_id, current_price, exit_reason)
    
    def _update_price_tracking(self, position_id: int, current_price: float,
                               entry_price: float, old_peak: float):
        """Update price tracking for a position"""
        new_peak = max(old_peak, current_price)
        with self._get_connection() as conn:
            conn.execute("""
                UPDATE learning_positions
                SET current_price = ?, peak_price = ?,
                    lowest_price = MIN(COALESCE(lowest_price, ?), ?),
                    peak_unrealized_pct = ?,
                    last_price_update = ?
                WHERE id = ?
            """, (current_price, new_peak, current_price, current_price,
                  ((new_peak / entry_price) - 1) * 100 if entry_price > 0 else 0,
                  datetime.utcnow(), position_id))
    
    def _get_token_price(self, token_address: str) -> float:
        """Get current token price from DexScreener"""
        try:
            url = f"https://api.dexscreener.com/latest/dex/tokens/{token_address}"
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                pairs = data.get('pairs', [])
                if pairs:
                    return float(pairs[0].get('priceUsd', 0))
        except Exception as e:
            logger.debug(f"Price fetch error for {token_address[:8]}: {e}")
        return 0
    
    # =========================================================================
    # LEARNING SYSTEM
    # =========================================================================
    
    def run_learning_iteration(self, force: bool = False) -> Dict:
        """
        Run a learning iteration to improve the strategy.
        
        This analyzes recent trades and adjusts:
        1. Entry thresholds (conviction, wallet WR, liquidity)
        2. Time restrictions (block unprofitable hours)
        3. Exit parameters
        4. Phase progression
        """
        logger.info("\n" + "=" * 70)
        logger.info("🎓 RUNNING LEARNING ITERATION")
        logger.info("=" * 70)
        
        results = {
            'timestamp': datetime.utcnow().isoformat(),
            'iteration': self.config.iteration_count + 1,
            'phase': self.config.current_phase,
            'status': 'completed'
        }
        
        # Get recent trades
        trades = self.get_closed_positions(days=7)
        
        if len(trades) < self.config.min_trades_for_learning:
            results['status'] = 'insufficient_data'
            results['message'] = f"Need {self.config.min_trades_for_learning} trades, have {len(trades)}"
            logger.info(f"⚠️ Insufficient data: {len(trades)} trades (need {self.config.min_trades_for_learning})")
            return results
        
        results['total_trades'] = len(trades)
        
        # 1. Calculate overall performance
        performance = self._calculate_performance(trades)
        results['performance'] = performance
        logger.info(f"\n📊 Overall Performance:")
        logger.info(f"   Trades: {performance['total_trades']}")
        logger.info(f"   Win Rate: {performance['win_rate']:.1%}")
        logger.info(f"   Total PnL: {performance['total_pnl']:.4f} SOL")
        logger.info(f"   Avg Hold: {performance['avg_hold_minutes']:.0f} min")
        
        # 2. Analyze by entry hour (DIURNAL ANALYSIS)
        hourly_analysis = self._analyze_by_hour(trades)
        results['hourly_analysis'] = hourly_analysis
        
        logger.info(f"\n⏰ Hourly Analysis:")
        for hour, data in sorted(hourly_analysis.items()):
            if data['count'] >= 3:
                status = "✅" if data['win_rate'] >= 0.5 else "❌"
                logger.info(f"   {hour:02d}:00 UTC: {data['count']} trades, "
                           f"{data['win_rate']:.0%} WR, {data['pnl']:.4f} SOL {status}")
        
        # 3. Identify hours to block
        blocked_hours = self._identify_blocked_hours(hourly_analysis)
        preferred_hours = self._identify_preferred_hours(hourly_analysis)
        
        results['blocked_hours'] = blocked_hours
        results['preferred_hours'] = preferred_hours
        
        if blocked_hours:
            logger.info(f"\n🚫 Hours to BLOCK (consistently losing):")
            for h in blocked_hours:
                data = hourly_analysis.get(h, {})
                logger.info(f"   {h:02d}:00 UTC - WR: {data.get('win_rate', 0):.0%}, PnL: {data.get('pnl', 0):.4f} SOL")
        
        if preferred_hours:
            logger.info(f"\n⭐ Preferred hours (consistently winning):")
            for h in preferred_hours:
                data = hourly_analysis.get(h, {})
                logger.info(f"   {h:02d}:00 UTC - WR: {data.get('win_rate', 0):.0%}, PnL: {data.get('pnl', 0):.4f} SOL")
        
        # 4. Analyze by conviction level
        conviction_analysis = self._analyze_by_conviction(trades)
        results['conviction_analysis'] = conviction_analysis
        
        logger.info(f"\n📈 Conviction Analysis:")
        for bucket, data in sorted(conviction_analysis.items()):
            if data['count'] >= 3:
                logger.info(f"   {bucket}: {data['count']} trades, "
                           f"{data['win_rate']:.0%} WR, {data['pnl']:.4f} SOL")
        
        # 5. Analyze by wallet win rate
        wallet_wr_analysis = self._analyze_by_wallet_wr(trades)
        results['wallet_wr_analysis'] = wallet_wr_analysis
        
        logger.info(f"\n👛 Wallet Win Rate Analysis:")
        for bucket, data in sorted(wallet_wr_analysis.items()):
            if data['count'] >= 3:
                logger.info(f"   {bucket}: {data['count']} trades, "
                           f"{data['win_rate']:.0%} WR, {data['pnl']:.4f} SOL")
        
        # 6. Analyze by exit reason
        exit_analysis = self._analyze_by_exit_reason(trades)
        results['exit_analysis'] = exit_analysis
        
        logger.info(f"\n🚪 Exit Reason Analysis:")
        for reason, data in exit_analysis.items():
            logger.info(f"   {reason}: {data['count']} trades, "
                       f"{data['win_rate']:.0%} WR, {data['avg_pnl']:.1f}% avg")
        
        # 7. Generate recommendations and update config
        recommendations = self._generate_recommendations(
            performance, hourly_analysis, conviction_analysis, 
            wallet_wr_analysis, exit_analysis
        )
        results['recommendations'] = recommendations
        
        logger.info(f"\n💡 Recommendations:")
        for rec in recommendations:
            logger.info(f"   • {rec}")
        
        # 8. Apply learned parameters
        self._apply_learning(results)
        
        # 9. Check for phase progression
        new_phase = self._check_phase_progression(performance)
        if new_phase != self.config.current_phase:
            results['phase_changed'] = True
            results['new_phase'] = new_phase
            logger.info(f"\n🎯 PHASE PROGRESSION: {self.config.current_phase} → {new_phase}")
        
        # 10. Save iteration results
        self.config.iteration_count += 1
        self._save_iteration(results)
        self._save_config()
        
        logger.info(f"\n✅ Learning iteration {self.config.iteration_count} complete")
        logger.info("=" * 70 + "\n")
        
        return results
    
    def _calculate_performance(self, trades: List[Dict]) -> Dict:
        """Calculate overall performance metrics"""
        if not trades:
            return {'total_trades': 0, 'win_rate': 0, 'total_pnl': 0}
        
        wins = sum(1 for t in trades if (t.get('pnl_sol') or 0) > 0)
        total_pnl = sum(t.get('pnl_sol', 0) or 0 for t in trades)
        hold_times = [t.get('hold_duration_minutes', 0) or 0 for t in trades]
        
        return {
            'total_trades': len(trades),
            'winning_trades': wins,
            'win_rate': wins / len(trades) if trades else 0,
            'total_pnl': total_pnl,
            'avg_pnl': total_pnl / len(trades) if trades else 0,
            'avg_hold_minutes': statistics.mean(hold_times) if hold_times else 0,
            'median_hold_minutes': statistics.median(hold_times) if hold_times else 0
        }
    
    def _analyze_by_hour(self, trades: List[Dict]) -> Dict:
        """Analyze performance by entry hour (UTC)"""
        by_hour = defaultdict(lambda: {'count': 0, 'wins': 0, 'pnl': 0, 'pnl_pcts': []})
        
        for t in trades:
            hour = t.get('entry_hour_utc')
            if hour is None:
                # Try to extract from entry_context
                try:
                    context = json.loads(t.get('entry_context_json', '{}'))
                    hour = context.get('entry_hour_utc')
                except:
                    pass
            
            if hour is not None:
                by_hour[hour]['count'] += 1
                if (t.get('pnl_sol') or 0) > 0:
                    by_hour[hour]['wins'] += 1
                by_hour[hour]['pnl'] += t.get('pnl_sol', 0) or 0
                by_hour[hour]['pnl_pcts'].append(t.get('pnl_pct', 0) or 0)
        
        # Calculate win rates
        for hour in by_hour:
            count = by_hour[hour]['count']
            by_hour[hour]['win_rate'] = by_hour[hour]['wins'] / count if count > 0 else 0
            by_hour[hour]['avg_pnl_pct'] = statistics.mean(by_hour[hour]['pnl_pcts']) if by_hour[hour]['pnl_pcts'] else 0
        
        return dict(by_hour)
    
    def _analyze_by_conviction(self, trades: List[Dict]) -> Dict:
        """Analyze performance by conviction score bucket"""
        buckets = {
            '30-50': {'min': 30, 'max': 50},
            '50-60': {'min': 50, 'max': 60},
            '60-70': {'min': 60, 'max': 70},
            '70-80': {'min': 70, 'max': 80},
            '80-90': {'min': 80, 'max': 90},
            '90+': {'min': 90, 'max': 101}
        }
        
        results = {b: {'count': 0, 'wins': 0, 'pnl': 0} for b in buckets}
        
        for t in trades:
            try:
                context = json.loads(t.get('entry_context_json', '{}'))
                conviction = context.get('conviction_score', 50)
                
                for bucket, bounds in buckets.items():
                    if bounds['min'] <= conviction < bounds['max']:
                        results[bucket]['count'] += 1
                        if (t.get('pnl_sol') or 0) > 0:
                            results[bucket]['wins'] += 1
                        results[bucket]['pnl'] += t.get('pnl_sol', 0) or 0
                        break
            except:
                pass
        
        for bucket in results:
            count = results[bucket]['count']
            results[bucket]['win_rate'] = results[bucket]['wins'] / count if count > 0 else 0
        
        return results
    
    def _analyze_by_wallet_wr(self, trades: List[Dict]) -> Dict:
        """Analyze performance by wallet win rate bucket"""
        buckets = {
            '40-50%': {'min': 0.40, 'max': 0.50},
            '50-55%': {'min': 0.50, 'max': 0.55},
            '55-60%': {'min': 0.55, 'max': 0.60},
            '60-65%': {'min': 0.60, 'max': 0.65},
            '65-70%': {'min': 0.65, 'max': 0.70},
            '70%+': {'min': 0.70, 'max': 1.01}
        }
        
        results = {b: {'count': 0, 'wins': 0, 'pnl': 0} for b in buckets}
        
        for t in trades:
            try:
                context = json.loads(t.get('entry_context_json', '{}'))
                wallet_wr = context.get('wallet_win_rate', 0.5)
                
                for bucket, bounds in buckets.items():
                    if bounds['min'] <= wallet_wr < bounds['max']:
                        results[bucket]['count'] += 1
                        if (t.get('pnl_sol') or 0) > 0:
                            results[bucket]['wins'] += 1
                        results[bucket]['pnl'] += t.get('pnl_sol', 0) or 0
                        break
            except:
                pass
        
        for bucket in results:
            count = results[bucket]['count']
            results[bucket]['win_rate'] = results[bucket]['wins'] / count if count > 0 else 0
        
        return results
    
    def _analyze_by_exit_reason(self, trades: List[Dict]) -> Dict:
        """Analyze performance by exit reason"""
        by_reason = defaultdict(lambda: {'count': 0, 'wins': 0, 'pnl': 0, 'pnl_pcts': []})
        
        for t in trades:
            reason = t.get('exit_reason', 'UNKNOWN')
            by_reason[reason]['count'] += 1
            if (t.get('pnl_sol') or 0) > 0:
                by_reason[reason]['wins'] += 1
            by_reason[reason]['pnl'] += t.get('pnl_sol', 0) or 0
            by_reason[reason]['pnl_pcts'].append(t.get('pnl_pct', 0) or 0)
        
        for reason in by_reason:
            count = by_reason[reason]['count']
            by_reason[reason]['win_rate'] = by_reason[reason]['wins'] / count if count > 0 else 0
            by_reason[reason]['avg_pnl'] = statistics.mean(by_reason[reason]['pnl_pcts']) if by_reason[reason]['pnl_pcts'] else 0
        
        return dict(by_reason)
    
    def _identify_blocked_hours(self, hourly_analysis: Dict) -> List[int]:
        """Identify hours that should be blocked (consistently losing)"""
        blocked = []
        for hour, data in hourly_analysis.items():
            # Block if: enough samples AND consistently losing
            if data['count'] >= 5 and data['win_rate'] < 0.35 and data['pnl'] < 0:
                blocked.append(hour)
        return sorted(blocked)
    
    def _identify_preferred_hours(self, hourly_analysis: Dict) -> List[int]:
        """Identify preferred trading hours (consistently winning)"""
        preferred = []
        for hour, data in hourly_analysis.items():
            # Prefer if: enough samples AND good performance
            if data['count'] >= 5 and data['win_rate'] >= 0.55 and data['pnl'] > 0:
                preferred.append(hour)
        return sorted(preferred)
    
    def _generate_recommendations(self, performance: Dict, hourly: Dict,
                                  conviction: Dict, wallet_wr: Dict, 
                                  exit_reasons: Dict) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        # Overall performance assessment
        wr = performance['win_rate']
        pnl = performance['total_pnl']
        
        if wr < 0.45:
            recommendations.append(f"⚠️ Win rate too low ({wr:.0%}). Need to tighten entry filters.")
        elif wr >= 0.55:
            recommendations.append(f"✅ Win rate good ({wr:.0%}). Continue current approach.")
        
        if pnl < 0:
            recommendations.append(f"⚠️ Negative PnL ({pnl:.4f} SOL). Review stop loss placement.")
        
        # Time-based recommendations
        blocked = self._identify_blocked_hours(hourly)
        if blocked:
            recommendations.append(f"🚫 Block hours {blocked} - consistently unprofitable")
        
        preferred = self._identify_preferred_hours(hourly)
        if preferred:
            recommendations.append(f"⭐ Focus on hours {preferred} - best performance")
        
        # Conviction recommendations
        low_conviction = conviction.get('30-50', {})
        high_conviction = conviction.get('80-90', {}) or conviction.get('90+', {})
        
        if low_conviction.get('count', 0) >= 5:
            if low_conviction.get('win_rate', 0) < 0.4:
                recommendations.append("📈 Raise minimum conviction threshold - low scores underperform")
        
        if high_conviction.get('count', 0) >= 5:
            if high_conviction.get('win_rate', 0) >= 0.6:
                recommendations.append("💪 High conviction trades performing well - consider larger sizes")
        
        # Wallet WR recommendations
        low_wr_wallets = wallet_wr.get('40-50%', {})
        high_wr_wallets = wallet_wr.get('70%+', {})
        
        if low_wr_wallets.get('count', 0) >= 5:
            if low_wr_wallets.get('win_rate', 0) < 0.4:
                recommendations.append("👛 Raise minimum wallet WR threshold - low WR wallets underperform")
        
        # Exit reason recommendations
        stop_loss = exit_reasons.get('STOP_LOSS', {})
        if stop_loss.get('count', 0) > performance['total_trades'] * 0.4:
            recommendations.append("🛑 Too many stop losses - consider widening or better entry timing")
        
        time_stop = exit_reasons.get('TIME_STOP', {})
        if time_stop.get('count', 0) >= 5 and time_stop.get('win_rate', 0) < 0.3:
            recommendations.append("⏰ Time stops underperforming - positions held too long without profits")
        
        take_profit = exit_reasons.get('TAKE_PROFIT', {})
        if take_profit.get('count', 0) >= 5 and take_profit.get('win_rate', 1) > 0.9:
            recommendations.append("🎯 Take profits working well - current TP level appropriate")
        
        return recommendations
    
    def _apply_learning(self, results: Dict):
        """Apply learned parameters to config"""
        # Apply blocked hours
        if 'blocked_hours' in results:
            self.config.blocked_hours_utc = results['blocked_hours']
        
        if 'preferred_hours' in results:
            self.config.preferred_hours_utc = results['preferred_hours']
        
        # Progressive threshold tightening based on phase and data
        phase = self.config.current_phase
        performance = results.get('performance', {})
        
        if phase == 'exploration':
            # In exploration, only block clearly bad hours
            pass
        
        elif phase == 'initial':
            # Start raising thresholds
            conviction_analysis = results.get('conviction_analysis', {})
            
            # Find first bucket with >50% win rate
            for bucket in ['50-60', '60-70', '70-80']:
                data = conviction_analysis.get(bucket, {})
                if data.get('count', 0) >= 5 and data.get('win_rate', 0) >= 0.5:
                    new_min = int(bucket.split('-')[0])
                    if new_min > self.config.min_conviction_score:
                        self.config.min_conviction_score = new_min
                        logger.info(f"📊 Raised min conviction to {new_min}")
                    break
        
        elif phase == 'refinement':
            # More aggressive threshold raising
            wallet_analysis = results.get('wallet_wr_analysis', {})
            
            # Find first bucket with >55% win rate
            for bucket in ['50-55%', '55-60%', '60-65%']:
                data = wallet_analysis.get(bucket, {})
                if data.get('count', 0) >= 5 and data.get('win_rate', 0) >= 0.55:
                    new_min = float(bucket.split('-')[0].replace('%', '')) / 100
                    if new_min > self.config.min_wallet_win_rate:
                        self.config.min_wallet_win_rate = new_min
                        logger.info(f"👛 Raised min wallet WR to {new_min:.0%}")
                    break
        
        elif phase in ['optimization', 'production']:
            # Focus on profitability - only trade best setups
            if performance.get('win_rate', 0) < 0.5:
                # Tighten everything
                self.config.min_conviction_score = min(80, self.config.min_conviction_score + 5)
                self.config.min_wallet_win_rate = min(0.65, self.config.min_wallet_win_rate + 0.05)
                logger.info(f"⚡ Tightened thresholds: Conv={self.config.min_conviction_score}, WR={self.config.min_wallet_win_rate:.0%}")
    
    def _check_phase_progression(self, performance: Dict) -> str:
        """Check if we should progress to next learning phase"""
        current = self.config.current_phase
        trades = performance.get('total_trades', 0)
        win_rate = performance.get('win_rate', 0)
        pnl = performance.get('total_pnl', 0)
        
        # Calculate days since phase started
        try:
            phase_start = datetime.fromisoformat(self.config.phase_started_at)
            days_in_phase = (datetime.utcnow() - phase_start).days
        except:
            days_in_phase = 0
        
        if current == 'exploration':
            # Progress after 2 days OR 50+ trades
            if days_in_phase >= 2 or trades >= 50:
                self.config.phase_started_at = datetime.utcnow().isoformat()
                return 'initial'
        
        elif current == 'initial':
            # Progress after 2 more days OR 100+ total trades
            if days_in_phase >= 2 or trades >= 100:
                self.config.phase_started_at = datetime.utcnow().isoformat()
                return 'refinement'
        
        elif current == 'refinement':
            # Progress after 2 more days OR win rate > 50%
            if days_in_phase >= 2 or win_rate >= 0.50:
                self.config.phase_started_at = datetime.utcnow().isoformat()
                return 'optimization'
        
        elif current == 'optimization':
            # Progress to production when profitable and consistent
            if win_rate >= 0.55 and pnl > 0 and trades >= 30:
                self.config.phase_started_at = datetime.utcnow().isoformat()
                return 'production'
        
        return current
    
    def _save_iteration(self, results: Dict):
        """Save learning iteration results"""
        with self._get_connection() as conn:
            conn.execute("""
                INSERT INTO learning_iterations (
                    iteration_number, timestamp, phase,
                    total_trades, win_rate, total_pnl_sol, avg_hold_minutes,
                    new_min_conviction, new_min_wallet_wr, new_min_liquidity,
                    blocked_hours, preferred_hours,
                    analysis_json, recommendations_json,
                    phase_changed, new_phase
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                results.get('iteration', 0),
                results.get('timestamp'),
                results.get('phase'),
                results.get('performance', {}).get('total_trades', 0),
                results.get('performance', {}).get('win_rate', 0),
                results.get('performance', {}).get('total_pnl', 0),
                results.get('performance', {}).get('avg_hold_minutes', 0),
                self.config.min_conviction_score,
                self.config.min_wallet_win_rate,
                self.config.min_liquidity_usd,
                json.dumps(results.get('blocked_hours', [])),
                json.dumps(results.get('preferred_hours', [])),
                json.dumps({
                    'hourly': results.get('hourly_analysis', {}),
                    'conviction': results.get('conviction_analysis', {}),
                    'wallet_wr': results.get('wallet_wr_analysis', {}),
                    'exit': results.get('exit_analysis', {})
                }),
                json.dumps(results.get('recommendations', [])),
                results.get('phase_changed', False),
                results.get('new_phase')
            ))
    
    # =========================================================================
    # ANALYTICS & REPORTING
    # =========================================================================
    
    def get_performance_summary(self) -> Dict:
        """Get comprehensive performance summary"""
        with self._get_connection() as conn:
            account = conn.execute(
                "SELECT * FROM learning_account WHERE id = 1"
            ).fetchone()
            
            if not account:
                return {}
            
            account = dict(account)
            total = account['total_trades'] or 0
            
            if total == 0:
                return {
                    'balance': account['current_balance'],
                    'starting_balance': account['starting_balance'],
                    'return_pct': 0,
                    'total_trades': 0,
                    'win_rate': 0,
                    'phase': self.config.current_phase,
                    'iteration': self.config.iteration_count
                }
            
            return_pct = ((account['current_balance'] / account['starting_balance']) - 1) * 100
            
            return {
                'balance': account['current_balance'],
                'starting_balance': account['starting_balance'],
                'return_pct': return_pct,
                'total_pnl_sol': account['total_pnl_sol'],
                'total_trades': total,
                'winning_trades': account['winning_trades'],
                'win_rate': account['winning_trades'] / total,
                'best_trade_pct': account['best_trade_pnl_pct'],
                'worst_trade_pct': account['worst_trade_pnl_pct'],
                'max_drawdown_pct': account['max_drawdown_pct'],
                'current_streak': account['current_streak'],
                'open_positions': self.open_position_count,
                'phase': self.config.current_phase,
                'iteration': self.config.iteration_count,
                'blocked_hours': self.config.blocked_hours_utc,
                'preferred_hours': self.config.preferred_hours_utc,
                'min_conviction': self.config.min_conviction_score,
                'min_wallet_wr': self.config.min_wallet_win_rate
            }
    
    def get_diurnal_report(self) -> Dict:
        """Get detailed diurnal (time-of-day) performance report"""
        with self._get_connection() as conn:
            rows = conn.execute("""
                SELECT hour_utc, 
                       SUM(trades_count) as trades,
                       SUM(wins) as wins,
                       SUM(losses) as losses,
                       SUM(total_pnl_sol) as pnl
                FROM hourly_performance
                WHERE day_of_week IS NULL
                GROUP BY hour_utc
                ORDER BY hour_utc
            """).fetchall()
            
            report = {}
            for row in rows:
                row = dict(row)
                hour = row['hour_utc']
                trades = row['trades'] or 0
                wins = row['wins'] or 0
                
                report[hour] = {
                    'trades': trades,
                    'wins': wins,
                    'losses': row['losses'] or 0,
                    'pnl_sol': row['pnl'] or 0,
                    'win_rate': wins / trades if trades > 0 else 0,
                    'is_blocked': hour in self.config.blocked_hours_utc,
                    'is_preferred': hour in self.config.preferred_hours_utc
                }
            
            return report
    
    def export_for_strategist(self, filepath: str = "learning_export.json") -> str:
        """Export comprehensive data for external strategy analysis"""
        data = {
            'exported_at': datetime.utcnow().isoformat(),
            'summary': self.get_performance_summary(),
            'diurnal_report': self.get_diurnal_report(),
            'config': asdict(self.config),
            'recent_trades': self.get_closed_positions(days=14, limit=500),
            'learning_iterations': self._get_iteration_history()
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        logger.info(f"📊 Exported learning data to {filepath}")
        return filepath
    
    def _get_iteration_history(self) -> List[Dict]:
        """Get history of learning iterations"""
        with self._get_connection() as conn:
            rows = conn.execute("""
                SELECT * FROM learning_iterations
                ORDER BY iteration_number DESC
                LIMIT 20
            """).fetchall()
            return [dict(r) for r in rows]
    
    def print_status(self):
        """Print comprehensive status"""
        summary = self.get_performance_summary()
        
        print("\n" + "=" * 70)
        print("🎓 LEARNING PAPER TRADER V4 STATUS")
        print("=" * 70)
        
        print(f"\n📊 ACCOUNT:")
        print(f"   Balance: {summary.get('balance', 0):.4f} SOL")
        print(f"   Starting: {summary.get('starting_balance', 0):.4f} SOL")
        print(f"   Return: {summary.get('return_pct', 0):+.1f}%")
        print(f"   Total PnL: {summary.get('total_pnl_sol', 0):+.4f} SOL")
        
        print(f"\n📈 PERFORMANCE:")
        print(f"   Trades: {summary.get('total_trades', 0)}")
        print(f"   Win Rate: {summary.get('win_rate', 0):.1%}")
        print(f"   Best Trade: {summary.get('best_trade_pct', 0):+.1f}%")
        print(f"   Worst Trade: {summary.get('worst_trade_pct', 0):+.1f}%")
        print(f"   Max Drawdown: {summary.get('max_drawdown_pct', 0):.1f}%")
        
        print(f"\n🎓 LEARNING STATUS:")
        print(f"   Phase: {summary.get('phase', 'unknown')}")
        print(f"   Iteration: {summary.get('iteration', 0)}")
        print(f"   Min Conviction: {summary.get('min_conviction', 0)}")
        print(f"   Min Wallet WR: {summary.get('min_wallet_wr', 0):.0%}")
        
        blocked = summary.get('blocked_hours', [])
        preferred = summary.get('preferred_hours', [])
        print(f"\n⏰ TIME RESTRICTIONS:")
        print(f"   Blocked Hours: {blocked if blocked else 'None yet'}")
        print(f"   Preferred Hours: {preferred if preferred else 'None yet'}")
        
        print(f"\n📍 CURRENT:")
        print(f"   Open Positions: {summary.get('open_positions', 0)}")
        print(f"   Current Streak: {summary.get('current_streak', 0)}")
        
        # Show diurnal summary if available
        diurnal = self.get_diurnal_report()
        if diurnal:
            print(f"\n⏰ HOURLY PERFORMANCE (Top 5):")
            sorted_hours = sorted(
                diurnal.items(), 
                key=lambda x: x[1].get('win_rate', 0) * min(5, x[1].get('trades', 0)),
                reverse=True
            )[:5]
            for hour, data in sorted_hours:
                if data['trades'] >= 3:
                    status = "✅" if data['win_rate'] >= 0.5 else "❌"
                    print(f"   {hour:02d}:00 UTC: {data['trades']} trades, "
                          f"{data['win_rate']:.0%} WR, {data['pnl_sol']:+.4f} SOL {status}")
        
        print("\n" + "=" * 70)


# =============================================================================
# INTEGRATION WITH MASTER SYSTEM
# =============================================================================

class LearningPaperTradingEngine:
    """
    Drop-in replacement for PaperTradingEngine that uses the Learning Paper Trader.
    
    Compatible with master_v2.py interface but with learning capabilities.
    """
    
    def __init__(self, db=None, starting_balance: float = 10.0, max_positions: int = 999):
        """
        Initialize the learning paper trading engine.
        
        Note: max_positions is ignored in learning mode (we want unlimited).
        """
        self.db = db  # Keep for compatibility
        
        config = LearningConfig(
            starting_balance_sol=starting_balance,
            max_open_positions=999,  # No limit in learning mode
            enable_auto_exits=True
        )
        
        self._trader = LearningPaperTrader(
            db_path="learning_paper_trades.db",
            config=config
        )
        
        print(f"🎓 Learning Paper Trading Engine initialized")
        print(f"   Phase: {self._trader.config.current_phase}")
        print(f"   Balance: {self._trader.balance:.4f} SOL")
    
    @property
    def balance(self) -> float:
        return self._trader.balance
    
    @property
    def available_balance(self) -> float:
        return self._trader.balance - self._trader.reserved_balance
    
    def can_open_position(self, signal: Dict = None) -> Tuple[bool, str]:
        """Check if we can open a position"""
        return self._trader.can_open_position(signal)
    
    def open_position(self, signal: Dict, decision: Dict, price: float) -> Optional[int]:
        """
        Open a paper position.
        
        Compatible with master_v2.py call signature.
        """
        return self._trader.open_position(
            token_address=signal.get('token_address', ''),
            token_symbol=signal.get('token_symbol', 'UNKNOWN'),
            entry_price=price,
            signal=signal,
            wallet_data={'address': signal.get('wallet', ''), 
                        'win_rate': signal.get('wallet_win_rate', 0.5)},
            decision=decision,
            size_sol=decision.get('position_size_sol'),
            stop_loss=decision.get('stop_loss', -0.15) * 100 if decision.get('stop_loss') else None,
            take_profit=decision.get('take_profit', 0.30) * 100 if decision.get('take_profit') else None
        )
    
    def check_exit(self, position: Dict, current_price: float) -> Optional[str]:
        """Check if position should exit - handled by background monitor"""
        # The LearningPaperTrader handles this automatically
        return None
    
    def close_position(self, position_id: int, exit_reason: str, exit_price: float) -> Optional[Dict]:
        """Close a paper position"""
        reason_map = {
            'STOP_LOSS': ExitReason.STOP_LOSS,
            'TAKE_PROFIT': ExitReason.TAKE_PROFIT,
            'TRAILING_STOP': ExitReason.TRAILING_STOP,
            'TIME_STOP': ExitReason.TIME_STOP,
            'SMART_EXIT': ExitReason.SMART_EXIT,
            'STRATEGIST_EXIT': ExitReason.STRATEGIST_EXIT,
            'MANUAL': ExitReason.MANUAL,
        }
        
        reason_enum = reason_map.get(exit_reason, ExitReason.MANUAL)
        
        result = self._trader.close_position(
            position_id=position_id,
            exit_price=exit_price,
            exit_reason=reason_enum
        )
        
        if result:
            return {
                'pnl_pct': result['pnl_pct'],
                'pnl_sol': result['pnl_sol'],
                'hold_minutes': result['hold_minutes'],
                'hold_duration_minutes': result['hold_minutes']  # Compatibility alias
            }
        return None
    
    def get_open_positions(self) -> List[Dict]:
        """Get all open positions"""
        return self._trader.get_open_positions()
    
    def get_stats(self) -> Dict:
        """Get trading statistics"""
        summary = self._trader.get_performance_summary()
        return {
            'balance': summary.get('balance', 0),
            'starting_balance': summary.get('starting_balance', 0),
            'total_pnl': summary.get('total_pnl_sol', 0),
            'return_pct': summary.get('return_pct', 0),
            'open_positions': summary.get('open_positions', 0),
            'total_trades': summary.get('total_trades', 0),
            'win_rate': summary.get('win_rate', 0),
            'phase': summary.get('phase', 'exploration'),
            'iteration': summary.get('iteration', 0),
            'blocked_hours': summary.get('blocked_hours', []),
            'preferred_hours': summary.get('preferred_hours', [])
        }
    
    def run_learning(self, force: bool = False) -> Dict:
        """Run learning iteration"""
        return self._trader.run_learning_iteration(force=force)
    
    def get_strategy_feedback(self) -> Dict:
        """Get detailed feedback for strategy improvement"""
        return {
            'summary': self._trader.get_performance_summary(),
            'diurnal': self._trader.get_diurnal_report(),
            'recent_trades': self._trader.get_closed_positions(days=7)
        }
    
    def print_status(self):
        """Print detailed status"""
        self._trader.print_status()
    
    def stop(self):
        """Stop the background monitor"""
        self._trader.stop_monitor()


# =============================================================================
# CLI
# =============================================================================

def main():
    import sys
    
    trader = LearningPaperTrader()
    
    if len(sys.argv) < 2:
        trader.print_status()
        return
    
    command = sys.argv[1].lower()
    
    if command == 'status':
        trader.print_status()
    
    elif command == 'learn':
        result = trader.run_learning_iteration(force=True)
        print(f"\nLearning result: {result.get('status')}")
    
    elif command == 'diurnal':
        report = trader.get_diurnal_report()
        print("\n⏰ DIURNAL PERFORMANCE REPORT")
        print("=" * 50)
        for hour in range(24):
            data = report.get(hour, {})
            if data.get('trades', 0) >= 1:
                status = "🚫" if data.get('is_blocked') else ("⭐" if data.get('is_preferred') else "  ")
                print(f"{status} {hour:02d}:00 UTC: {data.get('trades', 0):3d} trades, "
                      f"{data.get('win_rate', 0):5.0%} WR, {data.get('pnl_sol', 0):+8.4f} SOL")
    
    elif command == 'export':
        filepath = sys.argv[2] if len(sys.argv) > 2 else "learning_export.json"
        trader.export_for_strategist(filepath)
    
    elif command == 'help':
        print("""
Learning Paper Trader V4 Commands:
  status    - Show current status
  learn     - Run learning iteration
  diurnal   - Show hourly performance
  export    - Export data for analysis
  help      - Show this help
        """)
    
    else:
        print(f"Unknown command: {command}")
        print("Use 'help' to see available commands")


if __name__ == "__main__":
    main()
