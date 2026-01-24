"""
EFFECTIVE PAPER TRADER V3
=========================

A robust paper trading system designed to:
1. ENFORCE position limits strictly (no more 56 positions when limit is 5)
2. MONITOR exit conditions continuously (stop loss, take profit, trailing, time)
3. TRACK comprehensive data for strategy optimization
4. PROVIDE analytics that inform future trading strategies

Key Improvements:
- Position limits checked at multiple levels with atomic operations
- Background monitor thread for exit condition checking
- Rich analytics schema capturing entry context, market conditions, exit reasons
- Detailed trade journal for post-mortem analysis
- Strategy tagging for A/B testing different approaches
"""

import os
import json
import sqlite3
import threading
import time
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
from contextlib import contextmanager
from collections import defaultdict
from enum import Enum
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)


class ExitReason(Enum):
    """Exit reasons for tracking"""
    STOP_LOSS = "STOP_LOSS"
    TAKE_PROFIT = "TAKE_PROFIT"
    TRAILING_STOP = "TRAILING_STOP"
    TIME_STOP = "TIME_STOP"
    SMART_EXIT = "SMART_EXIT"           # Multiple wallets exiting
    MANUAL = "MANUAL"
    LIQUIDITY_DRIED = "LIQUIDITY_DRIED"
    RUG_DETECTED = "RUG_DETECTED"


class TradeStatus(Enum):
    """Position status"""
    OPEN = "open"
    CLOSED = "closed"
    STOPPED = "stopped"
    EXPIRED = "expired"


@dataclass
class PaperTraderConfig:
    """Configuration with sensible defaults"""
    starting_balance_sol: float = 10.0
    max_open_positions: int = 5
    max_position_size_sol: float = 0.5
    min_position_size_sol: float = 0.1
    default_position_size_sol: float = 0.3
    
    # Exit parameters
    default_stop_loss_pct: float = -12.0    # -12%
    default_take_profit_pct: float = 30.0   # +30%
    default_trailing_stop_pct: float = 8.0  # 8% from peak
    max_hold_hours: int = 12
    
    # Monitoring
    price_check_interval_seconds: int = 30
    enable_auto_exits: bool = True
    
    # Risk management
    max_daily_loss_sol: float = 2.0
    max_consecutive_losses: int = 3
    cooldown_after_losses_minutes: int = 30


@dataclass
class EntryContext:
    """Rich context captured at entry"""
    wallet_source: str
    wallet_cluster: str = "UNKNOWN"
    wallet_win_rate: float = 0.0
    wallet_roi_7d: float = 0.0
    
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
    
    # Market context
    sol_price_usd: float = 0.0
    market_regime: str = "NEUTRAL"
    
    # Strategy metadata
    strategy_name: str = "default"
    strategy_version: str = "1.0"
    entry_reason: str = ""


@dataclass
class Position:
    """A paper trading position with full tracking"""
    id: int
    token_address: str
    token_symbol: str
    
    # Entry data
    entry_price: float
    entry_time: datetime
    size_sol: float
    tokens_bought: float
    
    # Exit parameters (set at entry, can be adjusted)
    stop_loss_pct: float
    take_profit_pct: float
    trailing_stop_pct: float
    max_hold_hours: int
    
    # Live tracking
    current_price: float = 0.0
    peak_price: float = 0.0
    peak_unrealized_pct: float = 0.0
    lowest_price: float = 0.0
    last_price_update: Optional[datetime] = None
    
    # Status
    status: TradeStatus = TradeStatus.OPEN
    exit_price: float = 0.0
    exit_time: Optional[datetime] = None
    exit_reason: Optional[ExitReason] = None
    
    # Final results
    pnl_sol: float = 0.0
    pnl_pct: float = 0.0
    hold_duration_minutes: float = 0.0
    
    # Context (stored as JSON in DB)
    entry_context: Optional[EntryContext] = None
    notes: str = ""


class EffectivePaperTrader:
    """
    Paper trading engine that actually works.
    
    Key features:
    - Atomic position limit checks
    - Continuous exit monitoring
    - Comprehensive analytics
    - Strategy backtesting support
    """
    
    def __init__(self, db_path: str = "paper_trades_v3.db", 
                 config: PaperTraderConfig = None):
        self.db_path = db_path
        self.config = config or PaperTraderConfig()
        self._lock = threading.RLock()
        self._monitor_thread = None
        self._stop_monitoring = threading.Event()
        
        # Initialize database
        self._init_database()
        
        # Load account state
        self._load_account()
        
        # Start monitoring if enabled
        if self.config.enable_auto_exits:
            self._start_monitor()
        
        logger.info(f"📊 Effective Paper Trader initialized")
        logger.info(f"   Balance: {self.balance:.4f} SOL")
        logger.info(f"   Open positions: {self.open_position_count}/{self.config.max_open_positions}")
        logger.info(f"   Lifetime trades: {self.total_trades}")
    
    def _init_database(self):
        """Create comprehensive schema for analytics"""
        with self._get_connection() as conn:
            # Account state (singleton)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS paper_account_v3 (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    starting_balance REAL NOT NULL,
                    current_balance REAL NOT NULL,
                    reserved_balance REAL DEFAULT 0,
                    total_trades INTEGER DEFAULT 0,
                    winning_trades INTEGER DEFAULT 0,
                    total_pnl_sol REAL DEFAULT 0,
                    best_trade_pnl_pct REAL DEFAULT 0,
                    worst_trade_pnl_pct REAL DEFAULT 0,
                    max_drawdown_pct REAL DEFAULT 0,
                    peak_balance REAL,
                    current_streak INTEGER DEFAULT 0,
                    longest_win_streak INTEGER DEFAULT 0,
                    longest_lose_streak INTEGER DEFAULT 0,
                    last_trade_time TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Positions with full tracking
            conn.execute("""
                CREATE TABLE IF NOT EXISTS paper_positions_v3 (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    token_address TEXT NOT NULL,
                    token_symbol TEXT NOT NULL,
                    
                    -- Entry data
                    entry_price REAL NOT NULL,
                    entry_time TIMESTAMP NOT NULL,
                    size_sol REAL NOT NULL,
                    tokens_bought REAL NOT NULL,
                    
                    -- Exit parameters
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
                    
                    -- Rich context (JSON)
                    entry_context_json TEXT,
                    notes TEXT,
                    
                    -- Indexing
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    
                    UNIQUE(token_address, entry_time)
                )
            """)
            
            # Price snapshots for analysis
            conn.execute("""
                CREATE TABLE IF NOT EXISTS position_price_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    position_id INTEGER NOT NULL,
                    price REAL NOT NULL,
                    unrealized_pct REAL NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (position_id) REFERENCES paper_positions_v3(id)
                )
            """)
            
            # Trade journal for detailed notes
            conn.execute("""
                CREATE TABLE IF NOT EXISTS trade_journal (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    position_id INTEGER,
                    event_type TEXT NOT NULL,
                    event_data TEXT,
                    notes TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (position_id) REFERENCES paper_positions_v3(id)
                )
            """)
            
            # Daily performance summaries
            conn.execute("""
                CREATE TABLE IF NOT EXISTS daily_performance (
                    date DATE PRIMARY KEY,
                    starting_balance REAL,
                    ending_balance REAL,
                    trades_opened INTEGER DEFAULT 0,
                    trades_closed INTEGER DEFAULT 0,
                    wins INTEGER DEFAULT 0,
                    losses INTEGER DEFAULT 0,
                    pnl_sol REAL DEFAULT 0,
                    max_drawdown_pct REAL DEFAULT 0,
                    best_trade_pct REAL,
                    worst_trade_pct REAL,
                    avg_hold_minutes REAL,
                    notes TEXT
                )
            """)
            
            # Strategy performance tracking
            conn.execute("""
                CREATE TABLE IF NOT EXISTS strategy_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    strategy_name TEXT NOT NULL,
                    strategy_version TEXT,
                    period_start DATE,
                    period_end DATE,
                    total_trades INTEGER DEFAULT 0,
                    wins INTEGER DEFAULT 0,
                    total_pnl_sol REAL DEFAULT 0,
                    avg_win_pct REAL,
                    avg_loss_pct REAL,
                    profit_factor REAL,
                    sharpe_ratio REAL,
                    max_drawdown_pct REAL,
                    avg_hold_minutes REAL,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(strategy_name, strategy_version, period_start)
                )
            """)
            
            # Create indexes
            conn.execute("CREATE INDEX IF NOT EXISTS idx_positions_status ON paper_positions_v3(status)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_positions_token ON paper_positions_v3(token_address)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_positions_time ON paper_positions_v3(entry_time)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_price_history_pos ON position_price_history(position_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_journal_pos ON trade_journal(position_id)")
    
    @contextmanager
    def _get_connection(self):
        """Thread-safe connection"""
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()
    
    def _load_account(self):
        """Load or create account"""
        with self._get_connection() as conn:
            account = conn.execute(
                "SELECT * FROM paper_account_v3 WHERE id = 1"
            ).fetchone()
            
            if account:
                self.balance = account['current_balance']
                self.reserved_balance = account['reserved_balance']
                self.total_trades = account['total_trades']
                self.winning_trades = account['winning_trades']
                self.total_pnl = account['total_pnl_sol']
                self.peak_balance = account['peak_balance'] or self.config.starting_balance_sol
                self.current_streak = account['current_streak']
            else:
                # Initialize new account
                conn.execute("""
                    INSERT INTO paper_account_v3 
                    (id, starting_balance, current_balance, peak_balance)
                    VALUES (1, ?, ?, ?)
                """, (self.config.starting_balance_sol, 
                      self.config.starting_balance_sol,
                      self.config.starting_balance_sol))
                
                self.balance = self.config.starting_balance_sol
                self.reserved_balance = 0
                self.total_trades = 0
                self.winning_trades = 0
                self.total_pnl = 0
                self.peak_balance = self.config.starting_balance_sol
                self.current_streak = 0
        
        # Count open positions
        self.open_position_count = len(self.get_open_positions())
    
    def _start_monitor(self):
        """Start background exit monitoring"""
        def monitor_loop():
            logger.info("🔄 Exit monitor started")
            while not self._stop_monitoring.is_set():
                try:
                    self._check_all_exits()
                except Exception as e:
                    logger.error(f"Monitor error: {e}")
                time.sleep(self.config.price_check_interval_seconds)
        
        self._monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self._monitor_thread.start()
    
    def stop_monitor(self):
        """Stop the background monitor"""
        self._stop_monitoring.set()
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
    
    # =========================================================================
    # POSITION MANAGEMENT
    # =========================================================================
    
    def can_open_position(self, token_address: str = None) -> Tuple[bool, str]:
        """
        Check if we can open a new position.
        Returns (can_open, reason)
        """
        with self._lock:
            # Check position limit
            open_count = len(self.get_open_positions())
            if open_count >= self.config.max_open_positions:
                return False, f"Max positions reached ({open_count}/{self.config.max_open_positions})"
            
            # Check available balance
            available = self.balance - self.reserved_balance
            if available < self.config.min_position_size_sol:
                return False, f"Insufficient balance ({available:.4f} SOL available)"
            
            # Check if already in this token
            if token_address:
                existing = self._get_position_by_token(token_address)
                if existing:
                    return False, f"Already have position in this token"
            
            # Check daily loss limit
            today_pnl = self._get_today_pnl()
            if today_pnl <= -self.config.max_daily_loss_sol:
                return False, f"Daily loss limit reached ({today_pnl:.4f} SOL)"
            
            # Check consecutive losses
            if self.current_streak <= -self.config.max_consecutive_losses:
                last_loss_time = self._get_last_loss_time()
                if last_loss_time:
                    cooldown_end = last_loss_time + timedelta(
                        minutes=self.config.cooldown_after_losses_minutes
                    )
                    if datetime.now() < cooldown_end:
                        remaining = (cooldown_end - datetime.now()).seconds // 60
                        return False, f"Cooling down after {-self.current_streak} losses ({remaining}min left)"
            
            return True, "OK"
    
    def open_position(self, 
                      token_address: str,
                      token_symbol: str,
                      entry_price: float,
                      size_sol: float = None,
                      context: EntryContext = None,
                      stop_loss_pct: float = None,
                      take_profit_pct: float = None,
                      trailing_stop_pct: float = None,
                      max_hold_hours: int = None,
                      notes: str = "") -> Optional[int]:
        """
        Open a new paper position.
        
        Returns position ID if successful, None otherwise.
        """
        with self._lock:
            # Validate
            can_open, reason = self.can_open_position(token_address)
            if not can_open:
                logger.warning(f"⚠️ Cannot open position: {reason}")
                return None
            
            # Determine position size
            size_sol = size_sol or self.config.default_position_size_sol
            size_sol = min(size_sol, self.config.max_position_size_sol)
            size_sol = max(size_sol, self.config.min_position_size_sol)
            
            available = self.balance - self.reserved_balance
            if size_sol > available:
                size_sol = available
            
            if size_sol < self.config.min_position_size_sol:
                logger.warning(f"⚠️ Position size too small: {size_sol:.4f} SOL")
                return None
            
            # Calculate tokens bought
            if entry_price <= 0:
                logger.error("Entry price must be positive")
                return None
            tokens_bought = size_sol / entry_price
            
            # Set exit parameters
            stop_loss = stop_loss_pct or self.config.default_stop_loss_pct
            take_profit = take_profit_pct or self.config.default_take_profit_pct
            trailing = trailing_stop_pct or self.config.default_trailing_stop_pct
            max_hold = max_hold_hours or self.config.max_hold_hours
            
            # Create context if not provided
            if context is None:
                context = EntryContext(wallet_source="unknown")
            
            # Insert into database
            with self._get_connection() as conn:
                cursor = conn.execute("""
                    INSERT INTO paper_positions_v3
                    (token_address, token_symbol, entry_price, entry_time, size_sol, tokens_bought,
                     stop_loss_pct, take_profit_pct, trailing_stop_pct, max_hold_hours,
                     current_price, peak_price, lowest_price, entry_context_json, notes, status)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'open')
                """, (
                    token_address, token_symbol, entry_price, datetime.now(), size_sol, tokens_bought,
                    stop_loss, take_profit, trailing, max_hold,
                    entry_price, entry_price, entry_price, 
                    json.dumps(asdict(context)), notes
                ))
                
                position_id = cursor.lastrowid
                
                # Update account balance
                self.balance -= size_sol
                self.reserved_balance += size_sol
                conn.execute("""
                    UPDATE paper_account_v3 
                    SET current_balance = ?, reserved_balance = ?, updated_at = ?
                    WHERE id = 1
                """, (self.balance, self.reserved_balance, datetime.now()))
                
                # Log to journal
                conn.execute("""
                    INSERT INTO trade_journal (position_id, event_type, event_data, notes)
                    VALUES (?, 'ENTRY', ?, ?)
                """, (position_id, json.dumps({
                    'price': entry_price,
                    'size_sol': size_sol,
                    'conviction': context.conviction_score,
                    'wallet': context.wallet_source[:20] if context.wallet_source else 'unknown'
                }), notes))
                
                # Update daily stats
                self._update_daily_stats(conn, trades_opened=1)
            
            self.open_position_count += 1
            
            logger.info(f"📥 OPENED: {token_symbol}")
            logger.info(f"   Size: {size_sol:.4f} SOL @ ${entry_price:.8f}")
            logger.info(f"   Stop: {stop_loss}% | TP: {take_profit}% | Trail: {trailing}%")
            logger.info(f"   Balance: {self.balance:.4f} SOL | Positions: {self.open_position_count}/{self.config.max_open_positions}")
            
            return position_id
    
    def close_position(self, 
                       position_id: int, 
                       exit_price: float,
                       exit_reason: ExitReason = ExitReason.MANUAL,
                       notes: str = "") -> Optional[Dict]:
        """
        Close a paper position.
        
        Returns trade result dict or None.
        """
        with self._lock:
            with self._get_connection() as conn:
                # Get position
                pos = conn.execute("""
                    SELECT * FROM paper_positions_v3 WHERE id = ? AND status = 'open'
                """, (position_id,)).fetchone()
                
                if not pos:
                    logger.warning(f"Position {position_id} not found or already closed")
                    return None
                
                # Calculate PnL
                entry_price = pos['entry_price']
                size_sol = pos['size_sol']
                tokens = pos['tokens_bought']
                entry_time = datetime.fromisoformat(pos['entry_time'])
                
                exit_value = tokens * exit_price
                pnl_sol = exit_value - size_sol
                pnl_pct = ((exit_price / entry_price) - 1) * 100 if entry_price > 0 else 0
                hold_minutes = (datetime.now() - entry_time).total_seconds() / 60
                
                # Determine status
                status = TradeStatus.CLOSED.value
                if exit_reason in [ExitReason.STOP_LOSS, ExitReason.TRAILING_STOP]:
                    status = TradeStatus.STOPPED.value
                elif exit_reason == ExitReason.TIME_STOP:
                    status = TradeStatus.EXPIRED.value
                
                # Update position
                conn.execute("""
                    UPDATE paper_positions_v3
                    SET status = ?, exit_price = ?, exit_time = ?, exit_reason = ?,
                        pnl_sol = ?, pnl_pct = ?, hold_duration_minutes = ?
                    WHERE id = ?
                """, (status, exit_price, datetime.now(), exit_reason.value,
                      pnl_sol, pnl_pct, hold_minutes, position_id))
                
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
                
                # Track peak balance and drawdown
                if self.balance > self.peak_balance:
                    self.peak_balance = self.balance
                drawdown = ((self.peak_balance - self.balance) / self.peak_balance) * 100
                
                conn.execute("""
                    UPDATE paper_account_v3
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
                      datetime.now(), datetime.now()))
                
                # Log to journal
                conn.execute("""
                    INSERT INTO trade_journal (position_id, event_type, event_data, notes)
                    VALUES (?, 'EXIT', ?, ?)
                """, (position_id, json.dumps({
                    'price': exit_price,
                    'pnl_sol': pnl_sol,
                    'pnl_pct': pnl_pct,
                    'reason': exit_reason.value,
                    'hold_minutes': hold_minutes
                }), notes))
                
                # Update daily stats
                self._update_daily_stats(
                    conn, 
                    trades_closed=1,
                    wins=1 if is_win else 0,
                    losses=0 if is_win else 1,
                    pnl_sol=pnl_sol,
                    trade_pct=pnl_pct,
                    hold_minutes=hold_minutes
                )
            
            self.open_position_count -= 1
            
            emoji = "✅" if is_win else "❌"
            logger.info(f"📤 CLOSED: {pos['token_symbol']} ({exit_reason.value})")
            logger.info(f"   {emoji} PnL: {pnl_sol:+.4f} SOL ({pnl_pct:+.1f}%)")
            logger.info(f"   Hold time: {hold_minutes:.1f} min")
            logger.info(f"   Balance: {self.balance:.4f} SOL | Positions: {self.open_position_count}/{self.config.max_open_positions}")
            
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
                'hold_minutes': hold_minutes,
                'is_win': is_win
            }
    
    # =========================================================================
    # EXIT MONITORING
    # =========================================================================
    
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
        hold_hours = (datetime.now() - entry_time).total_seconds() / 3600
        
        # Check exit conditions in priority order
        exit_reason = None
        
        # 1. Stop loss
        if pnl_pct <= pos['stop_loss_pct']:
            exit_reason = ExitReason.STOP_LOSS
        
        # 2. Take profit
        elif pnl_pct >= pos['take_profit_pct']:
            exit_reason = ExitReason.TAKE_PROFIT
        
        # 3. Trailing stop (only if we're in profit)
        elif pnl_pct > 0 and from_peak_pct <= -pos['trailing_stop_pct']:
            exit_reason = ExitReason.TRAILING_STOP
        
        # 4. Time stop
        elif hold_hours >= pos['max_hold_hours']:
            exit_reason = ExitReason.TIME_STOP
        
        # Execute exit if triggered
        if exit_reason:
            logger.info(f"🚨 Exit triggered for {pos['token_symbol']}: {exit_reason.value}")
            self.close_position(position_id, current_price, exit_reason)
    
    def _update_price_tracking(self, position_id: int, current_price: float, 
                                entry_price: float, old_peak: float):
        """Update price tracking for a position"""
        new_peak = max(old_peak, current_price)
        unrealized_pct = ((current_price / entry_price) - 1) * 100 if entry_price > 0 else 0
        peak_unrealized = ((new_peak / entry_price) - 1) * 100 if entry_price > 0 else 0
        
        with self._get_connection() as conn:
            conn.execute("""
                UPDATE paper_positions_v3
                SET current_price = ?, peak_price = ?, peak_unrealized_pct = ?,
                    lowest_price = MIN(COALESCE(lowest_price, ?), ?),
                    last_price_update = ?
                WHERE id = ?
            """, (current_price, new_peak, peak_unrealized, current_price, 
                  current_price, datetime.now(), position_id))
            
            # Record price snapshot (every 5 minutes max)
            last_snapshot = conn.execute("""
                SELECT timestamp FROM position_price_history 
                WHERE position_id = ? ORDER BY timestamp DESC LIMIT 1
            """, (position_id,)).fetchone()
            
            should_snapshot = True
            if last_snapshot:
                last_time = datetime.fromisoformat(last_snapshot['timestamp'])
                if (datetime.now() - last_time).total_seconds() < 300:
                    should_snapshot = False
            
            if should_snapshot:
                conn.execute("""
                    INSERT INTO position_price_history (position_id, price, unrealized_pct)
                    VALUES (?, ?, ?)
                """, (position_id, current_price, unrealized_pct))
    
    def _get_token_price(self, token_address: str) -> float:
        """Get current token price from DexScreener"""
        try:
            url = f"https://api.dexscreener.com/latest/dex/tokens/{token_address}"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                pairs = data.get('pairs', [])
                
                if pairs:
                    # Get highest liquidity pair
                    pair = max(pairs, key=lambda p: float(p.get('liquidity', {}).get('usd', 0) or 0))
                    return float(pair.get('priceUsd', 0) or 0)
            
            return 0
        except:
            return 0
    
    # =========================================================================
    # QUERIES
    # =========================================================================
    
    def get_open_positions(self) -> List[Dict]:
        """Get all open positions"""
        with self._get_connection() as conn:
            rows = conn.execute("""
                SELECT * FROM paper_positions_v3 
                WHERE status = 'open'
                ORDER BY entry_time DESC
            """).fetchall()
            return [dict(row) for row in rows]
    
    def _get_position_by_token(self, token_address: str) -> Optional[Dict]:
        """Get open position for a token"""
        with self._get_connection() as conn:
            row = conn.execute("""
                SELECT * FROM paper_positions_v3 
                WHERE token_address = ? AND status = 'open'
                LIMIT 1
            """, (token_address,)).fetchone()
            return dict(row) if row else None
    
    def get_closed_positions(self, days: int = 30, limit: int = 100) -> List[Dict]:
        """Get closed positions"""
        cutoff = datetime.now() - timedelta(days=days)
        with self._get_connection() as conn:
            rows = conn.execute("""
                SELECT * FROM paper_positions_v3 
                WHERE status != 'open' AND exit_time > ?
                ORDER BY exit_time DESC
                LIMIT ?
            """, (cutoff, limit)).fetchall()
            return [dict(row) for row in rows]
    
    def get_position(self, position_id: int) -> Optional[Dict]:
        """Get a specific position"""
        with self._get_connection() as conn:
            row = conn.execute("""
                SELECT * FROM paper_positions_v3 WHERE id = ?
            """, (position_id,)).fetchone()
            return dict(row) if row else None
    
    def _get_today_pnl(self) -> float:
        """Get today's realized PnL"""
        today = datetime.now().date()
        with self._get_connection() as conn:
            result = conn.execute("""
                SELECT COALESCE(SUM(pnl_sol), 0) as total
                FROM paper_positions_v3
                WHERE DATE(exit_time) = ? AND status != 'open'
            """, (today,)).fetchone()
            return result['total']
    
    def _get_last_loss_time(self) -> Optional[datetime]:
        """Get time of last losing trade"""
        with self._get_connection() as conn:
            result = conn.execute("""
                SELECT exit_time FROM paper_positions_v3
                WHERE status != 'open' AND pnl_sol < 0
                ORDER BY exit_time DESC LIMIT 1
            """).fetchone()
            if result:
                return datetime.fromisoformat(result['exit_time'])
            return None
    
    def _update_daily_stats(self, conn, trades_opened: int = 0, trades_closed: int = 0,
                           wins: int = 0, losses: int = 0, pnl_sol: float = 0,
                           trade_pct: float = None, hold_minutes: float = None):
        """Update daily performance stats"""
        today = datetime.now().date()
        
        existing = conn.execute(
            "SELECT * FROM daily_performance WHERE date = ?", (today,)
        ).fetchone()
        
        if existing:
            updates = []
            params = []
            
            if trades_opened:
                updates.append("trades_opened = trades_opened + ?")
                params.append(trades_opened)
            if trades_closed:
                updates.append("trades_closed = trades_closed + ?")
                params.append(trades_closed)
            if wins:
                updates.append("wins = wins + ?")
                params.append(wins)
            if losses:
                updates.append("losses = losses + ?")
                params.append(losses)
            if pnl_sol:
                updates.append("pnl_sol = pnl_sol + ?")
                params.append(pnl_sol)
            if trade_pct is not None:
                updates.append("best_trade_pct = MAX(COALESCE(best_trade_pct, ?), ?)")
                params.extend([trade_pct, trade_pct])
                updates.append("worst_trade_pct = MIN(COALESCE(worst_trade_pct, ?), ?)")
                params.extend([trade_pct, trade_pct])
            
            updates.append("ending_balance = ?")
            params.append(self.balance)
            params.append(today)
            
            if updates:
                conn.execute(f"""
                    UPDATE daily_performance 
                    SET {', '.join(updates)}
                    WHERE date = ?
                """, params)
        else:
            conn.execute("""
                INSERT INTO daily_performance 
                (date, starting_balance, ending_balance, trades_opened, trades_closed,
                 wins, losses, pnl_sol, best_trade_pct, worst_trade_pct)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (today, self.balance, self.balance, trades_opened, trades_closed,
                  wins, losses, pnl_sol, trade_pct, trade_pct))
    
    # =========================================================================
    # ANALYTICS FOR STRATEGIST
    # =========================================================================
    
    def get_performance_summary(self) -> Dict:
        """Get comprehensive performance summary"""
        with self._get_connection() as conn:
            account = conn.execute("SELECT * FROM paper_account_v3 WHERE id = 1").fetchone()
            
            if not account:
                return {}
            
            total = account['total_trades'] or 1
            return_pct = ((account['current_balance'] / account['starting_balance']) - 1) * 100
            
            # Get recent trades for analysis
            recent = conn.execute("""
                SELECT * FROM paper_positions_v3 
                WHERE status != 'open'
                ORDER BY exit_time DESC LIMIT 100
            """).fetchall()
            
            # Calculate detailed metrics
            wins = [dict(r) for r in recent if r['pnl_sol'] > 0]
            losses = [dict(r) for r in recent if r['pnl_sol'] <= 0]
            
            avg_win = sum(r['pnl_pct'] for r in wins) / len(wins) if wins else 0
            avg_loss = sum(r['pnl_pct'] for r in losses) / len(losses) if losses else 0
            
            # Profit factor
            gross_profit = sum(r['pnl_sol'] for r in wins)
            gross_loss = abs(sum(r['pnl_sol'] for r in losses))
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
            
            # By exit reason
            by_reason = defaultdict(lambda: {'count': 0, 'wins': 0, 'pnl': 0})
            for r in recent:
                reason = r['exit_reason'] or 'unknown'
                by_reason[reason]['count'] += 1
                by_reason[reason]['pnl'] += r['pnl_sol']
                if r['pnl_sol'] > 0:
                    by_reason[reason]['wins'] += 1
            
            return {
                'balance': account['current_balance'],
                'starting_balance': account['starting_balance'],
                'return_pct': return_pct,
                'total_pnl_sol': account['total_pnl_sol'],
                'total_trades': account['total_trades'],
                'winning_trades': account['winning_trades'],
                'win_rate': account['winning_trades'] / total,
                'avg_win_pct': avg_win,
                'avg_loss_pct': avg_loss,
                'profit_factor': profit_factor,
                'best_trade_pct': account['best_trade_pnl_pct'],
                'worst_trade_pct': account['worst_trade_pnl_pct'],
                'max_drawdown_pct': account['max_drawdown_pct'],
                'current_streak': account['current_streak'],
                'open_positions': len(self.get_open_positions()),
                'by_exit_reason': dict(by_reason)
            }
    
    def get_strategy_analysis(self, days: int = 14) -> Dict:
        """
        Get detailed analysis for strategy optimization.
        This is what the Strategist needs to learn and improve.
        """
        cutoff = datetime.now() - timedelta(days=days)
        
        with self._get_connection() as conn:
            trades = conn.execute("""
                SELECT * FROM paper_positions_v3 
                WHERE status != 'open' AND exit_time > ?
                ORDER BY exit_time DESC
            """, (cutoff,)).fetchall()
            
            trades = [dict(t) for t in trades]
        
        if not trades:
            return {'status': 'no_data', 'trades': 0}
        
        # Parse entry contexts
        for t in trades:
            if t.get('entry_context_json'):
                try:
                    t['context'] = json.loads(t['entry_context_json'])
                except:
                    t['context'] = {}
            else:
                t['context'] = {}
        
        analysis = {
            'period_days': days,
            'total_trades': len(trades),
            'wins': sum(1 for t in trades if t['pnl_sol'] > 0),
            'total_pnl_sol': sum(t['pnl_sol'] for t in trades),
            
            # By conviction bucket
            'by_conviction': self._analyze_by_conviction(trades),
            
            # By exit reason
            'by_exit_reason': self._analyze_by_exit_reason(trades),
            
            # By liquidity bucket
            'by_liquidity': self._analyze_by_liquidity(trades),
            
            # By token age
            'by_token_age': self._analyze_by_token_age(trades),
            
            # By wallet cluster
            'by_cluster': self._analyze_by_cluster(trades),
            
            # By hold time
            'by_hold_time': self._analyze_by_hold_time(trades),
            
            # By time of day
            'by_hour': self._analyze_by_hour(trades),
            
            # Optimal ranges (from winning trades)
            'optimal_ranges': self._find_optimal_ranges(trades),
            
            # Problem patterns
            'problems': self._find_problems(trades),
            
            # Recommendations
            'recommendations': self._generate_recommendations(trades)
        }
        
        return analysis
    
    def _analyze_by_conviction(self, trades: List[Dict]) -> Dict:
        """Analyze by conviction score buckets"""
        buckets = {
            '50-60': (50, 60),
            '60-75': (60, 75),
            '75-90': (75, 90),
            '90+': (90, 101)
        }
        
        results = {}
        for name, (lo, hi) in buckets.items():
            matching = [t for t in trades 
                       if lo <= t['context'].get('conviction_score', 50) < hi]
            if matching:
                wins = sum(1 for t in matching if t['pnl_sol'] > 0)
                results[name] = {
                    'count': len(matching),
                    'win_rate': wins / len(matching),
                    'avg_pnl_pct': sum(t['pnl_pct'] for t in matching) / len(matching),
                    'total_pnl_sol': sum(t['pnl_sol'] for t in matching)
                }
        return results
    
    def _analyze_by_exit_reason(self, trades: List[Dict]) -> Dict:
        """Analyze by exit reason"""
        by_reason = defaultdict(lambda: {'count': 0, 'wins': 0, 'total_pnl': 0, 'avg_hold': 0})
        
        for t in trades:
            reason = t.get('exit_reason', 'unknown')
            by_reason[reason]['count'] += 1
            by_reason[reason]['total_pnl'] += t['pnl_sol']
            by_reason[reason]['avg_hold'] += t['hold_duration_minutes']
            if t['pnl_sol'] > 0:
                by_reason[reason]['wins'] += 1
        
        for reason in by_reason:
            count = by_reason[reason]['count']
            if count:
                by_reason[reason]['win_rate'] = by_reason[reason]['wins'] / count
                by_reason[reason]['avg_hold'] /= count
        
        return dict(by_reason)
    
    def _analyze_by_liquidity(self, trades: List[Dict]) -> Dict:
        """Analyze by liquidity bucket"""
        buckets = {
            '30k-50k': (30000, 50000),
            '50k-100k': (50000, 100000),
            '100k+': (100000, float('inf'))
        }
        
        results = {}
        for name, (lo, hi) in buckets.items():
            matching = [t for t in trades 
                       if lo <= t['context'].get('liquidity_usd', 0) < hi]
            if matching:
                wins = sum(1 for t in matching if t['pnl_sol'] > 0)
                results[name] = {
                    'count': len(matching),
                    'win_rate': wins / len(matching),
                    'avg_pnl_pct': sum(t['pnl_pct'] for t in matching) / len(matching)
                }
        return results
    
    def _analyze_by_token_age(self, trades: List[Dict]) -> Dict:
        """Analyze by token age"""
        buckets = {
            '0-2h': (0, 2),
            '2-12h': (2, 12),
            '12-48h': (12, 48),
            '48h+': (48, float('inf'))
        }
        
        results = {}
        for name, (lo, hi) in buckets.items():
            matching = [t for t in trades 
                       if lo <= t['context'].get('token_age_hours', 0) < hi]
            if matching:
                wins = sum(1 for t in matching if t['pnl_sol'] > 0)
                results[name] = {
                    'count': len(matching),
                    'win_rate': wins / len(matching),
                    'avg_pnl_pct': sum(t['pnl_pct'] for t in matching) / len(matching)
                }
        return results
    
    def _analyze_by_cluster(self, trades: List[Dict]) -> Dict:
        """Analyze by wallet cluster"""
        by_cluster = defaultdict(lambda: {'count': 0, 'wins': 0, 'total_pnl': 0})
        
        for t in trades:
            cluster = t['context'].get('wallet_cluster', 'UNKNOWN')
            by_cluster[cluster]['count'] += 1
            by_cluster[cluster]['total_pnl'] += t['pnl_sol']
            if t['pnl_sol'] > 0:
                by_cluster[cluster]['wins'] += 1
        
        for cluster in by_cluster:
            count = by_cluster[cluster]['count']
            if count:
                by_cluster[cluster]['win_rate'] = by_cluster[cluster]['wins'] / count
        
        return dict(by_cluster)
    
    def _analyze_by_hold_time(self, trades: List[Dict]) -> Dict:
        """Analyze by hold duration"""
        buckets = {
            '0-30min': (0, 30),
            '30min-2h': (30, 120),
            '2h-6h': (120, 360),
            '6h-12h': (360, 720),
            '12h+': (720, float('inf'))
        }
        
        results = {}
        for name, (lo, hi) in buckets.items():
            matching = [t for t in trades 
                       if lo <= t['hold_duration_minutes'] < hi]
            if matching:
                wins = sum(1 for t in matching if t['pnl_sol'] > 0)
                results[name] = {
                    'count': len(matching),
                    'win_rate': wins / len(matching),
                    'avg_pnl_pct': sum(t['pnl_pct'] for t in matching) / len(matching)
                }
        return results
    
    def _analyze_by_hour(self, trades: List[Dict]) -> Dict:
        """Analyze by hour of entry"""
        by_hour = defaultdict(lambda: {'count': 0, 'wins': 0, 'total_pnl': 0})
        
        for t in trades:
            entry_time = datetime.fromisoformat(t['entry_time'])
            hour = entry_time.hour
            by_hour[hour]['count'] += 1
            by_hour[hour]['total_pnl'] += t['pnl_sol']
            if t['pnl_sol'] > 0:
                by_hour[hour]['wins'] += 1
        
        for hour in by_hour:
            count = by_hour[hour]['count']
            if count:
                by_hour[hour]['win_rate'] = by_hour[hour]['wins'] / count
        
        return dict(sorted(by_hour.items()))
    
    def _find_optimal_ranges(self, trades: List[Dict]) -> Dict:
        """Find optimal parameter ranges from winning trades"""
        winners = [t for t in trades if t['pnl_sol'] > 0]
        
        if len(winners) < 5:
            return {}
        
        import statistics
        
        def get_range(values):
            if not values:
                return None
            return {
                'min': min(values),
                'p25': sorted(values)[len(values)//4] if len(values) >= 4 else min(values),
                'median': statistics.median(values),
                'p75': sorted(values)[3*len(values)//4] if len(values) >= 4 else max(values),
                'max': max(values)
            }
        
        return {
            'conviction': get_range([t['context'].get('conviction_score', 50) for t in winners]),
            'liquidity': get_range([t['context'].get('liquidity_usd', 0) for t in winners if t['context'].get('liquidity_usd')]),
            'token_age_hours': get_range([t['context'].get('token_age_hours', 0) for t in winners if t['context'].get('token_age_hours')]),
            'hold_minutes': get_range([t['hold_duration_minutes'] for t in winners])
        }
    
    def _find_problems(self, trades: List[Dict]) -> List[str]:
        """Identify problematic patterns"""
        problems = []
        losers = [t for t in trades if t['pnl_sol'] <= 0]
        
        if not losers:
            return problems
        
        # Early stop-outs
        early_stops = [t for t in losers 
                      if t['exit_reason'] == 'STOP_LOSS' 
                      and t['hold_duration_minutes'] < 30]
        if len(early_stops) > len(losers) * 0.3:
            problems.append(f"HIGH_EARLY_STOPOUT: {len(early_stops)}/{len(losers)} losers hit stop within 30min")
        
        # Time stops with unrealized profit
        time_stops = [t for t in losers if t['exit_reason'] == 'TIME_STOP']
        profitable_timeouts = [t for t in time_stops if t['peak_unrealized_pct'] > 15]
        if len(profitable_timeouts) >= 3:
            problems.append(f"MISSED_EXITS: {len(profitable_timeouts)} trades had 15%+ unrealized but hit time stop")
        
        # Low conviction losses
        low_conv = [t for t in losers if t['context'].get('conviction_score', 50) < 60]
        if len(low_conv) > len(losers) * 0.5:
            problems.append(f"LOW_CONVICTION_LOSSES: {len(low_conv)}/{len(losers)} losses were low conviction")
        
        return problems
    
    def _generate_recommendations(self, trades: List[Dict]) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        if len(trades) < 10:
            recommendations.append("Need more trades for reliable analysis (current: {len(trades)})")
            return recommendations
        
        # Analyze patterns
        by_reason = self._analyze_by_exit_reason(trades)
        by_conviction = self._analyze_by_conviction(trades)
        
        # Stop loss analysis
        if 'STOP_LOSS' in by_reason:
            sl_data = by_reason['STOP_LOSS']
            if sl_data['count'] > len(trades) * 0.4:
                recommendations.append("Consider widening stop loss - too many stops triggered")
        
        # Time stop analysis
        if 'TIME_STOP' in by_reason:
            ts_data = by_reason['TIME_STOP']
            if ts_data['win_rate'] < 0.3:
                recommendations.append("Time stops are mostly losers - consider tighter trailing stops")
        
        # Conviction analysis
        if '50-60' in by_conviction and by_conviction['50-60']['win_rate'] < 0.35:
            recommendations.append("Low conviction trades underperforming - raise entry threshold")
        
        if '90+' in by_conviction and by_conviction['90+']['win_rate'] > 0.6:
            recommendations.append("High conviction trades performing well - consider larger position sizes")
        
        return recommendations
    
    def export_for_strategist(self, filepath: str = "paper_trade_export.json"):
        """Export all data for strategist analysis"""
        data = {
            'exported_at': datetime.now().isoformat(),
            'summary': self.get_performance_summary(),
            'analysis': self.get_strategy_analysis(days=30),
            'daily_performance': self._get_daily_data(30),
            'all_trades': self.get_closed_positions(days=30, limit=500)
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        logger.info(f"📊 Exported data to {filepath}")
        return filepath
    
    def _get_daily_data(self, days: int) -> List[Dict]:
        """Get daily performance data"""
        with self._get_connection() as conn:
            rows = conn.execute("""
                SELECT * FROM daily_performance 
                ORDER BY date DESC LIMIT ?
            """, (days,)).fetchall()
            return [dict(row) for row in rows]
    
    # =========================================================================
    # CLI INTERFACE
    # =========================================================================
    
    def print_status(self):
        """Print comprehensive status"""
        stats = self.get_performance_summary()
        
        print("\n" + "="*60)
        print("📊 PAPER TRADING STATUS")
        print("="*60)
        
        print(f"\n💰 Account:")
        print(f"   Balance: {stats['balance']:.4f} SOL ({stats['return_pct']:+.1f}%)")
        print(f"   Total PnL: {stats['total_pnl_sol']:+.4f} SOL")
        print(f"   Max Drawdown: {stats['max_drawdown_pct']:.1f}%")
        
        print(f"\n📈 Trading:")
        print(f"   Trades: {stats['total_trades']} ({stats['win_rate']:.1%} win rate)")
        print(f"   Avg Win: {stats['avg_win_pct']:+.1f}% | Avg Loss: {stats['avg_loss_pct']:+.1f}%")
        print(f"   Profit Factor: {stats['profit_factor']:.2f}")
        print(f"   Current Streak: {stats['current_streak']}")
        
        print(f"\n📦 Open Positions: {stats['open_positions']}/{self.config.max_open_positions}")
        
        open_pos = self.get_open_positions()
        if open_pos:
            for pos in open_pos:
                entry_time = datetime.fromisoformat(pos['entry_time'])
                hold_hours = (datetime.now() - entry_time).total_seconds() / 3600
                current = pos['current_price'] or pos['entry_price']
                pnl_pct = ((current / pos['entry_price']) - 1) * 100 if pos['entry_price'] > 0 else 0
                emoji = "✅" if pnl_pct > 0 else "❌"
                print(f"   {emoji} {pos['token_symbol']}: {pos['size_sol']:.3f} SOL | {pnl_pct:+.1f}% | {hold_hours:.1f}h")
        
        # Exit reason breakdown
        if stats['by_exit_reason']:
            print(f"\n🚪 Exit Reasons:")
            for reason, data in stats['by_exit_reason'].items():
                wr = data['wins'] / data['count'] if data['count'] else 0
                print(f"   {reason}: {data['count']} trades, {wr:.1%} win rate, {data['pnl']:+.4f} SOL")


# =========================================================================
# CLI ENTRY POINT
# =========================================================================

def main():
    """CLI entry point"""
    import sys
    
    trader = EffectivePaperTrader()
    
    if len(sys.argv) < 2:
        trader.print_status()
        return
    
    command = sys.argv[1].lower()
    
    if command == 'status':
        trader.print_status()
    
    elif command == 'analyze':
        analysis = trader.get_strategy_analysis(days=14)
        print("\n📊 STRATEGY ANALYSIS (14 days)")
        print("="*60)
        print(f"Total Trades: {analysis['total_trades']}")
        print(f"Win Rate: {analysis['wins']/max(1, analysis['total_trades']):.1%}")
        print(f"Total PnL: {analysis['total_pnl_sol']:+.4f} SOL")
        
        print("\n📈 By Conviction:")
        for bucket, data in analysis.get('by_conviction', {}).items():
            print(f"   {bucket}: {data['count']} trades, {data['win_rate']:.1%} win, {data['avg_pnl_pct']:+.1f}%")
        
        print("\n🚪 By Exit Reason:")
        for reason, data in analysis.get('by_exit_reason', {}).items():
            print(f"   {reason}: {data['count']} trades, {data['win_rate']:.1%} win")
        
        if analysis.get('problems'):
            print("\n⚠️ Problems Detected:")
            for prob in analysis['problems']:
                print(f"   • {prob}")
        
        if analysis.get('recommendations'):
            print("\n💡 Recommendations:")
            for rec in analysis['recommendations']:
                print(f"   • {rec}")
    
    elif command == 'export':
        filepath = sys.argv[2] if len(sys.argv) > 2 else "paper_trade_export.json"
        trader.export_for_strategist(filepath)
    
    elif command == 'close-stale':
        positions = trader.get_open_positions()
        closed = 0
        for pos in positions:
            entry_time = datetime.fromisoformat(pos['entry_time'])
            hold_hours = (datetime.now() - entry_time).total_seconds() / 3600
            if hold_hours > pos['max_hold_hours']:
                current_price = trader._get_token_price(pos['token_address'])
                if current_price > 0:
                    trader.close_position(pos['id'], current_price, ExitReason.TIME_STOP)
                    closed += 1
        print(f"Closed {closed} stale positions")
    
    elif command == 'help':
        print("""
Effective Paper Trader Commands:
  status       - Show current status
  analyze      - Run strategy analysis
  export [path] - Export data for strategist
  close-stale  - Close positions past max hold time
  help         - Show this help
        """)
    
    else:
        print(f"Unknown command: {command}")
        print("Use 'help' to see available commands")


if __name__ == "__main__":
    main()
