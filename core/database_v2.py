"""
DATABASE V2 - Unified Database Layer with Schema Migrations
Trading System V2 - Optimized for consistency and performance

Features:
- Automatic schema migration
- Connection pooling via context managers
- All tables required by Historian, Strategist, Executioner
- Cached queries for frequently accessed data
"""

import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from contextlib import contextmanager
from functools import lru_cache
import threading
import json


class DatabaseV2:
    """
    Unified database layer with automatic schema management.
    Thread-safe with connection pooling.
    """
    
    SCHEMA_VERSION = 3
    
    def __init__(self, db_path: str = 'swing_traders.db'):
        self.db_path = db_path
        self._local = threading.local()
        self._lock = threading.Lock()
        
        # Initialize/migrate schema
        self._init_schema()
        self._run_migrations()
        
        # In-memory caches
        self._signature_cache = set()
        self._wallet_cache = {}
        self._cache_ttl = 300  # 5 minutes
        self._cache_time = datetime.now()
        
        print(f"✅ Database V2 initialized: {db_path}")
    
    @contextmanager
    def connection(self):
        """Thread-safe connection context manager"""
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()
    
    def _init_schema(self):
        """Create all tables if they don't exist"""
        with self.connection() as conn:
            # Schema version tracking
            conn.execute("""
                CREATE TABLE IF NOT EXISTS schema_version (
                    version INTEGER PRIMARY KEY,
                    applied_at TIMESTAMP
                )
            """)
            
            # 1. Verified Wallets (The Historian)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS verified_wallets (
                    address TEXT PRIMARY KEY,
                    pnl_7d REAL DEFAULT 0,
                    roi_7d REAL DEFAULT 0,
                    win_rate REAL DEFAULT 0,
                    completed_swings INTEGER DEFAULT 0,
                    avg_hold_hours REAL DEFAULT 0,
                    risk_reward_ratio REAL DEFAULT 0,
                    best_trade_pct REAL DEFAULT 0,
                    worst_trade_pct REAL DEFAULT 0,
                    total_volume_sol REAL DEFAULT 0,
                    avg_position_size_sol REAL DEFAULT 0,
                    cluster TEXT DEFAULT 'BALANCED',
                    cluster_updated TIMESTAMP,
                    discovered_at TIMESTAMP,
                    last_updated TIMESTAMP,
                    is_active INTEGER DEFAULT 1
                )
            """)
            
            # 2. Tracked Positions (Core trading data)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS tracked_positions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    wallet_address TEXT NOT NULL,
                    token_address TEXT NOT NULL,
                    token_symbol TEXT,
                    
                    -- Entry data
                    entry_time TIMESTAMP NOT NULL,
                    entry_price REAL,
                    entry_market_cap REAL,
                    entry_liquidity REAL,
                    entry_volume_24h REAL,
                    token_age_hours REAL,
                    position_size_sol REAL,
                    position_size_tokens REAL,
                    
                    -- Strategy data (The Strategist)
                    strategy_name TEXT,
                    conviction_score REAL,
                    stop_loss_pct REAL DEFAULT -0.12,
                    take_profit_pct REAL DEFAULT 0.30,
                    trailing_stop_pct REAL DEFAULT 0.08,
                    max_hold_hours INTEGER DEFAULT 12,
                    wallet_count INTEGER DEFAULT 1,
                    clusters_json TEXT,
                    
                    -- Market context
                    regime TEXT,
                    sol_price_usd REAL,
                    
                    -- Exit data
                    status TEXT DEFAULT 'open',
                    exit_time TIMESTAMP,
                    exit_price REAL,
                    exit_reason TEXT,
                    exit_signature TEXT,
                    detection_method TEXT,
                    profit_pct REAL,
                    profit_sol REAL,
                    hold_duration_minutes REAL,
                    peak_price REAL,
                    peak_unrealized_pct REAL,
                    
                    -- Tracking metadata
                    source TEXT DEFAULT 'live',
                    notes TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    
                    UNIQUE(wallet_address, token_address, entry_time)
                )
            """)
            
            # 3. Position Snapshots (price history)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS position_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    position_id INTEGER,
                    snapshot_time TIMESTAMP,
                    price_usd REAL,
                    market_cap REAL,
                    liquidity REAL,
                    volume_24h REAL,
                    unrealized_pnl_pct REAL,
                    minutes_since_entry REAL,
                    FOREIGN KEY (position_id) REFERENCES tracked_positions(id)
                )
            """)
            
            # 4. Market Conditions (regime tracking)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS market_conditions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TIMESTAMP,
                    sol_price_usd REAL,
                    sol_24h_change_pct REAL,
                    sol_volume_24h REAL,
                    regime TEXT,
                    regime_confidence REAL,
                    btc_dominance REAL,
                    trending_tokens_count INTEGER
                )
            """)
            
            # 5. Processed Signatures (deduplication)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS processed_sigs (
                    signature TEXT PRIMARY KEY,
                    processed_at TIMESTAMP,
                    wallet_address TEXT,
                    trade_type TEXT,
                    token_address TEXT
                )
            """)
            
            # 6. Strategy Registry (The Strategist)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS strategy_registry (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    status TEXT DEFAULT 'CHALLENGER',
                    config_json TEXT,
                    performance_json TEXT,
                    created_at TIMESTAMP,
                    promoted_at TIMESTAMP,
                    evolved_from TEXT,
                    evolution_reason TEXT
                )
            """)
            
            # 7. Paper Trades (testing)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS paper_trades (
                    id TEXT PRIMARY KEY,
                    token_address TEXT,
                    token_symbol TEXT,
                    wallet_address TEXT,
                    entry_time TIMESTAMP,
                    entry_price REAL,
                    entry_liquidity REAL,
                    position_size_sol REAL,
                    conviction_score REAL,
                    stop_loss_pct REAL,
                    take_profit_pct REAL,
                    trailing_stop_pct REAL,
                    max_hold_hours INTEGER,
                    strategy_used TEXT,
                    exit_time TIMESTAMP,
                    exit_price REAL,
                    exit_reason TEXT,
                    final_pnl_pct REAL,
                    final_pnl_sol REAL,
                    hold_duration_minutes INTEGER,
                    peak_price REAL,
                    status TEXT DEFAULT 'open',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # 8. Exit Signals (wallet exit tracking)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS exit_signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    token_address TEXT,
                    token_symbol TEXT,
                    wallet_address TEXT,
                    wallet_cluster TEXT,
                    exit_price REAL,
                    exit_time TIMESTAMP,
                    signature TEXT,
                    UNIQUE(signature)
                )
            """)
            
            # 9. LLM Call Log (cost tracking)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS llm_calls (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TIMESTAMP,
                    call_type TEXT,
                    token_symbol TEXT,
                    input_tokens INTEGER,
                    output_tokens INTEGER,
                    cost_usd REAL,
                    result_summary TEXT
                )
            """)
            
            # Create indices for performance
            conn.execute("CREATE INDEX IF NOT EXISTS idx_positions_status ON tracked_positions(status)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_positions_wallet ON tracked_positions(wallet_address)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_positions_token ON tracked_positions(token_address)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_positions_exit_time ON tracked_positions(exit_time)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_snapshots_position ON position_snapshots(position_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_sigs_wallet ON processed_sigs(wallet_address)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_paper_status ON paper_trades(status)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_exit_signals_token ON exit_signals(token_address)")
    
    def _run_migrations(self):
        """Run any pending schema migrations"""
        with self.connection() as conn:
            # Check current version
            try:
                current = conn.execute("SELECT MAX(version) FROM schema_version").fetchone()[0] or 0
            except:
                current = 0
            
            if current < self.SCHEMA_VERSION:
                print(f"  📦 Running migrations: v{current} → v{self.SCHEMA_VERSION}")
                
                # Migration 1→3: Add missing columns to all tables
                self._migrate_v2(conn)
                
                # Record migration
                conn.execute(
                    "INSERT OR REPLACE INTO schema_version (version, applied_at) VALUES (?, ?)",
                    (self.SCHEMA_VERSION, datetime.now())
                )
    
    def _migrate_v2(self, conn):
        """Migration to V2 schema - add all missing columns"""
        # Get existing columns for tracked_positions
        columns = {row[1] for row in conn.execute("PRAGMA table_info(tracked_positions)").fetchall()}
        
        migrations = [
            ("tracked_positions", "strategy_name", "TEXT"),
            ("tracked_positions", "conviction_score", "REAL"),
            ("tracked_positions", "stop_loss_pct", "REAL DEFAULT -0.12"),
            ("tracked_positions", "take_profit_pct", "REAL DEFAULT 0.30"),
            ("tracked_positions", "trailing_stop_pct", "REAL DEFAULT 0.08"),
            ("tracked_positions", "max_hold_hours", "INTEGER DEFAULT 12"),
            ("tracked_positions", "wallet_count", "INTEGER DEFAULT 1"),
            ("tracked_positions", "clusters_json", "TEXT"),
            ("tracked_positions", "regime", "TEXT"),
            ("tracked_positions", "exit_reason", "TEXT"),
            ("tracked_positions", "exit_signature", "TEXT"),
            ("tracked_positions", "detection_method", "TEXT"),
            ("tracked_positions", "peak_price", "REAL"),
            ("tracked_positions", "peak_unrealized_pct", "REAL"),
            ("tracked_positions", "source", "TEXT DEFAULT 'live'"),
            ("tracked_positions", "entry_volume_24h", "REAL"),
        ]
        
        for table, column, col_type in migrations:
            if column not in columns:
                try:
                    conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {col_type}")
                    print(f"    + Added {table}.{column}")
                except sqlite3.OperationalError:
                    pass  # Column already exists
        
        # Add cluster column to wallets if missing
        wallet_cols = {row[1] for row in conn.execute("PRAGMA table_info(verified_wallets)").fetchall()}
        if "cluster" not in wallet_cols:
            conn.execute("ALTER TABLE verified_wallets ADD COLUMN cluster TEXT DEFAULT 'BALANCED'")
            print(f"    + Added verified_wallets.cluster")
        if "cluster_updated" not in wallet_cols:
            conn.execute("ALTER TABLE verified_wallets ADD COLUMN cluster_updated TIMESTAMP")
            print(f"    + Added verified_wallets.cluster_updated")
        if "is_active" not in wallet_cols:
            conn.execute("ALTER TABLE verified_wallets ADD COLUMN is_active INTEGER DEFAULT 1")
            print(f"    + Added verified_wallets.is_active")
        
        # Add missing columns to market_conditions
        market_cols = {row[1] for row in conn.execute("PRAGMA table_info(market_conditions)").fetchall()}
        market_migrations = [
            ("market_conditions", "regime", "TEXT"),
            ("market_conditions", "regime_confidence", "REAL"),
            ("market_conditions", "btc_dominance", "REAL"),
        ]
        
        for table, column, col_type in market_migrations:
            if column not in market_cols:
                try:
                    conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {col_type}")
                    print(f"    + Added {table}.{column}")
                except sqlite3.OperationalError:
                    pass
    
    # =========================================================================
    # WALLET METHODS
    # =========================================================================
    
    def is_wallet_tracked(self, address: str) -> bool:
        """Check if wallet is in verified_wallets"""
        # Check cache first
        if self._wallet_cache and address in self._wallet_cache:
            return True
        
        with self.connection() as conn:
            result = conn.execute(
                "SELECT 1 FROM verified_wallets WHERE address = ?", 
                (address,)
            ).fetchone()
            return result is not None
    
    def add_verified_wallet(self, address: str, stats: Dict):
        """Add or update a verified wallet"""
        with self.connection() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO verified_wallets 
                (address, pnl_7d, roi_7d, win_rate, completed_swings, avg_hold_hours,
                 risk_reward_ratio, best_trade_pct, worst_trade_pct, total_volume_sol,
                 avg_position_size_sol, discovered_at, last_updated) 
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, COALESCE(
                    (SELECT discovered_at FROM verified_wallets WHERE address = ?), ?
                ), ?)
            """, (
                address, 
                stats.get('pnl', 0), 
                stats.get('roi_7d', 0),
                stats.get('win_rate', 0), 
                stats.get('completed_swings', 0),
                stats.get('avg_hold_hours', 0), 
                stats.get('risk_reward_ratio', 0),
                stats.get('best_trade_pct', 0), 
                stats.get('worst_trade_pct', 0),
                stats.get('total_volume_sol', 0), 
                stats.get('avg_position_size_sol', 0),
                address, datetime.now(), datetime.now()
            ))
        
        # Invalidate cache
        self._wallet_cache.pop(address, None)
    
    def get_all_verified_wallets(self) -> List[Dict]:
        """Get all verified wallets"""
        with self.connection() as conn:
            rows = conn.execute(
                "SELECT * FROM verified_wallets WHERE is_active = 1"
            ).fetchall()
            return [dict(r) for r in rows]
    
    def get_wallet(self, address: str) -> Optional[Dict]:
        """Get single wallet data"""
        with self.connection() as conn:
            row = conn.execute(
                "SELECT * FROM verified_wallets WHERE address = ?",
                (address,)
            ).fetchone()
            return dict(row) if row else None
    
    def update_wallet_cluster(self, address: str, cluster: str):
        """Update wallet cluster assignment"""
        with self.connection() as conn:
            conn.execute(
                "UPDATE verified_wallets SET cluster = ?, cluster_updated = ? WHERE address = ?",
                (cluster, datetime.now(), address)
            )
    
    def get_wallet_count(self) -> int:
        """Get count of active wallets"""
        with self.connection() as conn:
            return conn.execute(
                "SELECT COUNT(*) FROM verified_wallets WHERE is_active = 1"
            ).fetchone()[0]
    
    # =========================================================================
    # POSITION METHODS
    # =========================================================================
    
    def is_position_tracked(self, wallet: str, token: str) -> bool:
        """Check if position already exists"""
        with self.connection() as conn:
            result = conn.execute(
                """SELECT 1 FROM tracked_positions 
                   WHERE wallet_address = ? AND token_address = ? AND status = 'open'""",
                (wallet, token)
            ).fetchone()
            return result is not None
    
    def add_tracked_position(self, wallet: str, token_address: str, 
                            entry_time: datetime, entry_price: float,
                            amount: float, token_symbol: str,
                            market_data: Dict = None) -> int:
        """Add a new tracked position"""
        market_data = market_data or {}
        
        with self.connection() as conn:
            cursor = conn.execute("""
                INSERT INTO tracked_positions 
                (wallet_address, token_address, token_symbol, entry_time, entry_price,
                 position_size_tokens, entry_market_cap, entry_liquidity, entry_volume_24h,
                 token_age_hours, position_size_sol, sol_price_usd, conviction_score,
                 stop_loss_pct, take_profit_pct, trailing_stop_pct, max_hold_hours,
                 strategy_name, wallet_count, clusters_json, regime, source)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                wallet, token_address, token_symbol, entry_time, entry_price,
                amount,
                market_data.get('market_cap', 0),
                market_data.get('liquidity', 0),
                market_data.get('volume_24h', 0),
                market_data.get('token_age_hours', 0),
                market_data.get('position_size_sol', 0),
                market_data.get('sol_price_usd', 0),
                market_data.get('conviction_score', 0),
                market_data.get('stop_loss', -0.12),
                market_data.get('take_profit', 0.30),
                market_data.get('trailing_stop', 0.08),
                market_data.get('max_hold_hours', 12),
                market_data.get('strategy', 'unknown'),
                market_data.get('wallet_count', 1),
                json.dumps(market_data.get('clusters', {})),
                market_data.get('regime', 'UNKNOWN'),
                market_data.get('source', 'live')
            ))
            return cursor.lastrowid
    
    def get_open_positions(self, wallet: str = None, token: str = None) -> List[Dict]:
        """Get open positions with optional filters"""
        with self.connection() as conn:
            query = "SELECT * FROM tracked_positions WHERE status = 'open'"
            params = []
            
            if wallet:
                query += " AND wallet_address = ?"
                params.append(wallet)
            if token:
                query += " AND token_address = ?"
                params.append(token)
            
            query += " ORDER BY entry_time DESC"
            rows = conn.execute(query, params).fetchall()
            return [dict(r) for r in rows]
    
    def get_position(self, wallet: str = None, token: str = None, 
                    position_id: int = None) -> Optional[Dict]:
        """Get single position by various criteria"""
        with self.connection() as conn:
            if position_id:
                row = conn.execute(
                    "SELECT * FROM tracked_positions WHERE id = ?",
                    (position_id,)
                ).fetchone()
            else:
                row = conn.execute(
                    """SELECT * FROM tracked_positions 
                       WHERE wallet_address = ? AND token_address = ? AND status = 'open'""",
                    (wallet, token)
                ).fetchone()
            return dict(row) if row else None
    
    def close_position(self, position_id: int, exit_data: Dict):
        """Close a position with exit data"""
        with self.connection() as conn:
            conn.execute("""
                UPDATE tracked_positions SET
                    status = 'closed',
                    exit_time = ?,
                    exit_price = ?,
                    exit_reason = ?,
                    exit_signature = ?,
                    detection_method = ?,
                    profit_pct = ?,
                    profit_sol = ?,
                    hold_duration_minutes = ?,
                    peak_price = ?,
                    peak_unrealized_pct = ?
                WHERE id = ?
            """, (
                exit_data.get('exit_time', datetime.now()),
                exit_data.get('exit_price', 0),
                exit_data.get('exit_reason', 'unknown'),
                exit_data.get('exit_signature'),
                exit_data.get('detection_method'),
                exit_data.get('profit_pct', 0),
                exit_data.get('profit_sol', 0),
                exit_data.get('hold_duration_minutes', 0),
                exit_data.get('peak_price'),
                exit_data.get('peak_unrealized_pct'),
                position_id
            ))
    
    def update_position_peak(self, position_id: int, peak_price: float, peak_pct: float):
        """Update position peak tracking"""
        with self.connection() as conn:
            conn.execute(
                "UPDATE tracked_positions SET peak_price = ?, peak_unrealized_pct = ? WHERE id = ?",
                (peak_price, peak_pct, position_id)
            )
    
    def get_closed_positions(self, days: int = 7, strategy: str = None) -> List[Dict]:
        """Get closed positions for analysis"""
        with self.connection() as conn:
            query = """
                SELECT * FROM tracked_positions 
                WHERE status = 'closed' 
                AND exit_time >= datetime('now', ? || ' days')
            """
            params = [f'-{days}']
            
            if strategy:
                query += " AND strategy_name = ?"
                params.append(strategy)
            
            query += " ORDER BY exit_time DESC"
            rows = conn.execute(query, params).fetchall()
            return [dict(r) for r in rows]
    
    # =========================================================================
    # SIGNATURE DEDUPLICATION
    # =========================================================================
    
    def is_signature_processed(self, signature: str) -> bool:
        """Check if signature was already processed (with caching)"""
        # Check cache first
        if signature in self._signature_cache:
            return True
        
        with self.connection() as conn:
            result = conn.execute(
                "SELECT 1 FROM processed_sigs WHERE signature = ?",
                (signature,)
            ).fetchone()
            
            if result:
                self._signature_cache.add(signature)
                return True
            return False
    
    def mark_signature_processed(self, signature: str, wallet: str = None,
                                 trade_type: str = None, token: str = None) -> bool:
        """Mark signature as processed, return False if already exists"""
        if signature in self._signature_cache:
            return False
        
        try:
            with self.connection() as conn:
                conn.execute("""
                    INSERT INTO processed_sigs (signature, processed_at, wallet_address, trade_type, token_address)
                    VALUES (?, ?, ?, ?, ?)
                """, (signature, datetime.now(), wallet, trade_type, token))
            
            self._signature_cache.add(signature)
            return True
        except sqlite3.IntegrityError:
            return False  # Already exists
    
    # =========================================================================
    # MARKET CONDITIONS
    # =========================================================================
    
    def log_market_conditions(self, conditions: Dict):
        """Log market conditions"""
        with self.connection() as conn:
            conn.execute("""
                INSERT INTO market_conditions 
                (timestamp, sol_price_usd, sol_24h_change_pct, sol_volume_24h,
                 regime, regime_confidence, btc_dominance, trending_tokens_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                datetime.now(),
                conditions.get('sol_price_usd', 0),
                conditions.get('sol_24h_change_pct', 0),
                conditions.get('sol_volume_24h', 0),
                conditions.get('regime', 'UNKNOWN'),
                conditions.get('regime_confidence', 0.5),
                conditions.get('btc_dominance', 0),
                conditions.get('trending_tokens_count', 0)
            ))
    
    def get_recent_market_conditions(self, hours: int = 24) -> List[Dict]:
        """Get recent market conditions for regime analysis"""
        with self.connection() as conn:
            rows = conn.execute("""
                SELECT * FROM market_conditions
                WHERE timestamp >= datetime('now', ? || ' hours')
                ORDER BY timestamp DESC
            """, (f'-{hours}',)).fetchall()
            return [dict(r) for r in rows]
    
    # =========================================================================
    # EXIT SIGNALS
    # =========================================================================
    
    def add_exit_signal(self, token: str, token_symbol: str, wallet: str,
                       cluster: str, price: float, signature: str):
        """Record a wallet exit signal"""
        try:
            with self.connection() as conn:
                conn.execute("""
                    INSERT INTO exit_signals (token_address, token_symbol, wallet_address,
                                             wallet_cluster, exit_price, exit_time, signature)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (token, token_symbol, wallet, cluster, price, datetime.now(), signature))
        except sqlite3.IntegrityError:
            pass  # Duplicate
    
    def get_exit_signals(self, token: str, minutes: int = 30) -> List[Dict]:
        """Get recent exit signals for a token"""
        with self.connection() as conn:
            rows = conn.execute("""
                SELECT * FROM exit_signals
                WHERE token_address = ?
                AND exit_time >= datetime('now', ? || ' minutes')
                ORDER BY exit_time DESC
            """, (token, f'-{minutes}')).fetchall()
            return [dict(r) for r in rows]
    
    # =========================================================================
    # LLM TRACKING
    # =========================================================================
    
    def log_llm_call(self, call_type: str, token_symbol: str = None,
                    input_tokens: int = 0, output_tokens: int = 0,
                    result_summary: str = None):
        """Log an LLM API call for cost tracking"""
        # Approximate cost: $3/M input, $15/M output for Sonnet
        cost = (input_tokens * 3 + output_tokens * 15) / 1_000_000
        
        with self.connection() as conn:
            conn.execute("""
                INSERT INTO llm_calls (timestamp, call_type, token_symbol, 
                                      input_tokens, output_tokens, cost_usd, result_summary)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (datetime.now(), call_type, token_symbol, input_tokens, output_tokens, cost, result_summary))
    
    def get_llm_cost_summary(self, days: int = 1) -> Dict:
        """Get LLM cost summary"""
        with self.connection() as conn:
            row = conn.execute("""
                SELECT 
                    COUNT(*) as calls,
                    SUM(cost_usd) as total_cost,
                    SUM(input_tokens) as total_input,
                    SUM(output_tokens) as total_output
                FROM llm_calls
                WHERE timestamp >= datetime('now', ? || ' days')
            """, (f'-{days}',)).fetchone()
            
            return {
                'calls': row[0] or 0,
                'cost_usd': row[1] or 0,
                'input_tokens': row[2] or 0,
                'output_tokens': row[3] or 0
            }
    
    # =========================================================================
    # STRATEGY REGISTRY
    # =========================================================================
    
    def get_strategy(self, strategy_id: str) -> Optional[Dict]:
        """Get strategy by ID"""
        with self.connection() as conn:
            row = conn.execute(
                "SELECT * FROM strategy_registry WHERE id = ?",
                (strategy_id,)
            ).fetchone()
            
            if row:
                result = dict(row)
                result['config'] = json.loads(result.get('config_json') or '{}')
                result['performance'] = json.loads(result.get('performance_json') or '{}')
                return result
            return None
    
    def save_strategy(self, strategy_id: str, data: Dict):
        """Save/update a strategy"""
        with self.connection() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO strategy_registry 
                (id, name, status, config_json, performance_json, created_at, 
                 promoted_at, evolved_from, evolution_reason)
                VALUES (?, ?, ?, ?, ?, COALESCE(
                    (SELECT created_at FROM strategy_registry WHERE id = ?), ?
                ), ?, ?, ?)
            """, (
                strategy_id,
                data.get('name'),
                data.get('status', 'CHALLENGER'),
                json.dumps(data.get('config', {})),
                json.dumps(data.get('performance', {})),
                strategy_id, datetime.now(),
                data.get('promoted_at'),
                data.get('evolved_from'),
                data.get('evolution_reason')
            ))
    
    def get_all_strategies(self) -> Dict[str, Dict]:
        """Get all strategies"""
        with self.connection() as conn:
            rows = conn.execute("SELECT * FROM strategy_registry").fetchall()
            
            result = {}
            for row in rows:
                data = dict(row)
                data['config'] = json.loads(data.get('config_json') or '{}')
                data['performance'] = json.loads(data.get('performance_json') or '{}')
                result[data['id']] = data
            
            return result
    
    # =========================================================================
    # ANALYTICS
    # =========================================================================
    
    def get_performance_summary(self, days: int = 7) -> Dict:
        """Get overall performance summary"""
        with self.connection() as conn:
            row = conn.execute("""
                SELECT 
                    COUNT(*) as total_trades,
                    SUM(CASE WHEN profit_pct > 0 THEN 1 ELSE 0 END) as wins,
                    SUM(CASE WHEN profit_pct <= 0 THEN 1 ELSE 0 END) as losses,
                    SUM(profit_sol) as total_pnl,
                    AVG(profit_pct) as avg_profit_pct,
                    AVG(hold_duration_minutes) as avg_hold_minutes,
                    MAX(profit_pct) as best_trade,
                    MIN(profit_pct) as worst_trade
                FROM tracked_positions
                WHERE status = 'closed'
                AND exit_time >= datetime('now', ? || ' days')
            """, (f'-{days}',)).fetchone()
            
            total = row[0] or 0
            wins = row[1] or 0
            
            return {
                'total_trades': total,
                'wins': wins,
                'losses': row[2] or 0,
                'win_rate': wins / total if total > 0 else 0,
                'total_pnl_sol': row[3] or 0,
                'avg_profit_pct': row[4] or 0,
                'avg_hold_minutes': row[5] or 0,
                'best_trade_pct': row[6] or 0,
                'worst_trade_pct': row[7] or 0
            }
    
    def cleanup_old_data(self, days: int = 30):
        """Clean up old data to keep database size manageable"""
        with self.connection() as conn:
            # Clean old snapshots
            conn.execute("""
                DELETE FROM position_snapshots 
                WHERE snapshot_time < datetime('now', ? || ' days')
            """, (f'-{days}',))
            
            # Clean old processed signatures
            conn.execute("""
                DELETE FROM processed_sigs 
                WHERE processed_at < datetime('now', ? || ' days')
            """, (f'-{days}',))
            
            # Clean old exit signals
            conn.execute("""
                DELETE FROM exit_signals 
                WHERE exit_time < datetime('now', '-7 days')
            """)
        
        # Clear caches
        self._signature_cache.clear()
        self._wallet_cache.clear()


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("DATABASE V2 - Schema Test")
    print("="*60)
    
    db = DatabaseV2("test_v2.db")
    
    # Test wallet operations
    print("\n1. Testing wallet operations...")
    db.add_verified_wallet("test_wallet_1", {
        'win_rate': 0.65,
        'pnl': 5.5,
        'completed_swings': 10
    })
    assert db.is_wallet_tracked("test_wallet_1")
    print("   ✅ Wallet operations OK")
    
    # Test position operations
    print("\n2. Testing position operations...")
    pos_id = db.add_tracked_position(
        wallet="test_wallet_1",
        token_address="test_token",
        entry_time=datetime.now(),
        entry_price=0.001,
        amount=1000,
        token_symbol="TEST",
        market_data={'conviction_score': 75, 'strategy': 'test_strategy'}
    )
    assert pos_id > 0
    print("   ✅ Position operations OK")
    
    # Test signature dedup
    print("\n3. Testing signature deduplication...")
    assert db.mark_signature_processed("sig_test_1", "wallet", "BUY", "token") == True
    assert db.is_signature_processed("sig_test_1") == True
    assert db.mark_signature_processed("sig_test_1") == False  # Duplicate
    print("   ✅ Signature deduplication OK")
    
    print("\n✅ All database tests passed!")
    
    # Cleanup
    import os
    os.remove("test_v2.db")
