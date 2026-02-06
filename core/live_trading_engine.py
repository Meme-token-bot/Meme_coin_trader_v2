"""
LIVE TRADING ENGINE V2 - Production Ready
==========================================

Builds on executioner_v1.py with enhancements:
1. Jito bundle integration for MEV protection and speed
2. Parallel paper/live execution with comparison
3. Enhanced safety controls for 3 SOL capital
4. AWS Secrets Manager integration (optional)
5. Kill switch with position liquidation

CAPITAL: 3 SOL
POSITION SIZE: 0.05 SOL (recommended)
MAX POSITIONS: 2
MAX DEPLOYED: 0.80 SOL (27% of capital)

Author: Claude
"""

import os
import json
import time
import threading
import hashlib
import base64
import base58
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict
from decimal import Decimal
from enum import Enum
from contextlib import contextmanager
import sqlite3
import requests
import logging
import random

from core.live_exit_manager import LiveExitManager, ExitConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s'
)
logger = logging.getLogger("LiveTrader")

# Import secrets manager if available
try:
    from core.secrets_manager import get_secret
    SECRETS_AVAILABLE = True
except ImportError:
    SECRETS_AVAILABLE = False
    def get_secret(key, default=None):
        return os.getenv(key, default)

# Solana imports
try:
    from solders.keypair import Keypair
    from solders.pubkey import Pubkey
    from solders.transaction import VersionedTransaction, Transaction
    from solders.signature import Signature
    from solders.message import MessageV0, Message
    from solders.system_program import transfer, TransferParams
    from solders.hash import Hash
    from solders.instruction import Instruction, CompiledInstruction, AccountMeta
    from solana.rpc.api import Client as SolanaClient
    from solana.rpc.commitment import Confirmed
    from solders.system_program import TransferParams, transfer
    from solders.message import Message
    from solders.transaction import Transaction
    SOLANA_AVAILABLE = True
except ImportError:
    SOLANA_AVAILABLE = False
    logger.warning("⚠️ Solana libraries not installed. Run: pip install solana solders --break-system-packages")


# =============================================================================
# CONSTANTS
# =============================================================================

SOL_MINT = "So11111111111111111111111111111111111111112"
LAMPORTS_PER_SOL = 1_000_000_000

# API Endpoints
DEFAULT_JUPITER_QUOTE_URL = os.getenv("JUPITER_QUOTE_URL", "https://public.jupiterapi.com/quote")
DEFAULT_JUPITER_SWAP_URL = os.getenv("JUPITER_SWAP_URL", "https://public.jupiterapi.com/swap")
JUPITER_QUOTE_URL_FALLBACK = "https://api.jup.ag/v6/quote"
JUPITER_SWAP_URL_FALLBACK = "https://api.jup.ag/v6/swap"
JUPITER_PRICE_URL = "https://api.jup.ag/price/v2/full"

# Jito endpoints
JITO_BLOCK_ENGINE_URLS = [
    "https://ny.mainnet.block-engine.jito.wtf",
    "https://amsterdam.mainnet.block-engine.jito.wtf",
    "https://frankfurt.mainnet.block-engine.jito.wtf",
    "https://slc.mainnet.block-engine.jito.wtf"
]
JITO_BUNDLE_URLS = [f"{url}/api/v1/bundles" for url in JITO_BLOCK_ENGINE_URLS]
JITO_TIP_ACCOUNTS = [
    "96gYZGLnJYVFmbjzopPSU6QiEV5fGqZNyN9nmNhvrZU5",
    "HFqU5x63VTqvQss8hp11i4bVmkdzGTbQrWMT7wekGuLt",
    "Cw8CFyM9FkoMi7K7Crf6HNQqf4uEMzpKw6QNghXLvLkY",
    "ADaUMid9yfUytqMBgopwjb2DTLSokTSzL1zt6iGPaS49",
    "DfXygSm4jCyNCybVYYK6DwvWqjKee8pbDmJGcLWNDXjh",
    "ADuUkR4vqLUMWXxW9gh6D6L8pMSawimctcNZ5pGwDcEt",
    "DttWaMuVvTiduZRnguLF7jNxTgiMBZ1hyAumKUiL2KRL",
    "3AVi9Tg9Uo68tJfuvoKvqKNWKkC5wPdSSdeBnizKZ6jT"
]
HELIUS_TIP_ACCOUNTS = [
    "4ACfpUFoaSD9bfPdeu6DBt89gB6ENTeHBXCAi87NhDEE",
    "D2L6yPZ2FmmmTKPgzaMKdhu6EWZcTpLy1Vhx8uvZe7NZ",
    "9bnz4RShgq1hAnLnZbP8kbgBg1kEmcJBYQq3gQbmnSta",
    "5VY91ws6B2hMmBFRsXkoAAdsPHBJwRfBht4DXox3xkwn",
    "2nyhqdwKcJZR2vcqCyrYsaPVdAnFoJjiksCXJ7hfEYgD",
    "2q5pghRs6arqVjRvT5gfgWfWcHWmw1ZuCzphgd5KfWGJ",
    "wyvPkWjVZz1M8fHQnMMCDTQDbkManefNNhweYk5WkcF",
    "3KCKozbAaF75qEU33jtzozcJ29yJuaLJTy2jFdzUY8bT",
    "4vieeGHPYPG2MmyPRcYjdiDmmhN3ww7hsFNap8pVN3Ey",
    "4TQLFNWK8AovT1gFvda5jfw2oJeRMKEmw7aH6MGBJ3or"
]

# Price APIs
COINGECKO_URL = "https://api.coingecko.com/api/v3/simple/price"
DEXSCREENER_URL = "https://api.dexscreener.com/latest/dex/tokens/{}"


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class LiveTradingConfig:
    """Production configuration for 3 SOL capital"""
    
    # Capital management
    starting_capital_sol: float = 2.5724235
    position_size_sol: float = 0.05          # Per trade
    max_open_positions: int = 2             # Max concurrent
    max_deployed_sol: float = 0.80           # Max SOL in positions
    fee_reserve_sol: float = 0.30            # Reserved for fees
    
    # Risk management
    max_daily_loss_sol: float = 0.25         # Stop trading for day
    max_consecutive_losses: int = 15         # Pause and reassess
    min_balance_sol: float = 1.50            # Emergency stop if below
    cool_down_minutes: int = 30              # After consecutive losses
    
    # Execution
    default_slippage_bps: int = 300          # 3% slippage
    max_slippage_bps: int = 500              # 5% max
    jito_tip_lamports: int = 5_000_000       # 0.001 SOL tip
    priority_fee_lamports: int = 100_000     # Priority fee
    enable_dynamic_priority_fees: bool = True  # Fetch priority fees from RPC
    priority_fee_level: str = "unsafeMax"    # low/medium/high/veryHigh/unsafeMax
    min_priority_fee_lamports: int = 200_000 # Floor when dynamic fees enabled
    max_priority_fee_lamports: int = 10_000_000 # Ceiling when dynamic fees enabled
    dynamic_compute_unit_limit: bool = True # Let Jupiter size CU automatically
    compute_unit_limit: Optional[int] = None # Explicit CU limit if set
    helius_sender_retry_seconds: int = 12    # Retry window for Helius sender
    helius_sender_retry_interval: float = 2  # Seconds between retries
    confirmation_timeout: int = 60           # Seconds
    retry_attempts: int = 2
    exit_monitor_interval_seconds: int = 2   # Exit monitor check interval
    use_helius_sender_for_buys: bool = True # Route buys via Helius sender
    use_helius_sender_for_sells: bool = True # Route sells via Helius sender
    enable_exit_websocket: bool = False      # Use Helius WS to trigger exits
    exit_websocket_ping_seconds: int = 30    # WS ping interval
    exit_websocket_reconnect_seconds: int = 5 # WS reconnect delay
    
    # Entry filters (from paper trading analysis)
    min_conviction: int = 60
    min_liquidity_usd: float = 5000
    blocked_hours_utc: List[int] = field(default_factory=lambda: [1, 3, 5, 19, 23])
    
    # Exit parameters (from paper trading)
    stop_loss_pct: float = -0.15             # -15%
    take_profit_pct: float = 0.30            # +30%
    trailing_stop_pct: float = 0.10          # 10% from peak
    max_hold_hours: int = 12
    
    # Feature flags
    enable_live_trading: bool = False        # MUST be True to trade
    enable_jito_bundles: bool = True         # Use Jito for speed
    parallel_paper_trading: bool = True      # Run paper alongside
    use_helius_sender: bool = True           # Use Helius RPC sender instead of Jito
    
    # Tax (NZ)
    tax_year_start_month: int = 4            # April
    cost_basis_method: str = "FIFO"


# =============================================================================
# PRICE SERVICE
# =============================================================================

class PriceService:
    """Get token prices with caching"""
    
    def __init__(self):
        self._cache = {}
        self._cache_ttl = 2  # seconds
        self._sol_nzd_rate = None
        self._sol_usd_rate = None
        self._last_sol_update = None
    
    def get_sol_prices(self) -> Tuple[float, float]:
        """Get SOL price in USD and NZD"""
        now = datetime.now()
        
        if (self._last_sol_update and 
            (now - self._last_sol_update).total_seconds() < 60):
            return self._sol_usd_rate or 0, self._sol_nzd_rate or 0
        
        try:
            resp = requests.get(
                f"{COINGECKO_URL}?ids=solana&vs_currencies=usd,nzd",
                timeout=10
            )
            if resp.status_code == 200:
                data = resp.json()
                self._sol_usd_rate = data['solana']['usd']
                self._sol_nzd_rate = data['solana']['nzd']
                self._last_sol_update = now
        except Exception as e:
            logger.warning(f"CoinGecko error: {e}")
        
        return self._sol_usd_rate or 0, self._sol_nzd_rate or 0
    
    def get_token_price(self, token_address: str) -> Optional[float]:
        """Get token price in USD"""
        cache_key = f"price_{token_address}"
        
        if cache_key in self._cache:
            entry = self._cache[cache_key]
            if (datetime.now() - entry['time']).total_seconds() < self._cache_ttl:
                return entry['price']
        
        try:
            resp = requests.get(
                f"{JUPITER_PRICE_URL}?ids={token_address}",
                timeout=10
            )
            if resp.status_code == 200:
                data = resp.json()
                price = data.get('data', {}).get(token_address, {}).get('price')
                if price:
                    self._cache[cache_key] = {'price': float(price), 'time': datetime.now()}
                    return float(price)
        except Exception as e:
            logger.warning(f"Jupiter price error: {e}")
        
        # Fallback to DexScreener
        try:
            resp = requests.get(DEXSCREENER_URL.format(token_address), timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                pairs = data.get('pairs', [])
                if pairs:
                    price = float(pairs[0].get('priceUsd', 0))
                    self._cache[cache_key] = {'price': price, 'time': datetime.now()}
                    return price
        except:
            pass
        
        return None

    def get_main_pool_vault(self, token_address: str) -> Optional[str]:
        """Fetch the most liquid pool vault address for a token."""
        try:
            resp = requests.get(DEXSCREENER_URL.format(token_address), timeout=10)
            if resp.status_code != 200:
                return None
            data = resp.json()
            pairs = data.get('pairs', [])
            if not pairs:
                return None
            best_pair = max(
                pairs,
                key=lambda p: float(p.get('liquidity', {}).get('usd', 0) or 0)
            )
            return best_pair.get('pairAddress')
        except Exception as exc:
            logger.warning(f"DexScreener vault lookup failed for {token_address}: {exc}")
            return None


# =============================================================================
# JITO BUNDLE SERVICE
# =============================================================================

class JitoBundleService:
    """Jito bundle submission for MEV protection and fast execution"""
    
    def __init__(self, keypair: Keypair):
        self.keypair = keypair
        self._tip_account_index = 0
        self.last_status_code: Optional[int] = None
    
    def get_tip_account(self) -> str:
        """Rotate through tip accounts for load balancing"""
        account = JITO_TIP_ACCOUNTS[self._tip_account_index]
        self._tip_account_index = (self._tip_account_index + 1) % len(JITO_TIP_ACCOUNTS)
        return account
    
    def submit_bundle(self, transactions: List[str], tip_lamports: int = None) -> Optional[str]:
        """
        Submit a bundle to Jito block engine.
        
        Args:
            transactions: List of base64 encoded signed transactions
                         [swap_tx, tip_tx] - swap first, tip second
            tip_lamports: (deprecated, tip should be in transactions)
        
        Returns:
            Bundle ID if successful
        """
        try:
            logger.info(f"📤 Submitting Jito bundle with {len(transactions)} transactions...")
            
            payload = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "sendBundle",
                "params": [transactions]
            }
            
            for bundle_url in JITO_BUNDLE_URLS:
                resp = requests.post(
                    bundle_url,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                    timeout=30
                )
                self.last_status_code = resp.status_code
                
                if resp.status_code == 200:
                    result = resp.json()
                    if 'result' in result:
                        bundle_id = result['result']
                        logger.info(f"🚀 Jito bundle accepted: {bundle_id}")
                        return bundle_id
                    elif 'error' in result:
                        error_msg = result['error']
                        logger.error(f"❌ Jito bundle rejected: {error_msg}")
                        # Common errors:
                        # - "Bundle contains invalid transactions" - signatures wrong
                        # - "Bundle too old" - blockhash expired
                        # - "Insufficient tip" - need higher tip
                else:
                    logger.error(
                        f"❌ Jito HTTP error: {resp.status_code} - {resp.text[:200]}"
                    )
        
        except requests.exceptions.Timeout:
            logger.error("⏱️ Jito request timed out")
        except Exception as e:
            logger.error(f"❌ Jito bundle submission failed: {e}")
        
        return None
    
    def get_bundle_status(self, bundle_id: str) -> Optional[str]:
        """
        Check bundle status from Jito.
        
        Returns:
            Status string: 'pending', 'landed', 'finalized', 'invalid', 'failed', or None
        """
        try:
            payload = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "getBundleStatuses",
                "params": [[bundle_id]]
            }
            
            for bundle_url in JITO_BUNDLE_URLS:
                resp = requests.post(
                    bundle_url,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                    timeout=10
                )
                
                if resp.status_code == 200:
                    result = resp.json()
                    statuses = result.get('result', {}).get('value', [])
                    if statuses and len(statuses) > 0:
                        bundle_status = statuses[0]
                        if bundle_status:
                            # Jito returns confirmation_status or status
                            status = bundle_status.get('confirmation_status') or bundle_status.get('status')
                            if status:
                                logger.debug(f"Bundle {bundle_id[:8]}... status: {status}")
                                return status.lower()
        except Exception as e:
            logger.warning(f"Bundle status check failed: {e}")
        
        return None


# =============================================================================
# TAX DATABASE
# =============================================================================

class TaxDatabase:
    """NZ Tax compliant transaction recording"""
    
    def __init__(self, db_path: str = "live_trades_tax.db"):
        self.db_path = db_path
        self._init_database()
    
    @contextmanager
    def _get_connection(self):
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()
    
    def _init_database(self):
        """Initialize tax tables"""
        with self._get_connection() as conn:
            # Main transactions table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS tax_transactions (
                    id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    transaction_type TEXT NOT NULL,
                    token_address TEXT NOT NULL,
                    token_symbol TEXT,
                    token_amount REAL,
                    sol_amount REAL,
                    price_per_token_usd REAL,
                    price_per_token_nzd REAL,
                    sol_price_usd REAL,
                    sol_price_nzd REAL,
                    total_value_usd REAL,
                    total_value_nzd REAL,
                    fee_sol REAL,
                    fee_nzd REAL,
                    cost_basis_nzd REAL,
                    gain_loss_nzd REAL,
                    signature TEXT,
                    notes TEXT,
                    is_live BOOLEAN DEFAULT 1
                )
            """)
            
            # Cost basis lots for FIFO
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cost_basis_lots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    token_address TEXT NOT NULL,
                    acquisition_date TEXT NOT NULL,
                    tokens_acquired REAL NOT NULL,
                    tokens_remaining REAL NOT NULL,
                    cost_per_token_nzd REAL NOT NULL,
                    total_cost_nzd REAL NOT NULL,
                    acquisition_signature TEXT
                )
            """)
            
            # Live positions
            conn.execute("""
                CREATE TABLE IF NOT EXISTS live_positions (
                    token_address TEXT PRIMARY KEY,
                    token_symbol TEXT,
                    tokens_held REAL,
                    entry_price_usd REAL,
                    entry_time TEXT,
                    total_cost_sol REAL,
                    total_cost_nzd REAL,
                    stop_loss_pct REAL,
                    take_profit_pct REAL,
                    trailing_stop_pct REAL,
                    peak_price_usd REAL,
                    liquidity_pool_vault TEXT,
                    entry_signature TEXT,
                    conviction_score INTEGER
                )
            """)
            columns = {
                row[1] for row in conn.execute("PRAGMA table_info(live_positions)").fetchall()
            }
            if "liquidity_pool_vault" not in columns:
                conn.execute("ALTER TABLE live_positions ADD COLUMN liquidity_pool_vault TEXT")
            
            # Daily stats
            conn.execute("""
                CREATE TABLE IF NOT EXISTS daily_stats (
                    date TEXT PRIMARY KEY,
                    trades INTEGER DEFAULT 0,
                    wins INTEGER DEFAULT 0,
                    losses INTEGER DEFAULT 0,
                    pnl_sol REAL DEFAULT 0,
                    pnl_nzd REAL DEFAULT 0,
                    fees_sol REAL DEFAULT 0
                )
            """)
            
            # Execution log
            conn.execute("""
                CREATE TABLE IF NOT EXISTS execution_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    action TEXT,
                    token_address TEXT,
                    token_symbol TEXT,
                    status TEXT,
                    signature TEXT,
                    error TEXT,
                    details TEXT
                )
            """)
    
    def record_trade(self, trade_data: Dict) -> str:
        """Record a trade for tax purposes"""
        record_id = self._generate_id()
        
        with self._get_connection() as conn:
            conn.execute("""
                INSERT INTO tax_transactions (
                    id, timestamp, transaction_type, token_address, token_symbol,
                    token_amount, sol_amount, price_per_token_usd, price_per_token_nzd,
                    sol_price_usd, sol_price_nzd, total_value_usd, total_value_nzd,
                    fee_sol, fee_nzd, cost_basis_nzd, gain_loss_nzd, signature, notes, is_live
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                record_id,
                trade_data.get('timestamp', datetime.now(timezone.utc).isoformat()),
                trade_data.get('transaction_type', 'UNKNOWN'),
                trade_data.get('token_address', ''),
                trade_data.get('token_symbol', 'UNKNOWN'),
                trade_data.get('token_amount', 0),
                trade_data.get('sol_amount', 0),
                trade_data.get('price_per_token_usd', 0),
                trade_data.get('price_per_token_nzd', 0),
                trade_data.get('sol_price_usd', 0),
                trade_data.get('sol_price_nzd', 0),
                trade_data.get('total_value_usd', 0),
                trade_data.get('total_value_nzd', 0),
                trade_data.get('fee_sol', 0),
                trade_data.get('fee_nzd', 0),
                trade_data.get('cost_basis_nzd'),
                trade_data.get('gain_loss_nzd'),
                trade_data.get('signature', ''),
                trade_data.get('notes', ''),
                trade_data.get('is_live', True)
            ))
            
            # If it's a buy, create a cost basis lot
            if trade_data.get('transaction_type') == 'BUY':
                conn.execute("""
                    INSERT INTO cost_basis_lots (
                        token_address, acquisition_date, tokens_acquired, tokens_remaining,
                        cost_per_token_nzd, total_cost_nzd, acquisition_signature
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    trade_data.get('token_address'),
                    trade_data.get('timestamp', datetime.now(timezone.utc).isoformat()),
                    trade_data.get('token_amount', 0),
                    trade_data.get('token_amount', 0),
                    trade_data.get('price_per_token_nzd', 0),
                    trade_data.get('total_value_nzd', 0),
                    trade_data.get('signature', '')
                ))
        
        return record_id
    
    def add_position(self, position: Dict):
        """Add or update a live position"""
        with self._get_connection() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO live_positions (
                    token_address, token_symbol, tokens_held, entry_price_usd,
                    entry_time, total_cost_sol, total_cost_nzd, stop_loss_pct,
                    take_profit_pct, trailing_stop_pct, peak_price_usd,
                    liquidity_pool_vault, entry_signature, conviction_score
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                position['token_address'],
                position.get('token_symbol', 'UNKNOWN'),
                position.get('tokens_held', 0),
                position.get('entry_price_usd', 0),
                position.get('entry_time', datetime.now(timezone.utc).isoformat()),
                position.get('total_cost_sol', 0),
                position.get('total_cost_nzd', 0),
                position.get('stop_loss_pct', -0.15),
                position.get('take_profit_pct', 0.30),
                position.get('trailing_stop_pct', 0.10),
                position.get('peak_price_usd', position.get('entry_price_usd', 0)),
                position.get('liquidity_pool_vault'),
                position.get('entry_signature', ''),
                position.get('conviction_score', 0)
            ))
    
    def remove_position(self, token_address: str):
        """Remove a closed position"""
        with self._get_connection() as conn:
            conn.execute("DELETE FROM live_positions WHERE token_address = ?", (token_address,))
    
    def get_positions(self) -> List[Dict]:
        """Get all open positions"""
        with self._get_connection() as conn:
            rows = conn.execute("SELECT * FROM live_positions").fetchall()
            return [dict(r) for r in rows]
    
    def update_position_peak(self, token_address: str, peak_price: float):
        """Update peak price for trailing stop"""
        with self._get_connection() as conn:
            conn.execute(
                "UPDATE live_positions SET peak_price_usd = ? WHERE token_address = ?",
                (peak_price, token_address)
            )
    
    def get_fifo_cost_basis(self, token_address: str, tokens_to_sell: float) -> Tuple[float, float]:
        """
        Calculate FIFO cost basis for a sale.
        Returns (cost_basis_nzd, tokens_consumed)
        """
        with self._get_connection() as conn:
            lots = conn.execute("""
                SELECT * FROM cost_basis_lots 
                WHERE token_address = ? AND tokens_remaining > 0
                ORDER BY acquisition_date ASC
            """, (token_address,)).fetchall()
            
            cost_basis = 0.0
            tokens_remaining = tokens_to_sell
            
            for lot in lots:
                lot = dict(lot)
                if tokens_remaining <= 0:
                    break
                
                consume = min(lot['tokens_remaining'], tokens_remaining)
                cost_basis += consume * lot['cost_per_token_nzd']
                tokens_remaining -= consume
                
                # Update the lot
                new_remaining = lot['tokens_remaining'] - consume
                conn.execute(
                    "UPDATE cost_basis_lots SET tokens_remaining = ? WHERE id = ?",
                    (new_remaining, lot['id'])
                )
            
            return cost_basis, tokens_to_sell - tokens_remaining
    
    def update_daily_stats(self, pnl_sol: float, is_win: bool, fee_sol: float = 0):
        """Update daily statistics"""
        today = datetime.now(timezone.utc).strftime('%Y-%m-%d')
        
        with self._get_connection() as conn:
            existing = conn.execute(
                "SELECT * FROM daily_stats WHERE date = ?", (today,)
            ).fetchone()
            
            if existing:
                conn.execute("""
                    UPDATE daily_stats SET
                        trades = trades + 1,
                        wins = wins + ?,
                        losses = losses + ?,
                        pnl_sol = pnl_sol + ?,
                        fees_sol = fees_sol + ?
                    WHERE date = ?
                """, (1 if is_win else 0, 0 if is_win else 1, pnl_sol, fee_sol, today))
            else:
                conn.execute("""
                    INSERT INTO daily_stats (date, trades, wins, losses, pnl_sol, fees_sol)
                    VALUES (?, 1, ?, ?, ?, ?)
                """, (today, 1 if is_win else 0, 0 if is_win else 1, pnl_sol, fee_sol))
    
    def get_daily_stats(self) -> Dict:
        """Get today's stats"""
        today = datetime.now(timezone.utc).strftime('%Y-%m-%d')
        
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM daily_stats WHERE date = ?", (today,)
            ).fetchone()
            
            if row:
                return dict(row)
            return {'date': today, 'trades': 0, 'wins': 0, 'losses': 0, 'pnl_sol': 0, 'fees_sol': 0}
    
    def log_execution(self, action: str, token_address: str, token_symbol: str,
                      status: str, signature: str = None, error: str = None, details: Dict = None):
        """Log an execution attempt"""
        with self._get_connection() as conn:
            conn.execute("""
                INSERT INTO execution_log (timestamp, action, token_address, token_symbol, status, signature, error, details)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                datetime.now(timezone.utc).isoformat(),
                action, token_address, token_symbol, status, signature, error,
                json.dumps(details) if details else None
            ))
    
    def _generate_id(self) -> str:
        """Generate unique ID"""
        timestamp = datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S%f')
        random_part = hashlib.sha256(os.urandom(8)).hexdigest()[:8]
        return f"TX_{timestamp}_{random_part}"


# =============================================================================
# LIVE TRADING ENGINE
# =============================================================================

class LiveTradingEngine:
    """
    Production live trading engine.
    
    Features:
    - Jito bundle submission for fast, MEV-protected execution
    - Parallel paper trading for comparison
    - NZ tax compliant record keeping
    - Kill switch with position liquidation
    - Daily loss limits and cool downs
    """
    
    def __init__(self, 
                 private_key: str = None,
                 helius_key: str = None,
                 config: LiveTradingConfig = None):
        
        self.config = config or LiveTradingConfig()
        if not self.config.enable_live_trading:
            self.config.enable_live_trading = get_secret('ENABLE_LIVE_TRADING', '').lower() == 'true'
        if not self.config.use_helius_sender:
            self.config.use_helius_sender = get_secret('USE_HELIUS_SENDER', '').lower() == 'true'
        sender_buys_override = get_secret('USE_HELIUS_SENDER_FOR_BUYS')
        if sender_buys_override:
            self.config.use_helius_sender_for_buys = sender_buys_override.lower() == 'true'
        sender_sells_override = get_secret('USE_HELIUS_SENDER_FOR_SELLS')
        if sender_sells_override:
            self.config.use_helius_sender_for_sells = sender_sells_override.lower() == 'true'
        slippage_override = get_secret('DEFAULT_SLIPPAGE_BPS')
        if slippage_override:
            try:
                self.config.default_slippage_bps = int(slippage_override)
            except ValueError:
                logger.warning(f"Invalid DEFAULT_SLIPPAGE_BPS: {slippage_override}")
        jito_tip_override = get_secret('JITO_TIP_LAMPORTS')
        if jito_tip_override:
            try:
                self.config.jito_tip_lamports = int(jito_tip_override)
            except ValueError:
                logger.warning(f"Invalid JITO_TIP_LAMPORTS: {jito_tip_override}")
        priority_override = get_secret('PRIORITY_FEE_LAMPORTS')
        if priority_override:
            try:
                self.config.priority_fee_lamports = int(priority_override)
            except ValueError:
                logger.warning(f"Invalid PRIORITY_FEE_LAMPORTS: {priority_override}")
        dynamic_fee_override = get_secret('ENABLE_DYNAMIC_PRIORITY_FEES')
        if dynamic_fee_override:
            self.config.enable_dynamic_priority_fees = dynamic_fee_override.lower() == 'true'
        fee_level_override = get_secret('PRIORITY_FEE_LEVEL')
        if fee_level_override:
            self.config.priority_fee_level = fee_level_override
        min_fee_override = get_secret('MIN_PRIORITY_FEE_LAMPORTS')
        if min_fee_override:
            try:
                self.config.min_priority_fee_lamports = int(min_fee_override)
            except ValueError:
                logger.warning(f"Invalid MIN_PRIORITY_FEE_LAMPORTS: {min_fee_override}")
        max_fee_override = get_secret('MAX_PRIORITY_FEE_LAMPORTS')
        if max_fee_override:
            try:
                self.config.max_priority_fee_lamports = int(max_fee_override)
            except ValueError:
                logger.warning(f"Invalid MAX_PRIORITY_FEE_LAMPORTS: {max_fee_override}")
        dynamic_cu_override = get_secret('DYNAMIC_COMPUTE_UNIT_LIMIT')
        if dynamic_cu_override:
            self.config.dynamic_compute_unit_limit = dynamic_cu_override.lower() == 'true'
        compute_unit_override = get_secret('COMPUTE_UNIT_LIMIT')
        if compute_unit_override:
            try:
                self.config.compute_unit_limit = int(compute_unit_override)
            except ValueError:
                logger.warning(f"Invalid COMPUTE_UNIT_LIMIT: {compute_unit_override}")
        exit_interval_override = get_secret('EXIT_MONITOR_INTERVAL_SECONDS')
        if exit_interval_override:
            try:
                self.config.exit_monitor_interval_seconds = int(exit_interval_override)
            except ValueError:
                logger.warning(f"Invalid EXIT_MONITOR_INTERVAL_SECONDS: {exit_interval_override}")
        exit_ws_override = get_secret('ENABLE_EXIT_WEBSOCKET')
        if exit_ws_override:
            self.config.enable_exit_websocket = exit_ws_override.lower() == 'true'
        exit_ws_ping_override = get_secret('EXIT_WEBSOCKET_PING_SECONDS')
        if exit_ws_ping_override:
            try:
                self.config.exit_websocket_ping_seconds = int(exit_ws_ping_override)
            except ValueError:
                logger.warning(f"Invalid EXIT_WEBSOCKET_PING_SECONDS: {exit_ws_ping_override}")
        exit_ws_reconnect_override = get_secret('EXIT_WEBSOCKET_RECONNECT_SECONDS')
        if exit_ws_reconnect_override:
            try:
                self.config.exit_websocket_reconnect_seconds = int(exit_ws_reconnect_override)
            except ValueError:
                logger.warning(f"Invalid EXIT_WEBSOCKET_RECONNECT_SECONDS: {exit_ws_reconnect_override}")
        retry_seconds_override = get_secret('HELIUS_SENDER_RETRY_SECONDS')
        if retry_seconds_override:
            try:
                self.config.helius_sender_retry_seconds = int(retry_seconds_override)
            except ValueError:
                logger.warning(f"Invalid HELIUS_SENDER_RETRY_SECONDS: {retry_seconds_override}")
        retry_interval_override = get_secret('HELIUS_SENDER_RETRY_INTERVAL')
        if retry_interval_override:
            try:
                self.config.helius_sender_retry_interval = float(retry_interval_override)
            except ValueError:
                logger.warning(f"Invalid HELIUS_SENDER_RETRY_INTERVAL: {retry_interval_override}")
        sender_tip_override = get_secret('HELIUS_SENDER_TIP_LAMPORTS')
        if sender_tip_override:
            try:
                self.config.helius_sender_tip_lamports = int(sender_tip_override)
            except ValueError:
                logger.warning(f"Invalid HELIUS_SENDER_TIP_LAMPORTS: {sender_tip_override}")
        if self.config.use_helius_sender and self.config.use_helius_sender_for_buys:
            self.config.enable_jito_bundles = False
        self.price_service = PriceService()
        self.tax_db = TaxDatabase("live_trades_tax.db")
        self._last_sender_tip_error = False
        self.helius_key = helius_key or get_secret('HELIUS_KEY')
        exit_config = ExitConfig(
            price_check_interval_seconds=self.config.exit_monitor_interval_seconds,
            enable_auto_exits=True,
        )
        helius_ws_url = get_secret("HELIUS_WS_URL")
        if not helius_ws_url and self.helius_key:
            helius_ws_url = f"wss://mainnet.helius-rpc.com/?api-key={self.helius_key}"
        self.exit_monitor = LiveExitManager(
            self,
            config=exit_config,
            enable_websocket=self.config.enable_exit_websocket,
            helius_ws_url=helius_ws_url,
            websocket_ping_seconds=self.config.exit_websocket_ping_seconds,
            websocket_reconnect_seconds=self.config.exit_websocket_reconnect_seconds,
        )
        self.exit_monitor.start_monitoring()
        
        # Load wallet
        self.keypair = None
        self.wallet_pubkey = None

        if private_key is None:
            hot_wallet_keys = [
                get_secret(f'HOT_WALLET_{i}') for i in range(1, 6)
            ]
            hot_wallet_keys = [key for key in hot_wallet_keys if key]
            if hot_wallet_keys:
                private_key = random.choice(hot_wallet_keys)
            else:
                private_key = get_secret('SOLANA_PRIVATE_KEY')
        
        if private_key:
            try:
                if isinstance(private_key, list):
                    self.keypair = Keypair.from_bytes(bytes(private_key))
                elif isinstance(private_key, dict):
                    key_value = private_key.get('private_key') or private_key.get('value')
                    if not key_value:
                        raise ValueError("Unsupported key dict format")
                    self.keypair = Keypair.from_base58_string(key_value)
                else:
                    self.keypair = Keypair.from_base58_string(private_key)
                
                self.wallet_pubkey = str(self.keypair.pubkey())
                logger.info(f"🔐 Wallet loaded: {self.wallet_pubkey[:8]}...{self.wallet_pubkey[-4:]}")
            except Exception as e:
                logger.error(f"Failed to load wallet: {e}")
        
        # Solana client
        if self.helius_key:
            self.rpc_url = f"https://mainnet.helius-rpc.com/?api-key={self.helius_key}"
            self.solana_client = SolanaClient(self.rpc_url) if SOLANA_AVAILABLE else None
        else:
            self.solana_client = None
            logger.warning("⚠️ No Helius key - RPC unavailable")
        
        # Jito service
        self.jito = JitoBundleService(self.keypair) if self.keypair else None
        
        # State tracking
        self._consecutive_losses = 0
        self._cool_down_until = None
        self._kill_switch_active = False
        self._lock = threading.Lock()
        
        # Monitoring thread
        self._monitor_thread = None
        self._stop_monitoring = threading.Event()
        
        logger.info("🚀 Live Trading Engine initialized")
        logger.info(f"   Live trading: {'ENABLED' if self.config.enable_live_trading else 'DISABLED'}")
        logger.info(f"   Position size: {self.config.position_size_sol} SOL")
        logger.info(f"   Max positions: {self.config.max_open_positions}")
        logger.info(f"   Jito bundles: {'ENABLED' if self.config.enable_jito_bundles else 'DISABLED'}")
        if self.config.use_helius_sender:
            logger.info("   Helius sender: ENABLED")
    
    def get_sol_balance(self) -> float:
        """Get current SOL balance"""
        if not self.solana_client or not self.keypair:
            return 0.0
        
        try:
            resp = self.solana_client.get_balance(self.keypair.pubkey())
            return resp.value / LAMPORTS_PER_SOL
        except Exception as e:
            logger.error(f"Balance check failed: {e}")
            return 0.0
    
    def can_open_position(self, signal: Dict = None) -> Tuple[bool, str]:
        """Check if we can open a new position"""
        
        if self._kill_switch_active:
            return False, "Kill switch active"
        
        if not self.config.enable_live_trading:
            return False, "Live trading disabled"
        
        if self._cool_down_until and datetime.now(timezone.utc) < self._cool_down_until:
            remaining = (self._cool_down_until - datetime.now(timezone.utc)).seconds // 60
            return False, f"Cool down active ({remaining}m remaining)"
        
        # Check balance
        balance = self.get_sol_balance()
        if balance < self.config.min_balance_sol:
            self._activate_kill_switch("Balance below minimum")
            return False, f"Balance too low: {balance:.4f} SOL"
        
        if balance < self.config.position_size_sol + self.config.fee_reserve_sol:
            return False, "Insufficient balance for position + fees"
        
        # Check position count
        positions = self.tax_db.get_positions()
        if len(positions) >= self.config.max_open_positions:
            return False, f"Max positions reached ({self.config.max_open_positions})"
        
        # Check deployed capital
        deployed = sum(p.get('total_cost_sol', 0) for p in positions)
        if deployed + self.config.position_size_sol > self.config.max_deployed_sol:
            return False, f"Max deployed capital reached ({self.config.max_deployed_sol} SOL)"
        
        # Check daily stats
        daily = self.tax_db.get_daily_stats()
        if daily.get('pnl_sol', 0) <= -self.config.max_daily_loss_sol:
            return False, f"Daily loss limit reached ({self.config.max_daily_loss_sol} SOL)"
        
        # Check time filters
        if signal:
            current_hour = datetime.now(timezone.utc).hour
            if current_hour in self.config.blocked_hours_utc:
                return False, f"Hour {current_hour} UTC is blocked"
            
            conviction = signal.get('conviction_score', signal.get('conviction', 0))
            if conviction < self.config.min_conviction:
                return False, f"Conviction {conviction} < {self.config.min_conviction}"
            
            #liquidity = signal.get('liquidity', 0)
            #if liquidity < self.config.min_liquidity_usd:
            #    return False, f"Liquidity ${liquidity:,.0f} < ${self.config.min_liquidity_usd:,.0f}"
            # Liquidity filter intentionally disabled
            
        return True, "OK"
    
    def validate_jito_setup(self) -> Tuple[bool, str]:
        """
        Validate that Jito bundle execution is properly configured.
        Call this before enabling live trading!
        
        Returns:
            (success, message)
        """
        issues = []
        
        # Check Solana libraries
        if not SOLANA_AVAILABLE:
            issues.append("Solana libraries not installed")
        
        # Check keypair
        if not self.keypair:
            issues.append("No wallet keypair configured")
        
        # Check Jito service
        if not self.jito:
            issues.append("Jito service not initialized")
        else:
            # Test tip account rotation
            tip_account = self.jito.get_tip_account()
            if not tip_account:
                issues.append("Failed to get Jito tip account")
        
        # Check RPC client
        if not self.solana_client:
            issues.append("Solana RPC client not configured")
        else:
            # Test RPC connection
            try:
                if hasattr(self.solana_client, 'get_health'):
                    resp = self.solana_client.get_health()
                    if resp != 'ok':
                        issues.append(f"RPC health check returned: {resp}")
                else:
                    self.solana_client.get_latest_blockhash()
            except Exception as e:
                issues.append(f"RPC health check failed: {e}")
        
        # Check balance
        balance = self.get_sol_balance()
        if balance < self.config.position_size_sol + self.config.jito_tip_lamports / LAMPORTS_PER_SOL:
            issues.append(f"Insufficient balance: {balance:.4f} SOL")
        
        # Check tip amount is reasonable
        tip_sol = self.config.jito_tip_lamports / LAMPORTS_PER_SOL
        if tip_sol < 0.0001:
            issues.append(f"Tip too low ({tip_sol} SOL) - may not be prioritized")
        if tip_sol > 0.01:
            issues.append(f"Tip very high ({tip_sol} SOL) - consider reducing")
        
        if issues:
            return False, "; ".join(issues)
        
        return True, f"✅ Jito setup validated. Balance: {balance:.4f} SOL, Tip: {tip_sol:.4f} SOL"
    
    def test_jito_connection(self) -> bool:
        """Test Jito block engine connectivity"""
        try:
            # Simple connectivity test to Jito
            resp = requests.get(
                f"{JITO_BLOCK_ENGINE_URL}/api/v1/bundles",
                f"{JITO_BLOCK_ENGINE_URLS[0]}/api/v1/bundles",
                timeout=10
            )
            # Even if we get an error response, connectivity is working
            logger.info(f"Jito connectivity test: {resp.status_code}")
            return resp.status_code in [200, 400, 405]  # Various valid responses
        except Exception as e:
            logger.error(f"Jito connectivity test failed: {e}")
            return False
    
    def execute_buy(self, signal: Dict) -> Dict:
        """
        Execute a buy order.
        
        Returns execution result dict.
        """
        token_address = signal.get('token_address')
        token_symbol = signal.get('token_symbol', 'UNKNOWN')
        
        result = {
            'success': False,
            'action': 'BUY',
            'token_address': token_address,
            'token_symbol': token_symbol,
            'signature': None,
            'error': None,
            'is_live': True
        }
        
        # Pre-checks
        can_trade, reason = self.can_open_position(signal)
        if not can_trade:
            result['error'] = reason
            self.tax_db.log_execution('BUY', token_address, token_symbol, 'REJECTED', error=reason)
            return result
        
        # Get prices
        sol_usd, sol_nzd = self.price_service.get_sol_prices()
        token_price = self.price_service.get_token_price(token_address)
        
        if not token_price:
            result['error'] = "Could not get token price"
            self.tax_db.log_execution('BUY', token_address, token_symbol, 'FAILED', error=result['error'])
            return result
        
        # Calculate amounts
        sol_amount = self.config.position_size_sol
        input_lamports = int(sol_amount * LAMPORTS_PER_SOL)
        
        try:
            # Get Jupiter quote
            quote = self._get_jupiter_quote(
                input_mint=SOL_MINT,
                output_mint=token_address,
                amount=input_lamports,
                slippage_bps=self.config.default_slippage_bps
            )
            
            if not quote:
                result['error'] = "Failed to get quote"
                self.tax_db.log_execution('BUY', token_address, token_symbol, 'FAILED', error=result['error'])
                return result
            
            # Get swap transaction
            use_sender = self.config.use_helius_sender and self.config.use_helius_sender_for_buys
            swap_tx = self._get_jupiter_swap(quote, as_legacy=use_sender)
            
            if not swap_tx:
                result['error'] = "Failed to get swap transaction"
                self.tax_db.log_execution('BUY', token_address, token_symbol, 'FAILED', error=result['error'])
                return result
            
            # Execute via configured transport (Helius sender or standard RPC).
            signed_payload = None
            signature, signed_payload = self._execute_swap_transport(swap_tx, use_sender=use_sender)
            
            if not signature:
                result['error'] = "Transaction failed"
                self.tax_db.log_execution('BUY', token_address, token_symbol, 'FAILED', error=result['error'])
                return result
            
            # Wait for confirmation
            confirmed = self._wait_for_confirmation(signature, signed_payload)
            
            if not confirmed:
                result['error'] = "Transaction not confirmed"
                self.tax_db.log_execution('BUY', token_address, token_symbol, 'UNCONFIRMED', 
                                          signature=signature, error=result['error'])
                return result
            
            # Calculate received tokens
            out_amount = int(quote.get('outAmount', 0))
            tokens_received = out_amount / (10 ** 6)  # Assume 6 decimals, adjust as needed
            
            # Calculate values
            token_price_nzd = token_price * (sol_nzd / sol_usd) if sol_usd > 0 else 0
            total_value_usd = sol_amount * sol_usd
            total_value_nzd = sol_amount * sol_nzd
            
            # Estimate fee
            fee_sol = 0.002  # ~0.002 SOL typical
            
            # Record trade
            trade_data = {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'transaction_type': 'BUY',
                'token_address': token_address,
                'token_symbol': token_symbol,
                'token_amount': tokens_received,
                'sol_amount': sol_amount,
                'price_per_token_usd': token_price,
                'price_per_token_nzd': token_price_nzd,
                'sol_price_usd': sol_usd,
                'sol_price_nzd': sol_nzd,
                'total_value_usd': total_value_usd,
                'total_value_nzd': total_value_nzd,
                'fee_sol': fee_sol,
                'fee_nzd': fee_sol * sol_nzd,
                'signature': signature,
                'is_live': True
            }
            self.tax_db.record_trade(trade_data)
            
            # Add position
            liquidity_pool_vault = self.price_service.get_main_pool_vault(token_address)
            position_data = {
                'token_address': token_address,
                'token_symbol': token_symbol,
                'entry_timestamp': datetime.now(timezone.utc).isoformat(),
                'entry_time': datetime.now(timezone.utc).isoformat(),
                'entry_price_usd': token_price,
                'tokens_held': tokens_received,
                'total_cost_sol': sol_amount,
                'total_cost_usd': total_value_usd,
                'total_cost_nzd': total_value_nzd,
                'entry_signature': signature,
                'liquidity_pool_vault': liquidity_pool_vault,
                'conviction_score': signal.get('conviction_score', 0),
                'stop_loss_pct': signal.get('stop_loss_pct', self.config.stop_loss_pct),
                'take_profit_pct': signal.get('take_profit_pct', self.config.take_profit_pct),
                'trailing_stop_pct': signal.get('trailing_stop_pct', self.config.trailing_stop_pct),
                'peak_price_usd': token_price,
            }
            self.tax_db.add_position(position_data)
            if liquidity_pool_vault and self.exit_monitor:
                self.exit_monitor.refresh_websocket_subscriptions()
            
            # Log success
            self.tax_db.log_execution('BUY', token_address, token_symbol, 'SUCCESS', 
                                      signature=signature, details={'sol_spent': sol_amount})
            
            result['success'] = True
            result['signature'] = signature
            result['sol_amount'] = sol_amount
            result['tokens_received'] = tokens_received
            
            logger.info(f"✅ BUY executed: {token_symbol} | {sol_amount} SOL | Sig: {signature[:16]}...")
            
        except Exception as e:
            result['error'] = str(e)
            self.tax_db.log_execution('BUY', token_address, token_symbol, 'ERROR', error=str(e))
            logger.error(f"❌ BUY failed: {e}")
        
        return result
    
    def execute_sell(self, token_address: str, exit_reason: str = "MANUAL") -> Dict:
        """Execute a sell order for an open position"""
        
        result = {
            'success': False,
            'action': 'SELL',
            'token_address': token_address,
            'exit_reason': exit_reason,
            'signature': None,
            'error': None,
            'pnl_sol': 0,
            'pnl_pct': 0
        }
        
        # Get position
        positions = self.tax_db.get_positions()
        position = next((p for p in positions if p['token_address'] == token_address), None)
        
        if not position:
            result['error'] = "Position not found"
            return result
        
        token_symbol = position.get('token_symbol', 'UNKNOWN')
        tokens_held = position.get('tokens_held', 0)
        entry_cost_sol = position.get('total_cost_sol', 0)
        entry_cost_nzd = position.get('total_cost_nzd', 0)
        
        try:
            # Get prices
            sol_usd, sol_nzd = self.price_service.get_sol_prices()
            token_price = self.price_service.get_token_price(token_address)
            
            if not token_price:
                result['error'] = "Could not get token price"
                return result
            
            # Calculate token amount in smallest unit
            decimals = self._get_token_decimals(token_address)
            token_amount = int(Decimal(str(tokens_held)) * (10 ** decimals))
            if token_amount <= 0:
                result['error'] = "Token amount is too small to swap"
                return result
            
            # Get Jupiter quote (sell token for SOL)
            quote = self._get_jupiter_quote(
                input_mint=token_address,
                output_mint=SOL_MINT,
                amount=token_amount,
                slippage_bps=self.config.default_slippage_bps
            )
            
            if not quote:
                result['error'] = "Failed to get quote"
                return result
            
            # Get swap transaction
            use_sender = self.config.use_helius_sender and self.config.use_helius_sender_for_sells
            swap_tx = self._get_jupiter_swap(quote, as_legacy=use_sender)
            
            if not swap_tx:
                result['error'] = "Failed to get swap transaction"
                return result
            
            # Execute via configured transport (Helius sender or standard RPC).
            signed_payload = None
            signature, signed_payload = self._execute_swap_transport(swap_tx, use_sender=use_sender)
            
            if not signature:
                result['error'] = "Transaction failed"
                return result
            
            # Wait for confirmation
            confirmed = self._wait_for_confirmation(signature, signed_payload)

            if not confirmed:
                # Retry: get a new quote and try again
                logger.warning("Exit not confirmed; retrying with fresh quote...")
                retry_quote = self._get_jupiter_quote(
                    input_mint=token_address,
                    output_mint=SOL_MINT,
                    amount=token_amount,
                    slippage_bps=self.config.default_slippage_bps
                )
                if retry_quote:
                    retry_swap = self._get_jupiter_swap(
                        retry_quote,
                        as_legacy=(self.config.use_helius_sender and self.config.use_helius_sender_for_sells),
                    )
                    if retry_swap:
                        retry_sig, retry_payload = self._execute_swap_transport(
                            retry_swap,
                            use_sender=(self.config.use_helius_sender and self.config.use_helius_sender_for_sells),
                        )
                        if retry_sig:
                            retry_confirmed = self._wait_for_confirmation(retry_sig, retry_payload)
                            if retry_confirmed:
                                signature = retry_sig
                                confirmed = True
            
            if not confirmed:
                result['error'] = "Transaction not confirmed"
                return result
            
            # Calculate received SOL
            out_amount = int(quote.get('outAmount', 0))
            sol_received = out_amount / LAMPORTS_PER_SOL
            
            # Calculate P&L
            fee_sol = 0.002
            net_sol_received = sol_received - fee_sol
            pnl_sol = net_sol_received - entry_cost_sol
            pnl_pct = (pnl_sol / entry_cost_sol * 100) if entry_cost_sol > 0 else 0
            
            # Get FIFO cost basis
            cost_basis_nzd, _ = self.tax_db.get_fifo_cost_basis(token_address, tokens_held)
            proceeds_nzd = sol_received * sol_nzd
            gain_loss_nzd = proceeds_nzd - cost_basis_nzd
            
            # Record trade
            trade_data = {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'transaction_type': 'SELL',
                'token_address': token_address,
                'token_symbol': token_symbol,
                'token_amount': tokens_held,
                'sol_amount': sol_received,
                'price_per_token_usd': token_price,
                'price_per_token_nzd': token_price * (sol_nzd / sol_usd) if sol_usd > 0 else 0,
                'sol_price_usd': sol_usd,
                'sol_price_nzd': sol_nzd,
                'total_value_usd': sol_received * sol_usd,
                'total_value_nzd': proceeds_nzd,
                'fee_sol': fee_sol,
                'fee_nzd': fee_sol * sol_nzd,
                'cost_basis_nzd': cost_basis_nzd,
                'gain_loss_nzd': gain_loss_nzd,
                'signature': signature,
                'notes': f"Exit reason: {exit_reason}",
                'is_live': True
            }
            self.tax_db.record_trade(trade_data)
            
            # Update daily stats
            is_win = pnl_sol > 0
            self.tax_db.update_daily_stats(pnl_sol, is_win, fee_sol)
            
            # Update consecutive losses
            with self._lock:
                if is_win:
                    self._consecutive_losses = 0
                else:
                    self._consecutive_losses += 1
                    if self._consecutive_losses >= self.config.max_consecutive_losses:
                        self._trigger_cool_down()
            
            # Remove position
            self.tax_db.remove_position(token_address)
            
            # Log
            self.tax_db.log_execution('SELL', token_address, token_symbol, 'SUCCESS',
                                      signature=signature, details={
                                          'exit_reason': exit_reason,
                                          'pnl_sol': pnl_sol,
                                          'pnl_pct': pnl_pct
                                      })
            
            result['success'] = True
            result['signature'] = signature
            result['sol_received'] = sol_received
            result['pnl_sol'] = pnl_sol
            result['pnl_pct'] = pnl_pct
            result['token_symbol'] = token_symbol
            
            emoji = "✅" if is_win else "❌"
            logger.info(f"{emoji} SELL executed: {token_symbol} | {exit_reason} | PnL: {pnl_pct:+.1f}% ({pnl_sol:+.4f} SOL)")
            
        except Exception as e:
            result['error'] = str(e)
            logger.error(f"❌ SELL failed: {e}")
        
        return result
    
    def check_exit_conditions(self) -> List[Dict]:
        """Check all positions for exit conditions"""
        exits = []
        positions = self.tax_db.get_positions()
        
        for pos in positions:
            token_address = pos['token_address']
            token_symbol = pos.get('token_symbol', 'UNKNOWN')
            entry_price = pos.get('entry_price_usd', 0)
            peak_price = pos.get('peak_price_usd', entry_price)
            entry_time = self._parse_entry_time(pos.get('entry_time'))
            hours_held = (datetime.now(timezone.utc) - entry_time).total_seconds() / 3600
            
            # Get current price
            current_price = self.price_service.get_token_price(token_address)
            
            if not current_price or not entry_price:
                if hours_held >= self.config.max_hold_hours:
                    result = self.execute_sell(token_address, 'TIME_STOP')
                    exits.append(result)
                continue
            
            # Update peak price
            if current_price > peak_price:
                if hasattr(self.tax_db, "update_position_peak"):
                    self.tax_db.update_position_peak(token_address, current_price)
                peak_price = current_price
            
            # Calculate P&L
            pnl_pct = (current_price - entry_price) / entry_price
            
            # Check conditions
            exit_reason = None
            
            # Stop loss
            if pnl_pct <= pos.get('stop_loss_pct', -0.15):
                exit_reason = 'STOP_LOSS'
            
            # Take profit
            elif pnl_pct >= pos.get('take_profit_pct', 0.30):
                exit_reason = 'TAKE_PROFIT'
            
            # Trailing stop
            elif peak_price > entry_price:
                drop_from_peak = (current_price - peak_price) / peak_price
                if drop_from_peak <= -pos.get('trailing_stop_pct', 0.10):
                    exit_reason = 'TRAILING_STOP'
            
            # Time stop
            if hours_held >= self.config.max_hold_hours:
                exit_reason = 'TIME_STOP'
            
            if exit_reason:
                result = self.execute_sell(token_address, exit_reason)
                exits.append(result)
        
        return exits
    
    def _get_jupiter_quote(self, input_mint: str, output_mint: str, 
                           amount: int, slippage_bps: int = 100) -> Optional[Dict]:
        """Get a quote from Jupiter"""
        params = {
            'inputMint': input_mint,
            'outputMint': output_mint,
            'amount': str(amount),
            'slippageBps': slippage_bps
        }
        metis_url = get_secret('QUICKNODE_METIS_URL')
        if metis_url:
            metis_url = metis_url.rstrip('/')
            quote_url = f"{metis_url}/quote"
        else:
            quote_url = DEFAULT_JUPITER_QUOTE_URL

        for url in (quote_url, JUPITER_QUOTE_URL_FALLBACK):
            try:
                resp = requests.get(url, params=params, timeout=10)
                if resp.status_code == 200:
                    return resp.json()
                logger.warning(f"Jupiter quote non-200 ({resp.status_code}) from {url}")
            except Exception as e:
                logger.error(f"Jupiter quote error via {url}: {e}")
        
        return None
    
    def _get_jupiter_swap(self, quote: Dict) -> Optional[str]:
        """Get swap transaction from Jupiter"""
        priority_fee_lamports = self._get_priority_fee_lamports()
        payload = {
            'quoteResponse': quote,
            'userPublicKey': self.wallet_pubkey,
            'wrapAndUnwrapSol': True,
            'dynamicComputeUnitLimit': True,
            'prioritizationFeeLamports': priority_fee_lamports,
            'dynamicSlippage': {"maxBps": 1500},
            # Keep Jupiter defaults to avoid token-program/account incompatibilities
            # on newer pools (e.g., token-2022/shared account routes).
            'asLegacyTransaction': False,
        }
        if self.config.compute_unit_limit:
            payload['computeUnitLimit'] = self.config.compute_unit_limit
        metis_url = get_secret('QUICKNODE_METIS_URL')
        if metis_url:
            metis_url = metis_url.rstrip('/')
            swap_url = f"{metis_url}/swap"
        else:
            swap_url = DEFAULT_JUPITER_SWAP_URL

        for url in (swap_url, JUPITER_SWAP_URL_FALLBACK):
            try:
                resp = requests.post(url, json=payload, timeout=30)
                if resp.status_code == 200:
                    data = resp.json()
                    return data.get('swapTransaction')
                logger.warning(f"Jupiter swap non-200 ({resp.status_code}) from {url}")
            except Exception as e:
                logger.error(f"Jupiter swap error via {url}: {e}")
        
        return None
    
    def _execute_via_jito(self, swap_tx_base64: str) -> Optional[str]:
        """
        Execute transaction via Jito bundle with proper tip.
        
        Jito requires a bundle with:
        1. Your swap transaction
        2. A tip transaction paying a Jito validator
        
        Without the tip, Jito won't prioritize your bundle!
        """
        try:
            # 1. Decode the swap transaction (unsigned from Jupiter)
            swap_tx_bytes = base64.b64decode(swap_tx_base64)
            swap_tx = VersionedTransaction.from_bytes(swap_tx_bytes)
            
            # 2. Get a fresh blockhash for the tip transaction
            blockhash_resp = self.solana_client.get_latest_blockhash()
            recent_blockhash = blockhash_resp.value.blockhash
            
            # 3. Create the tip transaction
            tip_account_pubkey = Pubkey.from_string(self.jito.get_tip_account())
            tip_lamports = self.config.jito_tip_lamports
            
            # Create transfer instruction for the tip
            tip_instruction = transfer(
                TransferParams(
                    from_pubkey=self.keypair.pubkey(),
                    to_pubkey=tip_account_pubkey,
                    lamports=tip_lamports
                )
            )
            
            # Create tip transaction message
            tip_message = Message.new_with_blockhash(
                [tip_instruction],
                self.keypair.pubkey(),
                recent_blockhash
            )
            
            # Create and sign tip transaction
            tip_tx = Transaction.new_unsigned(tip_message)
            tip_tx.sign([self.keypair], recent_blockhash)
            
            # 4. Sign the swap transaction
            swap_tx = self._sign_versioned_transaction(swap_tx)
            
            # 5. Encode both transactions for the bundle
            signed_swap = base64.b64encode(bytes(swap_tx)).decode('utf-8')
            signed_tip = base64.b64encode(bytes(tip_tx)).decode('utf-8')
            
            # 6. Submit bundle: [swap_tx, tip_tx]
            # Note: Swap first, tip second (tip is the "bribe" for including the swap)
            bundle_id = self.jito.submit_bundle([signed_swap, signed_tip])
            
            if bundle_id:
                logger.info(f"📦 Jito bundle submitted: {bundle_id[:16]}... (tip: {tip_lamports/LAMPORTS_PER_SOL:.4f} SOL)")
                
                # Wait for bundle confirmation
                for attempt in range(30):  # 30 second timeout
                    time.sleep(1)
                    status = self.jito.get_bundle_status(bundle_id)
                    
                    if status == 'landed':
                        logger.info(f"✅ Jito bundle landed!")
                        return str(swap_tx.signatures[0])
                    elif status == 'finalized':
                        logger.info(f"✅ Jito bundle finalized!")
                        return str(swap_tx.signatures[0])
                    elif status in ['invalid', 'failed']:
                        logger.error(f"❌ Jito bundle failed: {status}")
                        break
                    
                    # Also check if transaction is on chain directly
                    if attempt > 5:
                        try:
                            sig = Signature.from_string(str(swap_tx.signatures[0]))
                            tx_resp = self.solana_client.get_signature_statuses([sig])
                            if tx_resp.value and tx_resp.value[0]:
                                if tx_resp.value[0].confirmation_status:
                                    logger.info(f"✅ Transaction confirmed on chain!")
                                    return str(swap_tx.signatures[0])
                        except:
                            pass
                
                # If bundle didn't confirm, still return signature to check later
                logger.warning(f"⚠️ Jito bundle status unclear, returning signature for manual check")
                return str(swap_tx.signatures[0])
            
            # Bundle submission failed
            logger.error("❌ Failed to submit Jito bundle")
            if getattr(self.jito, "last_status_code", None) == 429:
                logger.warning("⚠️ Jito rate limited (429). Falling back to RPC execution.")
                return self._execute_via_rpc(swap_tx_base64)
            return None
            
        except Exception as e:
            logger.error(f"Jito execution error: {e}")
            import traceback
            traceback.print_exc()
            
            # Fallback to regular RPC if Jito fails
            logger.info("↩️ Falling back to regular RPC...")
            return self._execute_via_rpc(swap_tx_base64)
        
        return None
    
    def _execute_via_rpc(self, swap_tx_base64: str) -> Optional[str]:
        """Execute transaction via regular RPC"""
        if self.config.use_helius_sender:
            return self._execute_via_helius_sender(swap_tx_base64)
        try:
            # Decode and sign
            tx_bytes = base64.b64decode(swap_tx_base64)
            tx = VersionedTransaction.from_bytes(tx_bytes)
            tx = self._sign_versioned_transaction(tx)
            
            # Send
            resp = self.solana_client.send_transaction(tx)
            
            if resp.value:
                return str(resp.value)
            
        except Exception as e:
            logger.error(f"RPC execution error: {e}")
        
        return None
    
    def _sign_swap_transaction_base64(self, swap_tx_base64: str) -> Optional[str]:
        """Sign Jupiter-provided swap tx (legacy or v0) without mutating instructions."""
        try:
            tx_bytes = base64.b64decode(swap_tx_base64)
            try:
                tx = VersionedTransaction.from_bytes(tx_bytes)
                tx = self._sign_versioned_transaction(tx)
                return base64.b64encode(bytes(tx)).decode("utf-8")
            except Exception:
                tx = Transaction.from_bytes(tx_bytes)
                tx.sign([self.keypair], tx.message.recent_blockhash)
                return base64.b64encode(bytes(tx)).decode("utf-8")
        except Exception as e:
            logger.error(f"Swap signing error: {e}")
            return None

    def _send_signed_tx_via_rpc(self, signed_tx_base64: str) -> Optional[str]:
        """Send an already-signed tx via standard RPC sendTransaction."""
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "sendTransaction",
            "params": [
                signed_tx_base64,
                {"encoding": "base64", "skipPreflight": True, "maxRetries": 5},
            ],
        }
        try:
            resp = requests.post(self.rpc_url, json=payload, timeout=30)
            if resp.status_code != 200:
                logger.error(f"RPC send HTTP error: {resp.status_code} - {resp.text[:200]}")
                return None
            result = resp.json()
            if "result" in result:
                return str(result["result"])
            logger.error(f"RPC send error: {result.get('error')}")
            return None
        except Exception as e:
            logger.error(f"RPC send execution error: {e}")
            return None

    def _execute_swap_transport(self, swap_tx_base64: str, use_sender: bool) -> Tuple[Optional[str], Optional[str]]:
        """Sign Jupiter tx and submit via Helius sender or standard RPC."""
        signed_tx_base64 = None
        if use_sender:
            signed_tx_base64 = self._prepare_helius_signed_tx(swap_tx_base64)
            if not signed_tx_base64:
                logger.warning("Sender tx preparation failed; falling back to standard RPC signing path.")
                signed_tx_base64 = self._sign_swap_transaction_base64(swap_tx_base64)
        else:
            signed_tx_base64 = self._sign_swap_transaction_base64(swap_tx_base64)
        if not signed_tx_base64:
            return None, None

        self._last_sender_tip_error = False
        signature = self._send_helius_signed_tx(signed_tx_base64) if use_sender else self._send_signed_tx_via_rpc(signed_tx_base64)
        if signature:
            via = "Helius sender" if use_sender else "standard RPC"
            logger.info(f"📤 {via} accepted: {signature[:32]}...")
            return signature, signed_tx_base64

        if use_sender and self._last_sender_tip_error:
            logger.warning("⚠️ Helius sender rejected tx for missing tip; falling back to standard RPC for this swap.")
            fallback_sig = self._send_signed_tx_via_rpc(signed_tx_base64)
            if fallback_sig:
                logger.info(f"📤 standard RPC fallback accepted: {fallback_sig[:32]}...")
                return fallback_sig, signed_tx_base64

        return None, None
    
    def _execute_direct_rpc(self, swap_tx_base64: str) -> Tuple[Optional[str], Optional[str]]:
        """Execute a swap via standard Helius RPC with fresh blockhash.
        
        This avoids the Helius sender (which requires a tip instruction that
        forces decompiling and rebuilding the message, corrupting account metadata).
        
        Instead: replace blockhash bytes directly in Jupiter's serialized message,
        preserving ALL instructions and account flags exactly as Jupiter built them.
        Send via standard RPC sendTransaction which has no tip requirement.
        """
        try:
            tx_bytes = base64.b64decode(swap_tx_base64)
            
            # Parse as legacy transaction
            tx = Transaction.from_bytes(tx_bytes)
            message = tx.message
            
            # Fetch fresh blockhash
            blockhash_resp = self.solana_client.get_latest_blockhash()
            fresh_blockhash = blockhash_resp.value.blockhash
            
            # Replace blockhash directly in serialized message bytes
            msg_bytes = bytearray(bytes(message))
            old_bh_bytes = bytes(message.recent_blockhash)
            pos = msg_bytes.find(old_bh_bytes)
            if pos == -1:
                logger.error("Could not locate blockhash in message bytes")
                return None, None
            
            msg_bytes[pos:pos+32] = bytes(fresh_blockhash)
            new_message = Message.from_bytes(bytes(msg_bytes))
            new_tx = Transaction.new_unsigned(new_message)
            new_tx.sign([self.keypair], fresh_blockhash)
            
            # Send via standard RPC (no tip required)
            from solana.rpc.types import TxOpts
            opts = TxOpts(skip_preflight=True, max_retries=5)
            resp = self.solana_client.send_transaction(new_tx, opts=opts)
            
            if resp.value:
                sig = str(resp.value)
                signed_b64 = base64.b64encode(bytes(new_tx)).decode("utf-8")
                logger.info(f"📤 RPC accepted: {sig[:32]}... (blockhash={str(fresh_blockhash)[:16]}...)")
                return sig, signed_b64
            else:
                logger.error(f"RPC send failed: {resp}")
                return None, None
                
        except Exception as e:
            logger.error(f"Direct RPC execution error: {e}")
            import traceback
            traceback.print_exc()
            return None, None
    
    def _prepare_helius_signed_tx(self, swap_tx_base64: str) -> Optional[str]:
        """Prepare a signed legacy tx with mandatory Helius tip for sender endpoint."""
        try:
            tx_bytes = base64.b64decode(swap_tx_base64)
            try:
                tx = Transaction.from_bytes(tx_bytes)
                tx = self._inject_helius_tip(tx)
            except Exception:
                versioned_tx = VersionedTransaction.from_bytes(tx_bytes)
                tx = self._build_legacy_from_versioned(versioned_tx)

            blockhash_resp = self.solana_client.get_latest_blockhash()
            fresh_blockhash = blockhash_resp.value.blockhash

            new_message = Message.new_with_blockhash(
                list(self._compiled_to_instructions(tx.message)[0]),
                self.keypair.pubkey(),
                fresh_blockhash,
            )
            tx = Transaction.new_unsigned(new_message)
            tx.sign([self.keypair], fresh_blockhash)

            return base64.b64encode(bytes(tx)).decode("utf-8")
        except Exception as e:
            logger.error(f"Helius sender signing error: {e}")
            return None

    def _send_helius_signed_tx(self, signed_tx_base64: str) -> Optional[str]:
        """Send a signed legacy transaction via Helius sender."""
        sender_url = get_secret("HELIUS_SENDER_URL")
        if not sender_url:
            helius_key = get_secret("HELIUS_KEY")
            if helius_key:
                sender_url = f"https://sender.helius-rpc.com/fast?api-key={helius_key}"
        if not sender_url:
            logger.error("Helius sender unavailable: RPC URL not configured")
            return None
        
        header_name = get_secret("HELIUS_PRIORITY_HEADER", "x-helius-priority")
        header_value = get_secret("HELIUS_PRIORITY_VALUE", "true")

        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "sendTransaction",
            "params": [
                signed_tx_base64,
                {"encoding": "base64", "skipPreflight": True, "maxRetries": 5}
            ]
        }
        headers = {
            "Content-Type": "application/json",
            header_name: header_value
        }

        try:
            resp = requests.post(sender_url, json=payload, headers=headers, timeout=30)
            if resp.status_code != 200:
                body_snippet = (resp.text or "")[:400]
                if "transaction must send a tip of at least" in body_snippet.lower():
                    self._last_sender_tip_error = True
                logger.error(f"Helius sender HTTP error: {resp.status_code} - {body_snippet}")
                return None
            result = resp.json()
            if "result" in result:
                return str(result["result"])
            error_blob = str(result.get('error'))
            if "transaction must send a tip of at least" in error_blob.lower():
                self._last_sender_tip_error = True
            logger.error(f"Helius sender error: {result.get('error')}")
        except Exception as e:
            logger.error(f"Helius sender execution error: {e}")

        return None
    
    def _execute_via_helius_sender_with_payload(self, swap_tx_base64: str) -> Tuple[Optional[str], Optional[str]]:
        """Execute via Helius sender and return signature plus signed payload."""
        signed_tx_base64 = self._sign_swap_transaction_base64(swap_tx_base64)
        if not signed_tx_base64:
            return None, None
        signature = self._send_helius_signed_tx(signed_tx_base64)
        return signature, signed_tx_base64

    def _execute_via_helius_sender(self, swap_tx_base64: str) -> Optional[str]:
        """Execute transaction via Helius sender (priority header)."""
        signature, _ = self._execute_via_helius_sender_with_payload(swap_tx_base64)
        return signature
    
    def _execute_via_helius_sender_with_retries(self, swap_tx_base64: str) -> Optional[str]:
        """Retry Helius sender submissions to improve landing rate."""
        deadline = time.time() + max(self.config.helius_sender_retry_seconds, 0)
        attempt = 0
        while time.time() <= deadline or attempt == 0:
            attempt += 1
            signature = self._execute_via_helius_sender(swap_tx_base64)
            if signature:
                return signature
            if self.config.helius_sender_retry_interval <= 0:
                break
            time.sleep(self.config.helius_sender_retry_interval)
        return None

    def _inject_helius_tip(self, tx: Transaction) -> Transaction:
        """Inject mandatory Helius tip into a legacy transaction."""
        message = tx.message
        if not hasattr(message, "instructions"):
            raise ValueError("Helius tip injection requires legacy transaction message")

        instructions, payer, recent_blockhash = self._compiled_to_instructions(message)
        instructions.append(self._build_helius_tip_instruction())

        new_message = Message.new_with_blockhash(
            instructions,
            payer,
            recent_blockhash
        )
        return Transaction.new_unsigned(new_message)

    def _build_legacy_from_versioned(self, tx: VersionedTransaction) -> Transaction:
        """Rebuild a legacy transaction from a versioned message plus Helius tip."""
        message = tx.message
        instructions, payer, recent_blockhash = self._compiled_to_instructions(message)
        instructions.append(self._build_helius_tip_instruction())
        new_message = Message.new_with_blockhash(
            instructions,
            payer,
            recent_blockhash
        )
        return Transaction.new_unsigned(new_message)

    def _build_helius_tip_instruction(self) -> Instruction:
        tip_pubkey = Pubkey.from_string(random.choice(HELIUS_TIP_ACCOUNTS))
        tip_lamports = max(self.config.helius_sender_tip_lamports, 200_000)
        return transfer(
            TransferParams(
                from_pubkey=self.keypair.pubkey(),
                to_pubkey=tip_pubkey,
                lamports=tip_lamports
            )
        )

    def _compiled_to_instructions(self, message) -> Tuple[List[Instruction], Pubkey, Hash]:
        account_keys = list(message.account_keys)
        payer = account_keys[0]
        header = message.header
        num_required = header.num_required_signatures
        num_readonly_signed = getattr(header, "num_readonly_signed", None)
        if num_readonly_signed is None:
            num_readonly_signed = header.num_readonly_signed_accounts
        num_readonly_unsigned = getattr(header, "num_readonly_unsigned", None)
        if num_readonly_unsigned is None:
            num_readonly_unsigned = header.num_readonly_unsigned_accounts
        writable_signed_cutoff = num_required - num_readonly_signed
        writable_unsigned_cutoff = len(account_keys) - num_readonly_unsigned

        instructions = []
        for compiled in message.instructions:
            program_id = account_keys[compiled.program_id_index]
            accounts = [
                AccountMeta(
                    account_keys[i],
                    i < num_required,
                    (i < writable_signed_cutoff) if i < num_required else (i < writable_unsigned_cutoff)
                )
                for i in compiled.accounts
            ]
            instructions.append(Instruction(program_id, compiled.data, accounts))

        return instructions, payer, message.recent_blockhash

    def _sign_versioned_transaction(self, tx: VersionedTransaction) -> VersionedTransaction:
        """Sign a VersionedTransaction across solders API variants."""
        if hasattr(tx, "sign"):
            tx.sign([self.keypair])
            return tx
        message = tx.message
        try:
            message_bytes = bytes(message)
        except Exception:
            message_bytes = message.serialize()
        signature = self.keypair.sign_message(message_bytes)
        if not isinstance(signature, Signature):
            signature = Signature.from_bytes(signature)
        return VersionedTransaction.populate(message, [signature])
    
    def _get_priority_fee_lamports(self) -> int:
        """Fetch dynamic priority fee or return configured fallback."""
        if not self.config.enable_dynamic_priority_fees:
            return self.config.priority_fee_lamports
        if not self.rpc_url or not self.wallet_pubkey:
            return self.config.priority_fee_lamports

        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "getPriorityFeeEstimate",
            "params": [
                {
                    "accountKeys": [self.wallet_pubkey],
                    "options": {"includeAllPriorityFeeLevels": True}
                }
            ],
        }
        try:
            response = requests.post(self.rpc_url, json=payload, timeout=10)
            response.raise_for_status()
            result = response.json().get("result", {})
            levels = result.get("priorityFeeLevels", {}) or {}
            target_level = self.config.priority_fee_level or "high"
            fee = levels.get(target_level)
            if fee is None:
                fee = result.get("priorityFeeEstimate")
            if fee is None:
                return self.config.priority_fee_lamports
            fee_int = int(fee)
            fee_int = max(self.config.min_priority_fee_lamports, fee_int)
            fee_int = min(self.config.max_priority_fee_lamports, fee_int)
            return fee_int
        except Exception as exc:
            logger.warning(f"Priority fee fetch failed: {exc}")
            return self.config.priority_fee_lamports
    
    def _wait_for_confirmation(
        self,
        signature: str,
        signed_tx_base64: Optional[str] = None,
        timeout: int = None,
        spam_interval: float = 0.4,
        spam_duration: float = 8.0,
        status_interval: float = 0.5,
    ) -> bool:
        """Wait for transaction confirmation with optional resend spamming."""
        timeout = timeout or self.config.confirmation_timeout
        start = time.monotonic()
        spam_until = start + max(spam_duration, 0) if signed_tx_base64 else start
        next_spam = start
        next_status = start

        if not self.solana_client:
            logger.warning("Confirmation check skipped: Solana client unavailable.")
            return False

        while time.monotonic() - start < timeout:
            now = time.monotonic()

            if signed_tx_base64 and now <= spam_until and now >= next_spam:
                try:
                    # Resend via standard RPC
                    resend_bytes = base64.b64decode(signed_tx_base64)
                    resend_tx = Transaction.from_bytes(resend_bytes)
                    from solana.rpc.types import TxOpts
                    self.solana_client.send_transaction(resend_tx, opts=TxOpts(skip_preflight=True))
                except Exception:
                    pass  # Resend failures are expected, don't log
                next_spam = now + max(spam_interval, 0.05)

            if now >= next_status:
                try:
                    resp = self.solana_client.get_signature_statuses([Signature.from_string(signature)])
                    statuses = resp.value
                    if statuses and statuses[0]:
                        status = statuses[0]
                        conf_status = str(status.confirmation_status)
                        elapsed = now - start
                        if 'confirmed' in conf_status.lower() or 'finalized' in conf_status.lower():
                            if status.err is None:
                                logger.info(f"✅ TX CONFIRMED ({conf_status}) in {elapsed:.1f}s")
                                return True
                            else:
                                logger.error(f"❌ TX ON-CHAIN ERROR: {status.err}")
                                return False
                except Exception as e:
                    logger.warning(f"Confirmation check error: {e}")

                next_status = now + max(status_interval, 0.2)

            time.sleep(0.05)
        
        logger.warning(f"⏱️ TX TIMEOUT ({timeout}s): {signature[:32]}...")
        return False
    
    def _parse_entry_time(self, entry_time_value: Any) -> datetime:
        """Parse stored entry time safely, defaulting to now on bad data."""
        if not entry_time_value:
            logger.warning("Missing entry_time on position; defaulting to now.")
            return datetime.now(timezone.utc)
        if isinstance(entry_time_value, datetime):
            if entry_time_value.tzinfo is None:
                return entry_time_value.replace(tzinfo=timezone.utc)
            return entry_time_value
        if isinstance(entry_time_value, str):
            cleaned = entry_time_value.replace('Z', '+00:00')
            try:
                return datetime.fromisoformat(cleaned)
            except ValueError:
                logger.warning(f"Invalid entry_time format '{entry_time_value}'; defaulting to now.")
                return datetime.now(timezone.utc)
        logger.warning(f"Unsupported entry_time type {type(entry_time_value)}; defaulting to now.")
        return datetime.now(timezone.utc)

    def _get_token_decimals(self, mint: str) -> int:
        """Fetch token decimals from RPC (cached)."""
        if not hasattr(self, "_token_decimals_cache"):
            self._token_decimals_cache = {}
        if mint in self._token_decimals_cache:
            return self._token_decimals_cache[mint]
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "getTokenSupply",
            "params": [mint],
        }
        try:
            response = requests.post(self.rpc_url, json=payload, timeout=20)
            response.raise_for_status()
            result = response.json().get("result", {})
            value = result.get("value", {})
            decimals = value.get("decimals")
            if decimals is None:
                raise RuntimeError("Missing decimals in RPC response.")
            decimals_int = int(decimals)
            self._token_decimals_cache[mint] = decimals_int
            return decimals_int
        except Exception as exc:
            logger.warning(f"Failed to fetch decimals for {mint}: {exc}. Defaulting to 6.")
            return 6

    def _prepare_legacy_swap(self, swap_tx_base64: str) -> Optional[str]:
        """Ensure a swap transaction is legacy encoded for Helius sender."""
        try:
            tx_bytes = base64.b64decode(swap_tx_base64)
            try:
                legacy_tx = Transaction.from_bytes(tx_bytes)
            except Exception:
                versioned_tx = VersionedTransaction.from_bytes(tx_bytes)
                legacy_tx = self._build_legacy_from_versioned(versioned_tx)
            return base64.b64encode(bytes(legacy_tx)).decode("utf-8")
        except Exception as exc:
            logger.error(f"Failed to prepare legacy swap: {exc}")
            return None
    
    def _trigger_cool_down(self):
        """Trigger cool down period"""
        self._cool_down_until = datetime.now(timezone.utc) + timedelta(minutes=self.config.cool_down_minutes)
        logger.warning(f"⏸️ Cool down triggered for {self.config.cool_down_minutes} minutes")
    
    def _activate_kill_switch(self, reason: str):
        """Activate kill switch and close all positions"""
        logger.critical(f"🚨 KILL SWITCH ACTIVATED: {reason}")
        self._kill_switch_active = True
        
        # Close all positions
        positions = self.tax_db.get_positions()
        for pos in positions:
            try:
                self.execute_sell(pos['token_address'], 'KILL_SWITCH')
            except Exception as e:
                logger.error(f"Failed to close {pos.get('token_symbol')}: {e}")
    
    def start_monitoring(self):
        """Start background exit monitoring"""
        if self._monitor_thread and self._monitor_thread.is_alive():
            return
        
        self._stop_monitoring.clear()
        self._monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self._monitor_thread.start()
        logger.info("🔄 Exit monitoring started")
    
    def stop_monitoring(self):
        """Stop background monitoring"""
        self._stop_monitoring.set()
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
        logger.info("⏹️ Exit monitoring stopped")
    
    def _monitoring_loop(self):
        """Background monitoring loop"""
        while not self._stop_monitoring.is_set():
            try:
                if self.config.enable_live_trading and not self._kill_switch_active:
                    exits = self.check_exit_conditions()
                    for exit in exits:
                        if exit.get('success'):
                            logger.info(f"🚪 Auto-exit: {exit.get('token_symbol')} | {exit.get('exit_reason')}")
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
            
            self._stop_monitoring.wait(30)  # Check every 30 seconds
    
    def get_status(self) -> Dict:
        """Get engine status"""
        balance = self.get_sol_balance()
        positions = self.tax_db.get_positions()
        daily = self.tax_db.get_daily_stats()
        
        deployed = sum(p.get('total_cost_sol', 0) for p in positions)
        
        return {
            'live_trading_enabled': self.config.enable_live_trading,
            'kill_switch_active': self._kill_switch_active,
            'in_cool_down': self._cool_down_until is not None and datetime.now(timezone.utc) < self._cool_down_until,
            'wallet': self.wallet_pubkey[:8] + '...' if self.wallet_pubkey else None,
            'balance_sol': balance,
            'deployed_sol': deployed,
            'available_sol': balance - deployed - self.config.fee_reserve_sol,
            'open_positions': len(positions),
            'max_positions': self.config.max_open_positions,
            'position_size': self.config.position_size_sol,
            'daily_trades': daily.get('trades', 0),
            'daily_pnl_sol': daily.get('pnl_sol', 0),
            'daily_wins': daily.get('wins', 0),
            'daily_losses': daily.get('losses', 0),
            'consecutive_losses': self._consecutive_losses
        }
    
    def print_status(self):
        """Print formatted status"""
        status = self.get_status()
        
        print("\n" + "=" * 60)
        print("🚀 LIVE TRADING ENGINE STATUS")
        print("=" * 60)
        
        print(f"\n  Trading: {'✅ ENABLED' if status['live_trading_enabled'] else '❌ DISABLED'}")
        if status['kill_switch_active']:
            print(f"  ⚠️ KILL SWITCH ACTIVE")
        if status['in_cool_down']:
            print(f"  ⏸️ IN COOL DOWN")
        
        print(f"\n  Wallet: {status['wallet']}")
        print(f"  Balance: {status['balance_sol']:.4f} SOL")
        print(f"  Deployed: {status['deployed_sol']:.4f} SOL")
        print(f"  Available: {status['available_sol']:.4f} SOL")
        
        print(f"\n  Positions: {status['open_positions']}/{status['max_positions']}")
        print(f"  Position Size: {status['position_size']} SOL")
        
        print(f"\n  Today's Stats:")
        print(f"    Trades: {status['daily_trades']}")
        print(f"    Wins: {status['daily_wins']}")
        print(f"    Losses: {status['daily_losses']}")
        print(f"    PnL: {status['daily_pnl_sol']:+.4f} SOL")
        
        print("\n" + "=" * 60)


# =============================================================================
# CLI
# =============================================================================

def main():
    """CLI interface"""
    import argparse
    
    # Try to initialize secrets
    try:
        from core.secrets_manager import init_secrets
        init_secrets()
    except ImportError:
        from dotenv import load_dotenv
        load_dotenv()
    
    parser = argparse.ArgumentParser(description="Live Trading Engine")
    parser.add_argument('command', 
                       choices=['status', 'positions', 'validate', 'validate-jito', 'test-jito', 'enable', 'disable', 'kill'],
                       help='Command to run')
    
    args = parser.parse_args()
    
    # Initialize engine
    config = LiveTradingConfig(
        enable_live_trading=get_secret('ENABLE_LIVE_TRADING', '').lower() == 'true',
        position_size_sol=float(get_secret('POSITION_SIZE_SOL', '0.08'))
    )
    
    engine = LiveTradingEngine(
        private_key=get_secret('SOLANA_PRIVATE_KEY'),
        helius_key=get_secret('HELIUS_KEY'),
        config=config
    )
    
    if args.command == 'status':
        engine.print_status()
    
    elif args.command == 'positions':
        positions = engine.tax_db.get_positions()
        print(f"\n📊 Open Positions ({len(positions)})")
        for pos in positions:
            print(f"  {pos['token_symbol']}: {pos['tokens_held']:.4f} tokens")
            print(f"    Entry: ${pos['entry_price_usd']:.8f}")
            print(f"    Cost: {pos['total_cost_sol']:.4f} SOL")
    
    elif args.command == 'validate':
        can_trade, reason = engine.can_open_position()
        print(f"\nCan trade: {'✅ YES' if can_trade else '❌ NO'}")
        print(f"Reason: {reason}")
    
    elif args.command == 'validate-jito':
        print("\n🔧 VALIDATING JITO SETUP")
        print("=" * 50)
        
        success, message = engine.validate_jito_setup()
        print(f"\n{message}")
        
        if success:
            print("\n✅ Jito is properly configured!")
            print(f"   Tip per transaction: {engine.config.jito_tip_lamports / LAMPORTS_PER_SOL:.4f} SOL")
            print(f"   Jito bundles enabled: {engine.config.enable_jito_bundles}")
        else:
            print("\n❌ Jito setup has issues - fix before live trading!")
    
    elif args.command == 'test-jito':
        print("\n🧪 TESTING JITO CONNECTIVITY")
        print("=" * 50)
        
        if engine.test_jito_connection():
            print("✅ Jito block engine is reachable!")
        else:
            print("❌ Cannot reach Jito block engine")
            print("   Check your network connection and firewall")
    
    elif args.command == 'enable':
        print("To enable live trading:")
        print("1. In AWS Secrets Manager, set ENABLE_LIVE_TRADING=true")
        print("2. Restart the bot")
        print("\nOr via API: POST /live/enable?confirm=LIVE")
    
    elif args.command == 'disable':
        print("To disable live trading:")
        print("1. In AWS Secrets Manager, set ENABLE_LIVE_TRADING=false")
        print("2. Restart the bot")
        print("\nOr via API: POST /live/disable")
    
    elif args.command == 'kill':
        confirm = input("⚠️ This will close ALL positions. Type 'KILL' to confirm: ")
        if confirm == 'KILL':
            engine._activate_kill_switch("Manual kill switch")
        else:
            print("Cancelled")


if __name__ == "__main__":
    main()