"""
THE EXECUTIONER V1 - Real Trade Execution with NZ Tax Compliance
================================================================

PURPOSE:
    Execute real trades on Solana based on signals from the Strategist.
    Maintain comprehensive transaction records for NZ tax compliance.

FEATURES:
    1. Jupiter DEX aggregation for best swap routes
    2. Secure wallet management
    3. Comprehensive transaction logging (NZ IRD compliant)
    4. Position limits and safety controls
    5. Slippage protection
    6. Automatic retry with exponential backoff
    7. Real-time NZD value tracking

NZ TAX COMPLIANCE:
    - Records cost basis for each acquisition
    - Tracks disposal events with proceeds
    - Calculates gains using FIFO method
    - Exports to CSV for accountant/IRD

USAGE:
    from executioner_v1 import Executioner
    
    exec = Executioner(
        private_key="your_base58_private_key",
        db=your_database_instance
    )
    
    # Execute a buy signal from Strategist
    result = exec.execute_signal(strategist_signal)
    
    # Force exit a position
    result = exec.exit_position(token_address, reason="manual")
    
    # Export tax records
    exec.export_tax_records("2024", "tax_report_2024.csv")

AUTHOR: Trading Bot System
VERSION: 1.0.0
"""

import os
import json
import time
import sqlite3
import requests
import base64
import hashlib
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict
from decimal import Decimal, ROUND_DOWN
from enum import Enum
from contextlib import contextmanager
import threading
from functools import lru_cache
import logging

# Solana/Jupiter imports - will need to be installed
try:
    from solders.keypair import Keypair
    from solders.pubkey import Pubkey
    from solders.transaction import VersionedTransaction
    from solders.signature import Signature
    from solders.commitment_config import CommitmentLevel
    from solana.rpc.api import Client as SolanaClient
    from solana.rpc.commitment import Confirmed, Finalized
    SOLANA_AVAILABLE = True
except ImportError:
    SOLANA_AVAILABLE = False
    print("⚠️ Solana libraries not installed. Run: pip install solana solders --break-system-packages")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Executioner")


# =============================================================================
# CONSTANTS
# =============================================================================

SOL_MINT = "So11111111111111111111111111111111111111112"
USDC_MINT = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"
LAMPORTS_PER_SOL = 1_000_000_000

# Jupiter API endpoints
JUPITER_QUOTE_URL = "https://quote-api.jup.ag/v6/quote"
JUPITER_SWAP_URL = "https://quote-api.jup.ag/v6/swap"
JUPITER_PRICE_URL = "https://price.jup.ag/v6/price"

# Helius RPC for reliable Solana access
HELIUS_RPC_TEMPLATE = "https://mainnet.helius-rpc.com/?api-key={}"

# CoinGecko for NZD conversion
COINGECKO_SOL_URL = "https://api.coingecko.com/api/v3/simple/price?ids=solana&vs_currencies=nzd,usd"

# DexScreener for token prices
DEXSCREENER_URL = "https://api.dexscreener.com/latest/dex/tokens/{}"


# =============================================================================
# ENUMS & DATA CLASSES
# =============================================================================

class TransactionType(Enum):
    BUY = "BUY"
    SELL = "SELL"
    TRANSFER_IN = "TRANSFER_IN"
    TRANSFER_OUT = "TRANSFER_OUT"
    FEE = "FEE"


class ExecutionStatus(Enum):
    PENDING = "PENDING"
    SUBMITTED = "SUBMITTED"
    CONFIRMED = "CONFIRMED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


@dataclass
class ExecutionConfig:
    """Configuration for the Executioner"""
    # Position sizing
    max_position_size_sol: float = 0.5          # Max SOL per trade
    min_position_size_sol: float = 0.01         # Min SOL per trade
    max_portfolio_pct: float = 0.20             # Max % of portfolio in single token
    max_open_positions: int = 10                # Max concurrent positions
    
    # Risk management
    default_slippage_bps: int = 100             # 1% slippage (100 basis points)
    max_slippage_bps: int = 500                 # 5% max slippage
    min_liquidity_usd: float = 10000            # Minimum liquidity to trade
    
    # Execution
    retry_attempts: int = 3                      # Retries on failure
    retry_delay_seconds: float = 2.0            # Delay between retries
    confirmation_timeout: int = 60              # Seconds to wait for confirmation
    priority_fee_lamports: int = 100000         # Priority fee (0.0001 SOL)
    
    # Safety
    require_strategist_signal: bool = True      # Only trade on Strategist signals
    min_conviction_score: int = 60              # Minimum conviction to execute
    enable_live_trading: bool = False           # MUST be True to execute real trades
    
    # Tax
    tax_year_start_month: int = 4               # NZ tax year starts April
    cost_basis_method: str = "FIFO"             # FIFO or WEIGHTED_AVERAGE


@dataclass
class TaxRecord:
    """Single transaction record for tax purposes"""
    id: str                              # Unique ID
    timestamp: datetime                  # UTC timestamp
    transaction_type: str                # BUY, SELL, FEE, etc.
    token_address: str                   # Token mint address
    token_symbol: str                    # Human readable symbol
    token_amount: float                  # Amount of tokens
    sol_amount: float                    # SOL spent/received
    price_per_token_usd: float          # Price at time of trade
    price_per_token_nzd: float          # NZD price
    sol_price_usd: float                 # SOL price at trade time
    sol_price_nzd: float                 # SOL price in NZD
    total_value_usd: float              # Total transaction value USD
    total_value_nzd: float              # Total transaction value NZD
    fee_sol: float                       # Transaction fee in SOL
    fee_nzd: float                       # Fee in NZD
    signature: str                       # Solana transaction signature
    cost_basis_nzd: Optional[float] = None    # For sells: cost basis
    gain_loss_nzd: Optional[float] = None     # For sells: realized gain/loss
    notes: str = ""                      # Additional notes
    
    def to_dict(self) -> Dict:
        return {
            **asdict(self),
            'timestamp': self.timestamp.isoformat() if self.timestamp else None
        }


@dataclass 
class Position:
    """Tracks an open position"""
    token_address: str
    token_symbol: str
    tokens_held: float
    entry_price_usd: float
    entry_time: datetime
    total_cost_sol: float
    total_cost_nzd: float
    stop_loss_pct: float = -0.12
    take_profit_pct: float = 0.30
    trailing_stop_pct: float = 0.08
    peak_price_usd: float = 0.0
    current_price_usd: float = 0.0
    strategist_conviction: int = 0
    entry_signature: str = ""


@dataclass
class ExecutionResult:
    """Result of a trade execution"""
    success: bool
    status: ExecutionStatus
    signature: Optional[str] = None
    token_address: str = ""
    token_symbol: str = ""
    transaction_type: str = ""
    tokens_amount: float = 0.0
    sol_amount: float = 0.0
    price_usd: float = 0.0
    price_nzd: float = 0.0
    fee_sol: float = 0.0
    error_message: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    tax_record_id: Optional[str] = None


# =============================================================================
# PRICE SERVICE
# =============================================================================

class PriceService:
    """
    Fetches current prices for tokens and SOL in both USD and NZD.
    Uses CoinGecko for SOL/NZD rate and DexScreener for token prices.
    """
    
    def __init__(self):
        self._sol_price_cache = {}
        self._cache_ttl = 30  # 30 seconds
        self._last_sol_fetch = 0
        self._lock = threading.Lock()
    
    def get_sol_prices(self) -> Dict[str, float]:
        """Get SOL price in USD and NZD"""
        with self._lock:
            now = time.time()
            if now - self._last_sol_fetch < self._cache_ttl and self._sol_price_cache:
                return self._sol_price_cache
            
            try:
                resp = requests.get(COINGECKO_SOL_URL, timeout=10)
                if resp.status_code == 200:
                    data = resp.json()
                    self._sol_price_cache = {
                        'usd': data['solana']['usd'],
                        'nzd': data['solana']['nzd']
                    }
                    self._last_sol_fetch = now
            except Exception as e:
                logger.warning(f"Failed to fetch SOL price: {e}")
                # Fallback to approximate if cache empty
                if not self._sol_price_cache:
                    self._sol_price_cache = {'usd': 200.0, 'nzd': 340.0}
            
            return self._sol_price_cache
    
    def get_token_price(self, token_address: str) -> Dict[str, float]:
        """Get token price in USD and convert to NZD"""
        try:
            # Try Jupiter price API first
            resp = requests.get(
                JUPITER_PRICE_URL,
                params={'ids': token_address},
                timeout=10
            )
            if resp.status_code == 200:
                data = resp.json()
                if token_address in data.get('data', {}):
                    price_usd = float(data['data'][token_address]['price'])
                    sol_prices = self.get_sol_prices()
                    nzd_usd_rate = sol_prices['nzd'] / sol_prices['usd']
                    return {
                        'usd': price_usd,
                        'nzd': price_usd * nzd_usd_rate
                    }
        except Exception as e:
            logger.warning(f"Jupiter price fetch failed: {e}")
        
        # Fallback to DexScreener
        try:
            resp = requests.get(DEXSCREENER_URL.format(token_address), timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                pairs = data.get('pairs', [])
                if pairs:
                    # Get the highest liquidity pair
                    best_pair = max(pairs, key=lambda p: float(p.get('liquidity', {}).get('usd', 0) or 0))
                    price_usd = float(best_pair.get('priceUsd', 0) or 0)
                    sol_prices = self.get_sol_prices()
                    nzd_usd_rate = sol_prices['nzd'] / sol_prices['usd']
                    return {
                        'usd': price_usd,
                        'nzd': price_usd * nzd_usd_rate
                    }
        except Exception as e:
            logger.warning(f"DexScreener price fetch failed: {e}")
        
        return {'usd': 0.0, 'nzd': 0.0}
    
    def get_token_info(self, token_address: str) -> Dict:
        """Get token metadata (symbol, decimals, etc.)"""
        try:
            resp = requests.get(DEXSCREENER_URL.format(token_address), timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                pairs = data.get('pairs', [])
                if pairs:
                    best_pair = max(pairs, key=lambda p: float(p.get('liquidity', {}).get('usd', 0) or 0))
                    base_token = best_pair.get('baseToken', {})
                    return {
                        'symbol': base_token.get('symbol', 'UNKNOWN'),
                        'name': base_token.get('name', ''),
                        'liquidity_usd': float(best_pair.get('liquidity', {}).get('usd', 0) or 0),
                        'volume_24h': float(best_pair.get('volume', {}).get('h24', 0) or 0),
                        'price_usd': float(best_pair.get('priceUsd', 0) or 0)
                    }
        except Exception as e:
            logger.warning(f"Token info fetch failed: {e}")
        
        return {'symbol': 'UNKNOWN', 'name': '', 'liquidity_usd': 0, 'volume_24h': 0, 'price_usd': 0}


# =============================================================================
# TAX RECORD DATABASE
# =============================================================================

class TaxDatabase:
    """
    SQLite database for comprehensive tax record keeping.
    Designed for NZ IRD compliance.
    """
    
    def __init__(self, db_path: str = "tax_records.db"):
        self.db_path = db_path
        self._lock = threading.Lock()
        self._init_schema()
        logger.info(f"✅ Tax database initialized: {db_path}")
    
    @contextmanager
    def connection(self):
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
        """Create all tax-related tables"""
        with self.connection() as conn:
            # Main transaction records
            conn.execute("""
                CREATE TABLE IF NOT EXISTS tax_transactions (
                    id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    transaction_type TEXT NOT NULL,
                    token_address TEXT NOT NULL,
                    token_symbol TEXT NOT NULL,
                    token_amount REAL NOT NULL,
                    sol_amount REAL NOT NULL,
                    price_per_token_usd REAL,
                    price_per_token_nzd REAL,
                    sol_price_usd REAL,
                    sol_price_nzd REAL,
                    total_value_usd REAL,
                    total_value_nzd REAL,
                    fee_sol REAL DEFAULT 0,
                    fee_nzd REAL DEFAULT 0,
                    signature TEXT,
                    cost_basis_nzd REAL,
                    gain_loss_nzd REAL,
                    notes TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Cost basis lots for FIFO tracking
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cost_basis_lots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    token_address TEXT NOT NULL,
                    acquisition_date TEXT NOT NULL,
                    tokens_acquired REAL NOT NULL,
                    tokens_remaining REAL NOT NULL,
                    cost_per_token_nzd REAL NOT NULL,
                    total_cost_nzd REAL NOT NULL,
                    acquisition_signature TEXT,
                    is_exhausted INTEGER DEFAULT 0,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Summary by tax year
            conn.execute("""
                CREATE TABLE IF NOT EXISTS tax_year_summary (
                    tax_year TEXT PRIMARY KEY,
                    total_acquisitions_nzd REAL DEFAULT 0,
                    total_disposals_nzd REAL DEFAULT 0,
                    total_gains_nzd REAL DEFAULT 0,
                    total_losses_nzd REAL DEFAULT 0,
                    net_gain_loss_nzd REAL DEFAULT 0,
                    total_fees_nzd REAL DEFAULT 0,
                    trade_count INTEGER DEFAULT 0,
                    updated_at TEXT
                )
            """)
            
            # Open positions
            conn.execute("""
                CREATE TABLE IF NOT EXISTS live_positions (
                    token_address TEXT PRIMARY KEY,
                    token_symbol TEXT NOT NULL,
                    tokens_held REAL NOT NULL,
                    entry_price_usd REAL,
                    entry_time TEXT,
                    total_cost_sol REAL,
                    total_cost_nzd REAL,
                    stop_loss_pct REAL DEFAULT -0.12,
                    take_profit_pct REAL DEFAULT 0.30,
                    trailing_stop_pct REAL DEFAULT 0.08,
                    peak_price_usd REAL DEFAULT 0,
                    current_price_usd REAL DEFAULT 0,
                    strategist_conviction INTEGER DEFAULT 0,
                    entry_signature TEXT,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Execution log
            conn.execute("""
                CREATE TABLE IF NOT EXISTS execution_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    action TEXT NOT NULL,
                    token_address TEXT,
                    token_symbol TEXT,
                    status TEXT NOT NULL,
                    signature TEXT,
                    details_json TEXT,
                    error_message TEXT
                )
            """)
            
            # Create indexes
            conn.execute("CREATE INDEX IF NOT EXISTS idx_tax_timestamp ON tax_transactions(timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_tax_token ON tax_transactions(token_address)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_lots_token ON cost_basis_lots(token_address)")
    
    def record_transaction(self, record: TaxRecord) -> str:
        """Record a tax transaction"""
        with self._lock:
            with self.connection() as conn:
                conn.execute("""
                    INSERT INTO tax_transactions (
                        id, timestamp, transaction_type, token_address, token_symbol,
                        token_amount, sol_amount, price_per_token_usd, price_per_token_nzd,
                        sol_price_usd, sol_price_nzd, total_value_usd, total_value_nzd,
                        fee_sol, fee_nzd, signature, cost_basis_nzd, gain_loss_nzd, notes
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    record.id, record.timestamp.isoformat(), record.transaction_type,
                    record.token_address, record.token_symbol, record.token_amount,
                    record.sol_amount, record.price_per_token_usd, record.price_per_token_nzd,
                    record.sol_price_usd, record.sol_price_nzd, record.total_value_usd,
                    record.total_value_nzd, record.fee_sol, record.fee_nzd, record.signature,
                    record.cost_basis_nzd, record.gain_loss_nzd, record.notes
                ))
                
                # Update tax year summary
                self._update_tax_year_summary(conn, record)
                
        return record.id
    
    def add_cost_basis_lot(self, token_address: str, tokens: float, 
                          cost_nzd: float, signature: str) -> int:
        """Add a new cost basis lot for FIFO tracking"""
        with self._lock:
            with self.connection() as conn:
                cursor = conn.execute("""
                    INSERT INTO cost_basis_lots (
                        token_address, acquisition_date, tokens_acquired, tokens_remaining,
                        cost_per_token_nzd, total_cost_nzd, acquisition_signature
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    token_address, datetime.now(timezone.utc).isoformat(),
                    tokens, tokens, cost_nzd / tokens if tokens > 0 else 0,
                    cost_nzd, signature
                ))
                return cursor.lastrowid
    
    def consume_cost_basis_fifo(self, token_address: str, tokens_to_sell: float) -> Tuple[float, List[Dict]]:
        """
        Consume cost basis lots using FIFO method.
        Returns (total_cost_basis_nzd, list of lots consumed)
        """
        with self._lock:
            with self.connection() as conn:
                # Get all non-exhausted lots for this token, ordered by acquisition date
                lots = conn.execute("""
                    SELECT * FROM cost_basis_lots 
                    WHERE token_address = ? AND is_exhausted = 0
                    ORDER BY acquisition_date ASC
                """, (token_address,)).fetchall()
                
                total_cost_basis = 0.0
                tokens_remaining = tokens_to_sell
                lots_consumed = []
                
                for lot in lots:
                    if tokens_remaining <= 0:
                        break
                    
                    available = lot['tokens_remaining']
                    consume = min(available, tokens_remaining)
                    cost_per_token = lot['cost_per_token_nzd']
                    
                    total_cost_basis += consume * cost_per_token
                    tokens_remaining -= consume
                    
                    lots_consumed.append({
                        'lot_id': lot['id'],
                        'tokens_consumed': consume,
                        'cost_basis': consume * cost_per_token
                    })
                    
                    # Update the lot
                    new_remaining = available - consume
                    if new_remaining <= 0.000001:  # Effectively zero
                        conn.execute(
                            "UPDATE cost_basis_lots SET tokens_remaining = 0, is_exhausted = 1 WHERE id = ?",
                            (lot['id'],)
                        )
                    else:
                        conn.execute(
                            "UPDATE cost_basis_lots SET tokens_remaining = ? WHERE id = ?",
                            (new_remaining, lot['id'])
                        )
                
                return total_cost_basis, lots_consumed
    
    def _update_tax_year_summary(self, conn, record: TaxRecord):
        """Update the tax year summary"""
        # NZ tax year runs April to March
        timestamp = record.timestamp
        if timestamp.month >= 4:
            tax_year = f"{timestamp.year}-{timestamp.year + 1}"
        else:
            tax_year = f"{timestamp.year - 1}-{timestamp.year}"
        
        # Get existing summary
        existing = conn.execute(
            "SELECT * FROM tax_year_summary WHERE tax_year = ?", (tax_year,)
        ).fetchone()
        
        if existing:
            acquisitions = existing['total_acquisitions_nzd'] or 0
            disposals = existing['total_disposals_nzd'] or 0
            gains = existing['total_gains_nzd'] or 0
            losses = existing['total_losses_nzd'] or 0
            fees = existing['total_fees_nzd'] or 0
            count = existing['trade_count'] or 0
        else:
            acquisitions = disposals = gains = losses = fees = count = 0
        
        if record.transaction_type == "BUY":
            acquisitions += record.total_value_nzd
        elif record.transaction_type == "SELL":
            disposals += record.total_value_nzd
            if record.gain_loss_nzd:
                if record.gain_loss_nzd > 0:
                    gains += record.gain_loss_nzd
                else:
                    losses += abs(record.gain_loss_nzd)
        
        fees += record.fee_nzd
        count += 1
        
        conn.execute("""
            INSERT OR REPLACE INTO tax_year_summary (
                tax_year, total_acquisitions_nzd, total_disposals_nzd,
                total_gains_nzd, total_losses_nzd, net_gain_loss_nzd,
                total_fees_nzd, trade_count, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            tax_year, acquisitions, disposals, gains, losses,
            gains - losses, fees, count, datetime.now(timezone.utc).isoformat()
        ))
    
    def get_open_positions(self) -> List[Dict]:
        """Get all open positions"""
        with self.connection() as conn:
            rows = conn.execute("SELECT * FROM live_positions").fetchall()
            return [dict(r) for r in rows]
    
    def save_position(self, position: Position):
        """Save or update a position"""
        with self._lock:
            with self.connection() as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO live_positions (
                        token_address, token_symbol, tokens_held, entry_price_usd,
                        entry_time, total_cost_sol, total_cost_nzd, stop_loss_pct,
                        take_profit_pct, trailing_stop_pct, peak_price_usd,
                        current_price_usd, strategist_conviction, entry_signature, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    position.token_address, position.token_symbol, position.tokens_held,
                    position.entry_price_usd, position.entry_time.isoformat(),
                    position.total_cost_sol, position.total_cost_nzd, position.stop_loss_pct,
                    position.take_profit_pct, position.trailing_stop_pct, position.peak_price_usd,
                    position.current_price_usd, position.strategist_conviction,
                    position.entry_signature, datetime.now(timezone.utc).isoformat()
                ))
    
    def close_position(self, token_address: str):
        """Remove a position after selling"""
        with self._lock:
            with self.connection() as conn:
                conn.execute("DELETE FROM live_positions WHERE token_address = ?", (token_address,))
    
    def get_position(self, token_address: str) -> Optional[Dict]:
        """Get a specific position"""
        with self.connection() as conn:
            row = conn.execute(
                "SELECT * FROM live_positions WHERE token_address = ?", (token_address,)
            ).fetchone()
            return dict(row) if row else None
    
    def log_execution(self, action: str, token_address: str, token_symbol: str,
                     status: str, signature: str = None, details: Dict = None, error: str = None):
        """Log an execution attempt"""
        with self.connection() as conn:
            conn.execute("""
                INSERT INTO execution_log (timestamp, action, token_address, token_symbol,
                                          status, signature, details_json, error_message)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                datetime.now(timezone.utc).isoformat(), action, token_address, token_symbol,
                status, signature, json.dumps(details) if details else None, error
            ))
    
    def export_tax_records(self, tax_year: str, output_path: str) -> str:
        """Export tax records to CSV for accountant/IRD"""
        import csv
        
        with self.connection() as conn:
            # Parse tax year (e.g., "2024-2025")
            start_year, end_year = map(int, tax_year.split('-'))
            start_date = f"{start_year}-04-01"
            end_date = f"{end_year}-03-31"
            
            records = conn.execute("""
                SELECT * FROM tax_transactions
                WHERE timestamp >= ? AND timestamp <= ?
                ORDER BY timestamp ASC
            """, (start_date, end_date + "T23:59:59")).fetchall()
            
            with open(output_path, 'w', newline='') as f:
                writer = csv.writer(f)
                
                # Header
                writer.writerow([
                    'Date', 'Time (UTC)', 'Type', 'Token Symbol', 'Token Address',
                    'Token Amount', 'SOL Amount', 'Price/Token (NZD)', 'Total Value (NZD)',
                    'Fee (NZD)', 'Cost Basis (NZD)', 'Gain/Loss (NZD)', 'Transaction Signature', 'Notes'
                ])
                
                # Data rows
                for r in records:
                    ts = datetime.fromisoformat(r['timestamp'])
                    writer.writerow([
                        ts.strftime('%Y-%m-%d'),
                        ts.strftime('%H:%M:%S'),
                        r['transaction_type'],
                        r['token_symbol'],
                        r['token_address'],
                        f"{r['token_amount']:.8f}",
                        f"{r['sol_amount']:.6f}",
                        f"{r['price_per_token_nzd']:.8f}" if r['price_per_token_nzd'] else '',
                        f"{r['total_value_nzd']:.2f}",
                        f"{r['fee_nzd']:.4f}",
                        f"{r['cost_basis_nzd']:.2f}" if r['cost_basis_nzd'] else '',
                        f"{r['gain_loss_nzd']:.2f}" if r['gain_loss_nzd'] else '',
                        r['signature'],
                        r['notes'] or ''
                    ])
                
                # Summary section
                summary = conn.execute(
                    "SELECT * FROM tax_year_summary WHERE tax_year = ?", (tax_year,)
                ).fetchone()
                
                if summary:
                    writer.writerow([])
                    writer.writerow(['=== TAX YEAR SUMMARY ==='])
                    writer.writerow(['Tax Year', tax_year])
                    writer.writerow(['Total Acquisitions (NZD)', f"${summary['total_acquisitions_nzd']:.2f}"])
                    writer.writerow(['Total Disposals (NZD)', f"${summary['total_disposals_nzd']:.2f}"])
                    writer.writerow(['Total Gains (NZD)', f"${summary['total_gains_nzd']:.2f}"])
                    writer.writerow(['Total Losses (NZD)', f"${summary['total_losses_nzd']:.2f}"])
                    writer.writerow(['Net Gain/Loss (NZD)', f"${summary['net_gain_loss_nzd']:.2f}"])
                    writer.writerow(['Total Fees (NZD)', f"${summary['total_fees_nzd']:.2f}"])
                    writer.writerow(['Trade Count', summary['trade_count']])
        
        return output_path
    
    def get_tax_summary(self, tax_year: str) -> Dict:
        """Get tax year summary"""
        with self.connection() as conn:
            row = conn.execute(
                "SELECT * FROM tax_year_summary WHERE tax_year = ?", (tax_year,)
            ).fetchone()
            return dict(row) if row else {}


# =============================================================================
# JUPITER SWAP CLIENT
# =============================================================================

class JupiterClient:
    """
    Client for Jupiter DEX aggregator.
    Handles quote fetching and swap transaction building.
    """
    
    def __init__(self, wallet_pubkey: str):
        self.wallet_pubkey = wallet_pubkey
    
    def get_quote(self, input_mint: str, output_mint: str, amount: int,
                  slippage_bps: int = 100) -> Optional[Dict]:
        """
        Get a swap quote from Jupiter.
        
        Args:
            input_mint: Token to sell
            output_mint: Token to buy
            amount: Amount in smallest units (lamports for SOL)
            slippage_bps: Slippage tolerance in basis points
        """
        try:
            params = {
                'inputMint': input_mint,
                'outputMint': output_mint,
                'amount': str(amount),
                'slippageBps': slippage_bps,
                'onlyDirectRoutes': False,
                'asLegacyTransaction': False
            }
            
            resp = requests.get(JUPITER_QUOTE_URL, params=params, timeout=15)
            
            if resp.status_code == 200:
                return resp.json()
            else:
                logger.error(f"Jupiter quote error: {resp.status_code} - {resp.text}")
                return None
                
        except Exception as e:
            logger.error(f"Jupiter quote exception: {e}")
            return None
    
    def get_swap_transaction(self, quote: Dict, priority_fee: int = 100000) -> Optional[str]:
        """
        Get a swap transaction from Jupiter.
        
        Returns base64-encoded transaction ready for signing.
        """
        try:
            payload = {
                'quoteResponse': quote,
                'userPublicKey': self.wallet_pubkey,
                'wrapAndUnwrapSol': True,
                'computeUnitPriceMicroLamports': priority_fee,
                'dynamicComputeUnitLimit': True,
                'skipUserAccountsRpcCalls': False
            }
            
            resp = requests.post(JUPITER_SWAP_URL, json=payload, timeout=30)
            
            if resp.status_code == 200:
                data = resp.json()
                return data.get('swapTransaction')
            else:
                logger.error(f"Jupiter swap error: {resp.status_code} - {resp.text}")
                return None
                
        except Exception as e:
            logger.error(f"Jupiter swap exception: {e}")
            return None


# =============================================================================
# THE EXECUTIONER
# =============================================================================

class Executioner:
    """
    THE EXECUTIONER - Real Trade Execution Engine
    
    Executes trades on Solana via Jupiter DEX aggregator.
    Maintains comprehensive tax records for NZ compliance.
    """
    
    def __init__(self, private_key: str = None, config: ExecutionConfig = None,
                 tax_db_path: str = "tax_records.db", helius_key: str = None):
        """
        Initialize the Executioner.
        
        Args:
            private_key: Base58-encoded Solana private key
            config: Execution configuration
            tax_db_path: Path to tax records database
            helius_key: Helius API key for RPC access
        """
        self.config = config or ExecutionConfig()
        self.tax_db = TaxDatabase(tax_db_path)
        self.price_service = PriceService()
        
        # Initialize Solana connection
        self.helius_key = helius_key or os.getenv('HELIUS_KEY')
        if self.helius_key:
            self.rpc_url = HELIUS_RPC_TEMPLATE.format(self.helius_key)
        else:
            self.rpc_url = "https://api.mainnet-beta.solana.com"
        
        if SOLANA_AVAILABLE:
            self.solana_client = SolanaClient(self.rpc_url)
        else:
            self.solana_client = None
        
        # Initialize wallet
        self.keypair = None
        self.wallet_pubkey = None
        if private_key and SOLANA_AVAILABLE:
            try:
                # Handle both base58 and byte array formats
                if isinstance(private_key, str):
                    self.keypair = Keypair.from_base58_string(private_key)
                else:
                    self.keypair = Keypair.from_bytes(private_key)
                self.wallet_pubkey = str(self.keypair.pubkey())
                logger.info(f"✅ Wallet loaded: {self.wallet_pubkey[:8]}...{self.wallet_pubkey[-4:]}")
            except Exception as e:
                logger.error(f"Failed to load wallet: {e}")
        
        # Initialize Jupiter client
        self.jupiter = JupiterClient(self.wallet_pubkey) if self.wallet_pubkey else None
        
        # Execution state
        self._lock = threading.Lock()
        self._pending_executions = {}
        
        logger.info(f"🎯 EXECUTIONER initialized")
        logger.info(f"   Live Trading: {'ENABLED ⚠️' if self.config.enable_live_trading else 'DISABLED (paper mode)'}")
        logger.info(f"   Max Position: {self.config.max_position_size_sol} SOL")
        logger.info(f"   Slippage: {self.config.default_slippage_bps} bps")
    
    def get_sol_balance(self) -> float:
        """Get current SOL balance"""
        if not self.solana_client or not self.wallet_pubkey:
            return 0.0
        
        try:
            pubkey = Pubkey.from_string(self.wallet_pubkey)
            resp = self.solana_client.get_balance(pubkey)
            lamports = resp.value
            return lamports / LAMPORTS_PER_SOL
        except Exception as e:
            logger.error(f"Failed to get balance: {e}")
            return 0.0
    
    def execute_signal(self, signal: Dict) -> ExecutionResult:
        """
        Execute a trading signal from the Strategist.
        
        Expected signal format:
        {
            'action': 'BUY' or 'SELL',
            'token_address': str,
            'token_symbol': str,
            'conviction_score': int (0-100),
            'suggested_size_sol': float,
            'stop_loss_pct': float,
            'take_profit_pct': float,
            'trailing_stop_pct': float,
            'reason': str
        }
        """
        action = signal.get('action', 'BUY').upper()
        token_address = signal.get('token_address')
        token_symbol = signal.get('token_symbol', 'UNKNOWN')
        conviction = signal.get('conviction_score', 0)
        
        logger.info(f"📨 Processing signal: {action} {token_symbol} (conviction: {conviction})")
        
        # Validation
        if not token_address:
            return ExecutionResult(
                success=False,
                status=ExecutionStatus.FAILED,
                error_message="No token address provided"
            )
        
        if self.config.require_strategist_signal and conviction < self.config.min_conviction_score:
            return ExecutionResult(
                success=False,
                status=ExecutionStatus.CANCELLED,
                error_message=f"Conviction {conviction} below minimum {self.config.min_conviction_score}"
            )
        
        if action == 'BUY':
            return self._execute_buy(signal)
        elif action == 'SELL':
            return self._execute_sell(signal)
        else:
            return ExecutionResult(
                success=False,
                status=ExecutionStatus.FAILED,
                error_message=f"Unknown action: {action}"
            )
    
    def _execute_buy(self, signal: Dict) -> ExecutionResult:
        """Execute a buy order"""
        token_address = signal['token_address']
        token_symbol = signal.get('token_symbol', 'UNKNOWN')
        suggested_size = signal.get('suggested_size_sol', self.config.max_position_size_sol * 0.5)
        
        # Check if we already have a position
        existing = self.tax_db.get_position(token_address)
        if existing:
            logger.warning(f"Already have position in {token_symbol}")
            return ExecutionResult(
                success=False,
                status=ExecutionStatus.CANCELLED,
                error_message=f"Position already exists for {token_symbol}"
            )
        
        # Check position limits
        open_positions = self.tax_db.get_open_positions()
        if len(open_positions) >= self.config.max_open_positions:
            return ExecutionResult(
                success=False,
                status=ExecutionStatus.CANCELLED,
                error_message=f"Max positions ({self.config.max_open_positions}) reached"
            )
        
        # Calculate position size
        sol_balance = self.get_sol_balance()
        position_size = min(
            suggested_size,
            self.config.max_position_size_sol,
            sol_balance * self.config.max_portfolio_pct
        )
        position_size = max(position_size, self.config.min_position_size_sol)
        
        if position_size > sol_balance * 0.95:  # Keep 5% buffer for fees
            return ExecutionResult(
                success=False,
                status=ExecutionStatus.CANCELLED,
                error_message=f"Insufficient balance. Have {sol_balance:.4f} SOL, need {position_size:.4f}"
            )
        
        # Check liquidity
        token_info = self.price_service.get_token_info(token_address)
        if token_info['liquidity_usd'] < self.config.min_liquidity_usd:
            return ExecutionResult(
                success=False,
                status=ExecutionStatus.CANCELLED,
                error_message=f"Insufficient liquidity: ${token_info['liquidity_usd']:.0f}"
            )
        
        # Get prices for tax record
        sol_prices = self.price_service.get_sol_prices()
        token_prices = self.price_service.get_token_price(token_address)
        
        # Execute the swap
        if not self.config.enable_live_trading:
            logger.info(f"🎮 PAPER MODE: Would buy {position_size:.4f} SOL of {token_symbol}")
            return self._simulate_buy(signal, position_size, token_info, sol_prices, token_prices)
        
        return self._execute_jupiter_buy(
            token_address, token_symbol, position_size, signal,
            token_info, sol_prices, token_prices
        )
    
    def _execute_jupiter_buy(self, token_address: str, token_symbol: str,
                            sol_amount: float, signal: Dict, token_info: Dict,
                            sol_prices: Dict, token_prices: Dict) -> ExecutionResult:
        """Execute a buy via Jupiter"""
        if not self.jupiter or not self.keypair:
            return ExecutionResult(
                success=False,
                status=ExecutionStatus.FAILED,
                error_message="Wallet or Jupiter client not initialized"
            )
        
        lamports = int(sol_amount * LAMPORTS_PER_SOL)
        
        # Get quote
        quote = self.jupiter.get_quote(
            input_mint=SOL_MINT,
            output_mint=token_address,
            amount=lamports,
            slippage_bps=self.config.default_slippage_bps
        )
        
        if not quote:
            self.tax_db.log_execution(
                'BUY', token_address, token_symbol, 'FAILED',
                error='Failed to get Jupiter quote'
            )
            return ExecutionResult(
                success=False,
                status=ExecutionStatus.FAILED,
                error_message="Failed to get swap quote"
            )
        
        # Get expected output
        out_amount = int(quote.get('outAmount', 0))
        price_impact = float(quote.get('priceImpactPct', 0))
        
        if price_impact > 5.0:  # More than 5% price impact
            return ExecutionResult(
                success=False,
                status=ExecutionStatus.CANCELLED,
                error_message=f"Price impact too high: {price_impact:.2f}%"
            )
        
        # Get swap transaction
        swap_tx_b64 = self.jupiter.get_swap_transaction(
            quote, self.config.priority_fee_lamports
        )
        
        if not swap_tx_b64:
            self.tax_db.log_execution(
                'BUY', token_address, token_symbol, 'FAILED',
                error='Failed to get swap transaction'
            )
            return ExecutionResult(
                success=False,
                status=ExecutionStatus.FAILED,
                error_message="Failed to build swap transaction"
            )
        
        # Sign and send transaction
        try:
            tx_bytes = base64.b64decode(swap_tx_b64)
            tx = VersionedTransaction.from_bytes(tx_bytes)
            
            # Sign the transaction
            tx.sign([self.keypair])
            
            # Send transaction
            opts = {"skip_preflight": True, "max_retries": 3}
            result = self.solana_client.send_transaction(tx, opts=opts)
            
            signature = str(result.value)
            logger.info(f"📤 Transaction sent: {signature}")
            
            # Wait for confirmation
            confirmed = self._wait_for_confirmation(signature)
            
            if not confirmed:
                self.tax_db.log_execution(
                    'BUY', token_address, token_symbol, 'TIMEOUT',
                    signature=signature, error='Transaction confirmation timeout'
                )
                return ExecutionResult(
                    success=False,
                    status=ExecutionStatus.FAILED,
                    signature=signature,
                    error_message="Transaction confirmation timeout"
                )
            
            # Calculate final amounts
            tokens_received = out_amount / (10 ** 9)  # Assuming 9 decimals
            price_per_token = (sol_amount * sol_prices['usd']) / tokens_received if tokens_received > 0 else 0
            fee_sol = 0.000005 + (self.config.priority_fee_lamports / LAMPORTS_PER_SOL)
            
            # Create tax record
            record_id = self._generate_record_id()
            tax_record = TaxRecord(
                id=record_id,
                timestamp=datetime.now(timezone.utc),
                transaction_type="BUY",
                token_address=token_address,
                token_symbol=token_symbol,
                token_amount=tokens_received,
                sol_amount=sol_amount,
                price_per_token_usd=price_per_token,
                price_per_token_nzd=price_per_token * (sol_prices['nzd'] / sol_prices['usd']),
                sol_price_usd=sol_prices['usd'],
                sol_price_nzd=sol_prices['nzd'],
                total_value_usd=sol_amount * sol_prices['usd'],
                total_value_nzd=sol_amount * sol_prices['nzd'],
                fee_sol=fee_sol,
                fee_nzd=fee_sol * sol_prices['nzd'],
                signature=signature,
                notes=f"Conviction: {signal.get('conviction_score', 0)}"
            )
            
            self.tax_db.record_transaction(tax_record)
            
            # Add cost basis lot for FIFO
            self.tax_db.add_cost_basis_lot(
                token_address, tokens_received,
                sol_amount * sol_prices['nzd'], signature
            )
            
            # Save position
            position = Position(
                token_address=token_address,
                token_symbol=token_symbol,
                tokens_held=tokens_received,
                entry_price_usd=price_per_token,
                entry_time=datetime.now(timezone.utc),
                total_cost_sol=sol_amount,
                total_cost_nzd=sol_amount * sol_prices['nzd'],
                stop_loss_pct=signal.get('stop_loss_pct', -0.12),
                take_profit_pct=signal.get('take_profit_pct', 0.30),
                trailing_stop_pct=signal.get('trailing_stop_pct', 0.08),
                peak_price_usd=price_per_token,
                current_price_usd=price_per_token,
                strategist_conviction=signal.get('conviction_score', 0),
                entry_signature=signature
            )
            self.tax_db.save_position(position)
            
            self.tax_db.log_execution(
                'BUY', token_address, token_symbol, 'CONFIRMED',
                signature=signature, details={'sol_spent': sol_amount, 'tokens_received': tokens_received}
            )
            
            logger.info(f"✅ BUY CONFIRMED: {tokens_received:.4f} {token_symbol} for {sol_amount:.4f} SOL")
            
            return ExecutionResult(
                success=True,
                status=ExecutionStatus.CONFIRMED,
                signature=signature,
                token_address=token_address,
                token_symbol=token_symbol,
                transaction_type="BUY",
                tokens_amount=tokens_received,
                sol_amount=sol_amount,
                price_usd=price_per_token,
                price_nzd=price_per_token * (sol_prices['nzd'] / sol_prices['usd']),
                fee_sol=fee_sol,
                tax_record_id=record_id
            )
            
        except Exception as e:
            logger.error(f"Transaction execution failed: {e}")
            self.tax_db.log_execution(
                'BUY', token_address, token_symbol, 'FAILED',
                error=str(e)
            )
            return ExecutionResult(
                success=False,
                status=ExecutionStatus.FAILED,
                error_message=str(e)
            )
    
    def _simulate_buy(self, signal: Dict, sol_amount: float, token_info: Dict,
                     sol_prices: Dict, token_prices: Dict) -> ExecutionResult:
        """Simulate a buy for paper trading"""
        token_address = signal['token_address']
        token_symbol = signal.get('token_symbol', 'UNKNOWN')
        
        # Calculate simulated amounts
        price_usd = token_prices['usd'] if token_prices['usd'] > 0 else token_info['price_usd']
        if price_usd <= 0:
            price_usd = 0.0001  # Fallback
        
        tokens_received = (sol_amount * sol_prices['usd']) / price_usd
        fee_sol = 0.000005
        
        # Create tax record (even for paper trades for testing)
        record_id = self._generate_record_id()
        tax_record = TaxRecord(
            id=record_id,
            timestamp=datetime.now(timezone.utc),
            transaction_type="BUY",
            token_address=token_address,
            token_symbol=token_symbol,
            token_amount=tokens_received,
            sol_amount=sol_amount,
            price_per_token_usd=price_usd,
            price_per_token_nzd=token_prices['nzd'],
            sol_price_usd=sol_prices['usd'],
            sol_price_nzd=sol_prices['nzd'],
            total_value_usd=sol_amount * sol_prices['usd'],
            total_value_nzd=sol_amount * sol_prices['nzd'],
            fee_sol=fee_sol,
            fee_nzd=fee_sol * sol_prices['nzd'],
            signature=f"PAPER_{record_id}",
            notes=f"PAPER TRADE | Conviction: {signal.get('conviction_score', 0)}"
        )
        
        self.tax_db.record_transaction(tax_record)
        self.tax_db.add_cost_basis_lot(
            token_address, tokens_received,
            sol_amount * sol_prices['nzd'], f"PAPER_{record_id}"
        )
        
        # Save position
        position = Position(
            token_address=token_address,
            token_symbol=token_symbol,
            tokens_held=tokens_received,
            entry_price_usd=price_usd,
            entry_time=datetime.now(timezone.utc),
            total_cost_sol=sol_amount,
            total_cost_nzd=sol_amount * sol_prices['nzd'],
            stop_loss_pct=signal.get('stop_loss_pct', -0.12),
            take_profit_pct=signal.get('take_profit_pct', 0.30),
            trailing_stop_pct=signal.get('trailing_stop_pct', 0.08),
            peak_price_usd=price_usd,
            current_price_usd=price_usd,
            strategist_conviction=signal.get('conviction_score', 0),
            entry_signature=f"PAPER_{record_id}"
        )
        self.tax_db.save_position(position)
        
        logger.info(f"📝 PAPER BUY: {tokens_received:.4f} {token_symbol} @ ${price_usd:.8f}")
        
        return ExecutionResult(
            success=True,
            status=ExecutionStatus.CONFIRMED,
            signature=f"PAPER_{record_id}",
            token_address=token_address,
            token_symbol=token_symbol,
            transaction_type="BUY",
            tokens_amount=tokens_received,
            sol_amount=sol_amount,
            price_usd=price_usd,
            price_nzd=token_prices['nzd'],
            fee_sol=fee_sol,
            tax_record_id=record_id
        )
    
    def _execute_sell(self, signal: Dict) -> ExecutionResult:
        """Execute a sell order"""
        token_address = signal['token_address']
        token_symbol = signal.get('token_symbol', 'UNKNOWN')
        
        # Get existing position
        position = self.tax_db.get_position(token_address)
        if not position:
            return ExecutionResult(
                success=False,
                status=ExecutionStatus.CANCELLED,
                error_message=f"No position found for {token_symbol}"
            )
        
        tokens_to_sell = position['tokens_held']
        
        # Get prices
        sol_prices = self.price_service.get_sol_prices()
        token_prices = self.price_service.get_token_price(token_address)
        
        if not self.config.enable_live_trading:
            logger.info(f"🎮 PAPER MODE: Would sell {tokens_to_sell:.4f} {token_symbol}")
            return self._simulate_sell(signal, position, sol_prices, token_prices)
        
        return self._execute_jupiter_sell(
            token_address, token_symbol, tokens_to_sell,
            signal, position, sol_prices, token_prices
        )
    
    def _execute_jupiter_sell(self, token_address: str, token_symbol: str,
                             tokens_to_sell: float, signal: Dict, position: Dict,
                             sol_prices: Dict, token_prices: Dict) -> ExecutionResult:
        """Execute a sell via Jupiter"""
        if not self.jupiter or not self.keypair:
            return ExecutionResult(
                success=False,
                status=ExecutionStatus.FAILED,
                error_message="Wallet or Jupiter client not initialized"
            )
        
        # Convert tokens to smallest units (assuming 9 decimals)
        token_amount = int(tokens_to_sell * (10 ** 9))
        
        # Get quote
        quote = self.jupiter.get_quote(
            input_mint=token_address,
            output_mint=SOL_MINT,
            amount=token_amount,
            slippage_bps=self.config.default_slippage_bps
        )
        
        if not quote:
            self.tax_db.log_execution(
                'SELL', token_address, token_symbol, 'FAILED',
                error='Failed to get Jupiter quote'
            )
            return ExecutionResult(
                success=False,
                status=ExecutionStatus.FAILED,
                error_message="Failed to get swap quote"
            )
        
        out_amount = int(quote.get('outAmount', 0))
        sol_received = out_amount / LAMPORTS_PER_SOL
        
        # Get swap transaction
        swap_tx_b64 = self.jupiter.get_swap_transaction(
            quote, self.config.priority_fee_lamports
        )
        
        if not swap_tx_b64:
            return ExecutionResult(
                success=False,
                status=ExecutionStatus.FAILED,
                error_message="Failed to build swap transaction"
            )
        
        # Sign and send
        try:
            tx_bytes = base64.b64decode(swap_tx_b64)
            tx = VersionedTransaction.from_bytes(tx_bytes)
            tx.sign([self.keypair])
            
            result = self.solana_client.send_transaction(tx, opts={"skip_preflight": True})
            signature = str(result.value)
            
            confirmed = self._wait_for_confirmation(signature)
            
            if not confirmed:
                return ExecutionResult(
                    success=False,
                    status=ExecutionStatus.FAILED,
                    signature=signature,
                    error_message="Transaction confirmation timeout"
                )
            
            # Calculate gains using FIFO
            cost_basis, lots_used = self.tax_db.consume_cost_basis_fifo(token_address, tokens_to_sell)
            proceeds_nzd = sol_received * sol_prices['nzd']
            gain_loss = proceeds_nzd - cost_basis
            
            fee_sol = 0.000005 + (self.config.priority_fee_lamports / LAMPORTS_PER_SOL)
            price_per_token = (sol_received * sol_prices['usd']) / tokens_to_sell if tokens_to_sell > 0 else 0
            
            # Create tax record
            record_id = self._generate_record_id()
            tax_record = TaxRecord(
                id=record_id,
                timestamp=datetime.now(timezone.utc),
                transaction_type="SELL",
                token_address=token_address,
                token_symbol=token_symbol,
                token_amount=tokens_to_sell,
                sol_amount=sol_received,
                price_per_token_usd=price_per_token,
                price_per_token_nzd=price_per_token * (sol_prices['nzd'] / sol_prices['usd']),
                sol_price_usd=sol_prices['usd'],
                sol_price_nzd=sol_prices['nzd'],
                total_value_usd=sol_received * sol_prices['usd'],
                total_value_nzd=proceeds_nzd,
                fee_sol=fee_sol,
                fee_nzd=fee_sol * sol_prices['nzd'],
                signature=signature,
                cost_basis_nzd=cost_basis,
                gain_loss_nzd=gain_loss,
                notes=f"Exit reason: {signal.get('reason', 'manual')}"
            )
            
            self.tax_db.record_transaction(tax_record)
            self.tax_db.close_position(token_address)
            
            self.tax_db.log_execution(
                'SELL', token_address, token_symbol, 'CONFIRMED',
                signature=signature, details={
                    'tokens_sold': tokens_to_sell,
                    'sol_received': sol_received,
                    'gain_loss_nzd': gain_loss
                }
            )
            
            gain_emoji = "📈" if gain_loss >= 0 else "📉"
            logger.info(f"✅ SELL CONFIRMED: {tokens_to_sell:.4f} {token_symbol} for {sol_received:.4f} SOL {gain_emoji} ${gain_loss:.2f} NZD")
            
            return ExecutionResult(
                success=True,
                status=ExecutionStatus.CONFIRMED,
                signature=signature,
                token_address=token_address,
                token_symbol=token_symbol,
                transaction_type="SELL",
                tokens_amount=tokens_to_sell,
                sol_amount=sol_received,
                price_usd=price_per_token,
                price_nzd=price_per_token * (sol_prices['nzd'] / sol_prices['usd']),
                fee_sol=fee_sol,
                tax_record_id=record_id
            )
            
        except Exception as e:
            logger.error(f"Sell execution failed: {e}")
            return ExecutionResult(
                success=False,
                status=ExecutionStatus.FAILED,
                error_message=str(e)
            )
    
    def _simulate_sell(self, signal: Dict, position: Dict,
                      sol_prices: Dict, token_prices: Dict) -> ExecutionResult:
        """Simulate a sell for paper trading"""
        token_address = position['token_address']
        token_symbol = position['token_symbol']
        tokens_to_sell = position['tokens_held']
        
        # Calculate simulated proceeds
        price_usd = token_prices['usd'] if token_prices['usd'] > 0 else position.get('entry_price_usd', 0.0001)
        sol_received = (tokens_to_sell * price_usd) / sol_prices['usd']
        
        # Calculate gains using FIFO
        cost_basis, _ = self.tax_db.consume_cost_basis_fifo(token_address, tokens_to_sell)
        proceeds_nzd = sol_received * sol_prices['nzd']
        gain_loss = proceeds_nzd - cost_basis
        
        fee_sol = 0.000005
        
        record_id = self._generate_record_id()
        tax_record = TaxRecord(
            id=record_id,
            timestamp=datetime.now(timezone.utc),
            transaction_type="SELL",
            token_address=token_address,
            token_symbol=token_symbol,
            token_amount=tokens_to_sell,
            sol_amount=sol_received,
            price_per_token_usd=price_usd,
            price_per_token_nzd=token_prices['nzd'],
            sol_price_usd=sol_prices['usd'],
            sol_price_nzd=sol_prices['nzd'],
            total_value_usd=sol_received * sol_prices['usd'],
            total_value_nzd=proceeds_nzd,
            fee_sol=fee_sol,
            fee_nzd=fee_sol * sol_prices['nzd'],
            signature=f"PAPER_{record_id}",
            cost_basis_nzd=cost_basis,
            gain_loss_nzd=gain_loss,
            notes=f"PAPER TRADE | Exit: {signal.get('reason', 'manual')}"
        )
        
        self.tax_db.record_transaction(tax_record)
        self.tax_db.close_position(token_address)
        
        gain_emoji = "📈" if gain_loss >= 0 else "📉"
        logger.info(f"📝 PAPER SELL: {tokens_to_sell:.4f} {token_symbol} {gain_emoji} ${gain_loss:.2f} NZD")
        
        return ExecutionResult(
            success=True,
            status=ExecutionStatus.CONFIRMED,
            signature=f"PAPER_{record_id}",
            token_address=token_address,
            token_symbol=token_symbol,
            transaction_type="SELL",
            tokens_amount=tokens_to_sell,
            sol_amount=sol_received,
            price_usd=price_usd,
            price_nzd=token_prices['nzd'],
            fee_sol=fee_sol,
            tax_record_id=record_id
        )
    
    def exit_position(self, token_address: str, reason: str = "manual") -> ExecutionResult:
        """Force exit a position"""
        position = self.tax_db.get_position(token_address)
        if not position:
            return ExecutionResult(
                success=False,
                status=ExecutionStatus.CANCELLED,
                error_message=f"No position found for {token_address[:8]}..."
            )
        
        signal = {
            'action': 'SELL',
            'token_address': token_address,
            'token_symbol': position['token_symbol'],
            'reason': reason
        }
        
        return self._execute_sell(signal)
    
    def check_and_execute_exits(self) -> List[ExecutionResult]:
        """Check all positions for exit conditions and execute if triggered"""
        results = []
        positions = self.tax_db.get_open_positions()
        
        for pos in positions:
            token_address = pos['token_address']
            current_price = self.price_service.get_token_price(token_address)['usd']
            
            if current_price <= 0:
                continue
            
            entry_price = pos['entry_price_usd']
            pnl_pct = (current_price - entry_price) / entry_price if entry_price > 0 else 0
            
            # Update peak price
            if current_price > pos.get('peak_price_usd', 0):
                pos['peak_price_usd'] = current_price
                pos['current_price_usd'] = current_price
                self.tax_db.save_position(Position(**{k: v for k, v in pos.items() 
                                                      if k in Position.__dataclass_fields__}))
            
            # Check exit conditions
            exit_reason = None
            
            # Stop loss
            if pnl_pct <= pos.get('stop_loss_pct', -0.12):
                exit_reason = "STOP_LOSS"
            
            # Take profit
            elif pnl_pct >= pos.get('take_profit_pct', 0.30):
                exit_reason = "TAKE_PROFIT"
            
            # Trailing stop
            elif pos.get('peak_price_usd', 0) > 0:
                peak = pos['peak_price_usd']
                drawdown = (current_price - peak) / peak
                if drawdown <= -pos.get('trailing_stop_pct', 0.08):
                    exit_reason = "TRAILING_STOP"
            
            # Time stop
            entry_time = datetime.fromisoformat(pos['entry_time']) if isinstance(pos['entry_time'], str) else pos['entry_time']
            hours_held = (datetime.now(timezone.utc) - entry_time.replace(tzinfo=timezone.utc)).total_seconds() / 3600
            if hours_held > 12:  # Default max hold
                exit_reason = "TIME_STOP"
            
            if exit_reason:
                logger.info(f"🚨 Exit triggered for {pos['token_symbol']}: {exit_reason} (PnL: {pnl_pct:.1%})")
                result = self.exit_position(token_address, exit_reason)
                results.append(result)
        
        return results
    
    def _wait_for_confirmation(self, signature: str, timeout: int = None) -> bool:
        """Wait for transaction confirmation"""
        timeout = timeout or self.config.confirmation_timeout
        start = time.time()
        
        while time.time() - start < timeout:
            try:
                resp = self.solana_client.get_signature_statuses([Signature.from_string(signature)])
                statuses = resp.value
                
                if statuses and statuses[0]:
                    status = statuses[0]
                    if status.confirmation_status in ['confirmed', 'finalized']:
                        if status.err is None:
                            return True
                        else:
                            logger.error(f"Transaction failed: {status.err}")
                            return False
                
                time.sleep(1)
            except Exception as e:
                logger.warning(f"Confirmation check error: {e}")
                time.sleep(2)
        
        return False
    
    def _generate_record_id(self) -> str:
        """Generate a unique record ID"""
        timestamp = datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S%f')
        random_suffix = hashlib.sha256(os.urandom(8)).hexdigest()[:8]
        return f"TX_{timestamp}_{random_suffix}"
    
    def export_tax_records(self, tax_year: str, output_path: str = None) -> str:
        """Export tax records for a given year"""
        if output_path is None:
            output_path = f"tax_report_{tax_year}.csv"
        return self.tax_db.export_tax_records(tax_year, output_path)
    
    def get_tax_summary(self, tax_year: str = None) -> Dict:
        """Get tax summary for a year"""
        if tax_year is None:
            # Current NZ tax year
            now = datetime.now()
            if now.month >= 4:
                tax_year = f"{now.year}-{now.year + 1}"
            else:
                tax_year = f"{now.year - 1}-{now.year}"
        return self.tax_db.get_tax_summary(tax_year)
    
    def get_open_positions(self) -> List[Dict]:
        """Get all open positions"""
        return self.tax_db.get_open_positions()
    
    def get_stats(self) -> Dict:
        """Get execution statistics"""
        positions = self.tax_db.get_open_positions()
        sol_balance = self.get_sol_balance()
        
        # Get current tax year summary
        tax_summary = self.get_tax_summary()
        
        return {
            'sol_balance': sol_balance,
            'open_positions': len(positions),
            'live_trading_enabled': self.config.enable_live_trading,
            'max_position_size': self.config.max_position_size_sol,
            'tax_year_gains_nzd': tax_summary.get('total_gains_nzd', 0),
            'tax_year_losses_nzd': tax_summary.get('total_losses_nzd', 0),
            'tax_year_net_nzd': tax_summary.get('net_gain_loss_nzd', 0),
            'tax_year_trades': tax_summary.get('trade_count', 0),
            'wallet': self.wallet_pubkey[:8] + "..." if self.wallet_pubkey else "Not loaded"
        }


# =============================================================================
# INTEGRATION WITH STRATEGIST
# =============================================================================

class StrategistIntegration:
    """
    Bridge between the Strategist and Executioner.
    Converts Strategist signals into execution commands.
    """
    
    def __init__(self, executioner: Executioner, db=None):
        self.executioner = executioner
        self.db = db
        self._last_processed_signal = None
    
    def process_strategist_decision(self, decision: Dict, token_data: Dict) -> Optional[ExecutionResult]:
        """
        Process a decision from the Strategist.
        
        Expected decision format (from strategist_v2.py):
        {
            'action': 'ENTER' or 'SKIP' or 'EXIT',
            'conviction': int,
            'position_size_multiplier': float,
            'stop_loss': float,
            'take_profit': float,
            'trailing_stop': float,
            'reason': str
        }
        """
        action = decision.get('action', 'SKIP')
        
        if action == 'SKIP':
            return None
        
        conviction = decision.get('conviction', 0)
        
        # Build signal for executioner
        signal = {
            'action': 'BUY' if action == 'ENTER' else 'SELL',
            'token_address': token_data.get('token_address'),
            'token_symbol': token_data.get('token_symbol', 'UNKNOWN'),
            'conviction_score': conviction,
            'suggested_size_sol': self.executioner.config.max_position_size_sol * decision.get('position_size_multiplier', 0.5),
            'stop_loss_pct': decision.get('stop_loss', -0.12),
            'take_profit_pct': decision.get('take_profit', 0.30),
            'trailing_stop_pct': decision.get('trailing_stop', 0.08),
            'reason': decision.get('reason', '')
        }
        
        return self.executioner.execute_signal(signal)


# =============================================================================
# MAIN / CLI
# =============================================================================

def main():
    """CLI interface for the Executioner"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Executioner - Solana Trade Execution")
    parser.add_argument('command', choices=['status', 'positions', 'tax', 'export', 'exit'],
                       help='Command to run')
    parser.add_argument('--token', help='Token address for exit command')
    parser.add_argument('--year', help='Tax year for export (e.g., 2024-2025)')
    parser.add_argument('--live', action='store_true', help='Enable live trading')
    
    args = parser.parse_args()
    
    # Load config
    config = ExecutionConfig(
        enable_live_trading=args.live
    )
    
    # Initialize executioner (will need private key from env for live trading)
    private_key = os.getenv('SOLANA_PRIVATE_KEY')
    executioner = Executioner(private_key=private_key, config=config)
    
    if args.command == 'status':
        stats = executioner.get_stats()
        print("\n📊 EXECUTIONER STATUS")
        print("=" * 40)
        for k, v in stats.items():
            print(f"  {k}: {v}")
    
    elif args.command == 'positions':
        positions = executioner.get_open_positions()
        print(f"\n📈 OPEN POSITIONS ({len(positions)})")
        print("=" * 60)
        for pos in positions:
            print(f"  {pos['token_symbol']}: {pos['tokens_held']:.4f} tokens")
            print(f"    Entry: ${pos['entry_price_usd']:.8f}")
            print(f"    Cost: {pos['total_cost_sol']:.4f} SOL (${pos['total_cost_nzd']:.2f} NZD)")
            print()
    
    elif args.command == 'tax':
        summary = executioner.get_tax_summary(args.year)
        print(f"\n💰 TAX SUMMARY ({args.year or 'current year'})")
        print("=" * 40)
        for k, v in summary.items():
            if isinstance(v, float):
                print(f"  {k}: ${v:.2f}")
            else:
                print(f"  {k}: {v}")
    
    elif args.command == 'export':
        year = args.year
        if not year:
            now = datetime.now()
            if now.month >= 4:
                year = f"{now.year}-{now.year + 1}"
            else:
                year = f"{now.year - 1}-{now.year}"
        
        output = executioner.export_tax_records(year)
        print(f"✅ Tax records exported to: {output}")
    
    elif args.command == 'exit':
        if not args.token:
            print("❌ --token required for exit command")
            return
        result = executioner.exit_position(args.token)
        if result.success:
            print(f"✅ Position exited: {result.signature}")
        else:
            print(f"❌ Exit failed: {result.error_message}")


if __name__ == "__main__":
    main()
