#!/usr/bin/env python3
"""
TOKEN SWAP TO SOL UTILITY
=========================

Swaps SPL tokens back to SOL using Jupiter API and Helius Sender.
Designed to integrate with the Meme_coin_trader_v3 bot.

This utility:
- Uses the same transaction execution logic as live_trading_engine.py
- Properly records transactions in live_trades_tax.db
- Uses Helius Sender for ultra-low latency execution
- Supports dynamic Jito tips

USAGE:
    # Swap 10,000 SHARK to SOL
    python token_swap_to_sol.py --token SHARK --amount 10000
    
    # Swap using token mint address directly
    python token_swap_to_sol.py --mint SHARKSYJjqaNyxVfrpnBN9pjgkhwDhatnMyicWPnr1s --amount 10000
    
    # Dry run (simulation only)
    python token_swap_to_sol.py --token SHARK --amount 10000 --dry-run

Author: Meme_coin_trader_v3 System
"""

import os
import sys
import json
import time
import base64
import hashlib
import sqlite3
import logging
import argparse
import requests
from datetime import datetime, timezone
from typing import Dict, Optional, Tuple, Any
from contextlib import contextmanager
from dataclasses import dataclass

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("TokenSwap")

# =============================================================================
# CONSTANTS
# =============================================================================

SOL_MINT = "So11111111111111111111111111111111111111112"
LAMPORTS_PER_SOL = 1_000_000_000

# Known token mint addresses
# These are the tokens in your wallet based on spl-token accounts output
TOKEN_MINTS = {
    # YOUR WALLET TOKENS (from spl-token accounts)
    'TOKEN1': '14DieZPb3JAQwAopBZF5GLGeykjT7A3pjee2muexpump',  # 179,194 tokens
    'TOKEN2': '63Z3Q7JX3SBGDiiwqqnPTVvHcuUk6ixkzsYQbKzhpump',  # 23,843 tokens
    'TOKEN3': '8ryRQD6jWfxnSdvvVbQ8Tzwo1NgGP7w1X1nQPpb4pump',  # 154,223 tokens
    'TOKEN4': 'AtdqW9HYpx6bzuXAyuVRz6a3UiTHLoNTRJDN8buXHem7',  # 555,429 tokens
    'TOKEN5': 'FMMqsxKuP7PiCYG7dRhedWoNFfFGfUpXKRGZ5Fvgscaa',  # 49 tokens
    
    # Common tokens for reference
    'BONK': 'DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263',
    'WIF': 'EKpQGSJtjMFqKZ9KQanSqYXRcF8fBopzLHYxdM65zcjm',
    'USDC': 'EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v',
}

# Helius Sender tip accounts
TIP_ACCOUNTS = [
    "4ACfpUFoaSD9bfPdeu6DBt89gB6ENTeHBXCAi87NhDEE",
    "D2L6yPZ2FmmmTKPgzaMKdhu6EWZcTpLy1Vhx8uvZe7NZ",
    "9bnz4RShgq1hAnLnZbP8kbgBg1kEmcJBYQq3gQbmnSta",
    "5VY91ws6B2hMmBFRsXkoAAdsPHBJwRfBht4DXox3xkwn",
    "2nyhqdwKcJZR2vcqCyrYsaPVdAnFoJjiksCXJ7hfEYgD",
]

# Jupiter API endpoints
JUPITER_QUOTE_URL = "https://quote-api.jup.ag/v6/quote"
JUPITER_SWAP_URL = "https://quote-api.jup.ag/v6/swap"

# CoinGecko for prices
COINGECKO_URL = "https://api.coingecko.com/api/v3/simple/price"

# Default configuration
DEFAULT_SLIPPAGE_BPS = 100  # 1%
DEFAULT_PRIORITY_FEE = 50000  # 50k microlamports
MIN_TIP_SOL = 0.0002  # Minimum Helius Sender tip

# =============================================================================
# TRY TO IMPORT SOLANA LIBRARIES
# =============================================================================

try:
    from solana.rpc.api import Client as SolanaClient
    from solders.keypair import Keypair
    from solders.pubkey import Pubkey
    from solders.transaction import VersionedTransaction, Transaction
    from solders.message import Message
    from solders.system_program import TransferParams, transfer
    from solders.compute_budget import set_compute_unit_limit, set_compute_unit_price
    SOLANA_AVAILABLE = True
except ImportError:
    SOLANA_AVAILABLE = False
    logger.warning("⚠️ Solana libraries not installed. Run: pip install solana solders --break-system-packages")

# =============================================================================
# SECRETS MANAGEMENT
# =============================================================================

def get_secret(key: str, default: str = None) -> Optional[str]:
    """Get secret from AWS Secrets Manager or environment"""
    try:
        from core.secrets_manager import get_secret as aws_get_secret
        value = aws_get_secret(key)
        if value:
            return value
    except ImportError:
        pass
    
    # Fallback to environment
    return os.getenv(key, default)


def init_secrets():
    """Initialize secrets from AWS or environment"""
    try:
        from core.secrets_manager import init_secrets as aws_init
        aws_init()
        logger.info("✅ Loaded secrets from AWS Secrets Manager")
    except ImportError:
        try:
            from dotenv import load_dotenv
            load_dotenv()
            logger.info("📁 Loaded secrets from .env file")
        except ImportError:
            logger.info("📁 Using environment variables")


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class SwapConfig:
    """Configuration for token swaps"""
    slippage_bps: int = DEFAULT_SLIPPAGE_BPS
    priority_fee_lamports: int = DEFAULT_PRIORITY_FEE
    min_tip_sol: float = MIN_TIP_SOL
    use_dynamic_tip: bool = True
    max_retries: int = 3
    confirmation_timeout: int = 30
    db_path: str = "live_trades_tax.db"


# =============================================================================
# PRICE SERVICE
# =============================================================================

class PriceService:
    """Get token prices from various sources"""
    
    def __init__(self):
        self._cache = {}
        self._cache_ttl = 30
        self._sol_prices = {'usd': 0, 'nzd': 0}
        self._last_sol_update = None
    
    def get_sol_prices(self) -> Tuple[float, float]:
        """Get SOL price in USD and NZD"""
        now = datetime.now()
        
        if (self._last_sol_update and 
            (now - self._last_sol_update).total_seconds() < 60):
            return self._sol_prices['usd'], self._sol_prices['nzd']
        
        try:
            resp = requests.get(
                f"{COINGECKO_URL}?ids=solana&vs_currencies=usd,nzd",
                timeout=10
            )
            if resp.status_code == 200:
                data = resp.json()
                self._sol_prices['usd'] = data['solana']['usd']
                self._sol_prices['nzd'] = data['solana']['nzd']
                self._last_sol_update = now
        except Exception as e:
            logger.warning(f"CoinGecko error: {e}")
        
        return self._sol_prices['usd'], self._sol_prices['nzd']
    
    def get_token_price(self, token_address: str) -> Optional[float]:
        """Get token price in USD from DexScreener"""
        try:
            url = f"https://api.dexscreener.com/latest/dex/tokens/{token_address}"
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                pairs = data.get('pairs', [])
                if pairs:
                    # Get highest liquidity pair
                    best_pair = max(pairs, key=lambda p: float(p.get('liquidity', {}).get('usd', 0) or 0))
                    return float(best_pair.get('priceUsd', 0) or 0)
        except Exception as e:
            logger.warning(f"DexScreener price error: {e}")
        
        return None
    
    def get_token_info(self, token_address: str) -> Dict:
        """Get token info (symbol, name, price) from DexScreener"""
        try:
            url = f"https://api.dexscreener.com/latest/dex/tokens/{token_address}"
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                pairs = data.get('pairs', [])
                if pairs:
                    best_pair = max(pairs, key=lambda p: float(p.get('liquidity', {}).get('usd', 0) or 0))
                    base_token = best_pair.get('baseToken', {})
                    return {
                        'symbol': base_token.get('symbol', 'UNKNOWN'),
                        'name': base_token.get('name', ''),
                        'price_usd': float(best_pair.get('priceUsd', 0) or 0),
                        'liquidity_usd': float(best_pair.get('liquidity', {}).get('usd', 0) or 0),
                    }
        except Exception as e:
            logger.warning(f"DexScreener info error: {e}")
        
        return {'symbol': 'UNKNOWN', 'name': '', 'price_usd': 0, 'liquidity_usd': 0}


# =============================================================================
# TAX DATABASE
# =============================================================================

class TaxDatabase:
    """
    Database for recording trades with NZ tax compliance.
    
    IMPORTANT: This is the same schema as live_trading_engine.py to ensure
    all transactions are properly tracked.
    """
    
    def __init__(self, db_path: str = "live_trades_tax.db"):
        self.db_path = db_path
        self._init_database()
        logger.info(f"📊 Tax database initialized: {db_path}")
    
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
        """Initialize database tables if they don't exist"""
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
                    is_live INTEGER DEFAULT 1
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
                    is_exhausted INTEGER DEFAULT 0
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
                    entry_signature TEXT,
                    conviction_score INTEGER
                )
            """)
            
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
            
            # Create indexes
            conn.execute("CREATE INDEX IF NOT EXISTS idx_tax_timestamp ON tax_transactions(timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_tax_token ON tax_transactions(token_address)")
    
    def _generate_id(self) -> str:
        """Generate unique transaction ID"""
        timestamp = datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S%f')
        random_part = hashlib.sha256(os.urandom(8)).hexdigest()[:8]
        return f"TX_{timestamp}_{random_part}"
    
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
                trade_data.get('transaction_type', 'SELL'),
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
        
        logger.info(f"📝 Recorded trade: {record_id}")
        return record_id
    
    def get_fifo_cost_basis(self, token_address: str, tokens_to_sell: float) -> Tuple[float, int]:
        """
        Get FIFO cost basis for selling tokens.
        Returns (total_cost_basis_nzd, lots_consumed)
        """
        with self._get_connection() as conn:
            # Get all non-exhausted lots for this token, ordered by date (FIFO)
            lots = conn.execute("""
                SELECT id, tokens_remaining, cost_per_token_nzd, total_cost_nzd
                FROM cost_basis_lots
                WHERE token_address = ? AND is_exhausted = 0
                ORDER BY acquisition_date ASC
            """, (token_address,)).fetchall()
            
            if not lots:
                logger.warning(f"⚠️ No cost basis lots found for {token_address}")
                return 0.0, 0
            
            total_cost = 0.0
            tokens_remaining = tokens_to_sell
            lots_consumed = 0
            
            for lot in lots:
                if tokens_remaining <= 0:
                    break
                
                lot_tokens = lot['tokens_remaining']
                lot_cost_per = lot['cost_per_token_nzd']
                
                if lot_tokens <= tokens_remaining:
                    # Use entire lot
                    total_cost += lot_tokens * lot_cost_per
                    tokens_remaining -= lot_tokens
                    lots_consumed += 1
                    
                    # Mark lot as exhausted
                    conn.execute(
                        "UPDATE cost_basis_lots SET is_exhausted = 1, tokens_remaining = 0 WHERE id = ?",
                        (lot['id'],)
                    )
                else:
                    # Use partial lot
                    total_cost += tokens_remaining * lot_cost_per
                    new_remaining = lot_tokens - tokens_remaining
                    tokens_remaining = 0
                    
                    # Update lot
                    conn.execute(
                        "UPDATE cost_basis_lots SET tokens_remaining = ? WHERE id = ?",
                        (new_remaining, lot['id'])
                    )
            
            return total_cost, lots_consumed
    
    def remove_position(self, token_address: str):
        """Remove a position after selling"""
        with self._get_connection() as conn:
            conn.execute("DELETE FROM live_positions WHERE token_address = ?", (token_address,))
    
    def get_position(self, token_address: str) -> Optional[Dict]:
        """Get a position by token address"""
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM live_positions WHERE token_address = ?",
                (token_address,)
            ).fetchone()
            return dict(row) if row else None
    
    def update_daily_stats(self, pnl_sol: float, is_win: bool, fee_sol: float = 0):
        """Update daily trading statistics"""
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


# =============================================================================
# TOKEN SWAP ENGINE
# =============================================================================

class TokenSwapEngine:
    """
    Engine for swapping tokens back to SOL.
    Uses the same execution logic as live_trading_engine.py.
    """
    
    def __init__(self, config: SwapConfig = None):
        self.config = config or SwapConfig()
        self.price_service = PriceService()
        self.tax_db = TaxDatabase(self.config.db_path)
        
        # Initialize wallet
        self.keypair = None
        self.wallet_pubkey = None
        self.solana_client = None
        
        # Load credentials
        self._load_wallet()
        self._init_rpc()
        
        logger.info("🔄 Token Swap Engine initialized")
    
    def _load_wallet(self):
        """Load wallet from secrets"""
        if not SOLANA_AVAILABLE:
            logger.error("❌ Solana libraries not available")
            return
        
        private_key = get_secret('SOLANA_PRIVATE_KEY') or get_secret('HOT_WALLET_1')
        
        if private_key:
            try:
                if isinstance(private_key, list):
                    self.keypair = Keypair.from_bytes(bytes(private_key))
                elif isinstance(private_key, dict):
                    key_value = private_key.get('private_key') or private_key.get('value')
                    self.keypair = Keypair.from_base58_string(key_value)
                else:
                    self.keypair = Keypair.from_base58_string(private_key)
                
                self.wallet_pubkey = str(self.keypair.pubkey())
                logger.info(f"🔐 Wallet loaded: {self.wallet_pubkey[:8]}...{self.wallet_pubkey[-4:]}")
            except Exception as e:
                logger.error(f"Failed to load wallet: {e}")
        else:
            logger.error("❌ No wallet key found in secrets")
    
    def _init_rpc(self):
        """Initialize Solana RPC client"""
        if not SOLANA_AVAILABLE:
            return
        
        helius_key = get_secret('HELIUS_KEY')
        if helius_key:
            rpc_url = f"https://mainnet.helius-rpc.com/?api-key={helius_key}"
            self.solana_client = SolanaClient(rpc_url)
            logger.info("✅ Helius RPC connected")
        else:
            logger.error("❌ No HELIUS_KEY found")
    
    def get_token_balance(self, token_mint: str) -> Tuple[float, int]:
        """
        Get token balance for the wallet using direct RPC call.
        Returns (balance, decimals)
        """
        if not self.wallet_pubkey:
            return 0, 0
        
        try:
            # Use direct HTTP RPC call for reliable parsing
            helius_key = get_secret('HELIUS_KEY')
            rpc_url = f"https://mainnet.helius-rpc.com/?api-key={helius_key}"
            
            payload = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "getTokenAccountsByOwner",
                "params": [
                    self.wallet_pubkey,
                    {"mint": token_mint},
                    {"encoding": "jsonParsed"}
                ]
            }
            
            resp = requests.post(rpc_url, json=payload, timeout=30)
            data = resp.json()
            
            if 'error' in data:
                logger.error(f"RPC error: {data['error']}")
                return 0, 6
            
            accounts = data.get('result', {}).get('value', [])
            
            if accounts:
                for account in accounts:
                    try:
                        info = account.get('account', {}).get('data', {}).get('parsed', {}).get('info', {})
                        token_amount = info.get('tokenAmount', {})
                        
                        balance = float(token_amount.get('uiAmount') or 0)
                        decimals = int(token_amount.get('decimals', 6))
                        
                        if balance == 0:
                            # Try raw amount
                            raw = token_amount.get('amount', '0')
                            balance = int(raw) / (10 ** decimals) if raw else 0
                        
                        return balance, decimals
                        
                    except Exception as e:
                        logger.debug(f"Error parsing token account: {e}")
                        continue
            
            return 0, 6  # Not found
            
        except Exception as e:
            logger.error(f"Failed to get token balance: {e}")
            return 0, 6
    
    def get_sol_balance(self) -> float:
        """Get SOL balance"""
        if not self.solana_client or not self.wallet_pubkey:
            return 0
        
        try:
            pubkey = Pubkey.from_string(self.wallet_pubkey)
            resp = self.solana_client.get_balance(pubkey)
            return resp.value / LAMPORTS_PER_SOL
        except Exception as e:
            logger.error(f"Failed to get SOL balance: {e}")
            return 0
    
    def get_jupiter_quote(self, input_mint: str, output_mint: str, 
                          amount: int, slippage_bps: int) -> Optional[Dict]:
        """Get a quote from Jupiter"""
        params = {
            'inputMint': input_mint,
            'outputMint': output_mint,
            'amount': str(amount),
            'slippageBps': slippage_bps
        }
        
        try:
            resp = requests.get(JUPITER_QUOTE_URL, params=params, timeout=10)
            if resp.status_code == 200:
                return resp.json()
            else:
                logger.error(f"Jupiter quote error: {resp.status_code} - {resp.text[:200]}")
        except Exception as e:
            logger.error(f"Jupiter quote failed: {e}")
        
        return None
    
    def get_jupiter_swap(self, quote: Dict) -> Optional[str]:
        """Get swap transaction from Jupiter"""
        payload = {
            'quoteResponse': quote,
            'userPublicKey': self.wallet_pubkey,
            'wrapAndUnwrapSol': True,
            'prioritizationFeeLamports': self.config.priority_fee_lamports,
            'asLegacyTransaction': True,  # Required for Helius Sender
            'useSharedAccounts': False
        }
        
        try:
            resp = requests.post(JUPITER_SWAP_URL, json=payload, timeout=30)
            if resp.status_code == 200:
                data = resp.json()
                return data.get('swapTransaction')
            else:
                logger.error(f"Jupiter swap error: {resp.status_code} - {resp.text[:200]}")
        except Exception as e:
            logger.error(f"Jupiter swap failed: {e}")
        
        return None
    
    def get_dynamic_tip(self) -> float:
        """Get dynamic tip amount from Jito"""
        if not self.config.use_dynamic_tip:
            return self.config.min_tip_sol
        
        try:
            resp = requests.get(
                'https://bundles.jito.wtf/api/v1/bundles/tip_floor',
                timeout=5
            )
            if resp.status_code == 200:
                data = resp.json()
                if data and len(data) > 0:
                    tip = data[0].get('landed_tips_75th_percentile', self.config.min_tip_sol)
                    return max(tip, self.config.min_tip_sol)
        except Exception as e:
            logger.warning(f"Failed to get dynamic tip: {e}")
        
        return self.config.min_tip_sol
    
    def _inject_helius_tip(self, tx_bytes: bytes) -> bytes:
        """Inject Helius tip into transaction"""
        import random
        
        try:
            tx = Transaction.from_bytes(tx_bytes)
            message = tx.message
            
            # Get tip amount
            tip_sol = self.get_dynamic_tip()
            tip_lamports = int(tip_sol * LAMPORTS_PER_SOL)
            
            # Random tip account
            tip_account = Pubkey.from_string(random.choice(TIP_ACCOUNTS))
            
            # Create tip instruction
            tip_ix = transfer(TransferParams(
                from_pubkey=self.keypair.pubkey(),
                to_pubkey=tip_account,
                lamports=tip_lamports
            ))
            
            # Get existing instructions
            instructions = []
            for ix in message.instructions:
                program_id = message.account_keys[ix.program_id_index]
                accounts = [message.account_keys[i] for i in ix.accounts]
                instructions.append(type(ix)(
                    program_id=program_id,
                    accounts=accounts,
                    data=ix.data
                ))
            
            # Add tip instruction
            instructions.append(tip_ix)
            
            # Create new message with tip
            new_message = Message.new_with_blockhash(
                instructions,
                self.keypair.pubkey(),
                message.recent_blockhash
            )
            
            new_tx = Transaction.new_unsigned(new_message)
            new_tx.sign([self.keypair], message.recent_blockhash)
            
            logger.info(f"💰 Tip injected: {tip_sol:.4f} SOL")
            return bytes(new_tx)
            
        except Exception as e:
            logger.error(f"Failed to inject tip: {e}")
            return tx_bytes
    
    def execute_via_helius_sender(self, swap_tx_base64: str) -> Optional[str]:
        """Execute transaction via Helius Sender"""
        helius_key = get_secret('HELIUS_KEY')
        if not helius_key:
            logger.error("❌ No HELIUS_KEY for Sender")
            return None
        
        sender_url = f"https://sender.helius-rpc.com/fast?api-key={helius_key}"
        
        try:
            # Decode, inject tip, and re-encode
            tx_bytes = base64.b64decode(swap_tx_base64)
            
            try:
                tx = Transaction.from_bytes(tx_bytes)
                # Sign the transaction
                tx.sign([self.keypair], tx.message.recent_blockhash)
                tx_base64 = base64.b64encode(bytes(tx)).decode('utf-8')
            except Exception as e:
                logger.warning(f"Legacy transaction handling: {e}")
                # If it's already signed or versioned, use as-is
                tx_base64 = swap_tx_base64
            
            # Send via Helius Sender
            payload = {
                "jsonrpc": "2.0",
                "id": str(int(time.time() * 1000)),
                "method": "sendTransaction",
                "params": [
                    tx_base64,
                    {
                        "encoding": "base64",
                        "skipPreflight": True,
                        "maxRetries": 0
                    }
                ]
            }
            
            headers = {
                "Content-Type": "application/json"
            }
            
            resp = requests.post(sender_url, json=payload, headers=headers, timeout=30)
            
            if resp.status_code != 200:
                logger.error(f"Helius Sender HTTP error: {resp.status_code}")
                return None
            
            result = resp.json()
            
            if "error" in result:
                logger.error(f"Helius Sender error: {result['error']}")
                return None
            
            if "result" in result:
                signature = result["result"]
                logger.info(f"📤 Transaction sent via Helius Sender: {signature}")
                return signature
                
        except Exception as e:
            logger.error(f"Helius Sender execution error: {e}")
        
        return None
    
    def wait_for_confirmation(self, signature: str, timeout: int = None) -> bool:
        """Wait for transaction confirmation"""
        if not self.solana_client:
            return False
        
        timeout = timeout or self.config.confirmation_timeout
        start = time.time()
        
        while time.time() - start < timeout:
            try:
                from solders.signature import Signature
                sig = Signature.from_string(signature)
                resp = self.solana_client.get_signature_statuses([sig])
                
                if resp.value and resp.value[0]:
                    status = resp.value[0]
                    if status.confirmation_status:
                        conf_status = str(status.confirmation_status)
                        if 'confirmed' in conf_status.lower() or 'finalized' in conf_status.lower():
                            logger.info(f"✅ Transaction confirmed: {signature[:16]}...")
                            return True
                
                time.sleep(0.5)
                
            except Exception as e:
                logger.warning(f"Confirmation check error: {e}")
                time.sleep(1)
        
        logger.warning(f"⚠️ Confirmation timeout for {signature[:16]}...")
        return False
    
    def swap_token_to_sol(self, token_mint: str, token_symbol: str, 
                          amount: float, dry_run: bool = False) -> Dict:
        """
        Swap tokens to SOL.
        
        Args:
            token_mint: Token mint address
            token_symbol: Token symbol (e.g., 'SHARK')
            amount: Amount of tokens to swap
            dry_run: If True, only simulate (don't execute)
        
        Returns:
            Result dict with success status and details
        """
        result = {
            'success': False,
            'action': 'SWAP_TO_SOL',
            'token_address': token_mint,
            'token_symbol': token_symbol,
            'amount': amount,
            'signature': None,
            'sol_received': 0,
            'error': None,
            'is_live': not dry_run
        }
        
        logger.info(f"\n{'='*60}")
        logger.info(f"🔄 SWAP: {amount:,.0f} {token_symbol} → SOL")
        logger.info(f"{'='*60}")
        
        # Validate prerequisites
        if not self.keypair:
            result['error'] = "No wallet loaded"
            return result
        
        if not self.solana_client:
            result['error'] = "No RPC connection"
            return result
        
        # Check token balance
        balance, decimals = self.get_token_balance(token_mint)
        logger.info(f"💰 Token balance: {balance:,.2f} {token_symbol}")
        
        if balance < amount:
            result['error'] = f"Insufficient balance. Have {balance:,.2f}, need {amount:,.2f}"
            logger.error(result['error'])
            return result
        
        # Get prices
        sol_usd, sol_nzd = self.price_service.get_sol_prices()
        token_price = self.price_service.get_token_price(token_mint)
        
        logger.info(f"💵 SOL price: ${sol_usd:.2f} USD / ${sol_nzd:.2f} NZD")
        if token_price:
            logger.info(f"💵 {token_symbol} price: ${token_price:.10f} USD")
        
        # Convert amount to raw units
        raw_amount = int(amount * (10 ** decimals))
        
        # Get Jupiter quote
        logger.info(f"📊 Getting Jupiter quote...")
        quote = self.get_jupiter_quote(
            input_mint=token_mint,
            output_mint=SOL_MINT,
            amount=raw_amount,
            slippage_bps=self.config.slippage_bps
        )
        
        if not quote:
            result['error'] = "Failed to get Jupiter quote"
            self.tax_db.log_execution('SWAP', token_mint, token_symbol, 'FAILED', error=result['error'])
            return result
        
        # Parse quote
        out_amount = int(quote.get('outAmount', 0))
        sol_received = out_amount / LAMPORTS_PER_SOL
        price_impact = float(quote.get('priceImpactPct', 0))
        
        logger.info(f"📈 Quote: {amount:,.0f} {token_symbol} → {sol_received:.6f} SOL")
        logger.info(f"📉 Price impact: {price_impact:.2f}%")
        
        if price_impact > 5.0:
            logger.warning(f"⚠️ High price impact: {price_impact:.2f}%")
        
        # Calculate values
        total_value_usd = sol_received * sol_usd
        total_value_nzd = sol_received * sol_nzd
        token_price_nzd = token_price * (sol_nzd / sol_usd) if token_price and sol_usd > 0 else 0
        
        if dry_run:
            logger.info(f"\n🎮 DRY RUN - Would receive: {sol_received:.6f} SOL (${total_value_usd:.2f})")
            result['success'] = True
            result['sol_received'] = sol_received
            result['notes'] = "DRY_RUN"
            return result
        
        # Get swap transaction
        logger.info(f"🔧 Building swap transaction...")
        swap_tx = self.get_jupiter_swap(quote)
        
        if not swap_tx:
            result['error'] = "Failed to get swap transaction"
            self.tax_db.log_execution('SWAP', token_mint, token_symbol, 'FAILED', error=result['error'])
            return result
        
        # Execute via Helius Sender
        logger.info(f"🚀 Executing via Helius Sender...")
        signature = self.execute_via_helius_sender(swap_tx)
        
        if not signature:
            result['error'] = "Transaction failed"
            self.tax_db.log_execution('SWAP', token_mint, token_symbol, 'FAILED', error=result['error'])
            return result
        
        result['signature'] = signature
        
        # Wait for confirmation
        confirmed = self.wait_for_confirmation(signature)
        
        if not confirmed:
            result['error'] = "Transaction not confirmed"
            self.tax_db.log_execution('SWAP', token_mint, token_symbol, 'UNCONFIRMED', 
                                      signature=signature, error=result['error'])
            return result
        
        # Get FIFO cost basis
        cost_basis_nzd, lots_used = self.tax_db.get_fifo_cost_basis(token_mint, amount)
        
        # Calculate P&L
        fee_sol = 0.002  # Estimate
        proceeds_nzd = total_value_nzd
        gain_loss_nzd = proceeds_nzd - cost_basis_nzd if cost_basis_nzd > 0 else None
        
        # Record the trade
        trade_data = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'transaction_type': 'SELL',
            'token_address': token_mint,
            'token_symbol': token_symbol,
            'token_amount': amount,
            'sol_amount': sol_received,
            'price_per_token_usd': token_price or 0,
            'price_per_token_nzd': token_price_nzd,
            'sol_price_usd': sol_usd,
            'sol_price_nzd': sol_nzd,
            'total_value_usd': total_value_usd,
            'total_value_nzd': total_value_nzd,
            'fee_sol': fee_sol,
            'fee_nzd': fee_sol * sol_nzd,
            'cost_basis_nzd': cost_basis_nzd if cost_basis_nzd > 0 else None,
            'gain_loss_nzd': gain_loss_nzd,
            'signature': signature,
            'notes': f"Swap to SOL via Jupiter",
            'is_live': True
        }
        
        record_id = self.tax_db.record_trade(trade_data)
        
        # Remove position if exists
        self.tax_db.remove_position(token_mint)
        
        # Update daily stats
        pnl_sol = gain_loss_nzd / sol_nzd if gain_loss_nzd and sol_nzd > 0 else 0
        is_win = pnl_sol > 0
        self.tax_db.update_daily_stats(pnl_sol, is_win, fee_sol)
        
        # Log execution
        self.tax_db.log_execution('SWAP', token_mint, token_symbol, 'SUCCESS',
                                  signature=signature, details={
                                      'sol_received': sol_received,
                                      'cost_basis_nzd': cost_basis_nzd,
                                      'gain_loss_nzd': gain_loss_nzd
                                  })
        
        # Update result
        result['success'] = True
        result['sol_received'] = sol_received
        result['record_id'] = record_id
        
        # Final summary
        logger.info(f"\n{'='*60}")
        logger.info(f"✅ SWAP SUCCESSFUL")
        logger.info(f"{'='*60}")
        logger.info(f"  Sold: {amount:,.0f} {token_symbol}")
        logger.info(f"  Received: {sol_received:.6f} SOL (${total_value_usd:.2f})")
        logger.info(f"  Signature: {signature}")
        if gain_loss_nzd is not None:
            emoji = "📈" if gain_loss_nzd >= 0 else "📉"
            logger.info(f"  P&L: {emoji} ${gain_loss_nzd:.2f} NZD")
        logger.info(f"{'='*60}\n")
        
        return result


# =============================================================================
# CLI
# =============================================================================

def main():
    """CLI interface"""
    parser = argparse.ArgumentParser(
        description="Swap SPL tokens to SOL",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all tokens in your wallet with values
  python token_swap_to_sol.py --list
  
  # Swap using token mint address directly (RECOMMENDED)
  python token_swap_to_sol.py --mint 14DieZPb3JAQwAopBZF5GLGeykjT7A3pjee2muexpump --amount 10000
  
  # Swap all of a token
  python token_swap_to_sol.py --mint 14DieZPb3JAQwAopBZF5GLGeykjT7A3pjee2muexpump --all
  
  # Dry run (simulation only)
  python token_swap_to_sol.py --mint 14DieZPb3JAQwAopBZF5GLGeykjT7A3pjee2muexpump --amount 10000 --dry-run
  
  # Check balance only
  python token_swap_to_sol.py --mint 14DieZPb3JAQwAopBZF5GLGeykjT7A3pjee2muexpump --balance

Your wallet tokens:
  14DieZPb3JAQwAopBZF5GLGeykjT7A3pjee2muexpump  (179,194 tokens)
  63Z3Q7JX3SBGDiiwqqnPTVvHcuUk6ixkzsYQbKzhpump  (23,843 tokens)
  8ryRQD6jWfxnSdvvVbQ8Tzwo1NgGP7w1X1nQPpb4pump  (154,223 tokens)
  AtdqW9HYpx6bzuXAyuVRz6a3UiTHLoNTRJDN8buXHem7  (555,429 tokens)
  FMMqsxKuP7PiCYG7dRhedWoNFfFGfUpXKRGZ5Fvgscaa  (49 tokens)
        """
    )
    
    parser.add_argument('--token', '-t', type=str,
                        help='Token symbol (e.g., TOKEN1, BONK) - see --list')
    parser.add_argument('--mint', '-m', type=str,
                        help='Token mint address (use instead of --token)')
    parser.add_argument('--amount', '-a', type=float,
                        help='Amount of tokens to swap')
    parser.add_argument('--all', action='store_true',
                        help='Swap entire balance of the token')
    parser.add_argument('--slippage', '-s', type=int, default=DEFAULT_SLIPPAGE_BPS,
                        help=f'Slippage in basis points (default: {DEFAULT_SLIPPAGE_BPS})')
    parser.add_argument('--dry-run', action='store_true',
                        help='Simulate swap without executing')
    parser.add_argument('--balance', '-b', action='store_true',
                        help='Check token balance only')
    parser.add_argument('--list', '-l', action='store_true',
                        help='List all tokens in wallet with values')
    parser.add_argument('--debug', action='store_true',
                        help='Show debug info for RPC responses')
    parser.add_argument('--db', type=str, default='live_trades_tax.db',
                        help='Path to tax database')
    
    args = parser.parse_args()
    
    # Initialize secrets
    init_secrets()
    
    # Create config
    config = SwapConfig(
        slippage_bps=args.slippage,
        db_path=args.db
    )
    
    # Initialize engine
    engine = TokenSwapEngine(config)
    
    # List all tokens
    if args.list:
        list_wallet_tokens(engine, debug=getattr(args, 'debug', False))
        return
    
    # Resolve token mint address
    if args.mint:
        token_mint = args.mint
        # Try to get symbol from DexScreener
        info = engine.price_service.get_token_info(token_mint)
        token_symbol = info.get('symbol', 'UNKNOWN')
    elif args.token:
        token_symbol = args.token.upper()
        token_mint = TOKEN_MINTS.get(token_symbol)
        if not token_mint:
            logger.error(f"❌ Unknown token: {token_symbol}")
            logger.info(f"Known tokens: {', '.join(TOKEN_MINTS.keys())}")
            logger.info("Use --mint to specify the token address directly")
            logger.info("Use --list to see all tokens in your wallet")
            sys.exit(1)
    else:
        parser.print_help()
        sys.exit(1)
    
    # Check balance
    balance, decimals = engine.get_token_balance(token_mint)
    sol_balance = engine.get_sol_balance()
    
    print(f"\n💰 Wallet: {engine.wallet_pubkey}")
    print(f"   SOL: {sol_balance:.6f}")
    print(f"   {token_symbol}: {balance:,.2f}")
    print(f"   Token mint: {token_mint}")
    
    if args.balance:
        return
    
    # Determine amount
    if args.all:
        amount = balance
        if amount <= 0:
            logger.error(f"❌ No {token_symbol} balance to swap")
            sys.exit(1)
        logger.info(f"📊 Swapping entire balance: {amount:,.2f} {token_symbol}")
    elif args.amount:
        amount = args.amount
    else:
        logger.error("❌ Please specify --amount or --all")
        sys.exit(1)
    
    # Execute swap
    result = engine.swap_token_to_sol(
        token_mint=token_mint,
        token_symbol=token_symbol,
        amount=amount,
        dry_run=args.dry_run
    )
    
    if not result['success']:
        logger.error(f"❌ Swap failed: {result.get('error')}")
        sys.exit(1)
    
    logger.info("✅ Swap completed successfully!")


def list_wallet_tokens(engine: TokenSwapEngine):
    """List all tokens in the wallet with current values - using direct RPC"""
    if not engine.wallet_pubkey:
        logger.error("❌ Wallet not loaded")
        return
    
    print(f"\n💰 WALLET TOKEN BALANCES")
    print(f"   Wallet: {engine.wallet_pubkey}")
    print("=" * 80)
    
    sol_usd, sol_nzd = engine.price_service.get_sol_prices()
    sol_balance = engine.get_sol_balance()
    
    print(f"\n   SOL: {sol_balance:.6f} (${sol_balance * sol_usd:.2f} USD)")
    print("\n   TOKENS:")
    print("-" * 80)
    print(f"   {'Symbol':<12} {'Balance':>15} {'Price USD':>12} {'Value USD':>12} {'Mint Address':<44}")
    print("-" * 80)
    
    try:
        # Use direct HTTP RPC call for more reliable parsing
        helius_key = get_secret('HELIUS_KEY')
        rpc_url = f"https://mainnet.helius-rpc.com/?api-key={helius_key}"
        
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "getTokenAccountsByOwner",
            "params": [
                engine.wallet_pubkey,
                {"programId": "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA"},
                {"encoding": "jsonParsed"}
            ]
        }
        
        resp = requests.post(rpc_url, json=payload, timeout=30)
        data = resp.json()
        
        if 'error' in data:
            logger.error(f"RPC error: {data['error']}")
            return
        
        accounts = data.get('result', {}).get('value', [])
        
        total_value = 0
        tokens_found = []
        
        for account in accounts:
            try:
                info = account.get('account', {}).get('data', {}).get('parsed', {}).get('info', {})
                mint = info.get('mint', '')
                token_amount = info.get('tokenAmount', {})
                
                balance = float(token_amount.get('uiAmount') or 0)
                if balance <= 0:
                    # Try raw amount
                    raw = token_amount.get('amount', '0')
                    decimals = int(token_amount.get('decimals', 6))
                    balance = int(raw) / (10 ** decimals) if raw else 0
                
                if balance <= 0 or not mint:
                    continue
                
                tokens_found.append({
                    'mint': mint,
                    'balance': balance
                })
                
            except Exception as e:
                continue
        
        # Display tokens
        for token in tokens_found:
            mint = token['mint']
            balance = token['balance']
            
            # Get token info from DexScreener
            token_info = engine.price_service.get_token_info(mint)
            symbol = token_info.get('symbol', 'UNKNOWN')
            price = token_info.get('price_usd', 0)
            value = balance * price
            total_value += value
            
            # Truncate mint for display
            mint_display = mint[:20] + "..." + mint[-8:] if len(mint) > 30 else mint
            
            print(f"   {symbol:<12} {balance:>15,.2f} ${price:>11.8f} ${value:>11.2f} {mint_display}")
        
        if not tokens_found:
            print("   No tokens found with balance > 0")
        
        print("-" * 80)
        print(f"   {'TOTAL TOKEN VALUE':>39} ${total_value:>11.2f}")
        print(f"   {'TOTAL PORTFOLIO':>39} ${total_value + (sol_balance * sol_usd):>11.2f}")
        print("=" * 80)
        
    except Exception as e:
        logger.error(f"Failed to list tokens: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
