"""
MULTI-WALLET STEALTH TRADING SYSTEM
====================================

Professional-grade multi-wallet architecture to avoid whale detection:

1. 5 Hot Wallets - Randomly selected for trades, 1-week lifespan
2. Dynamic Jito Tips - Higher conviction = higher tip priority
3. Burner Wallets - Receive harvested profits from hot wallets
4. Harvest Pipeline - Burner → CEX with randomized timing/amounts
5. Dynamic Position Limits - Based on wallet balance (see scaling table)

ARCHITECTURE:
┌─────────────────────────────────────────────────────────────────┐
│                        AWS EC2 SERVER                            │
│  ┌──────────┬──────────┬──────────┬──────────┬──────────┐      │
│  │ Hot 1    │ Hot 2    │ Hot 3    │ Hot 4    │ Hot 5    │      │
│  │ 3-21 SOL │ 3-21 SOL │ 3-21 SOL │ 3-21 SOL │ 3-21 SOL │      │
│  └────┬─────┴────┬─────┴────┬─────┴────┬─────┴────┬─────┘      │
└───────┼──────────┼──────────┼──────────┼──────────┼─────────────┘
        │ Random   │ Random   │ Random   │ Random   │ Random
        │ amounts  │ amounts  │ amounts  │ amounts  │ amounts
        ▼          ▼          ▼          ▼          ▼
┌─────────────────────────────────────────────────────────────────┐
│                     LOCAL MACHINE (Burners)                      │
│  ┌──────────┬──────────┬──────────┬──────────┬──────────┐      │
│  │ Burner 1 │ Burner 2 │ Burner 3 │ Burner 4 │ Burner 5 │      │
│  │ →50 SOL  │ →50 SOL  │ →50 SOL  │ →50 SOL  │ →50 SOL  │      │
│  └────┬─────┴────┬─────┴────┬─────┴────┬─────┴────┬─────┘      │
└───────┼──────────┼──────────┼──────────┼──────────┼─────────────┘
        └──────────┴──────────┼──────────┴──────────┘
                              │ Telegram Alert
                              ▼
                    ┌─────────────────┐
                    │   KRAKEN (CEX)  │
                    │   Link Breaker  │
                    └─────────────────┘

POSITION SCALING TABLE (from analysis):
| Wallet_SOL | Trading_SOL | Gas_SOL | Reserve_SOL | Max_Positions |
|------------|-------------|---------|-------------|---------------|
| 3          | 2           | 0.5     | 0.5         | 5             |
| 6          | 4           | 1       | 1           | 11            |
| 9          | 6           | 1.5     | 1.5         | 17            |
| 12         | 8           | 2       | 2           | 22            |
| 15         | 10          | 2.5     | 2.5         | 28            |
| 18         | 12          | 3       | 3           | 34            |
| 21         | 14          | 3.5     | 3.5         | 40            |

Author: Trading Bot System
"""

import os
import json
import time
import random
import threading
import hashlib
import base64
import base58
import sqlite3
import requests
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict
from decimal import Decimal
from enum import Enum
from contextlib import contextmanager
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s'
)
logger = logging.getLogger("MultiWallet")

# Solana imports
try:
    from solders.keypair import Keypair
    from solders.pubkey import Pubkey
    from solders.transaction import VersionedTransaction, Transaction
    from solders.signature import Signature
    from solders.message import MessageV0, Message
    from solders.system_program import transfer, TransferParams
    from solders.hash import Hash
    from solana.rpc.api import Client as SolanaClient
    SOLANA_AVAILABLE = True
except ImportError:
    SOLANA_AVAILABLE = False
    logger.warning("⚠️ Solana libraries not installed")


# =============================================================================
# CONSTANTS
# =============================================================================

SOL_MINT = "So11111111111111111111111111111111111111112"
LAMPORTS_PER_SOL = 1_000_000_000

# Jito endpoints
JITO_BLOCK_ENGINE_URL = "https://mainnet.block-engine.jito.wtf"
JITO_BUNDLE_URL = f"{JITO_BLOCK_ENGINE_URL}/api/v1/bundles"
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

# Jupiter API
JUPITER_QUOTE_URL = "https://quote-api.jup.ag/v6/quote"
JUPITER_SWAP_URL = "https://quote-api.jup.ag/v6/swap"


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class MultiWalletConfig:
    """Configuration for multi-wallet stealth trading"""
    
    # Wallet management
    num_hot_wallets: int = 5
    wallet_lifespan_days: int = 7              # Rotate wallets weekly
    
    # Position sizing
    position_size_sol: float = 0.3             # Per trade
    max_positions_per_wallet: int = 40         # Max when wallet has 21+ SOL
    total_max_positions: int = 200             # Across all wallets
    
    # Balance thresholds (adjusted for single-wallet bootstrap)
    min_wallet_sol: float = 1.0                # Allow trading down to 1 SOL
    soft_min_wallet_sol: float = 3.0           # Preferred minimum (full capacity)
    max_wallet_sol: float = 21.0               # Trigger harvest above this
    harvest_threshold_sol: float = 21.0        # When to harvest to burner
    
    # Burner wallet settings
    burner_harvest_threshold: float = 50.0     # When burner should send to CEX
    
    # Jito tips (conviction-based)
    min_jito_tip_lamports: int = 500_000       # 0.0005 SOL (low conviction)
    max_jito_tip_lamports: int = 5_000_000     # 0.005 SOL (high conviction)
    
    # Timing randomization (anti-pattern detection)
    min_harvest_delay_hours: float = 1.0       # Minimum time between harvests
    max_harvest_delay_hours: float = 24.0      # Maximum time between harvests
    
    # Trading parameters
    default_slippage_bps: int = 200            # 2% for larger positions
    confirmation_timeout: int = 30
    
    # Entry filters
    min_conviction: int = 60
    min_liquidity_usd: float = 10000           # Higher for larger positions
    blocked_hours_utc: List[int] = field(default_factory=lambda: [1, 3, 5, 19, 23])
    
    # Exit parameters
    stop_loss_pct: float = -0.15
    take_profit_pct: float = 0.30
    trailing_stop_pct: float = 0.10
    max_hold_hours: int = 12


# =============================================================================
# POSITION SCALING CALCULATOR
# =============================================================================

class PositionScaler:
    """
    Calculates max positions based on wallet balance.
    
    Formula derived from the scaling table:
    max_positions ≈ floor(balance * 2 - 1), capped at 40
    
    Extended to support bootstrap mode (1-3 SOL):
    - 1 SOL → 1 position (bootstrap)
    - 2 SOL → 2 positions
    - 3 SOL → 5 positions (normal scaling starts)
    
    Breakdown of wallet balance:
    - Trading SOL: ~67% (for positions)
    - Gas SOL: ~17% (for transaction fees + Jito tips)
    - Reserve SOL: ~17% (safety buffer)
    """
    
    # Lookup table for exact values
    SCALING_TABLE = {
        3: {'trading': 2, 'gas': 0.5, 'reserve': 0.5, 'max_positions': 5},
        6: {'trading': 4, 'gas': 1.0, 'reserve': 1.0, 'max_positions': 11},
        9: {'trading': 6, 'gas': 1.5, 'reserve': 1.5, 'max_positions': 17},
        12: {'trading': 8, 'gas': 2.0, 'reserve': 2.0, 'max_positions': 22},
        15: {'trading': 10, 'gas': 2.5, 'reserve': 2.5, 'max_positions': 28},
        18: {'trading': 12, 'gas': 3.0, 'reserve': 3.0, 'max_positions': 34},
        21: {'trading': 14, 'gas': 3.5, 'reserve': 3.5, 'max_positions': 40},
    }
    
    @classmethod
    def get_max_positions(cls, balance_sol: float, position_size: float = 0.3) -> int:
        """
        Get maximum positions allowed for a given wallet balance.
        
        Supports bootstrap mode (1-3 SOL) for gradual scaling.
        """
        if balance_sol < 0.5:
            return 0  # Not enough for even gas
        
        # Bootstrap mode: 0.5-3 SOL
        # Be conservative - need gas reserve
        if balance_sol < 3.0:
            # Calculate how many positions we can afford
            # Reserve 0.3 SOL for gas (Jito tips + fees)
            available = balance_sol - 0.3
            max_pos = int(available / position_size)
            return max(0, min(3, max_pos))  # Cap at 3 during bootstrap
        
        # Normal mode: 3+ SOL
        # Cap at 21 SOL (40 positions max)
        balance_sol = min(balance_sol, 21.0)
        
        # Linear interpolation formula: positions ≈ balance * 2 - 1
        max_pos = int(balance_sol * 1.9 - 0.7)
        
        # Ensure within bounds
        return max(1, min(40, max_pos))
    
    @classmethod
    def get_trading_capital(cls, balance_sol: float) -> float:
        """Get the amount available for trading (excluding gas/reserve)"""
        if balance_sol < 0.5:
            return 0
        
        if balance_sol < 3.0:
            # Bootstrap: reserve 0.3 for gas
            return max(0, balance_sol - 0.3)
        
        # Normal: ~67% for trading
        return balance_sol * 0.67
    
    @classmethod
    def get_gas_reserve(cls, balance_sol: float) -> float:
        """Get gas reserve amount"""
        if balance_sol < 3.0:
            return min(0.3, balance_sol * 0.3)  # 30% or 0.3 SOL max
        return balance_sol * 0.17
    
    @classmethod
    def get_safety_reserve(cls, balance_sol: float) -> float:
        """Get safety reserve amount"""
        if balance_sol < 3.0:
            return 0  # No safety reserve in bootstrap
        return balance_sol * 0.17
    
    @classmethod
    def can_open_position(cls, balance_sol: float, open_positions: int, 
                          position_size: float = 0.3) -> Tuple[bool, str]:
        """Check if wallet can open another position"""
        max_pos = cls.get_max_positions(balance_sol, position_size)
        
        if max_pos == 0:
            return False, f"Balance too low ({balance_sol:.2f} SOL)"
        
        if open_positions >= max_pos:
            return False, f"Max positions reached ({max_pos}) for balance {balance_sol:.1f} SOL"
        
        trading_capital = cls.get_trading_capital(balance_sol)
        deployed = open_positions * position_size
        
        if deployed + position_size > trading_capital:
            return False, f"Insufficient trading capital: {trading_capital - deployed:.2f} SOL available"
        
        return True, "OK"
    
    @classmethod
    def get_mode(cls, balance_sol: float) -> str:
        """Get trading mode based on balance"""
        if balance_sol < 0.5:
            return "OFFLINE"
        elif balance_sol < 3.0:
            return "BOOTSTRAP"
        elif balance_sol < 21.0:
            return "SCALING"
        else:
            return "FULL"


# =============================================================================
# DYNAMIC JITO TIP CALCULATOR
# =============================================================================

class DynamicJitoTip:
    """
    Calculates Jito tip based on conviction score.
    
    Higher conviction = higher tip = faster execution = more likely to land
    
    Conviction 60 (minimum) → 0.0005 SOL tip
    Conviction 80 → 0.002 SOL tip
    Conviction 100 → 0.005 SOL tip
    """
    
    def __init__(self, config: MultiWalletConfig):
        self.config = config
    
    def calculate_tip(self, conviction_score: int) -> int:
        """
        Calculate Jito tip in lamports based on conviction score.
        
        Args:
            conviction_score: 0-100 score indicating trade quality
        
        Returns:
            Tip amount in lamports
        """
        # Clamp conviction to valid range
        conviction = max(0, min(100, conviction_score))
        
        # Linear interpolation between min and max tip
        # conviction 60 = min tip, conviction 100 = max tip
        if conviction < 60:
            return self.config.min_jito_tip_lamports
        
        # Scale from 60-100 to min-max tip
        ratio = (conviction - 60) / 40.0  # 0.0 to 1.0
        tip_range = self.config.max_jito_tip_lamports - self.config.min_jito_tip_lamports
        
        tip = int(self.config.min_jito_tip_lamports + (ratio * tip_range))
        
        return tip
    
    def get_tip_sol(self, conviction_score: int) -> float:
        """Get tip amount in SOL"""
        return self.calculate_tip(conviction_score) / LAMPORTS_PER_SOL


# =============================================================================
# HOT WALLET
# =============================================================================

@dataclass
class HotWallet:
    """Represents a single hot trading wallet"""
    
    id: int                                    # Wallet index (1-5)
    keypair: Keypair                           # Solana keypair
    public_key: str                            # Public address
    burner_address: str                        # Associated burner wallet address
    created_at: datetime                       # When wallet was created
    expires_at: datetime                       # When wallet should be rotated
    
    # Tracking
    total_trades: int = 0
    total_pnl_sol: float = 0.0
    last_trade_at: Optional[datetime] = None
    last_harvest_at: Optional[datetime] = None
    
    def is_expired(self) -> bool:
        """Check if wallet should be rotated"""
        return datetime.now(timezone.utc) > self.expires_at
    
    def days_until_expiry(self) -> float:
        """Days remaining before rotation"""
        delta = self.expires_at - datetime.now(timezone.utc)
        return max(0, delta.total_seconds() / 86400)
    
    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'public_key': self.public_key,
            'burner_address': self.burner_address,
            'created_at': self.created_at.isoformat(),
            'expires_at': self.expires_at.isoformat(),
            'total_trades': self.total_trades,
            'total_pnl_sol': self.total_pnl_sol,
            'days_until_expiry': self.days_until_expiry()
        }


# =============================================================================
# MULTI-WALLET MANAGER
# =============================================================================

class MultiWalletManager:
    """
    Manages multiple hot wallets for stealth trading.
    
    Features:
    - Random wallet selection for trades
    - Automatic balance-based position limits
    - Harvest scheduling with randomization
    - Wallet rotation after expiry
    """
    
    def __init__(self, config: MultiWalletConfig, helius_key: str,
                 db_path: str = "multi_wallet.db"):
        self.config = config
        self.helius_key = helius_key
        self.db_path = db_path
        
        # RPC client
        self.rpc_url = f"https://mainnet.helius-rpc.com/?api-key={helius_key}"
        self.solana_client = SolanaClient(self.rpc_url) if SOLANA_AVAILABLE else None
        
        # Jito tip calculator
        self.jito_tip_calc = DynamicJitoTip(config)
        
        # Wallet storage
        self.hot_wallets: Dict[int, HotWallet] = {}
        
        # Position tracking per wallet
        self.wallet_positions: Dict[int, List[Dict]] = {i: [] for i in range(1, 6)}
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Initialize database
        self._init_database()
        
        # Load existing wallets or create new ones
        self._load_or_create_wallets()
    
    def _init_database(self):
        """Initialize multi-wallet database"""
        with sqlite3.connect(self.db_path) as conn:
            # Wallet registry
            conn.execute("""
                CREATE TABLE IF NOT EXISTS wallets (
                    id INTEGER PRIMARY KEY,
                    public_key TEXT NOT NULL,
                    private_key_encrypted TEXT NOT NULL,
                    burner_address TEXT,
                    created_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    total_trades INTEGER DEFAULT 0,
                    total_pnl_sol REAL DEFAULT 0,
                    is_active BOOLEAN DEFAULT 1
                )
            """)
            
            # Positions per wallet
            conn.execute("""
                CREATE TABLE IF NOT EXISTS wallet_positions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    wallet_id INTEGER NOT NULL,
                    token_address TEXT NOT NULL,
                    token_symbol TEXT,
                    entry_price REAL,
                    entry_time TEXT,
                    position_size_sol REAL,
                    tokens_held REAL,
                    stop_loss_pct REAL,
                    take_profit_pct REAL,
                    conviction_score INTEGER,
                    status TEXT DEFAULT 'OPEN',
                    exit_price REAL,
                    exit_time TEXT,
                    pnl_sol REAL
                )
            """)
            
            # Harvest history
            conn.execute("""
                CREATE TABLE IF NOT EXISTS harvest_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    wallet_id INTEGER NOT NULL,
                    amount_sol REAL NOT NULL,
                    to_address TEXT NOT NULL,
                    signature TEXT,
                    timestamp TEXT NOT NULL,
                    status TEXT DEFAULT 'PENDING'
                )
            """)
            
            # Burner wallet balances (for notifications)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS burner_balances (
                    address TEXT PRIMARY KEY,
                    wallet_id INTEGER,
                    balance_sol REAL DEFAULT 0,
                    last_updated TEXT
                )
            """)
            
            conn.commit()
    
    def _load_or_create_wallets(self):
        """Load existing wallets from DB or create new ones"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM wallets WHERE is_active = 1"
            ).fetchall()
            
            if len(rows) >= self.config.num_hot_wallets:
                # Load existing wallets
                for row in rows[:self.config.num_hot_wallets]:
                    try:
                        # Decrypt private key (in production, use proper encryption)
                        private_key = self._decrypt_key(row['private_key_encrypted'])
                        keypair = Keypair.from_base58_string(private_key)
                        
                        wallet = HotWallet(
                            id=row['id'],
                            keypair=keypair,
                            public_key=row['public_key'],
                            burner_address=row['burner_address'] or '',
                            created_at=datetime.fromisoformat(row['created_at']),
                            expires_at=datetime.fromisoformat(row['expires_at']),
                            total_trades=row['total_trades'],
                            total_pnl_sol=row['total_pnl_sol']
                        )
                        self.hot_wallets[wallet.id] = wallet
                        logger.info(f"  ✅ Loaded wallet {wallet.id}: {wallet.public_key[:8]}...")
                    except Exception as e:
                        logger.error(f"  ❌ Failed to load wallet {row['id']}: {e}")
            
            # Create missing wallets
            existing_ids = set(self.hot_wallets.keys())
            for i in range(1, self.config.num_hot_wallets + 1):
                if i not in existing_ids:
                    logger.info(f"  🆕 Creating new hot wallet {i}...")
                    # Note: In production, you'd generate these externally
                    # and import via AWS Secrets Manager
                    wallet = self._create_wallet(i)
                    if wallet:
                        self.hot_wallets[i] = wallet
    
    def _create_wallet(self, wallet_id: int) -> Optional[HotWallet]:
        """Create a new hot wallet"""
        try:
            keypair = Keypair()
            public_key = str(keypair.pubkey())
            now = datetime.now(timezone.utc)
            expires = now + timedelta(days=self.config.wallet_lifespan_days)
            
            # Store in database
            encrypted_key = self._encrypt_key(base58.b58encode(bytes(keypair)).decode())
            
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO wallets (id, public_key, private_key_encrypted, created_at, expires_at)
                    VALUES (?, ?, ?, ?, ?)
                """, (wallet_id, public_key, encrypted_key, now.isoformat(), expires.isoformat()))
                conn.commit()
            
            wallet = HotWallet(
                id=wallet_id,
                keypair=keypair,
                public_key=public_key,
                burner_address='',  # Set via configure_burner()
                created_at=now,
                expires_at=expires
            )
            
            logger.info(f"  ✅ Created wallet {wallet_id}: {public_key}")
            return wallet
            
        except Exception as e:
            logger.error(f"  ❌ Failed to create wallet {wallet_id}: {e}")
            return None
    
    def _encrypt_key(self, private_key: str) -> str:
        """Encrypt private key for storage (basic - use proper encryption in production!)"""
        # TODO: Implement proper encryption with AWS KMS or similar
        # For now, just base64 encode (NOT SECURE - placeholder only)
        return base64.b64encode(private_key.encode()).decode()
    
    def _decrypt_key(self, encrypted: str) -> str:
        """Decrypt private key from storage"""
        # TODO: Implement proper decryption
        return base64.b64decode(encrypted.encode()).decode()
    
    def configure_burner(self, wallet_id: int, burner_address: str):
        """
        Configure the burner wallet address for a hot wallet.
        
        Args:
            wallet_id: Hot wallet ID (1-5)
            burner_address: Public address of the burner wallet
        """
        if wallet_id not in self.hot_wallets:
            raise ValueError(f"Wallet {wallet_id} not found")
        
        self.hot_wallets[wallet_id].burner_address = burner_address
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "UPDATE wallets SET burner_address = ? WHERE id = ?",
                (burner_address, wallet_id)
            )
            conn.execute("""
                INSERT OR REPLACE INTO burner_balances (address, wallet_id, balance_sol, last_updated)
                VALUES (?, ?, 0, ?)
            """, (burner_address, wallet_id, datetime.now(timezone.utc).isoformat()))
            conn.commit()
        
        logger.info(f"✅ Configured burner for wallet {wallet_id}: {burner_address[:8]}...")
    
    def get_wallet_balance(self, wallet_id: int) -> float:
        """Get SOL balance for a wallet"""
        if wallet_id not in self.hot_wallets:
            return 0.0
        
        if not self.solana_client:
            return 0.0
        
        wallet = self.hot_wallets[wallet_id]
        
        try:
            resp = self.solana_client.get_balance(wallet.keypair.pubkey())
            return resp.value / LAMPORTS_PER_SOL
        except Exception as e:
            logger.error(f"Balance check failed for wallet {wallet_id}: {e}")
            return 0.0
    
    def get_wallet_positions(self, wallet_id: int) -> List[Dict]:
        """Get open positions for a wallet"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("""
                SELECT * FROM wallet_positions 
                WHERE wallet_id = ? AND status = 'OPEN'
            """, (wallet_id,)).fetchall()
            return [dict(row) for row in rows]
    
    def select_wallet_for_trade(self, conviction_score: int = 60) -> Optional[Tuple[int, HotWallet]]:
        """
        Randomly select a wallet that can accept a new position.
        
        Supports bootstrap mode: Works with single wallet or wallets below 3 SOL.
        
        Returns:
            Tuple of (wallet_id, wallet) or None if no wallet available
        """
        with self._lock:
            available = []
            
            for wallet_id, wallet in self.hot_wallets.items():
                # Skip expired wallets
                if wallet.is_expired():
                    continue
                
                # Check balance and positions
                balance = self.get_wallet_balance(wallet_id)
                positions = self.get_wallet_positions(wallet_id)
                
                # Use PositionScaler which handles bootstrap mode
                can_trade, reason = PositionScaler.can_open_position(
                    balance, len(positions), self.config.position_size_sol
                )
                
                if can_trade:
                    available.append((wallet_id, wallet, balance))
                else:
                    logger.debug(f"Wallet {wallet_id} skipped: {reason}")
            
            if not available:
                logger.warning("No wallets available for trading")
                return None
            
            # If only one wallet available, just use it
            if len(available) == 1:
                return (available[0][0], available[0][1])
            
            # Random selection (weighted by available capacity)
            # Wallets with more room get higher probability
            weights = []
            for wid, w, bal in available:
                max_pos = PositionScaler.get_max_positions(bal, self.config.position_size_sol)
                current_pos = len(self.get_wallet_positions(wid))
                capacity = max_pos - current_pos
                weights.append(max(1, capacity))  # At least weight of 1
            
            total_weight = sum(weights)
            if total_weight == 0:
                return None
            
            # Weighted random selection
            r = random.uniform(0, total_weight)
            cumulative = 0
            for i, (wid, w, bal) in enumerate(available):
                cumulative += weights[i]
                if r <= cumulative:
                    mode = PositionScaler.get_mode(bal)
                    logger.info(f"Selected wallet {wid} | Balance: {bal:.2f} SOL | Mode: {mode}")
                    return (wid, w)
            
            # Fallback to last available
            return (available[-1][0], available[-1][1])
    
    def get_jito_tip_for_trade(self, conviction_score: int) -> int:
        """Get dynamic Jito tip based on conviction"""
        return self.jito_tip_calc.calculate_tip(conviction_score)
    
    def record_trade_entry(self, wallet_id: int, token_address: str, 
                          token_symbol: str, entry_price: float,
                          tokens_held: float, conviction_score: int) -> int:
        """Record a new position entry"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                INSERT INTO wallet_positions 
                (wallet_id, token_address, token_symbol, entry_price, entry_time,
                 position_size_sol, tokens_held, stop_loss_pct, take_profit_pct,
                 conviction_score, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'OPEN')
            """, (
                wallet_id, token_address, token_symbol, entry_price,
                datetime.now(timezone.utc).isoformat(),
                self.config.position_size_sol, tokens_held,
                self.config.stop_loss_pct, self.config.take_profit_pct,
                conviction_score
            ))
            conn.commit()
            
            # Update wallet stats
            conn.execute(
                "UPDATE wallets SET total_trades = total_trades + 1 WHERE id = ?",
                (wallet_id,)
            )
            conn.commit()
            
            return cursor.lastrowid
    
    def record_trade_exit(self, position_id: int, exit_price: float, pnl_sol: float):
        """Record position exit"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE wallet_positions 
                SET exit_price = ?, exit_time = ?, pnl_sol = ?, status = 'CLOSED'
                WHERE id = ?
            """, (exit_price, datetime.now(timezone.utc).isoformat(), pnl_sol, position_id))
            
            # Get wallet_id for stats update
            row = conn.execute(
                "SELECT wallet_id FROM wallet_positions WHERE id = ?", 
                (position_id,)
            ).fetchone()
            
            if row:
                conn.execute(
                    "UPDATE wallets SET total_pnl_sol = total_pnl_sol + ? WHERE id = ?",
                    (pnl_sol, row[0])
                )
            
            conn.commit()
    
    def check_harvest_needed(self) -> List[Dict]:
        """
        Check which wallets need to harvest profits to burners.
        
        Returns list of wallets that exceed harvest threshold.
        """
        harvest_needed = []
        
        for wallet_id, wallet in self.hot_wallets.items():
            if not wallet.burner_address:
                continue
            
            balance = self.get_wallet_balance(wallet_id)
            
            if balance > self.config.harvest_threshold_sol:
                excess = balance - self.config.max_wallet_sol
                # Randomize harvest amount (anti-pattern)
                harvest_amount = excess * random.uniform(0.8, 0.95)
                
                harvest_needed.append({
                    'wallet_id': wallet_id,
                    'balance': balance,
                    'harvest_amount': harvest_amount,
                    'burner_address': wallet.burner_address
                })
        
        return harvest_needed
    
    def execute_harvest(self, wallet_id: int, amount_sol: float) -> Optional[str]:
        """
        Execute harvest transfer from hot wallet to burner.
        
        Returns transaction signature if successful.
        """
        wallet = self.hot_wallets.get(wallet_id)
        if not wallet or not wallet.burner_address:
            return None
        
        if not self.solana_client:
            logger.warning("Harvest skipped: Solana RPC client unavailable")
            return None
        
        try:
            burner_pubkey = Pubkey.from_string(wallet.burner_address)
            amount_lamports = int(amount_sol * LAMPORTS_PER_SOL)
            
            # Create transfer instruction
            transfer_ix = transfer(
                TransferParams(
                    from_pubkey=wallet.keypair.pubkey(),
                    to_pubkey=burner_pubkey,
                    lamports=amount_lamports
                )
            )
            
            # Get recent blockhash
            blockhash_resp = self.solana_client.get_latest_blockhash()
            recent_blockhash = blockhash_resp.value.blockhash
            
            # Create and sign transaction
            message = Message.new_with_blockhash(
                [transfer_ix],
                wallet.keypair.pubkey(),
                recent_blockhash
            )
            tx = Transaction.new_unsigned(message)
            tx.sign([wallet.keypair], recent_blockhash)
            
            # Send
            resp = self.solana_client.send_transaction(tx)
            signature = str(resp.value)
            
            # Record harvest
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO harvest_history 
                    (wallet_id, amount_sol, to_address, signature, timestamp, status)
                    VALUES (?, ?, ?, ?, ?, 'COMPLETED')
                """, (
                    wallet_id, amount_sol, wallet.burner_address,
                    signature, datetime.now(timezone.utc).isoformat()
                ))
                conn.commit()
            
            logger.info(f"💸 Harvested {amount_sol:.2f} SOL from wallet {wallet_id} to burner")
            return signature
            
        except Exception as e:
            logger.error(f"Harvest failed for wallet {wallet_id}: {e}")
            return None
    
    def get_total_stats(self) -> Dict:
        """Get aggregate stats across all wallets"""
        total_balance = 0
        total_positions = 0
        total_trades = 0
        total_pnl = 0
        
        wallet_stats = []
        
        for wallet_id, wallet in self.hot_wallets.items():
            balance = self.get_wallet_balance(wallet_id)
            positions = self.get_wallet_positions(wallet_id)
            max_pos = PositionScaler.get_max_positions(balance)
            
            total_balance += balance
            total_positions += len(positions)
            total_trades += wallet.total_trades
            total_pnl += wallet.total_pnl_sol
            
            wallet_stats.append({
                'id': wallet_id,
                'address': wallet.public_key[:8] + '...',
                'balance': balance,
                'positions': f"{len(positions)}/{max_pos}",
                'trades': wallet.total_trades,
                'pnl': wallet.total_pnl_sol,
                'expires_in': f"{wallet.days_until_expiry():.1f} days"
            })
        
        return {
            'total_balance_sol': total_balance,
            'total_positions': total_positions,
            'max_total_positions': self.config.total_max_positions,
            'total_trades': total_trades,
            'total_pnl_sol': total_pnl,
            'wallets': wallet_stats
        }
    
    def print_status(self):
        """Print formatted status"""
        stats = self.get_total_stats()
        
        print("\n" + "=" * 70)
        print("🏦 MULTI-WALLET STEALTH TRADING STATUS")
        print("=" * 70)
        
        print(f"\n📊 AGGREGATE:")
        print(f"   Total Balance: {stats['total_balance_sol']:.2f} SOL")
        print(f"   Open Positions: {stats['total_positions']}/{stats['max_total_positions']}")
        print(f"   Total Trades: {stats['total_trades']}")
        print(f"   Total PnL: {stats['total_pnl_sol']:+.4f} SOL")
        
        print(f"\n🔥 HOT WALLETS:")
        for w in stats['wallets']:
            print(f"   Wallet {w['id']}: {w['address']} | "
                  f"{w['balance']:.2f} SOL | "
                  f"Pos: {w['positions']} | "
                  f"PnL: {w['pnl']:+.2f} | "
                  f"Expires: {w['expires_in']}")
        
        # Check for harvests needed
        harvests = self.check_harvest_needed()
        if harvests:
            print(f"\n⚠️ HARVEST NEEDED:")
            for h in harvests:
                print(f"   Wallet {h['wallet_id']}: {h['balance']:.2f} SOL → "
                      f"Harvest {h['harvest_amount']:.2f} SOL to burner")
        
        print("\n" + "=" * 70)


# =============================================================================
# TELEGRAM NOTIFIER FOR HARVESTS
# =============================================================================

class HarvestNotifier:
    """Sends Telegram alerts when burner wallets need harvesting to CEX"""
    
    def __init__(self, bot_token: str, chat_id: str):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.enabled = bool(bot_token and chat_id)
    
    def send(self, message: str):
        """Send Telegram message"""
        if not self.enabled:
            return
        
        try:
            url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
            requests.post(url, json={
                'chat_id': self.chat_id,
                'text': message,
                'parse_mode': 'HTML'
            }, timeout=10)
        except Exception as e:
            logger.error(f"Telegram error: {e}")
    
    def notify_hot_wallet_harvest(self, wallet_id: int, amount: float, burner_address: str):
        """Notify that hot wallet harvested to burner"""
        msg = f"""💸 <b>Hot Wallet Harvest</b>

Wallet {wallet_id} → Burner
Amount: {amount:.2f} SOL
Burner: {burner_address[:12]}...

<i>Burner accumulating funds...</i>"""
        self.send(msg)
    
    def notify_burner_ready(self, burner_address: str, balance: float, 
                           recommended_amount: float):
        """Notify that burner is ready to send to CEX"""
        # Randomize recommendation for anti-pattern
        random_amount = recommended_amount * random.uniform(0.85, 0.95)
        
        msg = f"""🏦 <b>BURNER READY FOR CEX</b>

Burner: {burner_address[:16]}...
Balance: {balance:.2f} SOL

<b>Action Required:</b>
Send ~{random_amount:.2f} SOL to your CEX

⏰ Do this at a random time today
💡 Don't use round numbers!
📊 Suggested: {random_amount:.3f} SOL"""
        
        self.send(msg)
    
    def notify_wallet_expiring(self, wallet_id: int, days_remaining: float,
                               balance: float):
        """Notify that a wallet is expiring soon"""
        msg = f"""⚠️ <b>Wallet Expiring Soon</b>

Wallet {wallet_id} expires in {days_remaining:.1f} days
Current balance: {balance:.2f} SOL

<b>Actions:</b>
1. Stop new trades to this wallet
2. Wait for positions to close
3. Harvest remaining balance
4. Generate new wallet"""
        
        self.send(msg)


# =============================================================================
# CLI & TESTING
# =============================================================================

def main():
    """CLI interface for multi-wallet manager"""
    import argparse
    
    # Try to initialize secrets
    try:
        from core.secrets_manager import init_secrets, get_secret
        init_secrets()
    except ImportError:
        from dotenv import load_dotenv
        load_dotenv()
        def get_secret(key, default=None):
            return os.getenv(key, default)
    
    parser = argparse.ArgumentParser(description="Multi-Wallet Stealth Trading System")
    parser.add_argument('command', 
                       choices=['status', 'create-wallets', 'set-burner', 
                               'check-harvest', 'execute-harvest', 'test-scaling'],
                       help='Command to run')
    parser.add_argument('--wallet-id', type=int, help='Wallet ID (1-5)')
    parser.add_argument('--burner', type=str, help='Burner wallet address')
    
    args = parser.parse_args()
    
    if args.command == 'test-scaling':
        print("\n📊 POSITION SCALING TABLE (with Bootstrap Mode)")
        print("=" * 70)
        print(f"{'Balance':>10} {'Mode':>12} {'Trading':>10} {'Max Pos':>10} {'Can Trade':>12}")
        print("-" * 70)
        
        for balance in [0.3, 0.5, 1, 1.5, 2, 2.5, 3, 6, 9, 12, 15, 18, 21, 25]:
            max_pos = PositionScaler.get_max_positions(balance, 0.3)
            trading = PositionScaler.get_trading_capital(balance)
            mode = PositionScaler.get_mode(balance)
            can, _ = PositionScaler.can_open_position(balance, 0, 0.3)
            print(f"{balance:>10.1f} {mode:>12} {trading:>10.2f} {max_pos:>10} {'✅' if can else '❌':>12}")
        
        print("\n📈 JITO TIP SCALING")
        print("=" * 60)
        print(f"{'Conviction':>12} {'Tip (SOL)':>12} {'Tip (lamports)':>15}")
        print("-" * 60)
        
        config = MultiWalletConfig()
        tip_calc = DynamicJitoTip(config)
        for conv in [50, 60, 70, 80, 90, 100]:
            tip = tip_calc.calculate_tip(conv)
            print(f"{conv:>12} {tip/LAMPORTS_PER_SOL:>12.4f} {tip:>15,}")
        
        return
    
    # Initialize manager
    helius_key = get_secret('HELIUS_KEY')
    if not helius_key:
        print("❌ HELIUS_KEY not found")
        return
    
    config = MultiWalletConfig()
    manager = MultiWalletManager(config, helius_key)
    
    if args.command == 'status':
        manager.print_status()
    
    elif args.command == 'create-wallets':
        print("Creating wallets... (done during initialization)")
        manager.print_status()
    
    elif args.command == 'set-burner':
        if not args.wallet_id or not args.burner:
            print("Usage: --wallet-id <1-5> --burner <address>")
            return
        manager.configure_burner(args.wallet_id, args.burner)
        print(f"✅ Burner set for wallet {args.wallet_id}")
    
    elif args.command == 'check-harvest':
        harvests = manager.check_harvest_needed()
        if harvests:
            print("\n⚠️ Wallets need harvesting:")
            for h in harvests:
                print(f"  Wallet {h['wallet_id']}: {h['balance']:.2f} SOL → "
                      f"Harvest {h['harvest_amount']:.2f} SOL")
        else:
            print("✅ No harvests needed")
    
    elif args.command == 'execute-harvest':
        if not args.wallet_id:
            print("Usage: --wallet-id <1-5>")
            return
        
        harvests = manager.check_harvest_needed()
        for h in harvests:
            if h['wallet_id'] == args.wallet_id:
                sig = manager.execute_harvest(h['wallet_id'], h['harvest_amount'])
                if sig:
                    print(f"✅ Harvest complete: {sig}")
                else:
                    print("❌ Harvest failed")
                return
        
        print(f"Wallet {args.wallet_id} doesn't need harvesting")


if __name__ == "__main__":
    main()
