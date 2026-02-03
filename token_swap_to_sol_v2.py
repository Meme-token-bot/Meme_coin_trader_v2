#!/usr/bin/env python3
"""
TOKEN SWAP TO SOL UTILITY (FIXED V4)
====================================

Swaps SPL tokens back to SOL using Jupiter API and Helius Sender.
Fixes:
1. Helius Sender requirement (200k lamport tip).
2. Solders instruction mutability.
3. VersionedTransaction signature type error (Signer vs Signature).
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
import random
import struct
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

TOKEN_MINTS = {
    'TOKEN1': '14DieZPb3JAQwAopBZF5GLGeykjT7A3pjee2muexpump',
    'TOKEN2': '63Z3Q7JX3SBGDiiwqqnPTVvHcuUk6ixkzsYQbKzhpump',
    'TOKEN3': '8ryRQD6jWfxnSdvvVbQ8Tzwo1NgGP7w1X1nQPpb4pump',
    'TOKEN4': 'AtdqW9HYpx6bzuXAyuVRz6a3UiTHLoNTRJDN8buXHem7',
    'TOKEN5': 'FMMqsxKuP7PiCYG7dRhedWoNFfFGfUpXKRGZ5Fvgscaa',
    'BONK': 'DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263',
    'WIF': 'EKpQGSJtjMFqKZ9KQanSqYXRcF8fBopzLHYxdM65zcjm',
    'USDC': 'EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v',
}

# Helius Sender Tip Accounts
HELIUS_TIP_ACCOUNTS = [
    "4ACfpUFoaSD9bfPdeu6DBt89gB6ENTeHBXCAi87NhDEE",
    "5VY91ws6B2hMmBFRsXkoAAdsPHBJwRfBht4DXox3xkwn",
    "wyvPkWjVZz1M8fHQnMMCDTQDbkManefNNhweYk5WkcF",
    "3KCKozbAaF75qEU33jtzozcJ29yJuaLJTy2jFdzUY8bT",
    "9bnz4RShgq1hAnLnZbP8kbgBg1kEmcJBYQq3gQbmnSta",
    "D1Mc6j9xQWgR1o1Z7yU5nVVXFQiAYx7FG9AW1aVfwrUM",
    "2q5pghRs6arqVjRvT5gfgWfWcHWmw1ZuCzphgd5KfWGJ",
    "4TQLFNWK8AovT1gFvda5jfw2oJeRMKEmw7aH6MGBJ3or",
]

# Jupiter API endpoints
JUPITER_QUOTE_URL = "https://public.jupiterapi.com/quote"
JUPITER_SWAP_URL = "https://public.jupiterapi.com/swap"
COINGECKO_URL = "https://api.coingecko.com/api/v3/simple/price"

# Default configuration
DEFAULT_SLIPPAGE_BPS = 100  # 1%
DEFAULT_PRIORITY_FEE = 200000 # 200k lamports minimum for Helius Sender
MIN_TIP_SOL = 0.0002

# =============================================================================
# IMPORT SOLANA LIBRARIES
# =============================================================================

try:
    from solana.rpc.api import Client as SolanaClient
    from solders.keypair import Keypair
    from solders.pubkey import Pubkey
    from solders.transaction import VersionedTransaction
    from solders.message import Message, MessageV0
    from solders.instruction import CompiledInstruction
    from solders.system_program import ID as SYS_PROGRAM_ID
    from solders.signature import Signature  # Required for populate
    SOLANA_AVAILABLE = True
except ImportError:
    SOLANA_AVAILABLE = False
    logger.warning("⚠️ Solana libraries not installed. Run: pip install solana solders")

# =============================================================================
# SECRETS MANAGEMENT
# =============================================================================

def get_secret(key: str, default: str = None) -> Optional[str]:
    try:
        from core.secrets_manager import get_secret as aws_get_secret
        value = aws_get_secret(key)
        if value: return value
    except ImportError:
        pass
    return os.getenv(key, default)

def init_secrets():
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
    slippage_bps: int = DEFAULT_SLIPPAGE_BPS
    priority_fee_lamports: int = DEFAULT_PRIORITY_FEE
    min_tip_sol: float = MIN_TIP_SOL
    use_dynamic_tip: bool = True
    max_retries: int = 3
    confirmation_timeout: int = 30
    db_path: str = "live_trades_tax.db"

# =============================================================================
# SERVICES (Price & Database)
# =============================================================================

class PriceService:
    def __init__(self):
        self._sol_prices = {'usd': 0, 'nzd': 0}
        self._last_sol_update = None
    
    def get_sol_prices(self) -> Tuple[float, float]:
        now = datetime.now()
        if (self._last_sol_update and (now - self._last_sol_update).total_seconds() < 60):
            return self._sol_prices['usd'], self._sol_prices['nzd']
        try:
            resp = requests.get(f"{COINGECKO_URL}?ids=solana&vs_currencies=usd,nzd", timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                self._sol_prices['usd'] = data['solana']['usd']
                self._sol_prices['nzd'] = data['solana']['nzd']
                self._last_sol_update = now
        except: pass
        return self._sol_prices['usd'], self._sol_prices['nzd']
    
    def get_token_price(self, token_address: str) -> Optional[float]:
        try:
            url = f"https://api.dexscreener.com/latest/dex/tokens/{token_address}"
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                pairs = data.get('pairs', [])
                if pairs:
                    best_pair = max(pairs, key=lambda p: float(p.get('liquidity', {}).get('usd', 0) or 0))
                    return float(best_pair.get('priceUsd', 0) or 0)
        except: pass
        return None

class TaxDatabase:
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
        with self._get_connection() as conn:
            conn.execute("""CREATE TABLE IF NOT EXISTS tax_transactions (id TEXT PRIMARY KEY, timestamp TEXT NOT NULL, transaction_type TEXT NOT NULL, token_address TEXT NOT NULL, token_symbol TEXT, token_amount REAL, sol_amount REAL, price_per_token_usd REAL, price_per_token_nzd REAL, sol_price_usd REAL, sol_price_nzd REAL, total_value_usd REAL, total_value_nzd REAL, fee_sol REAL, fee_nzd REAL, cost_basis_nzd REAL, gain_loss_nzd REAL, signature TEXT, notes TEXT, is_live INTEGER DEFAULT 1)""")
            conn.execute("""CREATE TABLE IF NOT EXISTS cost_basis_lots (id INTEGER PRIMARY KEY AUTOINCREMENT, token_address TEXT NOT NULL, acquisition_date TEXT NOT NULL, tokens_acquired REAL NOT NULL, tokens_remaining REAL NOT NULL, cost_per_token_nzd REAL NOT NULL, total_cost_nzd REAL NOT NULL, acquisition_signature TEXT, is_exhausted INTEGER DEFAULT 0)""")
            conn.execute("""CREATE TABLE IF NOT EXISTS live_positions (token_address TEXT PRIMARY KEY, token_symbol TEXT, tokens_held REAL, entry_price_usd REAL, entry_time TEXT, total_cost_sol REAL, total_cost_nzd REAL, stop_loss_pct REAL, take_profit_pct REAL, trailing_stop_pct REAL, peak_price_usd REAL, entry_signature TEXT, conviction_score INTEGER)""")
            conn.execute("""CREATE TABLE IF NOT EXISTS daily_stats (date TEXT PRIMARY KEY, trades INTEGER DEFAULT 0, wins INTEGER DEFAULT 0, losses INTEGER DEFAULT 0, pnl_sol REAL DEFAULT 0, pnl_nzd REAL DEFAULT 0, fees_sol REAL DEFAULT 0)""")
            conn.execute("""CREATE TABLE IF NOT EXISTS execution_log (id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT, action TEXT, token_address TEXT, token_symbol TEXT, status TEXT, signature TEXT, error TEXT, details TEXT)""")
    
    def _generate_id(self) -> str:
        timestamp = datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S%f')
        random_part = hashlib.sha256(os.urandom(8)).hexdigest()[:8]
        return f"TX_{timestamp}_{random_part}"
    
    def record_trade(self, trade_data: Dict) -> str:
        record_id = self._generate_id()
        with self._get_connection() as conn:
            conn.execute("""INSERT INTO tax_transactions (id, timestamp, transaction_type, token_address, token_symbol, token_amount, sol_amount, price_per_token_usd, price_per_token_nzd, sol_price_usd, sol_price_nzd, total_value_usd, total_value_nzd, fee_sol, fee_nzd, cost_basis_nzd, gain_loss_nzd, signature, notes, is_live) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""", 
                         (record_id, trade_data.get('timestamp'), trade_data.get('transaction_type'), trade_data.get('token_address'), trade_data.get('token_symbol'), trade_data.get('token_amount'), trade_data.get('sol_amount'), trade_data.get('price_per_token_usd'), trade_data.get('price_per_token_nzd'), trade_data.get('sol_price_usd'), trade_data.get('sol_price_nzd'), trade_data.get('total_value_usd'), trade_data.get('total_value_nzd'), trade_data.get('fee_sol'), trade_data.get('fee_nzd'), trade_data.get('cost_basis_nzd'), trade_data.get('gain_loss_nzd'), trade_data.get('signature'), trade_data.get('notes'), trade_data.get('is_live')))
        return record_id
    
    def get_fifo_cost_basis(self, token_address: str, tokens_to_sell: float) -> Tuple[float, int]:
        with self._get_connection() as conn:
            lots = conn.execute("SELECT id, tokens_remaining, cost_per_token_nzd, total_cost_nzd FROM cost_basis_lots WHERE token_address = ? AND is_exhausted = 0 ORDER BY acquisition_date ASC", (token_address,)).fetchall()
            if not lots: return 0.0, 0
            total_cost, tokens_remaining, lots_consumed = 0.0, tokens_to_sell, 0
            for lot in lots:
                if tokens_remaining <= 0: break
                lot_tokens = lot['tokens_remaining']
                if lot_tokens <= tokens_remaining:
                    total_cost += lot_tokens * lot['cost_per_token_nzd']
                    tokens_remaining -= lot_tokens
                    lots_consumed += 1
                    conn.execute("UPDATE cost_basis_lots SET is_exhausted = 1, tokens_remaining = 0 WHERE id = ?", (lot['id'],))
                else:
                    total_cost += tokens_remaining * lot['cost_per_token_nzd']
                    conn.execute("UPDATE cost_basis_lots SET tokens_remaining = ? WHERE id = ?", (lot_tokens - tokens_remaining, lot['id']))
                    tokens_remaining = 0
            return total_cost, lots_consumed
    
    def remove_position(self, token_address: str):
        with self._get_connection() as conn:
            conn.execute("DELETE FROM live_positions WHERE token_address = ?", (token_address,))
    
    def update_daily_stats(self, pnl_sol: float, is_win: bool, fee_sol: float = 0):
        today = datetime.now(timezone.utc).strftime('%Y-%m-%d')
        with self._get_connection() as conn:
            existing = conn.execute("SELECT * FROM daily_stats WHERE date = ?", (today,)).fetchone()
            if existing:
                conn.execute("UPDATE daily_stats SET trades = trades + 1, wins = wins + ?, losses = losses + ?, pnl_sol = pnl_sol + ?, fees_sol = fees_sol + ? WHERE date = ?", (1 if is_win else 0, 0 if is_win else 1, pnl_sol, fee_sol, today))
            else:
                conn.execute("INSERT INTO daily_stats (date, trades, wins, losses, pnl_sol, fees_sol) VALUES (?, 1, ?, ?, ?, ?)", (today, 1 if is_win else 0, 0 if is_win else 1, pnl_sol, fee_sol))

# =============================================================================
# TOKEN SWAP ENGINE
# =============================================================================

class TokenSwapEngine:
    def __init__(self, config: SwapConfig = None):
        self.config = config or SwapConfig()
        self.price_service = PriceService()
        self.tax_db = TaxDatabase(self.config.db_path)
        self.keypair = None
        self.wallet_pubkey = None
        self.solana_client = None
        self._load_wallet()
        self._init_rpc()
        logger.info("🔄 Token Swap Engine initialized")
    
    def _load_wallet(self):
        if not SOLANA_AVAILABLE: return
        private_key = get_secret('SOLANA_PRIVATE_KEY') or get_secret('HOT_WALLET_1')
        if private_key:
            try:
                if isinstance(private_key, list): self.keypair = Keypair.from_bytes(bytes(private_key))
                elif isinstance(private_key, dict): 
                    key_value = private_key.get('private_key') or private_key.get('value')
                    self.keypair = Keypair.from_base58_string(key_value)
                else: self.keypair = Keypair.from_base58_string(private_key)
                self.wallet_pubkey = str(self.keypair.pubkey())
                logger.info(f"🔐 Wallet loaded: {self.wallet_pubkey[:8]}...{self.wallet_pubkey[-4:]}")
            except Exception as e: logger.error(f"Failed to load wallet: {e}")
        else: logger.error("❌ No wallet key found in secrets")
    
    def _init_rpc(self):
        if not SOLANA_AVAILABLE: return
        helius_key = get_secret('HELIUS_KEY')
        if helius_key:
            rpc_url = f"https://mainnet.helius-rpc.com/?api-key={helius_key}"
            self.solana_client = SolanaClient(rpc_url)
            logger.info("✅ Helius RPC connected")
    
    def get_token_balance(self, token_mint: str) -> Tuple[float, int]:
        if not self.wallet_pubkey: return 0, 0
        try:
            helius_key = get_secret('HELIUS_KEY')
            rpc_url = f"https://mainnet.helius-rpc.com/?api-key={helius_key}"
            payload = {"jsonrpc": "2.0", "id": 1, "method": "getTokenAccountsByOwner", "params": [self.wallet_pubkey, {"mint": token_mint}, {"encoding": "jsonParsed"}]}
            resp = requests.post(rpc_url, json=payload, timeout=30)
            data = resp.json()
            accounts = data.get('result', {}).get('value', [])
            if accounts:
                info = accounts[0].get('account', {}).get('data', {}).get('parsed', {}).get('info', {})
                token_amount = info.get('tokenAmount', {})
                return float(token_amount.get('uiAmount') or 0), int(token_amount.get('decimals', 6))
            return 0, 6
        except Exception: return 0, 6

    def get_jupiter_quote(self, input_mint: str, output_mint: str, amount: int, slippage_bps: int) -> Optional[Dict]:
        params = {'inputMint': input_mint, 'outputMint': output_mint, 'amount': str(amount), 'slippageBps': slippage_bps}
        try:
            resp = requests.get(JUPITER_QUOTE_URL, params=params, timeout=10)
            if resp.status_code == 200: return resp.json()
        except Exception as e: logger.error(f"Jupiter quote failed: {e}")
        return None
    
    def get_jupiter_swap(self, quote: Dict) -> Optional[str]:
        payload = {
            'quoteResponse': quote,
            'userPublicKey': self.wallet_pubkey,
            'wrapAndUnwrapSol': True,
            'prioritizationFeeLamports': 'auto', # Manually injecting tip
            'asLegacyTransaction': False,
            'useSharedAccounts': False
            #'computeUnitPriceMicroLamports': 50_000
        }
        try:
            resp = requests.post(JUPITER_SWAP_URL, json=payload, timeout=30)
            if resp.status_code == 200:
                data = resp.json()
                return data.get('swapTransaction')
            else:
                logger.error(f"Jupiter swap error: {resp.status_code} - {resp.text[:200]}")
        except Exception as e: logger.error(f"Jupiter swap failed: {e}")
        return None

    def add_helius_tip_instruction(self, txn: VersionedTransaction, payer_pubkey: Pubkey, tip_lamports: int) -> VersionedTransaction:
        """
        Injects a SystemProgram.Transfer instruction to a Helius tip account into a compiled MessageV0.
        Safely handles account key insertion and index shifting by creating NEW instructions.
        """
        msg = txn.message
        
        # 1. Select a random Helius tip account
        tip_account_str = random.choice(HELIUS_TIP_ACCOUNTS)
        tip_pubkey = Pubkey.from_string(tip_account_str)
        
        # 2. Deconstruct the existing message
        account_keys = list(msg.account_keys)
        old_instructions = list(msg.instructions)
        
        # 3. Determine where to insert the new Tip Account
        header = msg.header
        num_signers = header.num_required_signatures
        num_readonly_signed = header.num_readonly_signed_accounts
        num_readonly_unsigned = header.num_readonly_unsigned_accounts
        
        # Calculate existing counts
        num_write_unsigned = len(account_keys) - num_signers - num_readonly_unsigned
        
        # Check if tip account already exists
        if tip_pubkey in account_keys:
             tip_index = account_keys.index(tip_pubkey)
             new_instructions = old_instructions
        else:
            # Insert the tip account into the [Unsigned Write] section.
            insert_index = num_signers + num_write_unsigned
            account_keys.insert(insert_index, tip_pubkey)
            tip_index = insert_index
            
            # CRITICAL FIX 1: Rebuild instructions with shifted indices using NEW objects
            new_instructions = []
            for ix in old_instructions:
                # Update program_id_index
                new_program_id_index = ix.program_id_index
                if new_program_id_index >= insert_index:
                    new_program_id_index += 1
                
                # Update account indices
                new_accounts_list = []
                for acc_idx in ix.accounts:
                    if acc_idx >= insert_index:
                        new_accounts_list.append(acc_idx + 1)
                    else:
                        new_accounts_list.append(acc_idx)
                
                # Create NEW CompiledInstruction (read-only fix)
                new_ix = CompiledInstruction(
                    program_id_index=new_program_id_index,
                    accounts=bytes(new_accounts_list),
                    data=ix.data
                )
                new_instructions.append(new_ix)

        # 4. Find System Program Index
        if SYS_PROGRAM_ID not in account_keys:
            account_keys.append(SYS_PROGRAM_ID)
            header.num_readonly_unsigned_accounts += 1
        
        sys_prog_index = account_keys.index(SYS_PROGRAM_ID)
        payer_index = account_keys.index(payer_pubkey)

        # 5. Create the Transfer Instruction
        # Layout: [u32 instruction_index] [u64 lamports]
        data = struct.pack("<IQ", 2, tip_lamports)
        
        tip_ix = CompiledInstruction(
            program_id_index=sys_prog_index,
            accounts=bytes([payer_index, tip_index]),
            data=data
        )
        
        new_instructions.append(tip_ix)
        
        # 6. Reconstruct MessageV0
        new_msg = MessageV0(
            header=header,
            account_keys=account_keys,
            recent_blockhash=msg.recent_blockhash,
            instructions=new_instructions,
            address_table_lookups=msg.address_table_lookups
        )
        
        # CRITICAL FIX 2: Use populate() with Signature objects, not bytes, and not the constructor
        dummy_sigs = [Signature.default() for _ in range(num_signers)]
        return VersionedTransaction.populate(new_msg, dummy_sigs)

    def execute_via_helius_sender(self, swap_tx_base64: str) -> Optional[str]:
        """Execute Versioned Transaction via Helius Sender with Tip Injection"""
        helius_key = get_secret('HELIUS_KEY')
        if not helius_key: return None
        sender_url = f"https://sender.helius-rpc.com/fast?api-key={helius_key}"
        
        try:
            # 1. Decode base64
            tx_bytes = base64.b64decode(swap_tx_base64)
            txn = VersionedTransaction.from_bytes(tx_bytes)

            # 2. INJECT HELIUS TIP
            logger.info(f"💉 Injecting Helius tip: {self.config.priority_fee_lamports} lamports")
            txn = self.add_helius_tip_instruction(
                txn, 
                self.keypair.pubkey(), 
                self.config.priority_fee_lamports
            )

            # 3. Sign transaction
            signature = self.keypair.sign_message(bytes(txn.message))
            signed_txn = VersionedTransaction.populate(txn.message, [signature])
            
            # 4. Re-encode to base64
            tx_base64 = base64.b64encode(bytes(signed_txn)).decode('utf-8')
            
            # 5. Send payload
            payload = {
                "jsonrpc": "2.0",
                "id": str(int(time.time() * 1000)),
                "method": "sendTransaction",
                "params": [
                    tx_base64,
                    {
                        "encoding": "base64",
                        "skipPreflight": True,
                        "maxRetries": 3
                    }
                ]
            }
            
            resp = requests.post(sender_url, json=payload, headers={"Content-Type": "application/json"}, timeout=30)
            
            if resp.status_code != 200:
                logger.error(f"Helius Sender HTTP error: {resp.status_code}")
                logger.error(f"Response: {resp.text}")
                return None
            
            result = resp.json()
            if "error" in result:
                logger.error(f"Helius Sender error: {result['error']}")
                return None
            
            if "result" in result:
                signature = result["result"]
                logger.info(f"📤 Transaction sent: {signature}")
                return signature
                
        except Exception as e:
            logger.error(f"Execution error: {e}")
            import traceback
            traceback.print_exc()
        
        return None
    
    def wait_for_confirmation(self, signature: str, timeout: int = None) -> bool:
        if not self.solana_client: return False
        timeout = timeout or self.config.confirmation_timeout
        start = time.time()
        time.sleep(1) 
        
        while time.time() - start < timeout:
            try:
                from solders.signature import Signature
                sig = Signature.from_string(signature)
                resp = self.solana_client.get_signature_statuses([sig])
                if resp.value and resp.value[0]:
                    status = resp.value[0]
                    if status.confirmation_status:
                        conf = str(status.confirmation_status)
                        if 'confirmed' in conf.lower() or 'finalized' in conf.lower():
                            logger.info(f"✅ Confirmed: {signature[:16]}...")
                            return True
                time.sleep(1)
            except Exception: time.sleep(1)
        return False
    
    def swap_token_to_sol(self, token_mint: str, token_symbol: str, amount: float, dry_run: bool = False) -> Dict:
        result = {'success': False, 'action': 'SWAP_TO_SOL', 'token_address': token_mint, 'token_symbol': token_symbol, 'amount': amount, 'signature': None, 'sol_received': 0}
        
        logger.info(f"\n{'='*60}\n🔄 SWAP: {amount:,.0f} {token_symbol} → SOL\n{'='*60}")
        
        if not self.keypair:
            logger.error("No wallet loaded"); return result
        
        balance, decimals = self.get_token_balance(token_mint)
        logger.info(f"💰 Balance: {balance:,.2f} {token_symbol}")
        
        if balance < amount:
            logger.error(f"Insufficient balance. Have {balance}, need {amount}"); return result
        
        sol_usd, sol_nzd = self.price_service.get_sol_prices()
        token_price = self.price_service.get_token_price(token_mint)
        
        raw_amount = int(amount * (10 ** decimals))
        
        logger.info(f"📊 Getting Jupiter quote...")
        quote = self.get_jupiter_quote(token_mint, SOL_MINT, raw_amount, self.config.slippage_bps)
        
        if not quote: logger.error("Quote failed"); return result
        
        out_amount = int(quote.get('outAmount', 0))
        sol_received = out_amount / LAMPORTS_PER_SOL
        logger.info(f"📈 Quote: {amount:,.0f} {token_symbol} → {sol_received:.6f} SOL")
        
        if dry_run:
            logger.info(f"🎮 DRY RUN - Would receive: {sol_received:.6f} SOL")
            result['success'] = True; return result
        
        logger.info(f"🔧 Building Versioned Transaction...")
        swap_tx = self.get_jupiter_swap(quote)
        
        if not swap_tx: logger.error("Swap build failed"); return result
        
        logger.info(f"🚀 Executing via Helius Sender...")
        signature = self.execute_via_helius_sender(swap_tx)
        
        if not signature: logger.error("Transaction failed"); return result
        
        result['signature'] = signature
        confirmed = self.wait_for_confirmation(signature)
        
        if not confirmed: logger.error("Not confirmed"); return result
        
        # Post-trade Logic
        cost_basis_nzd, _ = self.tax_db.get_fifo_cost_basis(token_mint, amount)
        fee_sol = self.config.priority_fee_lamports / LAMPORTS_PER_SOL
        proceeds_nzd = sol_received * sol_nzd
        gain_loss_nzd = proceeds_nzd - cost_basis_nzd if cost_basis_nzd > 0 else None
        
        self.tax_db.record_trade({
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'transaction_type': 'SELL', 'token_address': token_mint, 'token_symbol': token_symbol,
            'token_amount': amount, 'sol_amount': sol_received, 'price_per_token_usd': token_price or 0,
            'price_per_token_nzd': 0, 'sol_price_usd': sol_usd, 'sol_price_nzd': sol_nzd,
            'total_value_usd': sol_received * sol_usd, 'total_value_nzd': proceeds_nzd,
            'fee_sol': fee_sol, 'fee_nzd': fee_sol * sol_nzd, 'cost_basis_nzd': cost_basis_nzd,
            'gain_loss_nzd': gain_loss_nzd, 'signature': signature, 'notes': "Swap to SOL via Jupiter + Helius"
        })
        
        self.tax_db.remove_position(token_mint)
        self.tax_db.update_daily_stats(gain_loss_nzd/sol_nzd if gain_loss_nzd else 0, (gain_loss_nzd or 0) > 0, fee_sol)
        
        result['success'] = True
        result['sol_received'] = sol_received
        
        logger.info(f"\n✅ SWAP SUCCESSFUL\n   Sold: {amount:,.0f} {token_symbol}\n   Received: {sol_received:.6f} SOL\n   Signature: {signature}\n")
        return result

# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Swap SPL tokens to SOL")
    parser.add_argument('--token', '-t', type=str, help='Token symbol (e.g., TOKEN1)')
    parser.add_argument('--mint', '-m', type=str, help='Token mint address')
    parser.add_argument('--amount', '-a', type=float, help='Amount to swap')
    parser.add_argument('--all', action='store_true', help='Swap all')
    parser.add_argument('--slippage', '-s', type=int, default=DEFAULT_SLIPPAGE_BPS)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--list', '-l', action='store_true')
    
    args = parser.parse_args()
    init_secrets()
    config = SwapConfig(slippage_bps=args.slippage)
    engine = TokenSwapEngine(config)
    
    if args.list:
        print(f"\n💰 Wallet: {engine.wallet_pubkey}")
        return
        
    if args.mint: token_mint = args.mint
    elif args.token: token_mint = TOKEN_MINTS.get(args.token.upper())
    else: parser.print_help(); sys.exit(1)
    
    token_symbol = args.token.upper() if args.token else "TOKEN"
    
    balance, _ = engine.get_token_balance(token_mint)
    amount = balance if args.all else args.amount
    
    if not amount: logger.error("Specify amount"); sys.exit(1)
    
    engine.swap_token_to_sol(token_mint, token_symbol, amount, args.dry_run)

if __name__ == "__main__":
    main()
