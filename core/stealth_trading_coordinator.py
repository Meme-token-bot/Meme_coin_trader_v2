"""
STEALTH TRADING COORDINATOR
============================

Integrates the multi-wallet system with the live trading engine.

This is the main entry point for executing trades across multiple wallets
with proper stealth features.

Features:
- Random wallet selection for each trade
- Dynamic Jito tips based on conviction
- Automatic harvest scheduling
- Position tracking per wallet
- Telegram alerts for harvest readiness

Usage in master_v2.py:
    from core.stealth_trading_coordinator import StealthTradingCoordinator
    
    coordinator = StealthTradingCoordinator(helius_key, notifier)
    
    # When processing a signal:
    result = coordinator.execute_buy(signal)

Author: Trading Bot System
"""

from __future__ import annotations

import os
import json
import time
import random
import base64
import base58
import threading
import requests
import logging
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict

logger = logging.getLogger("StealthCoordinator")

# Import multi-wallet system
from multi_wallet_system import (
    MultiWalletConfig, MultiWalletManager, PositionScaler,
    DynamicJitoTip, HarvestNotifier, LAMPORTS_PER_SOL,
    JITO_TIP_ACCOUNTS, JITO_BUNDLE_URL
)

# Solana imports
try:
    from solders.keypair import Keypair
    from solders.pubkey import Pubkey
    from solders.transaction import VersionedTransaction, Transaction
    from solders.message import Message
    from solders.system_program import transfer, TransferParams
    from solana.rpc.api import Client as SolanaClient
    SOLANA_AVAILABLE = True
except ImportError:
    SOLANA_AVAILABLE = False


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class StealthConfig:
    """Configuration for stealth trading"""

    # Wallet rollout
    num_hot_wallets: int = 1
    
    # Position sizing
    position_size_sol: float = 0.3
    max_positions_per_wallet: int = 40
    total_max_positions: int = 200
    
    # Wallet thresholds
    min_wallet_sol: float = 3.0
    max_wallet_sol: float = 21.0
    harvest_threshold_sol: float = 21.0
    burner_harvest_threshold_sol: float = 50.0
    
    # Jito tips (conviction-based)
    min_jito_tip_lamports: int = 500_000       # 0.0005 SOL
    max_jito_tip_lamports: int = 5_000_000     # 0.005 SOL
    
    # Trading params
    default_slippage_bps: int = 200
    min_conviction: int = 60
    min_liquidity_usd: float = 10000
    blocked_hours_utc: List[int] = None
    
    # Exit params
    stop_loss_pct: float = -0.15
    take_profit_pct: float = 0.30
    trailing_stop_pct: float = 0.10
    max_hold_hours: int = 12
    
    def __post_init__(self):
        if self.blocked_hours_utc is None:
            self.blocked_hours_utc = [1, 3, 5, 19, 23]


# =============================================================================
# JITO BUNDLE EXECUTOR
# =============================================================================

class JitoBundleExecutor:
    """Executes trades via Jito bundles with dynamic tips"""
    
    def __init__(self, solana_client: Optional["SolanaClient"]):
        self.solana_client = solana_client
        self._tip_index = 0
    
    def get_tip_account(self) -> str:
        """Rotate through Jito tip accounts"""
        account = JITO_TIP_ACCOUNTS[self._tip_index]
        self._tip_index = (self._tip_index + 1) % len(JITO_TIP_ACCOUNTS)
        return account
    
    def execute_with_jito(
        self, 
        keypair: Keypair, 
        swap_tx_base64: str,
        tip_lamports: int
    ) -> Optional[str]:
        """
        Execute a swap transaction via Jito bundle with tip.
        
        Args:
            keypair: Wallet keypair for signing
            swap_tx_base64: Base64 encoded swap transaction from Jupiter
            tip_lamports: Jito tip amount in lamports
        
        Returns:
            Transaction signature if successful
        """
        if not self.solana_client:
            logger.warning("Jito execution skipped: Solana RPC client unavailable")
            return None
        
        try:
            # 1. Decode swap transaction
            swap_tx_bytes = base64.b64decode(swap_tx_base64)
            swap_tx = VersionedTransaction.from_bytes(swap_tx_bytes)
            
            # 2. Get blockhash for tip transaction
            blockhash_resp = self.solana_client.get_latest_blockhash()
            recent_blockhash = blockhash_resp.value.blockhash
            
            # 3. Create tip transaction
            tip_account = Pubkey.from_string(self.get_tip_account())
            tip_ix = transfer(
                TransferParams(
                    from_pubkey=keypair.pubkey(),
                    to_pubkey=tip_account,
                    lamports=tip_lamports
                )
            )
            
            tip_message = Message.new_with_blockhash(
                [tip_ix],
                keypair.pubkey(),
                recent_blockhash
            )
            tip_tx = Transaction.new_unsigned(tip_message)
            tip_tx.sign([keypair], recent_blockhash)
            
            # 4. Sign swap transaction
            swap_tx.sign([keypair])
            
            # 5. Encode both for bundle
            signed_swap = base64.b64encode(bytes(swap_tx)).decode('utf-8')
            signed_tip = base64.b64encode(bytes(tip_tx)).decode('utf-8')
            
            # 6. Submit bundle
            bundle_id = self._submit_bundle([signed_swap, signed_tip])
            
            if bundle_id:
                logger.info(f"📦 Jito bundle: {bundle_id[:16]}... | "
                           f"Tip: {tip_lamports/LAMPORTS_PER_SOL:.4f} SOL")
                
                # Wait for confirmation
                for _ in range(30):
                    time.sleep(1)
                    status = self._get_bundle_status(bundle_id)
                    
                    if status in ['landed', 'finalized']:
                        return str(swap_tx.signatures[0])
                    elif status in ['invalid', 'failed']:
                        logger.error(f"Bundle failed: {status}")
                        break
                
                # Check on-chain as fallback
                return str(swap_tx.signatures[0])
            
        except Exception as e:
            logger.error(f"Jito execution error: {e}")
        
        return None
    
    def _submit_bundle(self, transactions: List[str]) -> Optional[str]:
        """Submit bundle to Jito"""
        try:
            payload = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "sendBundle",
                "params": [transactions]
            }
            
            resp = requests.post(
                JITO_BUNDLE_URL,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            
            if resp.status_code == 200:
                result = resp.json()
                if 'result' in result:
                    return result['result']
                elif 'error' in result:
                    logger.error(f"Jito error: {result['error']}")
        except Exception as e:
            logger.error(f"Bundle submission failed: {e}")
        
        return None
    
    def _get_bundle_status(self, bundle_id: str) -> Optional[str]:
        """Check bundle status"""
        try:
            payload = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "getBundleStatuses",
                "params": [[bundle_id]]
            }
            
            resp = requests.post(
                JITO_BUNDLE_URL,
                json=payload,
                timeout=10
            )
            
            if resp.status_code == 200:
                result = resp.json()
                statuses = result.get('result', {}).get('value', [])
                if statuses:
                    status = statuses[0].get('confirmation_status') or statuses[0].get('status')
                    return status.lower() if status else None
        except:
            pass
        
        return None


# =============================================================================
# JUPITER INTEGRATION
# =============================================================================

class JupiterClient:
    """Client for Jupiter swap API"""
    
    QUOTE_URL = "https://quote-api.jup.ag/v6/quote"
    SWAP_URL = "https://quote-api.jup.ag/v6/swap"
    
    def __init__(self):
        self.session = requests.Session()
    
    def get_quote(
        self, 
        input_mint: str, 
        output_mint: str,
        amount: int,
        slippage_bps: int = 200
    ) -> Optional[Dict]:
        """Get swap quote from Jupiter"""
        try:
            params = {
                'inputMint': input_mint,
                'outputMint': output_mint,
                'amount': str(amount),
                'slippageBps': slippage_bps,
                'onlyDirectRoutes': 'false',
                'asLegacyTransaction': 'false'
            }
            
            resp = self.session.get(self.QUOTE_URL, params=params, timeout=10)
            
            if resp.status_code == 200:
                return resp.json()
            else:
                logger.error(f"Jupiter quote error: {resp.status_code}")
                
        except Exception as e:
            logger.error(f"Jupiter quote failed: {e}")
        
        return None
    
    def get_swap_transaction(
        self, 
        quote: Dict, 
        user_public_key: str,
        priority_fee: int = 0
    ) -> Optional[str]:
        """Get swap transaction from Jupiter"""
        try:
            payload = {
                'quoteResponse': quote,
                'userPublicKey': user_public_key,
                'wrapAndUnwrapSol': True,
                'dynamicComputeUnitLimit': True,
                'prioritizationFeeLamports': priority_fee
            }
            
            resp = self.session.post(self.SWAP_URL, json=payload, timeout=30)
            
            if resp.status_code == 200:
                data = resp.json()
                return data.get('swapTransaction')
            else:
                logger.error(f"Jupiter swap error: {resp.status_code}")
                
        except Exception as e:
            logger.error(f"Jupiter swap failed: {e}")
        
        return None


# =============================================================================
# STEALTH TRADING COORDINATOR
# =============================================================================

class StealthTradingCoordinator:
    """
    Main coordinator for stealth multi-wallet trading.
    
    Integrates:
    - Multi-wallet manager (wallet selection)
    - Dynamic Jito tips (conviction-based)
    - Jupiter swaps (DEX aggregator)
    - Harvest management (profit extraction)
    """
    
    SOL_MINT = "So11111111111111111111111111111111111111112"
    
    def __init__(
        self,
        helius_key: str,
        telegram_token: str = None,
        telegram_chat_id: str = None,
        config: StealthConfig = None
    ):
        self.config = config or StealthConfig()
        self.helius_key = helius_key
        
        # RPC client
        self.rpc_url = f"https://mainnet.helius-rpc.com/?api-key={helius_key}"
        self.solana_client = SolanaClient(self.rpc_url) if SOLANA_AVAILABLE else None
        
        # Initialize multi-wallet manager
        multi_config = MultiWalletConfig(
            num_hot_wallets=self.config.num_hot_wallets,
            position_size_sol=self.config.position_size_sol,
            max_positions_per_wallet=self.config.max_positions_per_wallet,
            total_max_positions=self.config.total_max_positions,
            min_wallet_sol=self.config.min_wallet_sol,
            max_wallet_sol=self.config.max_wallet_sol,
            harvest_threshold_sol=self.config.harvest_threshold_sol,
            min_jito_tip_lamports=self.config.min_jito_tip_lamports,
            max_jito_tip_lamports=self.config.max_jito_tip_lamports
        )
        self.wallet_manager = MultiWalletManager(multi_config, helius_key)
        
        # Jito executor
        self.jito = JitoBundleExecutor(self.solana_client)
        
        # Jupiter client
        self.jupiter = JupiterClient()
        
        # Notifier
        self.notifier = HarvestNotifier(telegram_token, telegram_chat_id) if telegram_token else None
        
        # Tip calculator
        self.tip_calc = DynamicJitoTip(multi_config)
        
        # Position scaler
        self.scaler = PositionScaler()
        
        # Thread for background tasks
        self._stop_event = threading.Event()
        self._background_thread = None
        
        logger.info("🎭 Stealth Trading Coordinator initialized")
        logger.info(f"   Position size: {self.config.position_size_sol} SOL")
        logger.info(f"   Max positions: {self.config.total_max_positions}")
    
    def load_wallet_keys(self, secrets: Dict[str, str]):
        """Load only operator-provided wallets from Secrets Manager."""
        loaded_ids = []
        for i in range(1, 6):
            key_name = f'HOT_WALLET_{i}'
            burner_name = f'BURNER_ADDRESS_{i}'

            raw_key: Any = secrets.get(key_name)
            if raw_key is None:
                continue
            if not isinstance(raw_key, str):
                logger.warning(f"  ⚠️ Skipping wallet {i}: {key_name} must be a string")
                continue

            private_key = raw_key.strip()
            if not private_key:
                continue

            burner_raw = secrets.get(burner_name, '')
            burner_address = burner_raw.strip() if isinstance(burner_raw, str) else ''

            try:
                wallet = self.wallet_manager.upsert_wallet_from_secret(
                    wallet_id=i,
                    private_key=private_key,
                    burner_address=burner_address
                )
                loaded_ids.append(i)
                logger.info(f"  ✅ Loaded hot wallet {i}: {wallet.public_key[:8]}...")

            except Exception as e:
                logger.error(f"  ❌ Failed to load wallet {i}: {e}")

        if not loaded_ids:
            logger.error("❌ No HOT_WALLET_n secrets loaded; stealth trading will be disabled")
            self.wallet_manager.hot_wallets = {}
            self.wallet_manager.set_active_wallet_ids([])
            self.config.num_hot_wallets = 0
            self.wallet_manager.config.num_hot_wallets = 0
            return

        self.config.num_hot_wallets = len(loaded_ids)
        self.wallet_manager.config.num_hot_wallets = len(loaded_ids)
        self.wallet_manager.set_active_wallet_ids(loaded_ids)

        # Keep in-memory registry restricted to the active wallet IDs only
        self.wallet_manager.hot_wallets = {
            wid: wallet for wid, wallet in self.wallet_manager.hot_wallets.items()
            if wid in set(loaded_ids)
        }
        self.wallet_manager.wallet_positions = {
            wid: self.wallet_manager.wallet_positions.get(wid, [])
            for wid in loaded_ids
        }

        logger.info(f"✅ Active hot wallets from Secrets Manager: {sorted(loaded_ids)}")
    
    def can_trade(self, signal: Dict) -> Tuple[bool, str]:
        """Check if we can execute a trade"""
        # Check conviction
        conviction = signal.get('conviction_score', signal.get('conviction', 0))
        if conviction < self.config.min_conviction:
            return False, f"Conviction {conviction} < {self.config.min_conviction}"
        
        # Check liquidity
        liquidity = signal.get('liquidity', 0)
        if liquidity < self.config.min_liquidity_usd:
            return False, f"Liquidity ${liquidity:,.0f} < ${self.config.min_liquidity_usd:,.0f}"
        
        # Check blocked hours
        current_hour = datetime.now(timezone.utc).hour
        if current_hour in self.config.blocked_hours_utc:
            return False, f"Hour {current_hour} UTC is blocked"
        
        # Check total position limit
        total_positions = sum(
            len(self.wallet_manager.get_wallet_positions(i))
            for i in self.wallet_manager.hot_wallets.keys()
        )
        if total_positions >= self.config.total_max_positions:
            return False, f"Total position limit reached ({total_positions})"
        
        return True, "OK"
    
    def execute_buy(self, signal: Dict) -> Dict:
        """
        Execute a buy order across the multi-wallet system.
        
        Workflow:
        1. Check if we can trade
        2. Select random wallet based on capacity
        3. Calculate dynamic Jito tip
        4. Get Jupiter quote
        5. Execute via Jito bundle
        6. Record position
        
        Returns execution result dict.
        """
        token_address = signal.get('token_address')
        token_symbol = signal.get('token_symbol', 'UNKNOWN')
        conviction = signal.get('conviction_score', signal.get('conviction', 60))
        
        result = {
            'success': False,
            'action': 'BUY',
            'token_address': token_address,
            'token_symbol': token_symbol,
            'wallet_id': None,
            'signature': None,
            'jito_tip_sol': 0,
            'error': None
        }
        
        # Pre-checks
        can_trade, reason = self.can_trade(signal)
        if not can_trade:
            result['error'] = reason
            logger.info(f"❌ Trade rejected: {reason}")
            return result
        
        # Select wallet
        selection = self.wallet_manager.select_wallet_for_trade(conviction)
        if not selection:
            result['error'] = "No wallet available"
            logger.warning("❌ No wallet available for trade")
            return result
        
        wallet_id, wallet = selection
        result['wallet_id'] = wallet_id
        
        # Calculate Jito tip
        jito_tip = self.tip_calc.calculate_tip(conviction)
        result['jito_tip_sol'] = jito_tip / LAMPORTS_PER_SOL
        
        try:
            # Get Jupiter quote
            amount_lamports = int(self.config.position_size_sol * LAMPORTS_PER_SOL)
            
            quote = self.jupiter.get_quote(
                input_mint=self.SOL_MINT,
                output_mint=token_address,
                amount=amount_lamports,
                slippage_bps=self.config.default_slippage_bps
            )
            
            if not quote:
                result['error'] = "Failed to get quote"
                return result
            
            # Get swap transaction
            swap_tx = self.jupiter.get_swap_transaction(
                quote=quote,
                user_public_key=wallet.public_key
            )
            
            if not swap_tx:
                result['error'] = "Failed to get swap transaction"
                return result
            
            # Execute via Jito
            signature = self.jito.execute_with_jito(
                keypair=wallet.keypair,
                swap_tx_base64=swap_tx,
                tip_lamports=jito_tip
            )
            
            if not signature:
                result['error'] = "Transaction failed"
                return result
            
            result['signature'] = signature
            
            # Calculate tokens received
            out_amount = int(quote.get('outAmount', 0))
            # Assume 6 decimals for most memecoins
            tokens_received = out_amount / (10 ** 6)
            
            # Get entry price
            in_amount = int(quote.get('inAmount', amount_lamports))
            entry_price = (in_amount / LAMPORTS_PER_SOL) / tokens_received if tokens_received > 0 else 0
            
            # Record position
            position_id = self.wallet_manager.record_trade_entry(
                wallet_id=wallet_id,
                token_address=token_address,
                token_symbol=token_symbol,
                entry_price=entry_price,
                tokens_held=tokens_received,
                conviction_score=conviction
            )
            
            result['success'] = True
            result['position_id'] = position_id
            result['tokens_received'] = tokens_received
            result['entry_price'] = entry_price
            
            logger.info(f"✅ BUY | Wallet {wallet_id} | {token_symbol} | "
                       f"{self.config.position_size_sol} SOL | "
                       f"Tip: {jito_tip/LAMPORTS_PER_SOL:.4f} | "
                       f"{signature[:16]}...")
            
            return result
            
        except Exception as e:
            result['error'] = str(e)
            logger.error(f"Buy execution error: {e}")
            return result
    
    def execute_sell(self, position: Dict, exit_reason: str = "MANUAL") -> Dict:
        """
        Execute a sell order for a position.
        
        Args:
            position: Position data from database
            exit_reason: Reason for exit (STOP_LOSS, TAKE_PROFIT, etc.)
        
        Returns execution result dict.
        """
        wallet_id = position.get('wallet_id')
        token_address = position.get('token_address')
        token_symbol = position.get('token_symbol', 'UNKNOWN')
        
        result = {
            'success': False,
            'action': 'SELL',
            'wallet_id': wallet_id,
            'token_address': token_address,
            'token_symbol': token_symbol,
            'exit_reason': exit_reason,
            'signature': None,
            'pnl_sol': 0,
            'error': None
        }
        
        if wallet_id not in self.wallet_manager.hot_wallets:
            result['error'] = f"Wallet {wallet_id} not found"
            return result
        
        wallet = self.wallet_manager.hot_wallets[wallet_id]
        
        # Use medium tip for sells (we want them to land)
        jito_tip = self.config.min_jito_tip_lamports + \
                  (self.config.max_jito_tip_lamports - self.config.min_jito_tip_lamports) // 2
        
        try:
            # Get token balance
            tokens_held = position.get('tokens_held', 0)
            token_amount = int(tokens_held * (10 ** 6))  # Assume 6 decimals
            
            # Get quote for sell
            quote = self.jupiter.get_quote(
                input_mint=token_address,
                output_mint=self.SOL_MINT,
                amount=token_amount,
                slippage_bps=self.config.default_slippage_bps
            )
            
            if not quote:
                result['error'] = "Failed to get sell quote"
                return result
            
            # Get swap transaction
            swap_tx = self.jupiter.get_swap_transaction(
                quote=quote,
                user_public_key=wallet.public_key
            )
            
            if not swap_tx:
                result['error'] = "Failed to get sell transaction"
                return result
            
            # Execute via Jito
            signature = self.jito.execute_with_jito(
                keypair=wallet.keypair,
                swap_tx_base64=swap_tx,
                tip_lamports=jito_tip
            )
            
            if not signature:
                result['error'] = "Sell transaction failed"
                return result
            
            result['signature'] = signature
            
            # Calculate PnL
            out_amount = int(quote.get('outAmount', 0))
            sol_received = out_amount / LAMPORTS_PER_SOL
            cost = position.get('position_size_sol', self.config.position_size_sol)
            pnl = sol_received - cost
            
            # Record exit
            self.wallet_manager.record_trade_exit(
                position_id=position.get('id'),
                exit_price=sol_received / tokens_held if tokens_held > 0 else 0,
                pnl_sol=pnl
            )
            
            result['success'] = True
            result['pnl_sol'] = pnl
            result['sol_received'] = sol_received
            
            logger.info(f"✅ SELL | Wallet {wallet_id} | {token_symbol} | "
                       f"PnL: {pnl:+.4f} SOL | {exit_reason} | "
                       f"{signature[:16]}...")
            
            return result
            
        except Exception as e:
            result['error'] = str(e)
            logger.error(f"Sell execution error: {e}")
            return result
    
    def check_and_harvest(self):
        """Check wallets and execute harvests if needed"""
        harvests = self.wallet_manager.check_harvest_needed()
        
        for harvest in harvests:
            wallet_id = harvest['wallet_id']
            amount = harvest['harvest_amount']
            burner = harvest['burner_address']
            
            if not burner:
                logger.warning(f"Wallet {wallet_id} has no burner configured")
                continue
            
            # Random delay before harvest (anti-pattern)
            delay_minutes = random.randint(5, 120)
            logger.info(f"💸 Scheduling harvest: {amount:.2f} SOL from wallet {wallet_id} "
                       f"in {delay_minutes} minutes")
            
            # In production, you'd schedule this for later
            # For now, execute with random delay simulation
            def _delayed_harvest():
                time.sleep(delay_minutes * 60)
                sig = self.wallet_manager.execute_harvest(wallet_id, amount)
                if sig and self.notifier:
                    self.notifier.notify_hot_wallet_harvest(wallet_id, amount, burner)
            
            threading.Thread(target=_delayed_harvest, daemon=True).start()
    
    def get_status(self) -> Dict:
        """Get coordinator status"""
        return self.wallet_manager.get_total_stats()
    
    def print_status(self):
        """Print formatted status"""
        self.wallet_manager.print_status()
    
    def start_background_tasks(self):
        """Start background monitoring"""
        def _monitor_loop():
            while not self._stop_event.is_set():
                try:
                    # Check for harvests every 15 minutes
                    self.check_and_harvest()
                except Exception as e:
                    logger.error(f"Background task error: {e}")
                
                self._stop_event.wait(900)  # 15 minutes
        
        self._background_thread = threading.Thread(target=_monitor_loop, daemon=True)
        self._background_thread.start()
        logger.info("🔄 Background monitoring started")
    
    def stop(self):
        """Stop background tasks"""
        self._stop_event.set()


# =============================================================================
# CLI
# =============================================================================

def main():
    """CLI for testing stealth coordinator"""
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
    
    parser = argparse.ArgumentParser(description="Stealth Trading Coordinator")
    parser.add_argument('command', choices=['status', 'test-tip', 'test-scaling'],
                       help='Command to run')
    
    args = parser.parse_args()
    
    helius_key = get_secret('HELIUS_KEY')
    telegram_token = get_secret('TELEGRAM_BOT_TOKEN')
    telegram_chat_id = get_secret('TELEGRAM_CHAT_ID')
    
    if args.command in ['status']:
        if not helius_key:
            print("❌ HELIUS_KEY required")
            return
        
        coordinator = StealthTradingCoordinator(
            helius_key=helius_key,
            telegram_token=telegram_token,
            telegram_chat_id=telegram_chat_id
        )
        coordinator.print_status()
    
    elif args.command == 'test-tip':
        print("\n📈 CONVICTION → JITO TIP MAPPING")
        print("=" * 50)
        
        config = MultiWalletConfig()
        calc = DynamicJitoTip(config)
        
        for conv in [50, 60, 70, 80, 90, 95, 100]:
            tip = calc.calculate_tip(conv)
            sol = tip / LAMPORTS_PER_SOL
            print(f"   Conviction {conv:3d}% → {sol:.4f} SOL tip")
    
    elif args.command == 'test-scaling':
        print("\n📊 BALANCE → MAX POSITIONS SCALING")
        print("=" * 50)
        
        for bal in [2, 3, 6, 9, 12, 15, 18, 21, 25]:
            max_pos = PositionScaler.get_max_positions(bal)
            trading = PositionScaler.get_trading_capital(bal)
            can, _ = PositionScaler.can_open_position(bal, 0, 0.3)
            print(f"   {bal:5.1f} SOL → {max_pos:2d} positions | "
                  f"Trading: {trading:.1f} SOL | "
                  f"{'✅' if can else '❌'}")


if __name__ == "__main__":
    main()
