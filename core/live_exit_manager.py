"""
LIVE EXIT MANAGER - Uses Existing LiveTradingEngine
====================================================

PURPOSE:
    Manages exit conditions for live trades using the EXISTING 
    LiveTradingEngine from core/live_trading_engine.py.

    This is a thin layer that adds:
    - Background monitoring for exit conditions
    - Stop Loss, Take Profit, Trailing Stop, Time Stop detection
    - Calls existing engine.execute_sell() for actual execution

INTEGRATION:
    Works with your existing:
    - core/live_trading_engine.py (LiveTradingEngine)
    - core/secrets_manager.py (get_secret)
    - live_trades_tax.db (via engine.tax_db)

AUTHOR: Trading Bot System
VERSION: 2.0.0
"""

import logging
import threading
import time
import json
import asyncio
import requests
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

try:
    import websockets
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    websockets = None
    WEBSOCKETS_AVAILABLE = False

logger = logging.getLogger(__name__)


class ExitReason(Enum):
    """Exit reasons for tracking and analytics"""
    TAKE_PROFIT = "TAKE_PROFIT"
    STOP_LOSS = "STOP_LOSS"
    TRAILING_STOP = "TRAILING_STOP"
    TIME_STOP = "TIME_STOP"
    MANUAL = "MANUAL"
    KILL_SWITCH = "KILL_SWITCH"
    SIGNAL = "SIGNAL"
    LIQUIDITY_DRIED = "LIQUIDITY_DRIED"
    RUG_DETECTED = "RUG_DETECTED"


@dataclass
class ExitConfig:
    """Configuration for exit management"""
    # Default exit thresholds (use engine config if not specified)
    default_stop_loss_pct: float = -15.0      # -15%
    default_take_profit_pct: float = 30.0     # +30%
    default_trailing_stop_pct: float = 10.0   # 10% from peak
    max_hold_hours: int = 12
    
    # Monitoring
    price_check_interval_seconds: int = 2
    enable_auto_exits: bool = True

     # Websocket monitoring
    enable_websocket: bool = False
    websocket_ping_seconds: int = 30
    websocket_reconnect_seconds: int = 5
    
    # Safety
    min_liquidity_exit_usd: float = 5000.0
    max_slippage_pct: float = 3.0
    
    # Notifications
    enable_notifications: bool = True


class LiveExitManager:
    """
    Manages exit conditions using the existing LiveTradingEngine.
    
    This class:
    1. Monitors all open positions from engine.tax_db
    2. Checks exit conditions (TP/SL/Trailing/Time)
    3. Calls engine.execute_sell() for exits
    """
    
    def __init__(self, 
                 trading_engine,  # Existing LiveTradingEngine instance
                 config: ExitConfig = None,
                 notifier = None,
                 enable_websocket: bool = False,
                 helius_ws_url: str = None,
                 websocket_ping_seconds: int = None,
                 websocket_reconnect_seconds: int = None):
        """
        Initialize exit manager.
        
        Args:
            trading_engine: Your existing LiveTradingEngine instance
            config: Exit configuration (optional)
            notifier: Telegram notifier (optional)
        """
        self.engine = trading_engine
        self.config = config or ExitConfig()
        self.notifier = notifier
        self.config.enable_websocket = enable_websocket
        if self.config.enable_websocket and not WEBSOCKETS_AVAILABLE:
            logger.warning(
                "⚠️ websockets package not installed - disabling websocket exit monitoring"
            )
            self.config.enable_websocket = False
        if websocket_ping_seconds is not None:
            self.config.websocket_ping_seconds = websocket_ping_seconds
        if websocket_reconnect_seconds is not None:
            self.config.websocket_reconnect_seconds = websocket_reconnect_seconds
        self.helius_ws_url = helius_ws_url
        
        # Use engine's tax_db for position tracking
        self.tax_db = trading_engine.tax_db
        
        # Thread management
        self._monitor_running = False
        self._monitor_thread = None
        self._ws_thread = None
        self._ws_stop = threading.Event()
        self._wake_event = threading.Event()
        self._vault_refresh = threading.Event()
        self._last_no_vault_log = None
        self._lock = threading.RLock()
        
        # State tracking
        self._consecutive_failures = 0
        self._last_check_time = None
        self._last_liquidity = {}  # Cache liquidity per token

        self._exit_in_flight = set()  # token addresses currently being exited
        self._exit_retry_after = {}   # token_address -> unix timestamp
        self._exit_failure_counts = {}
        
        logger.info("🔄 LiveExitManager initialized (using existing engine)")
    
    # =========================================================================
    # PRICE FETCHING
    # =========================================================================
    
    def get_token_price(self, token_address: str) -> Optional[float]:
        """Get current token price in USD from DexScreener"""
        try:
            url = f"https://api.dexscreener.com/latest/dex/tokens/{token_address}"
            resp = requests.get(url, timeout=10)
            
            if resp.status_code == 200:
                data = resp.json()
                pairs = data.get('pairs', [])
                
                if pairs:
                    # Get highest liquidity pair
                    best_pair = max(pairs, key=lambda p: float(p.get('liquidity', {}).get('usd', 0) or 0))
                    price = float(best_pair.get('priceUsd', 0) or 0)
                    liquidity = float(best_pair.get('liquidity', {}).get('usd', 0) or 0)
                    
                    # Cache liquidity
                    self._last_liquidity[token_address] = liquidity
                    return price
                    
        except Exception as e:
            logger.debug(f"Price fetch error for {token_address[:8]}: {e}")
        
        return None
    
    def get_token_liquidity(self, token_address: str) -> float:
        """Get cached or fresh liquidity in USD"""
        # Try cached value first
        if token_address in self._last_liquidity:
            return self._last_liquidity[token_address]
        
        # Fetch fresh
        self.get_token_price(token_address)
        return self._last_liquidity.get(token_address, 0)
    
    # =========================================================================
    # EXIT CONDITION CHECKING
    # =========================================================================
    
    def check_exit_conditions(self, position: Dict) -> Tuple[bool, Optional[ExitReason], Dict]:
        """
        Check if a position should be exited.
        
        Args:
            position: Position dict from tax_db.get_positions()
        
        Returns:
            (should_exit, reason, metrics_dict)
        """
        token_address = position.get('token_address')
        token_symbol = position.get('token_symbol', 'UNKNOWN')
        
        # Get position parameters (use position values or defaults)
        entry_price = position.get('entry_price_usd', 0)
        peak_price = position.get('peak_price_usd', entry_price)
        stop_loss_pct = position.get('stop_loss_pct', self.config.default_stop_loss_pct)
        take_profit_pct = position.get('take_profit_pct', self.config.default_take_profit_pct)
        trailing_stop_pct = position.get('trailing_stop_pct', self.config.default_trailing_stop_pct)
        max_hold_hours = position.get('max_hold_hours', self.config.max_hold_hours)
        
        # Normalize percentages (handle both -0.15 and -15 formats)
        if stop_loss_pct < -1:  # Likely -15 format
            stop_loss_pct = stop_loss_pct / 100
        if take_profit_pct > 1:  # Likely 30 format
            take_profit_pct = take_profit_pct / 100
        if trailing_stop_pct > 1:  # Likely 10 format
            trailing_stop_pct = trailing_stop_pct / 100
        
        # Build metrics dict
        metrics = {
            'token_symbol': token_symbol,
            'entry_price': entry_price,
            'stop_loss_pct': stop_loss_pct,
            'take_profit_pct': take_profit_pct,
            'trailing_stop_pct': trailing_stop_pct
        }
        
        # Calculate hold time
        entry_time_str = position.get('entry_time', '')
        try:
            if entry_time_str:
                entry_time = datetime.fromisoformat(entry_time_str.replace('Z', '+00:00'))
            else:
                entry_time = datetime.now(timezone.utc) - timedelta(hours=1)
        except:
            entry_time = datetime.now(timezone.utc) - timedelta(hours=1)
        
        hold_hours = (datetime.now(timezone.utc) - entry_time).total_seconds() / 3600
        metrics['hold_hours'] = hold_hours
        
        # 1. TIME STOP - Check first as it doesn't need price
        if hold_hours >= max_hold_hours:
            logger.info(f"⏰ TIME_STOP triggered for {token_symbol}: {hold_hours:.1f}h >= {max_hold_hours}h")
            return True, ExitReason.TIME_STOP, metrics
        
        # 2. Get current price
        current_price = self.get_token_price(token_address)
        
        if not current_price or current_price <= 0:
            # Can't get price - only exit on time
            if hold_hours >= max_hold_hours:
                return True, ExitReason.TIME_STOP, metrics
            return False, None, metrics
        
        metrics['current_price'] = current_price
        
        # 3. Check for low liquidity
        liquidity = self._last_liquidity.get(token_address, 0)
        metrics['liquidity_usd'] = liquidity
        
        if liquidity < self.config.min_liquidity_exit_usd:
            logger.warning(f"⚠️ Low liquidity for {token_symbol}: ${liquidity:.0f}")
            return True, ExitReason.LIQUIDITY_DRIED, metrics
        
        # 4. Calculate P&L
        if entry_price and entry_price > 0:
            pnl_pct = (current_price / entry_price) - 1  # Decimal format (-0.15 for -15%)
        else:
            pnl_pct = 0
        
        metrics['pnl_pct'] = pnl_pct * 100  # Store as percentage for display
        
        # 5. Update peak price
        new_peak = max(peak_price, current_price)
        if new_peak > peak_price:
            if hasattr(self.tax_db, "update_position_peak"):
                self.tax_db.update_position_peak(token_address, new_peak)
            peak_price = new_peak
        
        metrics['peak_price'] = peak_price
        
        # 6. Calculate drop from peak
        if peak_price and peak_price > 0:
            from_peak_pct = (current_price / peak_price) - 1  # Decimal format
        else:
            from_peak_pct = 0
        
        metrics['from_peak_pct'] = from_peak_pct * 100  # Percentage for display
        
        # === EXIT CONDITION CHECKS ===
        
        # STOP LOSS - Highest priority
        if pnl_pct <= stop_loss_pct:
            logger.info(f"🛑 STOP_LOSS triggered for {token_symbol}: {pnl_pct*100:.1f}% <= {stop_loss_pct*100:.1f}%")
            return True, ExitReason.STOP_LOSS, metrics
        
        # TAKE PROFIT
        if pnl_pct >= take_profit_pct:
            logger.info(f"🎯 TAKE_PROFIT triggered for {token_symbol}: {pnl_pct*100:.1f}% >= {take_profit_pct*100:.1f}%")
            return True, ExitReason.TAKE_PROFIT, metrics
        
        # TRAILING STOP (only if we've been in profit)
        if pnl_pct > 0 and from_peak_pct <= -trailing_stop_pct:
            logger.info(f"📉 TRAILING_STOP triggered for {token_symbol}: {from_peak_pct*100:.1f}% from peak")
            return True, ExitReason.TRAILING_STOP, metrics
        
        # No exit condition met
        return False, None, metrics
    
    # =========================================================================
    # EXIT EXECUTION
    # =========================================================================

    def _check_exit_throttle(self, token_address: str) -> Tuple[bool, Optional[str]]:
        """Check whether this token can attempt a new exit now."""
        now = time.time()
        with self._lock:
            if token_address in self._exit_in_flight:
                return False, "exit already in progress"

            retry_after = self._exit_retry_after.get(token_address, 0)
            if retry_after and now < retry_after:
                wait_s = max(1, int(retry_after - now))
                return False, f"cooldown active ({wait_s}s remaining)"

            self._exit_in_flight.add(token_address)
            return True, None

    def _record_exit_result(self, token_address: str, success: bool):
        """Update in-flight and backoff state after an exit attempt."""
        now = time.time()
        with self._lock:
            self._exit_in_flight.discard(token_address)

            if success:
                self._exit_failure_counts.pop(token_address, None)
                self._exit_retry_after.pop(token_address, None)
                return

            failures = self._exit_failure_counts.get(token_address, 0) + 1
            self._exit_failure_counts[token_address] = failures
            backoff_seconds = min(30, 2 ** min(failures, 4))  # 2,4,8,16,30...
            self._exit_retry_after[token_address] = now + backoff_seconds
    
    def execute_exit(self, position: Dict, reason: ExitReason, metrics: Dict = None) -> Dict:
        """
        Execute an exit using the existing engine.execute_sell().
        
        Args:
            position: Position dict
            reason: Exit reason enum
            metrics: Optional metrics dict
        
        Returns:
            Result dict from engine.execute_sell()
        """
        token_address = position.get('token_address')
        token_symbol = position.get('token_symbol', 'UNKNOWN')
        
        result = {
            'success': False,
            'token_address': token_address,
            'token_symbol': token_symbol,
            'exit_reason': reason.value,
            'pnl_sol': 0,
            'pnl_pct': 0,
            'signature': None,
            'error': None,
            'metrics': metrics or {}
        }

        allowed, throttle_reason = self._check_exit_throttle(token_address)
        if not allowed:
            result['error'] = throttle_reason
            logger.info(f"⏭️ Skipping exit for {token_symbol}: {throttle_reason}")
            return result
        
        try:
            logger.info(f"🔄 Executing exit for {token_symbol} | Reason: {reason.value}")
            
            # Use the existing engine's execute_sell method
            sell_result = self.engine.execute_sell(token_address, reason.value)
            
            if sell_result.get('success'):
                result['success'] = True
                result['signature'] = sell_result.get('signature')
                result['pnl_sol'] = sell_result.get('pnl_sol', 0)
                result['pnl_pct'] = sell_result.get('pnl_pct', 0)
                result['sol_received'] = sell_result.get('sol_received', 0)
                
                # Log success
                pnl_emoji = "📈" if result['pnl_sol'] > 0 else "📉"
                logger.info(
                    f"{pnl_emoji} EXIT COMPLETE: {token_symbol} | "
                    f"PnL: {result['pnl_pct']:+.1f}% ({result['pnl_sol']:+.4f} SOL) | "
                    f"Reason: {reason.value}"
                )
                
                # Send notification
                if self.notifier and self.config.enable_notifications:
                    is_win = result['pnl_sol'] > 0
                    emoji = "✅" if is_win else "❌"
                    self.notifier.send(
                        f"{emoji} EXIT: {token_symbol}\n"
                        f"PnL: {result['pnl_pct']:+.1f}%\n"
                        f"Reason: {reason.value}\n"
                        f"SOL: {result['pnl_sol']:+.4f}"
                    )
            else:
                result['error'] = sell_result.get('error', 'Unknown error')
                logger.error(f"❌ Exit failed for {token_symbol}: {result['error']}")
                
        except Exception as e:
            result['error'] = str(e)
            logger.error(f"❌ Exit exception for {token_symbol}: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            self._record_exit_result(token_address, result.get('success', False))
        
        return result
    
    def process_exit_signal(self, signal: Dict) -> Dict:
        """
        Process an exit signal from master_v2.py or strategist.
        
        Signal format:
        {
            'action': 'SELL',
            'token_address': '...',
            'token_symbol': '...',
            'reason': 'TAKE_PROFIT' | 'STOP_LOSS' | 'SIGNAL' | etc
        }
        """
        token_address = signal.get('token_address')
        token_symbol = signal.get('token_symbol', 'UNKNOWN')
        reason_str = signal.get('reason', 'SIGNAL')
        
        # Map reason string to enum
        try:
            reason = ExitReason[reason_str.upper()]
        except KeyError:
            reason = ExitReason.SIGNAL
        
        logger.info(f"📨 Received exit signal for {token_symbol}: {reason.value}")
        
        # Get position from tax_db
        positions = self.tax_db.get_positions()
        position = next((p for p in positions if p['token_address'] == token_address), None)
        
        if not position:
            return {
                'success': False,
                'error': f"Position not found for {token_symbol}",
                'token_address': token_address
            }
        
        return self.execute_exit(position, reason)
    
    # =========================================================================
    # MONITORING LOOP
    # =========================================================================
    
    def start_monitoring(self):
        """Start the background exit monitoring thread"""
        if self._monitor_running:
            logger.warning("Exit monitor already running")
            return
        
        self._monitor_running = True
        self._monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self._monitor_thread.start()
        logger.info("🔄 Exit monitoring started (checking every %ds)", self.config.price_check_interval_seconds)
        if self.config.enable_websocket and self.helius_ws_url:
            self._ws_stop.clear()
            self._ws_thread = threading.Thread(target=self._run_websocket_loop, daemon=True)
            self._ws_thread.start()
            logger.info("📡 Exit websocket monitoring enabled")
    
    def stop_monitoring(self):
        """Stop the background monitoring"""
        self._monitor_running = False
        self._ws_stop.set()
        if self._monitor_thread:
            self._monitor_thread.join(timeout=10)
        if self._ws_thread:
            self._ws_thread.join(timeout=10)
        logger.info("⏹️ Exit monitoring stopped")
    
    def _monitoring_loop(self):
        """Background loop that checks all positions for exit conditions"""
        while self._monitor_running:
            try:
                self._check_all_positions()
                self._last_check_time = datetime.now(timezone.utc)
                self._consecutive_failures = 0
                
            except Exception as e:
                self._consecutive_failures += 1
                logger.error(f"Monitor loop error ({self._consecutive_failures}): {e}")
                
                if self._consecutive_failures >= 5:
                    logger.critical("Too many consecutive monitoring failures!")
            
            # Wait for next check or websocket trigger
            self._wake_event.wait(timeout=self.config.price_check_interval_seconds)
            self._wake_event.clear()

    def _run_websocket_loop(self):
        try:
            asyncio.run(self._websocket_loop())
        except Exception as exc:
            logger.warning(f"Exit websocket loop error: {exc}")

    def refresh_websocket_subscriptions(self):
        """Trigger websocket subscription refresh and wake exit checks."""
        self._vault_refresh.set()
        self._wake_event.set()

    def _get_vault_addresses(self) -> List[str]:
        positions = self.tax_db.get_positions()
        vaults = []
        for pos in positions:
            for key in ("liquidity_pool_vault", "pool_vault_address", "vault_address"):
                vault = pos.get(key)
                if vault:
                    vaults.append(vault)
                    break
        return list({v for v in vaults if v})

    async def _websocket_loop(self):
        while not self._ws_stop.is_set():
            vaults = self._get_vault_addresses()
            if not vaults:
                now = time.time()
                if self._last_no_vault_log is None or now - self._last_no_vault_log >= 60:
                    logger.info("Exit websocket: no vault addresses found in open live positions; sleeping.")
                    self._last_no_vault_log = now
                await asyncio.sleep(self.config.websocket_reconnect_seconds)
                continue

            try:
                async with websockets.connect(self.helius_ws_url) as websocket:
                    for idx, vault in enumerate(vaults, start=1):
                        subscribe_msg = {
                            "jsonrpc": "2.0",
                            "id": idx,
                            "method": "accountSubscribe",
                            "params": [
                                vault,
                                {"encoding": "jsonParsed", "commitment": "confirmed"}
                            ],
                        }
                        await websocket.send(json.dumps(subscribe_msg))

                    while not self._ws_stop.is_set():
                        try:
                            message = await asyncio.wait_for(
                                websocket.recv(),
                                timeout=self.config.websocket_ping_seconds,
                            )
                            data = json.loads(message)
                            if "params" in data:
                                self._wake_event.set()
                            if self._vault_refresh.is_set():
                                self._vault_refresh.clear()
                                break
                        except asyncio.TimeoutError:
                            await websocket.ping()
            except Exception as exc:
                logger.warning(f"Exit websocket reconnecting after error: {exc}")
                await asyncio.sleep(self.config.websocket_reconnect_seconds)

    
    def _check_all_positions(self):
        """Check all open positions for exit conditions"""
        if not self.config.enable_auto_exits:
            return
        
        positions = self.tax_db.get_positions()
        
        if not positions:
            return
        
        logger.debug(f"Checking {len(positions)} open positions...")
        
        for position in positions:
            token_address = position.get('token_address')
            token_symbol = position.get('token_symbol')
            with self._lock:
                if token_address in self._exit_in_flight:
                    logger.debug(f"Skipping {token_symbol}: exit already in flight")
                    continue

            try:
                should_exit, reason, metrics = self.check_exit_conditions(position)
                
                if should_exit and reason:
                    result = self.execute_exit(position, reason, metrics)
                    
                    if not result.get('success') and result.get('error') not in (
                        'exit already in progress',
                    ) and not str(result.get('error', '')).startswith('cooldown active'):
                        logger.error(
                            f"Failed to exit {position.get('token_symbol')}: "
                            f"{result.get('error')}"
                        )
                        
            except Exception as e:
                logger.error(f"Error checking position {position.get('token_symbol')}: {e}")
    
    def check_exits_now(self) -> List[Dict]:
        """Manual check for exits (backup to automatic monitoring)"""
        results = []
        
        positions = self.tax_db.get_positions()
        
        for position in positions:
            token_address = position.get('token_address')
            with self._lock:
                if token_address in self._exit_in_flight:
                    continue

            try:
                should_exit, reason, metrics = self.check_exit_conditions(position)
                
                if should_exit and reason:
                    result = self.execute_exit(position, reason, metrics)
                    results.append(result)
                    
            except Exception as e:
                logger.error(f"Error in manual exit check: {e}")
        
        return results
    
    # =========================================================================
    # EMERGENCY CONTROLS
    # =========================================================================
    
    def emergency_exit_all(self, reason: str = "KILL_SWITCH") -> List[Dict]:
        """Emergency exit all open positions"""
        logger.critical(f"🚨 EMERGENCY EXIT ALL: {reason}")
        
        results = []
        positions = self.tax_db.get_positions()
        
        for position in positions:
            try:
                result = self.execute_exit(position, ExitReason.KILL_SWITCH)
                results.append(result)
            except Exception as e:
                logger.error(f"Emergency exit failed for {position.get('token_symbol')}: {e}")
                results.append({
                    'success': False,
                    'token_symbol': position.get('token_symbol'),
                    'error': str(e)
                })
        
        return results
    
    def force_exit(self, token_address: str, reason: str = "MANUAL") -> Dict:
        """Force exit a specific position"""
        try:
            exit_reason = ExitReason[reason.upper()]
        except KeyError:
            exit_reason = ExitReason.MANUAL
        
        positions = self.tax_db.get_positions()
        position = next((p for p in positions if p['token_address'] == token_address), None)
        
        if not position:
            return {'success': False, 'error': 'Position not found'}
        
        return self.execute_exit(position, exit_reason)
    
    # =========================================================================
    # STATUS & DIAGNOSTICS
    # =========================================================================
    
    def get_status(self) -> Dict:
        """Get exit manager status"""
        positions = self.tax_db.get_positions()
        
        return {
            'monitor_running': self._monitor_running,
            'last_check': self._last_check_time.isoformat() if self._last_check_time else None,
            'check_interval_seconds': self.config.price_check_interval_seconds,
            'open_positions': len(positions),
            'auto_exits_enabled': self.config.enable_auto_exits,
            'consecutive_failures': self._consecutive_failures,
            'exit_thresholds': {
                'stop_loss_pct': self.config.default_stop_loss_pct,
                'take_profit_pct': self.config.default_take_profit_pct,
                'trailing_stop_pct': self.config.default_trailing_stop_pct,
                'max_hold_hours': self.config.max_hold_hours
            }
        }
    
    def get_position_details(self) -> List[Dict]:
        """Get detailed info on all positions with current metrics"""
        positions = self.tax_db.get_positions()
        details = []
        
        for pos in positions:
            current_price = self.get_token_price(pos['token_address'])
            entry_price = pos.get('entry_price_usd', 0)
            
            pnl_pct = ((current_price / entry_price) - 1) * 100 if entry_price and current_price else 0
            
            details.append({
                'token_symbol': pos.get('token_symbol'),
                'token_address': pos.get('token_address'),
                'tokens_held': pos.get('tokens_held', 0),
                'entry_price': entry_price,
                'current_price': current_price,
                'pnl_pct': pnl_pct,
                'entry_time': pos.get('entry_time'),
                'stop_loss_pct': pos.get('stop_loss_pct', self.config.default_stop_loss_pct),
                'take_profit_pct': pos.get('take_profit_pct', self.config.default_take_profit_pct)
            })
        
        return details
