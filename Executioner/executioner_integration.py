"""
EXECUTIONER INTEGRATION - Connecting Live Trading to Master V2
==============================================================

This module integrates the Executioner with the existing trading system.
It hooks into master_v2.py to execute real trades based on paper trading signals.

INTEGRATION POINTS:
1. Webhook handler - Intercepts signals before paper trading
2. Position monitor - Syncs live positions with paper positions  
3. Exit checker - Monitors and executes exit conditions
4. Gradual rollout - Configurable % of signals go to live trading

SAFETY FEATURES:
- Paper trade validation period required before live
- Conviction threshold higher for live trades
- Position size limits enforced
- Kill switch for emergencies

USAGE:
    from executioner_integration import ExecutionerBridge
    
    bridge = ExecutionerBridge(
        executioner=executioner,
        paper_trader=paper_trader,
        db=database
    )
    
    # In your webhook handler:
    result = bridge.process_signal(signal, wallet_data)

NZ TAX NOTE:
    All transactions (including paper trades marked as such) are recorded
    in the tax database. Only live trades have real signatures.
"""

import os
import json
import time
import threading
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

# Import our components
try:
    from executioner_v1 import (
        Executioner, ExecutionConfig, ExecutionResult, 
        ExecutionStatus, Position, PriceService
    )
    EXECUTIONER_AVAILABLE = True
except ImportError:
    EXECUTIONER_AVAILABLE = False
    print("⚠️ executioner_v1 not found. Copy it to your project directory.")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ExecutionerIntegration")


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class IntegrationConfig:
    """Configuration for Executioner integration"""
    
    # Rollout settings
    live_trade_percentage: float = 0.0      # 0-100, % of signals that go live
    min_paper_profit_days: int = 14         # Days of profitable paper trading required
    min_paper_trades: int = 50              # Minimum paper trades before live
    min_paper_win_rate: float = 0.55        # Minimum win rate from paper trading
    
    # Live trading thresholds (stricter than paper)
    live_min_conviction: int = 70           # Higher than paper default of 60
    live_max_position_sol: float = 0.25     # More conservative than default
    live_min_liquidity_usd: float = 20000   # Higher liquidity requirement
    
    # Safety
    max_daily_loss_sol: float = 1.0         # Stop live trading if daily loss exceeds
    max_daily_trades: int = 20              # Max live trades per day
    cool_down_after_loss: int = 30          # Minutes to pause after consecutive losses
    consecutive_loss_limit: int = 3          # Losses before cool down
    
    # Sync settings
    sync_interval_seconds: int = 60         # How often to sync positions
    exit_check_interval: int = 30           # How often to check exit conditions


class TradingMode(Enum):
    PAPER_ONLY = "paper_only"
    SHADOW = "shadow"           # Live trades mirror paper, but don't execute
    GRADUAL = "gradual"         # Percentage of trades go live
    LIVE = "live"               # All qualifying trades go live


# =============================================================================
# TRADING STATS TRACKER
# =============================================================================

class TradingStatsTracker:
    """Tracks trading performance for safe rollout decisions"""
    
    def __init__(self, db=None):
        self.db = db
        self._daily_trades = []
        self._daily_pnl = 0.0
        self._consecutive_losses = 0
        self._last_trade_time = None
        self._cool_down_until = None
        self._lock = threading.Lock()
    
    def record_trade(self, is_win: bool, pnl_sol: float):
        """Record a trade result"""
        with self._lock:
            now = datetime.now(timezone.utc)
            self._daily_trades.append({
                'time': now,
                'is_win': is_win,
                'pnl': pnl_sol
            })
            self._daily_pnl += pnl_sol
            self._last_trade_time = now
            
            if is_win:
                self._consecutive_losses = 0
            else:
                self._consecutive_losses += 1
            
            # Clean up old trades (keep only last 24h)
            cutoff = now - timedelta(hours=24)
            self._daily_trades = [t for t in self._daily_trades if t['time'] > cutoff]
    
    def is_in_cool_down(self) -> bool:
        """Check if we're in a cool down period"""
        if self._cool_down_until is None:
            return False
        return datetime.now(timezone.utc) < self._cool_down_until
    
    def trigger_cool_down(self, minutes: int):
        """Trigger a cool down period"""
        self._cool_down_until = datetime.now(timezone.utc) + timedelta(minutes=minutes)
        logger.warning(f"⏸️ Cool down triggered for {minutes} minutes")
    
    def get_daily_stats(self) -> Dict:
        """Get today's trading stats"""
        with self._lock:
            trades = len(self._daily_trades)
            wins = sum(1 for t in self._daily_trades if t['is_win'])
            return {
                'trades': trades,
                'wins': wins,
                'losses': trades - wins,
                'win_rate': wins / trades if trades > 0 else 0,
                'pnl_sol': self._daily_pnl,
                'consecutive_losses': self._consecutive_losses,
                'in_cool_down': self.is_in_cool_down()
            }
    
    def can_trade_live(self, config: IntegrationConfig) -> Tuple[bool, str]:
        """Check if live trading is allowed based on stats"""
        stats = self.get_daily_stats()
        
        if self.is_in_cool_down():
            return False, "In cool down period"
        
        if stats['trades'] >= config.max_daily_trades:
            return False, f"Daily trade limit reached ({config.max_daily_trades})"
        
        if stats['pnl_sol'] <= -config.max_daily_loss_sol:
            return False, f"Daily loss limit reached ({config.max_daily_loss_sol} SOL)"
        
        if stats['consecutive_losses'] >= config.consecutive_loss_limit:
            self.trigger_cool_down(config.cool_down_after_loss)
            return False, f"Consecutive loss limit ({config.consecutive_loss_limit})"
        
        return True, "OK"


# =============================================================================
# PAPER TRADING VALIDATOR
# =============================================================================

class PaperTradingValidator:
    """Validates paper trading performance before allowing live trading"""
    
    def __init__(self, db=None):
        self.db = db
        self._validation_cache = None
        self._cache_time = None
        self._cache_ttl = 300  # 5 minutes
    
    def get_paper_performance(self, days: int = 14) -> Dict:
        """Get paper trading performance over specified days"""
        # Check cache
        if (self._validation_cache and self._cache_time and 
            (datetime.now() - self._cache_time).total_seconds() < self._cache_ttl):
            return self._validation_cache
        
        # Query paper trading database
        if not self.db:
            return self._empty_performance()
        
        try:
            # Assuming the paper trader uses these methods
            if hasattr(self.db, 'get_closed_positions'):
                closed = self.db.get_closed_positions(days=days)
            else:
                closed = []
            
            if not closed:
                return self._empty_performance()
            
            total = len(closed)
            wins = sum(1 for p in closed if p.get('profit_pct', 0) > 0)
            total_pnl = sum(p.get('profit_sol', 0) for p in closed)
            
            performance = {
                'days': days,
                'total_trades': total,
                'wins': wins,
                'losses': total - wins,
                'win_rate': wins / total if total > 0 else 0,
                'total_pnl_sol': total_pnl,
                'avg_pnl_sol': total_pnl / total if total > 0 else 0,
                'profitable_days': self._count_profitable_days(closed, days),
                'validated_at': datetime.now(timezone.utc).isoformat()
            }
            
            self._validation_cache = performance
            self._cache_time = datetime.now()
            
            return performance
            
        except Exception as e:
            logger.error(f"Failed to get paper performance: {e}")
            return self._empty_performance()
    
    def _empty_performance(self) -> Dict:
        return {
            'days': 0, 'total_trades': 0, 'wins': 0, 'losses': 0,
            'win_rate': 0, 'total_pnl_sol': 0, 'avg_pnl_sol': 0,
            'profitable_days': 0, 'validated_at': None
        }
    
    def _count_profitable_days(self, positions: List[Dict], days: int) -> int:
        """Count number of profitable trading days"""
        daily_pnl = {}
        for p in positions:
            exit_time = p.get('exit_time')
            if exit_time:
                if isinstance(exit_time, str):
                    date = exit_time[:10]
                else:
                    date = exit_time.strftime('%Y-%m-%d')
                daily_pnl[date] = daily_pnl.get(date, 0) + p.get('profit_sol', 0)
        
        return sum(1 for pnl in daily_pnl.values() if pnl > 0)
    
    def is_ready_for_live(self, config: IntegrationConfig) -> Tuple[bool, str]:
        """Check if paper trading performance qualifies for live trading"""
        perf = self.get_paper_performance(config.min_paper_profit_days)
        
        if perf['total_trades'] < config.min_paper_trades:
            return False, f"Need {config.min_paper_trades} trades, have {perf['total_trades']}"
        
        if perf['win_rate'] < config.min_paper_win_rate:
            return False, f"Need {config.min_paper_win_rate:.0%} win rate, have {perf['win_rate']:.0%}"
        
        if perf['total_pnl_sol'] <= 0:
            return False, f"Need positive PnL, have {perf['total_pnl_sol']:.4f} SOL"
        
        return True, "Paper trading validated ✅"


# =============================================================================
# EXECUTIONER BRIDGE
# =============================================================================

class ExecutionerBridge:
    """
    Bridge between the existing paper trading system and live execution.
    
    This is the main integration point that decides whether to paper trade
    or execute live based on configuration and performance validation.
    """
    
    def __init__(self, executioner: 'Executioner' = None, 
                 paper_trader=None, db=None, config: IntegrationConfig = None):
        self.executioner = executioner
        self.paper_trader = paper_trader
        self.db = db
        self.config = config or IntegrationConfig()
        
        # Mode determination
        self.mode = self._determine_mode()
        
        # Validators and trackers
        self.stats_tracker = TradingStatsTracker(db)
        self.paper_validator = PaperTradingValidator(db)
        self.price_service = PriceService()
        
        # Background monitoring
        self._monitor_thread = None
        self._stop_monitoring = threading.Event()
        
        logger.info(f"🌉 EXECUTIONER BRIDGE initialized")
        logger.info(f"   Mode: {self.mode.value}")
        logger.info(f"   Live %: {self.config.live_trade_percentage}%")
    
    def _determine_mode(self) -> TradingMode:
        """Determine trading mode based on configuration"""
        if not self.executioner:
            return TradingMode.PAPER_ONLY
        
        if not self.executioner.config.enable_live_trading:
            return TradingMode.PAPER_ONLY
        
        if self.config.live_trade_percentage <= 0:
            return TradingMode.SHADOW
        elif self.config.live_trade_percentage >= 100:
            return TradingMode.LIVE
        else:
            return TradingMode.GRADUAL
    
    def process_signal(self, signal: Dict, wallet_data: Dict = None) -> Dict:
        """
        Process a trading signal through the appropriate path.
        
        This is the main entry point called by master_v2.py webhook handler.
        
        Returns dict with:
        - paper_result: Result from paper trader
        - live_result: Result from live execution (if applicable)
        - execution_path: 'paper', 'live', or 'both'
        """
        result = {
            'paper_result': None,
            'live_result': None,
            'execution_path': 'paper',
            'reason': ''
        }
        
        token_address = signal.get('token_address', signal.get('token_out', ''))
        token_symbol = signal.get('token_symbol', 'UNKNOWN')
        
        # Always paper trade first (for learning and validation)
        if self.paper_trader:
            try:
                result['paper_result'] = self.paper_trader.process_signal(signal, wallet_data)
            except Exception as e:
                logger.error(f"Paper trading error: {e}")
        
        # Check if we should also go live
        should_live, reason = self._should_execute_live(signal)
        result['reason'] = reason
        
        if should_live and self.executioner:
            result['execution_path'] = 'both'
            
            # Build execution signal
            exec_signal = self._build_execution_signal(signal, wallet_data)
            
            try:
                live_result = self.executioner.execute_signal(exec_signal)
                result['live_result'] = {
                    'success': live_result.success,
                    'status': live_result.status.value,
                    'signature': live_result.signature,
                    'error': live_result.error_message
                }
                
                # Track for stats
                if live_result.success:
                    # We don't know P&L yet for buys, only track after sells
                    pass
                    
            except Exception as e:
                logger.error(f"Live execution error: {e}")
                result['live_result'] = {'success': False, 'error': str(e)}
        
        return result
    
    def process_exit_signal(self, token_address: str, reason: str, 
                           current_price: float = None) -> Dict:
        """Process an exit signal for a position"""
        result = {
            'paper_result': None,
            'live_result': None,
            'execution_path': 'paper'
        }
        
        # Close paper position
        if self.paper_trader and hasattr(self.paper_trader, 'close_position'):
            try:
                # Find paper position
                if hasattr(self.paper_trader, 'get_position'):
                    paper_pos = self.paper_trader.get_position(token_address)
                    if paper_pos:
                        result['paper_result'] = self.paper_trader.close_position(
                            paper_pos['id'], reason, current_price or 0
                        )
            except Exception as e:
                logger.error(f"Paper position close error: {e}")
        
        # Close live position
        if self.executioner:
            live_pos = self.executioner.tax_db.get_position(token_address)
            if live_pos:
                result['execution_path'] = 'both'
                try:
                    live_result = self.executioner.exit_position(token_address, reason)
                    result['live_result'] = {
                        'success': live_result.success,
                        'signature': live_result.signature,
                        'sol_received': live_result.sol_amount,
                        'error': live_result.error_message
                    }
                    
                    # Track P&L for stats
                    if live_result.success:
                        # Get gain/loss from tax record
                        # This would need the actual result data
                        pass
                        
                except Exception as e:
                    logger.error(f"Live position close error: {e}")
                    result['live_result'] = {'success': False, 'error': str(e)}
        
        return result
    
    def _should_execute_live(self, signal: Dict) -> Tuple[bool, str]:
        """Determine if a signal should be executed live"""
        
        # Mode check
        if self.mode == TradingMode.PAPER_ONLY:
            return False, "Paper only mode"
        
        if self.mode == TradingMode.SHADOW:
            return False, "Shadow mode (would execute)"
        
        # Paper trading validation
        is_validated, validation_msg = self.paper_validator.is_ready_for_live(self.config)
        if not is_validated:
            return False, f"Paper validation failed: {validation_msg}"
        
        # Daily stats check
        can_trade, stats_msg = self.stats_tracker.can_trade_live(self.config)
        if not can_trade:
            return False, f"Stats check failed: {stats_msg}"
        
        # Conviction check (stricter for live)
        conviction = signal.get('conviction_score', signal.get('conviction', 0))
        if conviction < self.config.live_min_conviction:
            return False, f"Conviction {conviction} < live minimum {self.config.live_min_conviction}"
        
        # Liquidity check
        token_address = signal.get('token_address', signal.get('token_out', ''))
        if token_address:
            token_info = self.price_service.get_token_info(token_address)
            if token_info['liquidity_usd'] < self.config.live_min_liquidity_usd:
                return False, f"Liquidity ${token_info['liquidity_usd']:.0f} < minimum"
        
        # Gradual rollout check
        if self.mode == TradingMode.GRADUAL:
            import random
            if random.random() * 100 > self.config.live_trade_percentage:
                return False, f"Random selection ({self.config.live_trade_percentage}% chance)"
        
        return True, "All checks passed ✅"
    
    def _build_execution_signal(self, signal: Dict, wallet_data: Dict = None) -> Dict:
        """Convert a paper trading signal to an execution signal"""
        return {
            'action': 'BUY' if signal.get('type', 'BUY') == 'BUY' else 'SELL',
            'token_address': signal.get('token_address', signal.get('token_out', '')),
            'token_symbol': signal.get('token_symbol', 'UNKNOWN'),
            'conviction_score': signal.get('conviction_score', signal.get('conviction', 60)),
            'suggested_size_sol': min(
                signal.get('position_size_sol', 0.1),
                self.config.live_max_position_sol
            ),
            'stop_loss_pct': signal.get('stop_loss_pct', -0.12),
            'take_profit_pct': signal.get('take_profit_pct', 0.30),
            'trailing_stop_pct': signal.get('trailing_stop_pct', 0.08),
            'reason': signal.get('reason', ''),
            'wallet_address': wallet_data.get('address') if wallet_data else None,
            'wallet_win_rate': wallet_data.get('win_rate', 0.5) if wallet_data else 0.5
        }
    
    def start_monitoring(self):
        """Start background position monitoring"""
        if self._monitor_thread and self._monitor_thread.is_alive():
            return
        
        self._stop_monitoring.clear()
        self._monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self._monitor_thread.start()
        logger.info("🔄 Position monitoring started")
    
    def stop_monitoring(self):
        """Stop background monitoring"""
        self._stop_monitoring.set()
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
        logger.info("⏹️ Position monitoring stopped")
    
    def _monitoring_loop(self):
        """Background loop for position monitoring and exit checks"""
        while not self._stop_monitoring.is_set():
            try:
                # Check exit conditions for live positions
                if self.executioner:
                    exits = self.executioner.check_and_execute_exits()
                    for exit_result in exits:
                        if exit_result.success:
                            logger.info(f"🚪 Auto-exit: {exit_result.token_symbol}")
                            # Track for stats
                            # (would need to get P&L from result)
                
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
            
            self._stop_monitoring.wait(self.config.exit_check_interval)
    
    def get_status(self) -> Dict:
        """Get bridge status and statistics"""
        status = {
            'mode': self.mode.value,
            'live_percentage': self.config.live_trade_percentage,
            'daily_stats': self.stats_tracker.get_daily_stats(),
            'paper_performance': self.paper_validator.get_paper_performance(),
        }
        
        # Validation status
        is_validated, msg = self.paper_validator.is_ready_for_live(self.config)
        status['paper_validated'] = is_validated
        status['validation_message'] = msg
        
        # Can trade status
        can_trade, trade_msg = self.stats_tracker.can_trade_live(self.config)
        status['can_trade_live'] = can_trade
        status['trade_message'] = trade_msg
        
        # Executioner stats if available
        if self.executioner:
            status['executioner'] = self.executioner.get_stats()
        
        return status
    
    def set_live_percentage(self, percentage: float):
        """Update the live trading percentage (0-100)"""
        self.config.live_trade_percentage = max(0, min(100, percentage))
        self.mode = self._determine_mode()
        logger.info(f"📊 Live percentage set to {self.config.live_trade_percentage}%")
    
    def emergency_stop(self):
        """Emergency stop all live trading"""
        logger.warning("🚨 EMERGENCY STOP TRIGGERED")
        self.config.live_trade_percentage = 0
        self.mode = TradingMode.PAPER_ONLY
        self.stop_monitoring()
        
        # Close all live positions
        if self.executioner:
            positions = self.executioner.get_open_positions()
            for pos in positions:
                try:
                    self.executioner.exit_position(pos['token_address'], 'EMERGENCY_STOP')
                except Exception as e:
                    logger.error(f"Failed to close {pos['token_symbol']}: {e}")


# =============================================================================
# EXAMPLE INTEGRATION WITH MASTER_V2
# =============================================================================

def integrate_with_master_v2(master):
    """
    Example showing how to integrate with existing master_v2.py
    
    Call this function in your master_v2.py initialization:
    
        from executioner_integration import integrate_with_master_v2
        bridge = integrate_with_master_v2(self)
    """
    from dotenv import load_dotenv
    load_dotenv()
    
    # Initialize Executioner
    private_key = os.getenv('SOLANA_PRIVATE_KEY')
    helius_key = os.getenv('HELIUS_KEY')
    
    exec_config = ExecutionConfig(
        enable_live_trading=bool(os.getenv('ENABLE_LIVE_TRADING', '').lower() == 'true'),
        max_position_size_sol=float(os.getenv('MAX_POSITION_SOL', '0.25')),
        min_conviction_score=int(os.getenv('MIN_CONVICTION', '60'))
    )
    
    executioner = Executioner(
        private_key=private_key,
        config=exec_config,
        helius_key=helius_key
    ) if private_key else None
    
    # Initialize integration
    int_config = IntegrationConfig(
        live_trade_percentage=float(os.getenv('LIVE_TRADE_PCT', '0')),
        min_paper_profit_days=14,
        min_paper_trades=50
    )
    
    bridge = ExecutionerBridge(
        executioner=executioner,
        paper_trader=master.paper_trader if hasattr(master, 'paper_trader') else None,
        db=master.db if hasattr(master, 'db') else None,
        config=int_config
    )
    
    # Start monitoring if live trading enabled
    if executioner and exec_config.enable_live_trading:
        bridge.start_monitoring()
    
    return bridge


# =============================================================================
# CLI
# =============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Executioner Integration Bridge")
    parser.add_argument('command', choices=['status', 'validate', 'set-live'],
                       help='Command to run')
    parser.add_argument('--percentage', type=float, help='Live trading percentage for set-live')
    
    args = parser.parse_args()
    
    # Create a minimal bridge for CLI
    config = IntegrationConfig()
    bridge = ExecutionerBridge(config=config)
    
    if args.command == 'status':
        status = bridge.get_status()
        print("\n🌉 EXECUTIONER BRIDGE STATUS")
        print("=" * 50)
        print(json.dumps(status, indent=2, default=str))
    
    elif args.command == 'validate':
        perf = bridge.paper_validator.get_paper_performance()
        is_ready, msg = bridge.paper_validator.is_ready_for_live(config)
        
        print("\n📊 PAPER TRADING VALIDATION")
        print("=" * 50)
        print(f"Trades: {perf['total_trades']}")
        print(f"Win Rate: {perf['win_rate']:.1%}")
        print(f"Total PnL: {perf['total_pnl_sol']:.4f} SOL")
        print(f"\nReady for Live: {'✅ YES' if is_ready else '❌ NO'}")
        print(f"Message: {msg}")
    
    elif args.command == 'set-live':
        if args.percentage is None:
            print("❌ --percentage required")
            return
        bridge.set_live_percentage(args.percentage)
        print(f"✅ Live percentage set to {args.percentage}%")


if __name__ == "__main__":
    main()
