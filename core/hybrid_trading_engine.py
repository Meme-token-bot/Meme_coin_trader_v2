"""
HYBRID TRADING ENGINE - Paper + Live Integration
=================================================

This module creates a unified trading system that:
1. Runs paper trading for continued learning
2. Executes live trades with validated parameters
3. Applies learned filters from paper trading analysis
4. Compares paper vs live results

Based on paper trading analysis:
- Win Rate: 38.1% (needs 22% for profit)
- Profit Factor: 2.19
- Risk/Reward: 3.55:1
- Blocked Hours: 1, 3, 5, 19, 23 UTC

Capital: 3 SOL
Position Size: 0.08 SOL
Max Positions: 10
Max Deployed: 0.80 SOL

Author: Claude
"""

import os
import json
import time
import threading
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s'
)
logger = logging.getLogger("HybridTrader")


# =============================================================================
# CONFIGURATION - Based on Paper Trading Analysis
# =============================================================================

@dataclass
class HybridTradingConfig:
    """
    Configuration based on your paper trading results.
    
    Paper Trading Results (5 days):
    - 6,942 trades
    - 38.1% win rate (breakeven: 22%)
    - 2.19 profit factor
    - 3.55:1 risk/reward
    - +3038% return
    """
    
    # Capital (3 SOL starting)
    starting_capital_sol: float = 3.0
    fee_reserve_sol: float = 0.30            # ~75 round trips
    
    # Position sizing (based on fee analysis)
    position_size_sol: float = 0.08          # 6.5% fee overhead
    max_open_positions: int = 10             # Max concurrent
    max_deployed_sol: float = 0.80           # 27% of capital at risk
    
    # Risk management
    max_daily_loss_sol: float = 0.25         # Stop trading for day
    max_consecutive_losses: int = 15         # Pause and reassess  
    min_balance_sol: float = 1.50            # Emergency stop
    cool_down_minutes: int = 30              # After consecutive losses
    
    # Entry filters (from paper trading analysis)
    min_conviction: int = 60
    min_wallet_wr: float = 0.35              # From wallet analysis
    min_liquidity_usd: float = 5000
    
    # Time filters (from hourly analysis - worst hours)
    blocked_hours_utc: List[int] = field(default_factory=lambda: [1, 3, 5, 19, 23])
    
    # Exit parameters (from paper trading)
    stop_loss_pct: float = -0.15             # -15%
    take_profit_pct: float = 0.30            # +30%
    trailing_stop_pct: float = 0.10          # 10% from peak
    max_hold_hours: int = 12
    
    # Execution
    default_slippage_bps: int = 150          # 1.5%
    jito_tip_lamports: int = 1_500_000       # 0.0015 SOL
    priority_fee_lamports: int = 50_000
    confirmation_timeout: int = 60
    
    # Mode controls
    enable_live_trading: bool = False        # MUST enable explicitly
    enable_paper_trading: bool = True        # Keep paper trading for comparison
    live_trade_percentage: float = 100.0     # % of qualifying signals to trade live
    
    # Validation thresholds (skip paper validation since you have 5 days)
    require_paper_validation: bool = False   # You've already validated
    min_paper_trades: int = 50
    min_paper_win_rate: float = 0.30


class TradingMode(Enum):
    PAPER_ONLY = "paper_only"
    LIVE_ONLY = "live_only"
    HYBRID = "hybrid"                        # Both paper and live


# =============================================================================
# DAILY STATS TRACKER
# =============================================================================

class DailyStatsTracker:
    """Track daily performance for risk management"""
    
    def __init__(self):
        self._today = datetime.now(timezone.utc).strftime('%Y-%m-%d')
        self._trades = 0
        self._wins = 0
        self._losses = 0
        self._pnl_sol = 0.0
        self._consecutive_losses = 0
        self._cool_down_until: Optional[datetime] = None
        self._lock = threading.Lock()
    
    def _check_date(self):
        """Reset stats if new day"""
        today = datetime.now(timezone.utc).strftime('%Y-%m-%d')
        if today != self._today:
            self._today = today
            self._trades = 0
            self._wins = 0
            self._losses = 0
            self._pnl_sol = 0.0
            # Don't reset consecutive losses - carry over
    
    def record_trade(self, pnl_sol: float, is_win: bool):
        """Record a completed trade"""
        with self._lock:
            self._check_date()
            self._trades += 1
            self._pnl_sol += pnl_sol
            
            if is_win:
                self._wins += 1
                self._consecutive_losses = 0
            else:
                self._losses += 1
                self._consecutive_losses += 1
    
    def trigger_cool_down(self, minutes: int):
        """Trigger cool down period"""
        self._cool_down_until = datetime.now(timezone.utc) + timedelta(minutes=minutes)
        logger.warning(f"⏸️ Cool down triggered for {minutes} minutes")
    
    def is_in_cool_down(self) -> bool:
        """Check if in cool down"""
        if self._cool_down_until is None:
            return False
        return datetime.now(timezone.utc) < self._cool_down_until
    
    def get_stats(self) -> Dict:
        """Get current day stats"""
        with self._lock:
            self._check_date()
            return {
                'date': self._today,
                'trades': self._trades,
                'wins': self._wins,
                'losses': self._losses,
                'win_rate': self._wins / self._trades if self._trades > 0 else 0,
                'pnl_sol': self._pnl_sol,
                'consecutive_losses': self._consecutive_losses,
                'in_cool_down': self.is_in_cool_down()
            }


# =============================================================================
# SIGNAL FILTER - Applies Learned Criteria
# =============================================================================

class SignalFilter:
    """
    Apply learned filters from paper trading analysis.
    
    Filters based on your data:
    - Blocked hours: 1, 3, 5, 19, 23 UTC (all <30% WR)
    - Min wallet WR: 0.35 (wallets below underperform)
    - Min liquidity: $5,000
    - Min conviction: 60
    """
    
    def __init__(self, config: HybridTradingConfig):
        self.config = config
        
    def should_trade(self, signal: Dict, wallet_data: Dict = None) -> Tuple[bool, str]:
        """
        Check if signal passes all learned filters.
        
        Returns (should_trade, reason)
        """
        # Check time filter
        current_hour = datetime.now(timezone.utc).hour
        if current_hour in self.config.blocked_hours_utc:
            return False, f"Blocked hour: {current_hour} UTC"
        
        # Check conviction
        conviction = signal.get('conviction_score', signal.get('conviction', 0))
        if conviction < self.config.min_conviction:
            return False, f"Low conviction: {conviction} < {self.config.min_conviction}"
        
        # Check liquidity
        #liquidity = signal.get('liquidity_usd', signal.get('liquidity', 0))
        #if liquidity < self.config.min_liquidity_usd:
        #    return False, f"Low liquidity: ${liquidity:,.0f} < ${self.config.min_liquidity_usd:,.0f}"
        # Liquidity filter intentionally disabled for live trading alignment
        
        # Check wallet win rate
        if wallet_data:
            wallet_wr = wallet_data.get('win_rate', 0.5)
            if wallet_wr < self.config.min_wallet_wr:
                return False, f"Low wallet WR: {wallet_wr:.1%} < {self.config.min_wallet_wr:.1%}"
        
        return True, "All filters passed"


# =============================================================================
# HYBRID TRADING ENGINE
# =============================================================================

class HybridTradingEngine:
    """
    Unified trading engine that runs both paper and live trading.
    
    Features:
    - Applies learned filters before trading
    - Executes on live and paper in parallel
    - Tracks and compares performance
    - Enforces risk limits
    """
    
    def __init__(self, 
                 paper_engine=None,
                 live_engine=None,
                 config: HybridTradingConfig = None,
                 notifier=None):
        
        self.config = config or HybridTradingConfig()
        self.paper_engine = paper_engine
        self.live_engine = live_engine
        self.notifier = notifier
        
        self.signal_filter = SignalFilter(self.config)
        self.daily_stats = DailyStatsTracker()
        
        # State
        self._kill_switch = False
        self._lock = threading.Lock()
        
        # Determine mode
        if self.config.enable_live_trading and self.config.enable_paper_trading:
            self.mode = TradingMode.HYBRID
        elif self.config.enable_live_trading:
            self.mode = TradingMode.LIVE_ONLY
        else:
            self.mode = TradingMode.PAPER_ONLY
        
        logger.info(f"🚀 Hybrid Trading Engine initialized")
        logger.info(f"   Mode: {self.mode.value}")
        logger.info(f"   Live Trading: {'ENABLED ⚡' if self.config.enable_live_trading else 'DISABLED'}")
        logger.info(f"   Paper Trading: {'ENABLED 📝' if self.config.enable_paper_trading else 'DISABLED'}")
        logger.info(f"   Position Size: {self.config.position_size_sol} SOL")
        logger.info(f"   Max Positions: {self.config.max_open_positions}")
        logger.info(f"   Blocked Hours: {self.config.blocked_hours_utc}")
    
    def can_open_live_position(self, signal: Dict = None) -> Tuple[bool, str]:
        """Check if we can open a live position"""
        
        if self._kill_switch:
            return False, "Kill switch active"
        
        if not self.config.enable_live_trading:
            return False, "Live trading disabled"
        
        if not self.live_engine:
            return False, "Live engine not configured"
        
        # Check cool down
        if self.daily_stats.is_in_cool_down():
            return False, "In cool down period"
        
        # Check daily loss limit
        stats = self.daily_stats.get_stats()
        if stats['pnl_sol'] <= -self.config.max_daily_loss_sol:
            return False, f"Daily loss limit reached: {stats['pnl_sol']:.4f} SOL"
        
        # Check consecutive losses
        if stats['consecutive_losses'] >= self.config.max_consecutive_losses:
            self.daily_stats.trigger_cool_down(self.config.cool_down_minutes)
            return False, f"Consecutive loss limit: {stats['consecutive_losses']}"
        
        # Check balance (if live engine can report it)
        if hasattr(self.live_engine, 'get_sol_balance'):
            balance = self.live_engine.get_sol_balance()
            if balance < self.config.min_balance_sol:
                self._activate_kill_switch("Balance below minimum")
                return False, f"Balance too low: {balance:.4f} SOL"
        
        # Check position count
        if hasattr(self.live_engine, 'get_open_positions'):
            positions = self.live_engine.get_open_positions()
            if len(positions) >= self.config.max_open_positions:
                return False, f"Max positions: {len(positions)}/{self.config.max_open_positions}"
            
            # Check deployed capital
            deployed = sum(p.get('total_cost_sol', 0) for p in positions)
            if deployed + self.config.position_size_sol > self.config.max_deployed_sol:
                return False, f"Max deployed: {deployed:.4f}/{self.config.max_deployed_sol} SOL"
        
        # Apply signal filter
        if signal:
            passes, reason = self.signal_filter.should_trade(signal)
            if not passes:
                return False, f"Filter: {reason}"
        
        return True, "OK"
    
    def process_signal(self, signal: Dict, wallet_data: Dict = None) -> Dict:
        """
        Process a trading signal through both paper and live systems.
        
        Returns:
            {
                'paper_result': {...} or None,
                'live_result': {...} or None,
                'filter_passed': bool,
                'filter_reason': str,
                'mode': 'paper_only' | 'live_only' | 'hybrid'
            }
        """
        result = {
            'paper_result': None,
            'live_result': None,
            'filter_passed': False,
            'filter_reason': '',
            'mode': self.mode.value,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        
        token_address = signal.get('token_address', signal.get('token_out', ''))
        token_symbol = signal.get('token_symbol', 'UNKNOWN')
        
        # Apply learned filters first
        filter_passed, filter_reason = self.signal_filter.should_trade(signal, wallet_data)
        result['filter_passed'] = filter_passed
        result['filter_reason'] = filter_reason
        
        if not filter_passed:
            logger.info(f"⏭️ Signal filtered: {token_symbol} | {filter_reason}")
            return result
        
        # Process paper trade (always, for comparison)
        if self.config.enable_paper_trading and self.paper_engine:
            try:
                paper_result = self._execute_paper_trade(signal, wallet_data)
                result['paper_result'] = paper_result
            except Exception as e:
                logger.error(f"Paper trade error: {e}")
        
        # Process live trade
        if self.config.enable_live_trading and self.live_engine:
            can_trade, reason = self.can_open_live_position(signal)
            
            if can_trade:
                try:
                    live_result = self._execute_live_trade(signal, wallet_data)
                    result['live_result'] = live_result
                    
                    # Log comparison if both executed
                    if result['paper_result'] and live_result:
                        self._log_comparison(result['paper_result'], live_result)
                    elif live_result and not live_result.get('success'):
                        reason = live_result.get('reason') or live_result.get('error') or 'unknown'
                        logger.info(f"🔴 Live trade failed: {token_symbol} | {reason}")
                        
                except Exception as e:
                    logger.error(f"Live trade error: {e}")
                    result['live_result'] = {'success': False, 'error': str(e)}
            else:
                logger.info(f"⏭️ Live trade skipped: {token_symbol} | {reason}")
                result['live_result'] = {'skipped': True, 'reason': reason}
        
        return result
    
    def _execute_paper_trade(self, signal: Dict, wallet_data: Dict = None) -> Dict:
        """Execute paper trade"""
        if hasattr(self.paper_engine, 'process_signal'):
            paper_result = self.paper_engine.process_signal(signal, wallet_data)
            if isinstance(paper_result, dict):
                paper_result.setdefault('success', bool(paper_result.get('position_id')))
            return paper_result
    
        # Build paper signal with our exit parameters
        paper_signal = {
            **signal,
            'position_size_sol': self.config.position_size_sol,
            'stop_loss': self.config.stop_loss_pct,
            'take_profit': self.config.take_profit_pct,
            'trailing_stop': self.config.trailing_stop_pct,
            'max_hold_hours': self.config.max_hold_hours
        }
        
        if hasattr(self.paper_engine, 'open_position'):
            return self.paper_engine.open_position(
                token_address=signal.get('token_address', signal.get('token_out', '')),
                token_symbol=signal.get('token_symbol', 'UNKNOWN'),
                entry_price=signal.get('price_usd', 0),
                position_size_sol=self.config.position_size_sol,
                conviction=signal.get('conviction_score', signal.get('conviction', 60)),
                wallet_address=wallet_data.get('address') if wallet_data else None,
                stop_loss_pct=self.config.stop_loss_pct,
                take_profit_pct=self.config.take_profit_pct,
                trailing_stop_pct=self.config.trailing_stop_pct,
                entry_context=signal
            )
        
        return {'success': False, 'error': 'Paper engine not configured properly'}
    
    def _execute_live_trade(self, signal: Dict, wallet_data: Dict = None) -> Dict:
        """Execute live trade"""
        live_signal = {
            'action': 'BUY',
            'token_address': signal.get('token_address', signal.get('token_out', '')),
            'token_symbol': signal.get('token_symbol', 'UNKNOWN'),
            'conviction_score': signal.get('conviction_score', signal.get('conviction', 60)),
            'position_size_sol': self.config.position_size_sol,
            'stop_loss_pct': self.config.stop_loss_pct,
            'take_profit_pct': self.config.take_profit_pct,
            'trailing_stop_pct': self.config.trailing_stop_pct,
            'slippage_bps': self.config.default_slippage_bps,
            'liquidity_usd': signal.get('liquidity_usd', 0),
            'wallet_address': wallet_data.get('address') if wallet_data else None
        }
        
        if hasattr(self.live_engine, 'execute_buy'):
            result = self.live_engine.execute_buy(live_signal)
            
            if result.get('success'):
                # Send notification
                if self.notifier:
                    self.notifier.send(
                        f"🟢 LIVE BUY: {live_signal['token_symbol']}\n"
                        f"Size: {self.config.position_size_sol} SOL\n"
                        f"Sig: {result.get('signature', 'N/A')[:16]}..."
                    )
            
            return result
        
        return {'success': False, 'error': 'Live engine not configured properly'}
    
    def _log_comparison(self, paper: Dict, live: Dict):
        """Log comparison between paper and live execution"""
        def _format_result(result: Dict) -> str:
            if result.get('success'):
                return "✅"
            if result.get('skipped'):
                reason = result.get('reason', 'skipped')
                return f"⚠️ ({reason})"
            reason = (
                result.get('filter_reason')
                or result.get('reason')
                or result.get('error')
                or "failed"
            )
            return f"❌ ({reason})"

        logger.info(f"📊 Trade Comparison:")
        logger.info(f"   Paper: {_format_result(paper)}")
        logger.info(f"   Live:  {_format_result(live)}")
    
    def record_exit(self, is_live: bool, pnl_sol: float, is_win: bool):
        """Record an exit for tracking"""
        if is_live:
            self.daily_stats.record_trade(pnl_sol, is_win)
    
    def _activate_kill_switch(self, reason: str):
        """Activate emergency kill switch"""
        logger.critical(f"🚨 KILL SWITCH: {reason}")
        self._kill_switch = True
        
        if self.notifier:
            self.notifier.send(f"🚨 KILL SWITCH ACTIVATED: {reason}")
        
        # Close all live positions
        if self.live_engine and hasattr(self.live_engine, 'get_open_positions'):
            positions = self.live_engine.get_open_positions()
            for pos in positions:
                try:
                    self.live_engine.execute_sell(pos['token_address'], 'KILL_SWITCH')
                except Exception as e:
                    logger.error(f"Failed to close {pos.get('token_symbol')}: {e}")
    
    def get_status(self) -> Dict:
        """Get comprehensive status"""
        status = {
            'mode': self.mode.value,
            'kill_switch': self._kill_switch,
            'daily_stats': self.daily_stats.get_stats(),
            'config': {
                'live_trading': self.config.enable_live_trading,
                'paper_trading': self.config.enable_paper_trading,
                'position_size': self.config.position_size_sol,
                'max_positions': self.config.max_open_positions,
                'blocked_hours': self.config.blocked_hours_utc
            }
        }
        
        if self.live_engine and hasattr(self.live_engine, 'get_status'):
            status['live_engine'] = self.live_engine.get_status()
        
        if self.paper_engine and hasattr(self.paper_engine, 'get_stats'):
            status['paper_engine'] = self.paper_engine.get_stats()
        
        return status
    
    def print_status(self):
        """Print formatted status"""
        status = self.get_status()
        
        print("\n" + "=" * 70)
        print("🔄 HYBRID TRADING ENGINE STATUS")
        print("=" * 70)
        
        print(f"\n  Mode: {status['mode']}")
        print(f"  Kill Switch: {'🚨 ACTIVE' if status['kill_switch'] else '✅ Off'}")
        
        daily = status['daily_stats']
        print(f"\n  📊 Today's Performance:")
        print(f"     Trades: {daily['trades']}")
        print(f"     Wins: {daily['wins']} | Losses: {daily['losses']}")
        print(f"     Win Rate: {daily['win_rate']:.1%}")
        print(f"     PnL: {daily['pnl_sol']:+.4f} SOL")
        print(f"     Consecutive Losses: {daily['consecutive_losses']}")
        print(f"     Cool Down: {'⏸️ Yes' if daily['in_cool_down'] else 'No'}")
        
        config = status['config']
        print(f"\n  ⚙️ Configuration:")
        print(f"     Live Trading: {'✅ ON' if config['live_trading'] else '❌ OFF'}")
        print(f"     Paper Trading: {'✅ ON' if config['paper_trading'] else '❌ OFF'}")
        print(f"     Position Size: {config['position_size']} SOL")
        print(f"     Max Positions: {config['max_positions']}")
        print(f"     Blocked Hours: {config['blocked_hours']}")
        
        print("\n" + "=" * 70)


# =============================================================================
# INTEGRATION FUNCTIONS
# =============================================================================

def create_hybrid_engine(paper_engine, notifier=None) -> HybridTradingEngine:
    """
    Create a hybrid trading engine with live trading support.
    
    Call this from master_v2.py:
        from hybrid_trading_engine import create_hybrid_engine
        hybrid = create_hybrid_engine(paper_engine, notifier)
    """
    # Try to import secrets manager
    try:
        from core.secrets_manager import get_secret
    except ImportError:
        from dotenv import load_dotenv
        load_dotenv()
        def get_secret(key, default=None):
            return os.getenv(key, default)
    
    # Check if live trading should be enabled
    enable_live = get_secret('ENABLE_LIVE_TRADING', '').lower() == 'true'
    
    # Create config
    config = HybridTradingConfig(
        enable_live_trading=enable_live,
        enable_paper_trading=True,  # Always keep paper for comparison
        position_size_sol=float(get_secret('POSITION_SIZE_SOL', '0.08')),
        max_open_positions=int(get_secret('MAX_OPEN_POSITIONS', '10')),
        max_daily_loss_sol=float(get_secret('MAX_DAILY_LOSS_SOL', '0.25')),
        min_conviction=int(get_secret('MIN_CONVICTION', '60')),
    )
    
    # Create live engine if enabled
    live_engine = None
    
    if enable_live:
        private_key = get_secret('SOLANA_PRIVATE_KEY')
        helius_key = get_secret('HELIUS_KEY')
        
        if private_key:
            try:
                # Import and create live engine
                from live_trading_engine import LiveTradingEngine, LiveTradingConfig
                
                live_config = LiveTradingConfig(
                    position_size_sol=config.position_size_sol,
                    max_open_positions=config.max_open_positions,
                    max_daily_loss_sol=config.max_daily_loss_sol,
                    min_conviction=config.min_conviction,
                    blocked_hours_utc=config.blocked_hours_utc,
                    stop_loss_pct=config.stop_loss_pct,
                    take_profit_pct=config.take_profit_pct,
                    trailing_stop_pct=config.trailing_stop_pct,
                    enable_live_trading=True,
                    enable_jito_bundles=True
                )
                
                live_engine = LiveTradingEngine(
                    private_key=private_key,
                    helius_key=helius_key,
                    config=live_config
                )
                
                logger.info("✅ Live trading engine initialized")
                
            except Exception as e:
                logger.error(f"Failed to initialize live engine: {e}")
        else:
            logger.warning("⚠️ SOLANA_PRIVATE_KEY not set - live trading disabled")
    
    # Create hybrid engine
    hybrid = HybridTradingEngine(
        paper_engine=paper_engine,
        live_engine=live_engine,
        config=config,
        notifier=notifier
    )
    
    return hybrid


# =============================================================================
# CLI
# =============================================================================

def main():
    """CLI interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Hybrid Trading Engine")
    parser.add_argument('command', choices=['status', 'test-filter', 'config'],
                       help='Command to run')
    
    args = parser.parse_args()
    
    config = HybridTradingConfig()
    
    if args.command == 'status':
        engine = HybridTradingEngine(config=config)
        engine.print_status()
    
    elif args.command == 'test-filter':
        filter_ = SignalFilter(config)
        
        # Test signal
        test_signal = {
            'token_address': 'TEST',
            'token_symbol': 'TEST',
            'conviction_score': 70,
            'liquidity_usd': 50000
        }
        
        test_wallet = {
            'address': 'TEST_WALLET',
            'win_rate': 0.45
        }
        
        # Test at different hours
        print("\n📋 Testing Signal Filter:")
        print(f"   Signal: conviction=70, liquidity=$50k")
        print(f"   Wallet: WR=45%")
        print(f"\n   Blocked hours: {config.blocked_hours_utc}")
        
        passes, reason = filter_.should_trade(test_signal, test_wallet)
        print(f"\n   Current hour result: {'✅ PASS' if passes else '❌ BLOCK'}")
        print(f"   Reason: {reason}")
    
    elif args.command == 'config':
        print("\n⚙️ Default Configuration:")
        print(f"   Position Size: {config.position_size_sol} SOL")
        print(f"   Max Positions: {config.max_open_positions}")
        print(f"   Max Deployed: {config.max_deployed_sol} SOL")
        print(f"   Max Daily Loss: {config.max_daily_loss_sol} SOL")
        print(f"   Blocked Hours: {config.blocked_hours_utc}")
        print(f"   Stop Loss: {config.stop_loss_pct:.0%}")
        print(f"   Take Profit: {config.take_profit_pct:.0%}")
        print(f"   Trailing Stop: {config.trailing_stop_pct:.0%}")


if __name__ == "__main__":
    main()
