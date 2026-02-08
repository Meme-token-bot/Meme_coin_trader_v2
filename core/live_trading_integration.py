"""
LIVE TRADING INTEGRATION - Clean Interface for master_v2.py
============================================================

PURPOSE:
    Provides a simple integration layer between:
    - master_v2.py (signal generation)
    - core/live_trading_engine.py (EXISTING trade execution)
    - live_exit_manager.py (exit monitoring)

    Uses YOUR EXISTING LiveTradingEngine - does not duplicate functionality.

USAGE:
    In master_v2.py:
    
    from core.live_trading_integration import LiveTradingIntegration
    
    # Initialize (uses your existing LiveTradingEngine)
    self.live_integration = LiveTradingIntegration(
        engine=self.live_engine,  # Your existing LiveTradingEngine
        notifier=self.notifier
    )
    
    # Start exit monitoring
    self.live_integration.start_exit_monitoring()
    
    # Process signals
    result = self.live_integration.process_buy_signal(signal, wallet_data)
    result = self.live_integration.process_sell_signal(exit_signal)
    
    # Cleanup
    self.live_integration.stop_exit_monitoring()

AUTHOR: Trading Bot System
VERSION: 2.0.0
"""

import logging
import threading
from datetime import datetime, timezone
from typing import Dict, List, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class IntegrationConfig:
    """Configuration for the integration layer"""
    # Signal filtering (additional to engine filters)
    min_conviction_score: float = 60.0
    min_wallet_win_rate: float = 0.4
    min_liquidity_usd: float = 20000.0
    blocked_hours_utc: List[int] = field(default_factory=list)
    
    # Exit parameters (passed to exit manager)
    stop_loss_pct: float = -15.0
    take_profit_pct: float = 30.0
    trailing_stop_pct: float = 10.0
    max_hold_hours: int = 12
    
    # Safety
    max_daily_loss_sol: float = 1.0
    enable_notifications: bool = True


class LiveTradingIntegration:
    """
    Integration layer connecting master_v2.py to the existing LiveTradingEngine.
    
    This class:
    1. Adds signal filtering before passing to engine
    2. Manages the exit monitor lifecycle
    3. Provides a clean API for master_v2.py
    """
    
    def __init__(self, 
                 engine,  # Your existing LiveTradingEngine
                 config: IntegrationConfig = None,
                 notifier = None):
        """
        Initialize the integration.
        
        Args:
            engine: Your existing LiveTradingEngine instance
            config: Optional configuration overrides
            notifier: Telegram notifier (optional)
        """
        self.engine = engine
        self.config = config or IntegrationConfig()
        self.notifier = notifier
        
        # Reuse engine exit manager when available to avoid duplicate monitors/threads
        if hasattr(engine, 'exit_monitor') and engine.exit_monitor is not None:
            self.exit_manager = engine.exit_monitor
            logger.info("♻️ Reusing engine LiveExitManager")
        else:
            from core.live_exit_manager import LiveExitManager, ExitConfig

            exit_config = ExitConfig(
                default_stop_loss_pct=self.config.stop_loss_pct,
                default_take_profit_pct=self.config.take_profit_pct,
                default_trailing_stop_pct=self.config.trailing_stop_pct,
                max_hold_hours=self.config.max_hold_hours,
                enable_auto_exits=True,
                enable_notifications=self.config.enable_notifications
            )

            self.exit_manager = LiveExitManager(
                trading_engine=engine,
                config=exit_config,
                notifier=notifier
            )
        
        # State
        self._lock = threading.RLock()
        
        logger.info("🚀 LiveTradingIntegration initialized")
        logger.info(f"   Using existing LiveTradingEngine")
        logger.info(f"   Exit monitoring: SL={self.config.stop_loss_pct}%, TP={self.config.take_profit_pct}%")
    
    # =========================================================================
    # SIGNAL FILTERING
    # =========================================================================
    
    def _should_trade(self, signal: Dict, wallet_data: Dict = None) -> tuple:
        """
        Additional filtering before passing to engine.
        
        The engine has its own checks (balance, position limits, etc.)
        This adds signal-specific filters.
        
        Returns:
            (should_trade, reason)
        """
        # Check conviction
        conviction = signal.get('conviction_score', signal.get('conviction', 0))
        if conviction < self.config.min_conviction_score:
            return False, f"Low conviction ({conviction:.0f} < {self.config.min_conviction_score:.0f})"
        
        # Check wallet win rate
        wallet_wr = wallet_data.get('win_rate', 0) if wallet_data else 0
        if wallet_wr < self.config.min_wallet_win_rate:
            return False, f"Low wallet WR ({wallet_wr:.1%} < {self.config.min_wallet_win_rate:.1%})"
        
        # Check liquidity
        liquidity = signal.get('liquidity_usd', signal.get('liquidity', 0))
        if liquidity < self.config.min_liquidity_usd:
            return False, f"Low liquidity (${liquidity:.0f} < ${self.config.min_liquidity_usd:.0f})"
        
        # Check blocked hours
        if self.config.blocked_hours_utc:
            current_hour = datetime.now(timezone.utc).hour
            if current_hour in self.config.blocked_hours_utc:
                return False, f"Blocked hour ({current_hour} UTC)"
        
        # Check daily loss limit (from engine's tax_db)
        if hasattr(self.engine, 'tax_db'):
            daily_stats = self.engine.tax_db.get_daily_stats()
            if daily_stats.get('pnl_sol', 0) <= -self.config.max_daily_loss_sol:
                return False, f"Daily loss limit ({self.config.max_daily_loss_sol} SOL)"
        
        return True, "OK"
    
    # =========================================================================
    # SIGNAL PROCESSING
    # =========================================================================
    
    def process_buy_signal(self, signal: Dict, wallet_data: Dict = None) -> Dict:
        """
        Process a buy signal from master_v2.py.
        
        Signal format:
        {
            'token_address': '...',
            'token_symbol': '...',
            'conviction_score': 70,
            'liquidity_usd': 50000,
            'price_usd': 0.001,
            ...
        }
        
        Wallet data format:
        {
            'address': '...',
            'win_rate': 0.5,
            'cluster': 'ALPHA',
            ...
        }
        
        Returns:
            {
                'success': bool,
                'filter_reason': str or None,
                'engine_result': dict from engine.execute_buy()
            }
        """
        result = {
            'success': False,
            'action': 'BUY',
            'filter_reason': None,
            'engine_result': None
        }
        
        token_symbol = signal.get('token_symbol', 'UNKNOWN')
        
        # Pre-trade filtering
        passes, reason = self._should_trade(signal, wallet_data)
        
        if not passes:
            result['filter_reason'] = reason
            logger.info(f"⏭️ Signal filtered: {token_symbol} | {reason}")
            return result
        
        # Build signal for engine (add exit parameters)
        engine_signal = {
            **signal,
            'stop_loss_pct': self.config.stop_loss_pct,
            'take_profit_pct': self.config.take_profit_pct,
            'trailing_stop_pct': self.config.trailing_stop_pct,
            'max_hold_hours': self.config.max_hold_hours
        }
        
        # Execute via existing engine
        try:
            engine_result = self.engine.execute_buy(engine_signal)
            result['engine_result'] = engine_result
            
            if engine_result.get('success'):
                result['success'] = True
                logger.info(f"🟢 LIVE BUY: {token_symbol} | Sig: {engine_result.get('signature', 'N/A')[:16]}...")
                
                # Send notification
                if self.notifier and self.config.enable_notifications:
                    self.notifier.send(
                        f"🟢 LIVE BUY: {token_symbol}\n"
                        f"Sig: {engine_result.get('signature', 'N/A')[:16]}..."
                    )
            else:
                logger.warning(f"🔴 LIVE BUY failed: {token_symbol} | {engine_result.get('error')}")
                
        except Exception as e:
            logger.error(f"Live buy exception: {e}")
            result['engine_result'] = {'success': False, 'error': str(e)}
        
        return result
    
    def process_sell_signal(self, signal: Dict) -> Dict:
        """
        Process a sell/exit signal.
        
        Signal format:
        {
            'token_address': '...',
            'token_symbol': '...',
            'reason': 'TAKE_PROFIT' | 'STOP_LOSS' | 'SIGNAL' | 'MANUAL'
        }
        
        Returns:
            Result from exit_manager.process_exit_signal()
        """
        return self.exit_manager.process_exit_signal(signal)
    
    # =========================================================================
    # EXIT MONITORING
    # =========================================================================
    
    def start_exit_monitoring(self):
        """Start background exit monitoring"""
        if getattr(self.exit_manager, '_monitor_running', False):
            logger.info("Exit monitoring already running; skipping duplicate start")
            return
        self.exit_manager.start_monitoring()
    
    def stop_exit_monitoring(self):
        """Stop background exit monitoring"""
        if not getattr(self.exit_manager, '_monitor_running', False):
            return
        self.exit_manager.stop_monitoring()
    
    def check_exits_now(self) -> List[Dict]:
        """Manual exit check (backup to automatic monitoring)"""
        return self.exit_manager.check_exits_now()
    
    # =========================================================================
    # EMERGENCY CONTROLS
    # =========================================================================
    
    def emergency_exit_all(self) -> List[Dict]:
        """Emergency exit all positions"""
        return self.exit_manager.emergency_exit_all()
    
    def force_exit(self, token_address: str, reason: str = "MANUAL") -> Dict:
        """Force exit a specific position"""
        return self.exit_manager.force_exit(token_address, reason)
    
    def activate_kill_switch(self):
        """Activate kill switch - exit all and disable trading"""
        logger.critical("🚨 KILL SWITCH ACTIVATED")
        
        # Exit all positions
        results = self.emergency_exit_all()
        
        # Disable engine if possible
        if hasattr(self.engine, 'config'):
            self.engine.config.enable_live_trading = False
        
        return results
    
    # =========================================================================
    # STATUS & DIAGNOSTICS
    # =========================================================================
    
    def get_status(self) -> Dict:
        """Get integration status"""
        engine_status = {}
        if hasattr(self.engine, 'get_status'):
            engine_status = self.engine.get_status()
        
        exit_status = self.exit_manager.get_status()
        
        return {
            'integration_active': True,
            'engine': engine_status,
            'exit_manager': exit_status,
            'filters': {
                'min_conviction': self.config.min_conviction_score,
                'min_wallet_wr': self.config.min_wallet_win_rate,
                'min_liquidity': self.config.min_liquidity_usd,
                'blocked_hours': self.config.blocked_hours_utc
            }
        }
    
    def get_open_positions(self) -> List[Dict]:
        """Get all open positions with details"""
        return self.exit_manager.get_position_details()
    
    def get_trade_history(self, limit: int = 50) -> List[Dict]:
        """Get recent trade history from tax_db"""
        if hasattr(self.engine, 'tax_db'):
            with self.engine.tax_db._get_connection() as conn:
                rows = conn.execute("""
                    SELECT * FROM tax_transactions 
                    ORDER BY timestamp DESC 
                    LIMIT ?
                """, (limit,)).fetchall()
                return [dict(row) for row in rows]
        return []
    
    def export_tax_report(self, start_date: str, end_date: str, output_path: str):
        """Export tax report"""
        if hasattr(self.engine, 'tax_db') and hasattr(self.engine.tax_db, 'export_tax_report'):
            self.engine.tax_db.export_tax_report(start_date, end_date, output_path)
            logger.info(f"📊 Tax report exported to {output_path}")
        else:
            logger.warning("Tax export not available")


# =============================================================================
# CONVENIENCE FACTORY
# =============================================================================

def create_live_integration(
    engine,
    notifier = None,
    stop_loss_pct: float = -15.0,
    take_profit_pct: float = 30.0,
    trailing_stop_pct: float = 10.0,
    max_hold_hours: int = 12,
    min_conviction: float = 60.0,
    min_wallet_wr: float = 0.4,
    min_liquidity: float = 20000.0
) -> LiveTradingIntegration:
    """
    Factory function to create a LiveTradingIntegration.
    
    Args:
        engine: Your existing LiveTradingEngine
        notifier: Telegram notifier (optional)
        ... exit and filter parameters
    
    Returns:
        Configured LiveTradingIntegration
    """
    config = IntegrationConfig(
        stop_loss_pct=stop_loss_pct,
        take_profit_pct=take_profit_pct,
        trailing_stop_pct=trailing_stop_pct,
        max_hold_hours=max_hold_hours,
        min_conviction_score=min_conviction,
        min_wallet_win_rate=min_wallet_wr,
        min_liquidity_usd=min_liquidity
    )
    
    return LiveTradingIntegration(engine=engine, config=config, notifier=notifier)


# =============================================================================
# CLI FOR TESTING
# =============================================================================

if __name__ == "__main__":
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(description="Live Trading Integration CLI")
    parser.add_argument('command', choices=['status', 'positions', 'history', 'exit', 'exit-all'],
                       help='Command to run')
    parser.add_argument('--token', help='Token address for exit command')
    parser.add_argument('--reason', default='MANUAL', help='Exit reason')
    parser.add_argument('--limit', type=int, default=20, help='History limit')
    
    args = parser.parse_args()
    
    # Initialize
    from core.live_trading_engine import LiveTradingEngine
    from core.secrets_manager import init_secrets
    
    init_secrets()
    engine = LiveTradingEngine()
    integration = LiveTradingIntegration(engine=engine)
    
    if args.command == 'status':
        import json
        print(json.dumps(integration.get_status(), indent=2, default=str))
    
    elif args.command == 'positions':
        positions = integration.get_open_positions()
        print(f"\n{'='*60}")
        print(f"OPEN POSITIONS ({len(positions)})")
        print(f"{'='*60}")
        for pos in positions:
            pnl = pos.get('pnl_pct', 0)
            emoji = "📈" if pnl > 0 else "📉"
            print(f"{emoji} {pos['token_symbol']}: {pnl:+.1f}%")
    
    elif args.command == 'history':
        history = integration.get_trade_history(limit=args.limit)
        print(f"\n{'='*60}")
        print(f"TRADE HISTORY (last {len(history)})")
        print(f"{'='*60}")
        for trade in history:
            print(f"{trade['timestamp'][:10]} | {trade['transaction_type']} | {trade['token_symbol']}")
    
    elif args.command == 'exit':
        if not args.token:
            print("Error: --token required for exit command")
            sys.exit(1)
        result = integration.force_exit(args.token, args.reason)
        print(f"Exit result: {result}")
    
    elif args.command == 'exit-all':
        confirm = input("Type 'CONFIRM' to exit all positions: ")
        if confirm == 'CONFIRM':
            results = integration.emergency_exit_all()
            print(f"Exited {len(results)} positions")
        else:
            print("Aborted")
