"""
DROP-IN REPLACEMENT FOR PaperTradingEngine
==========================================

This replaces the buggy PaperTradingEngine in master_v2.py with
the robust EffectivePaperTrader.

Usage in master_v2.py:
    # Replace this:
    # from paper_trading_engine import PaperTradingEngine
    
    # With this:
    from core.paper_engine_replacement import PaperTradingEngine

The interface is compatible with the existing master_v2.py code.
"""

import threading
from datetime import datetime
from typing import Dict, List, Optional

from core.effective_paper_trader import (
    EffectivePaperTrader, 
    PaperTraderConfig,
    EntryContext,
    ExitReason
)


class PaperTradingEngine:
    """
    Drop-in replacement for the original PaperTradingEngine.
    
    Same interface, but uses the robust EffectivePaperTrader under the hood.
    """
    
    def __init__(self, db, starting_balance: float = 10.0, max_positions: int = 5):
        """
        Initialize the paper trading engine.
        
        Args:
            db: DatabaseV2 instance (kept for compatibility, but we use our own DB)
            starting_balance: Starting balance in SOL
            max_positions: Maximum concurrent positions
        """
        self.db = db  # Keep reference for compatibility
        self.starting_balance = starting_balance
        
        # Create config
        config = PaperTraderConfig(
            starting_balance_sol=starting_balance,
            max_open_positions=max_positions,
            default_position_size_sol=0.3,
            default_stop_loss_pct=-12.0,
            default_take_profit_pct=30.0,
            default_trailing_stop_pct=8.0,
            max_hold_hours=12,
            enable_auto_exits=True,  # Background monitoring enabled!
            price_check_interval_seconds=30
        )
        
        # Initialize the robust trader
        self._trader = EffectivePaperTrader(
            db_path="paper_trades_v3.db",
            config=config
        )
        
        print(f"📊 PaperTradingEngine (V3) initialized")
        print(f"   Balance: {self._trader.balance:.4f} SOL")
        print(f"   Positions: {self._trader.open_position_count}/{max_positions}")
        print(f"   Auto-exit monitoring: ENABLED")
    
    @property
    def balance(self) -> float:
        return self._trader.balance
    
    @balance.setter
    def balance(self, value: float):
        # Balance is managed internally, but allow setting for compatibility
        pass
    
    @property
    def reserved(self) -> float:
        return self._trader.reserved_balance
    
    @property
    def available_balance(self) -> float:
        return self._trader.balance - self._trader.reserved_balance
    
    def _load_positions(self):
        """Compatibility method - positions are loaded automatically"""
        pass
    
    def open_position(self, signal: Dict, decision: Dict, price: float) -> Optional[int]:
        """
        Open a paper position.
        
        Compatible with master_v2.py's call signature.
        """
        token_address = signal.get('token_address', '')
        token_symbol = signal.get('token_symbol', 'UNKNOWN')
        
        # Check if we can open (atomic check inside)
        can_open, reason = self._trader.can_open_position(token_address)
        if not can_open:
            print(f"   ⚠️ Cannot open: {reason}")
            return None
        
        # Determine position size
        size_sol = min(
            decision.get('position_size_sol', 0.3),
            self.available_balance * 0.3,
            self._trader.config.max_position_size_sol
        )
        
        if size_sol < 0.1:
            print(f"   ⚠️ Position size too small: {size_sol:.4f}")
            return None
        
        # Build rich entry context
        context = EntryContext(
            wallet_source=decision.get('wallets', ['unknown'])[0] if decision.get('wallets') else 'unknown',
            wallet_cluster=decision.get('cluster', 'UNKNOWN'),
            wallet_win_rate=decision.get('wallet_win_rate', 0),
            wallet_roi_7d=decision.get('wallet_roi_7d', 0),
            liquidity_usd=signal.get('liquidity', 0),
            volume_24h_usd=signal.get('volume_24h', 0),
            market_cap_usd=signal.get('market_cap', 0),
            token_age_hours=signal.get('token_age_hours', 0),
            conviction_score=decision.get('conviction_score', 50),
            signal_wallets_count=decision.get('wallet_count', 1),
            clusters_detected=list(decision.get('clusters', {}).keys()) if decision.get('clusters') else [],
            aggregated_signal=decision.get('wallet_count', 1) > 1,
            sol_price_usd=signal.get('sol_price_usd', 0),
            market_regime=decision.get('regime', 'NEUTRAL'),
            strategy_name=decision.get('strategy', 'default'),
            entry_reason=decision.get('reason', '')
        )
        
        # Open position with the robust trader
        position_id = self._trader.open_position(
            token_address=token_address,
            token_symbol=token_symbol,
            entry_price=price,
            size_sol=size_sol,
            context=context,
            stop_loss_pct=decision.get('stop_loss', -0.12) * 100,  # Convert to percentage
            take_profit_pct=decision.get('take_profit', 0.25) * 100,
            trailing_stop_pct=decision.get('trailing_stop', 0.08) * 100,
            max_hold_hours=decision.get('max_hold_hours', 12)
        )
        
        return position_id
    
    def check_exit_conditions(self, position: Dict, current_price: float) -> Optional[str]:
        """
        Check if position should exit.
        
        Note: The EffectivePaperTrader does this automatically in background,
        but we keep this for compatibility and manual checks.
        """
        entry_price = position.get('entry_price', 0)
        if entry_price <= 0 or current_price <= 0:
            return None
        
        pnl_pct = ((current_price / entry_price) - 1) * 100
        
        # Get exit thresholds
        stop_loss = position.get('stop_loss_pct', -12)
        take_profit = position.get('take_profit_pct', 30)
        trailing_stop = position.get('trailing_stop_pct', 8)
        max_hold = position.get('max_hold_hours', 12)
        
        # Check stop loss
        if pnl_pct <= stop_loss:
            return 'STOP_LOSS'
        
        # Check take profit
        if pnl_pct >= take_profit:
            return 'TAKE_PROFIT'
        
        # Check trailing stop
        peak_price = position.get('peak_price', entry_price)
        if current_price > peak_price:
            peak_price = current_price
        
        if peak_price > entry_price:
            peak_pnl_pct = ((peak_price / entry_price) - 1) * 100
            if peak_pnl_pct >= 15:  # Only trail after 15% profit
                from_peak_pct = ((current_price / peak_price) - 1) * 100
                if from_peak_pct <= -trailing_stop:
                    return 'TRAILING_STOP'
        
        # Check time stop
        entry_time = position.get('entry_time')
        if entry_time:
            if isinstance(entry_time, str):
                entry_time = datetime.fromisoformat(entry_time.replace('Z', ''))
            hold_hours = (datetime.now() - entry_time).total_seconds() / 3600
            if hold_hours >= max_hold:
                return 'TIME_STOP'
        
        return None
    
    def close_position(self, position_id: int, exit_reason: str, exit_price: float) -> Optional[Dict]:
        """
        Close a paper position.
        
        Compatible with master_v2.py's call signature.
        """
        # Map string reason to enum
        reason_map = {
            'STOP_LOSS': ExitReason.STOP_LOSS,
            'TAKE_PROFIT': ExitReason.TAKE_PROFIT,
            'TRAILING_STOP': ExitReason.TRAILING_STOP,
            'TIME_STOP': ExitReason.TIME_STOP,
            'SMART_EXIT': ExitReason.SMART_EXIT,
            'MANUAL': ExitReason.MANUAL,
        }
        
        reason_enum = reason_map.get(exit_reason, ExitReason.MANUAL)
        
        result = self._trader.close_position(
            position_id=position_id,
            exit_price=exit_price,
            exit_reason=reason_enum
        )
        
        if result:
            return {
                'pnl_pct': result['pnl_pct'],
                'pnl_sol': result['pnl_sol'],
                'hold_minutes': result['hold_minutes']
            }
        return None
    
    def get_open_positions(self) -> List[Dict]:
        """Get all open positions."""
        return self._trader.get_open_positions()
    
    def get_stats(self) -> Dict:
        """Get trading statistics."""
        summary = self._trader.get_performance_summary()
        
        return {
            'balance': summary['balance'],
            'starting_balance': summary['starting_balance'],
            'total_pnl': summary['total_pnl_sol'],
            'return_pct': summary['return_pct'],
            'open_positions': summary['open_positions'],
            'total_trades': summary['total_trades'],
            'win_rate': summary['win_rate'],
            'profit_factor': summary['profit_factor']
        }
    
    def get_strategy_feedback(self) -> Dict:
        """
        Get detailed feedback for strategy improvement.
        
        This is the key method for the strategist to use.
        """
        return self._trader.get_strategy_analysis(days=14)
    
    def print_status(self):
        """Print detailed status."""
        self._trader.print_status()
    
    def stop(self):
        """Stop the background monitor (call on shutdown)."""
        self._trader.stop_monitor()


# For direct testing
if __name__ == "__main__":
    # Test that it works as a drop-in replacement
    engine = PaperTradingEngine(db=None, starting_balance=10.0, max_positions=5)
    engine.print_status()
    
    print("\n📊 Stats:", engine.get_stats())
