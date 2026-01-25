"""
LEARNING PAPER TRADER INTEGRATION
=================================

This module integrates the Learning Paper Trader with the existing master_v2.py system.

To use:
1. Copy learning_paper_trader.py and fixed_notifier.py to your core/ directory
2. In master_v2.py, replace the import:
   
   # OLD:
   # from core.paper_engine_replacement import PaperTradingEngine
   
   # NEW:
   from core.learning_integration import LearningTradingEngine, LearningNotifier

3. Replace the paper engine initialization:
   
   # OLD:
   # self.paper_engine = PaperTradingEngine(db, starting_balance=10.0, max_positions=5)
   
   # NEW:
   self.paper_engine = LearningTradingEngine(db, starting_balance=10.0)

4. Replace notifier:
   
   # OLD:
   # self.notifier = Notifier()
   
   # NEW:
   self.notifier = LearningNotifier()

The integration provides:
- Unlimited position opening (for data collection)
- Comprehensive time-of-day tracking
- Fixed hold time in notifications
- Automatic learning every 6 hours
- Progressive strategy refinement
"""

import os
import sys
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

# Import the learning components
from learning_paper_trader import (
    LearningPaperTrader, 
    LearningConfig, 
    LearningPaperTradingEngine,
    ExitReason
)
from fixed_notifier import FixedNotifier


class LearningTradingEngine:
    """
    Complete trading engine with integrated learning capabilities.
    
    This is a drop-in replacement for PaperTradingEngine that:
    1. Opens positions liberally (no limit)
    2. Tracks comprehensive data including time-of-day
    3. Learns from results every 6 hours
    4. Progressively tightens filters as it learns
    """
    
    def __init__(self, db=None, starting_balance: float = 10.0, max_positions: int = None):
        """
        Initialize the learning trading engine.
        
        Args:
            db: Database instance (kept for compatibility)
            starting_balance: Starting balance in SOL
            max_positions: Ignored - we allow unlimited positions for learning
        """
        self.db = db
        
        # Create config
        config = LearningConfig(
            starting_balance_sol=starting_balance,
            max_open_positions=999,  # Effectively unlimited
            enable_auto_exits=True,
            learning_interval_hours=6.0
        )
        
        # Initialize the learning trader
        self._trader = LearningPaperTrader(
            db_path="learning_paper_trades.db",
            config=config
        )
        
        # Track learning schedule
        self._last_learning = datetime.utcnow() - timedelta(hours=5)  # Start soon
        
        print(f"\n🎓 LEARNING TRADING ENGINE INITIALIZED")
        print(f"   Phase: {self._trader.config.current_phase}")
        print(f"   Iteration: {self._trader.config.iteration_count}")
        print(f"   Balance: {self._trader.balance:.4f} SOL")
        print(f"   Positions: {self._trader.open_position_count}")
        print(f"   Max positions: UNLIMITED (learning mode)")
    
    @property
    def balance(self) -> float:
        return self._trader.balance
    
    @property
    def available_balance(self) -> float:
        return self._trader.balance - self._trader.reserved_balance
    
    @property
    def open_position_count(self) -> int:
        return self._trader.open_position_count
    
    def can_open_position(self, signal: Dict = None) -> Tuple[bool, str]:
        """Check if we can open a position"""
        return self._trader.can_open_position(signal)
    
    def open_position(self, signal: Dict, decision: Dict, price: float) -> Optional[int]:
        """
        Open a paper position.
        
        Compatible with master_v2.py call signature.
        """
        # Build signal data with wallet info
        signal_data = {
            'token_address': signal.get('token_address', ''),
            'token_symbol': signal.get('token_symbol', 'UNKNOWN'),
            'liquidity': signal.get('liquidity', 0),
            'volume_24h': signal.get('volume_24h', 0),
            'market_cap': signal.get('market_cap', 0),
            'token_age_hours': signal.get('token_age_hours', 0),
            'holder_count': signal.get('holder_count', 0),
            'conviction_score': decision.get('conviction_score', 50),
            'wallet_win_rate': signal.get('wallet_win_rate', 0.5)
        }
        
        wallet_data = {
            'address': signal.get('wallet', signal.get('wallet_address', '')),
            'win_rate': signal.get('wallet_win_rate', 0.5),
            'cluster': signal.get('wallet_cluster', 'UNKNOWN'),
            'roi_7d': signal.get('wallet_roi', 0),
            'trades_count': signal.get('wallet_trades', 0)
        }
        
        # Calculate stop loss/take profit from decision
        stop_loss = decision.get('stop_loss', -0.15)
        if stop_loss > 0:
            stop_loss = -stop_loss
        stop_loss_pct = stop_loss * 100  # Convert to percentage
        
        take_profit = decision.get('take_profit', 0.30)
        take_profit_pct = take_profit * 100
        
        return self._trader.open_position(
            token_address=signal.get('token_address', ''),
            token_symbol=signal.get('token_symbol', 'UNKNOWN'),
            entry_price=price,
            signal=signal_data,
            wallet_data=wallet_data,
            decision=decision,
            size_sol=decision.get('position_size_sol'),
            stop_loss=stop_loss_pct,
            take_profit=take_profit_pct
        )
    
    def check_exit(self, position: Dict, current_price: float) -> Optional[str]:
        """
        Check if position should exit.
        
        Note: The LearningPaperTrader handles this automatically in background,
        but we keep this for compatibility and manual checks.
        """
        entry_price = position.get('entry_price', 0)
        if entry_price <= 0 or current_price <= 0:
            return None
        
        pnl_pct = ((current_price / entry_price) - 1) * 100
        
        # Check stop loss
        stop_loss = position.get('stop_loss_pct', -15)
        if pnl_pct <= stop_loss:
            return 'STOP_LOSS'
        
        # Check take profit
        take_profit = position.get('take_profit_pct', 30)
        if pnl_pct >= take_profit:
            return 'TAKE_PROFIT'
        
        # Check trailing stop
        peak_price = position.get('peak_price', entry_price)
        trailing_stop = position.get('trailing_stop_pct', 10)
        
        if peak_price > entry_price:
            peak_pnl_pct = ((peak_price / entry_price) - 1) * 100
            if peak_pnl_pct >= 15:  # Only trail after 15% profit
                from_peak_pct = ((current_price / peak_price) - 1) * 100
                if from_peak_pct <= -trailing_stop:
                    return 'TRAILING_STOP'
        
        # Check time stop
        entry_time = position.get('entry_time')
        max_hold = position.get('max_hold_hours', 12)
        
        if entry_time:
            if isinstance(entry_time, str):
                entry_time = datetime.fromisoformat(entry_time.replace('Z', ''))
            hold_hours = (datetime.utcnow() - entry_time).total_seconds() / 3600
            if hold_hours >= max_hold:
                return 'TIME_STOP'
        
        return None
    
    def close_position(self, position_id: int, exit_reason: str, exit_price: float) -> Optional[Dict]:
        """
        Close a paper position.
        
        Returns result dict with ACCURATE hold_minutes.
        """
        reason_map = {
            'STOP_LOSS': ExitReason.STOP_LOSS,
            'TAKE_PROFIT': ExitReason.TAKE_PROFIT,
            'TRAILING_STOP': ExitReason.TRAILING_STOP,
            'TIME_STOP': ExitReason.TIME_STOP,
            'SMART_EXIT': ExitReason.SMART_EXIT,
            'STRATEGIST_EXIT': ExitReason.STRATEGIST_EXIT,
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
                'hold_minutes': result['hold_minutes'],
                'hold_duration_minutes': result['hold_minutes'],  # Alias for compatibility
                'entry_hour_utc': result.get('entry_hour_utc'),
                'exit_hour_utc': result.get('exit_hour_utc'),
                'is_win': result['is_win']
            }
        return None
    
    def get_open_positions(self) -> List[Dict]:
        """Get all open positions"""
        return self._trader.get_open_positions()
    
    def get_stats(self) -> Dict:
        """Get trading statistics"""
        summary = self._trader.get_performance_summary()
        return {
            'balance': summary.get('balance', 0),
            'starting_balance': summary.get('starting_balance', 0),
            'total_pnl': summary.get('total_pnl_sol', 0),
            'return_pct': summary.get('return_pct', 0),
            'open_positions': summary.get('open_positions', 0),
            'total_trades': summary.get('total_trades', 0),
            'win_rate': summary.get('win_rate', 0),
            'profit_factor': 0,  # TODO: Calculate
            'phase': summary.get('phase', 'exploration'),
            'iteration': summary.get('iteration', 0),
            'blocked_hours': summary.get('blocked_hours', []),
            'preferred_hours': summary.get('preferred_hours', []),
            'min_conviction': summary.get('min_conviction', 30),
            'min_wallet_wr': summary.get('min_wallet_wr', 0.40)
        }
    
    def should_run_learning(self) -> bool:
        """Check if it's time to run the learning loop"""
        hours_since = (datetime.utcnow() - self._last_learning).total_seconds() / 3600
        return hours_since >= self._trader.config.learning_interval_hours
    
    def run_learning(self, force: bool = False, notifier=None) -> Dict:
        """
        Run learning iteration and optionally notify.
        
        Returns learning results.
        """
        if not force and not self.should_run_learning():
            return {'status': 'skipped', 'reason': 'not time yet'}
        
        self._last_learning = datetime.utcnow()
        results = self._trader.run_learning_iteration(force=True)
        
        # Send notification if enabled
        if notifier and hasattr(notifier, 'send_learning_alert'):
            try:
                notifier.send_learning_alert(results)
            except Exception as e:
                print(f"  ⚠️ Learning notification error: {e}")
        
        return results
    
    def get_strategy_feedback(self) -> Dict:
        """Get detailed feedback for strategy improvement"""
        return {
            'summary': self._trader.get_performance_summary(),
            'diurnal': self._trader.get_diurnal_report(),
            'config': {
                'phase': self._trader.config.current_phase,
                'iteration': self._trader.config.iteration_count,
                'min_conviction': self._trader.config.min_conviction_score,
                'min_wallet_wr': self._trader.config.min_wallet_win_rate,
                'blocked_hours': self._trader.config.blocked_hours_utc,
                'preferred_hours': self._trader.config.preferred_hours_utc
            }
        }
    
    def get_diurnal_report(self) -> Dict:
        """Get time-of-day performance report"""
        return self._trader.get_diurnal_report()
    
    def print_status(self):
        """Print detailed status"""
        self._trader.print_status()
    
    def stop(self):
        """Stop the background monitor"""
        self._trader.stop_monitor()


class LearningNotifier(FixedNotifier):
    """
    Extended notifier with learning-specific notifications.
    
    Inherits from FixedNotifier which has the hold time bug fix.
    """
    
    def send_entry_with_learning(self, signal: Dict, decision: Dict, learning_stats: Dict):
        """Send entry notification with learning context"""
        current_hour = datetime.utcnow().hour
        phase = learning_stats.get('phase', 'unknown')
        blocked = learning_stats.get('blocked_hours', [])
        
        # Warning if trading in a blocked hour (shouldn't happen but just in case)
        hour_warning = " ⚠️ BLOCKED HOUR!" if current_hour in blocked else ""
        
        msg = f"""🎯 <b>ENTRY SIGNAL</b>

Token: ${signal.get('token_symbol', 'UNKNOWN')}
Conviction: {decision.get('conviction_score', 0):.0f}/100
Wallets: {decision.get('wallet_count', 1)}
Regime: {decision.get('regime', 'UNKNOWN')}
Position: {decision.get('position_size_sol', 0):.3f} SOL
Stop: {decision.get('stop_loss', 0)*100:.0f}%
Target: {decision.get('take_profit', 0)*100:.0f}%

⏰ Hour (UTC): {current_hour:02d}:00{hour_warning}
🎓 Phase: {phase}
📊 Iteration: {learning_stats.get('iteration', 0)}"""
        
        self.send(msg)
    
    def send_phase_change(self, old_phase: str, new_phase: str, stats: Dict):
        """Send phase change notification"""
        msg = f"""🎓 <b>LEARNING PHASE CHANGE!</b>

{old_phase.upper()} → {new_phase.upper()}

📊 Current Stats:
• Trades: {stats.get('total_trades', 0)}
• Win Rate: {stats.get('win_rate', 0):.1%}
• PnL: {stats.get('total_pnl', 0):+.4f} SOL

This means the strategy is becoming more refined.
Expect fewer but higher quality trades."""
        
        self.send(msg)
    
    def send_daily_learning_summary(self, results: Dict):
        """Send comprehensive daily learning summary"""
        perf = results.get('performance', {})
        hourly = results.get('hourly_analysis', {})
        
        # Find best and worst hours
        best_hour = max(hourly.items(), key=lambda x: x[1].get('pnl', 0)) if hourly else (None, {})
        worst_hour = min(hourly.items(), key=lambda x: x[1].get('pnl', 0)) if hourly else (None, {})
        
        msg = f"""📊 <b>DAILY LEARNING SUMMARY</b>

📈 Performance (Last 7 Days):
• Total Trades: {perf.get('total_trades', 0)}
• Win Rate: {perf.get('win_rate', 0):.1%}
• Total PnL: {perf.get('total_pnl', 0):+.4f} SOL
• Avg Hold: {perf.get('avg_hold_minutes', 0):.0f} min

⏰ Time Analysis:"""
        
        if best_hour[0] is not None:
            msg += f"\n• Best Hour: {best_hour[0]:02d}:00 UTC ({best_hour[1].get('pnl', 0):+.4f} SOL)"
        if worst_hour[0] is not None:
            msg += f"\n• Worst Hour: {worst_hour[0]:02d}:00 UTC ({worst_hour[1].get('pnl', 0):+.4f} SOL)"
        
        blocked = results.get('blocked_hours', [])
        preferred = results.get('preferred_hours', [])
        
        if blocked:
            msg += f"\n\n🚫 Blocked Hours: {blocked}"
        if preferred:
            msg += f"\n⭐ Preferred Hours: {preferred}"
        
        # Phase info
        msg += f"\n\n🎓 Phase: {results.get('phase', 'unknown')}"
        msg += f"\n📊 Iteration: {results.get('iteration', 0)}"
        
        # Goal progress
        if perf.get('win_rate', 0) >= 0.50 and perf.get('total_pnl', 0) > 0:
            msg += "\n\n✅ ON TRACK for profitability goal!"
        elif perf.get('total_trades', 0) < 50:
            msg += "\n\n📚 Still collecting data..."
        else:
            msg += "\n\n⚠️ Need more refinement"
        
        self.send(msg)


# =============================================================================
# HELPER FUNCTIONS FOR MASTER_V2.PY MODIFICATION
# =============================================================================

def patch_master_for_learning():
    """
    Prints instructions for patching master_v2.py to use the learning system.
    """
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    LEARNING PAPER TRADER INTEGRATION GUIDE                   ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  To integrate the Learning Paper Trader with master_v2.py:                   ║
║                                                                              ║
║  1. Copy these files to your core/ directory:                                ║
║     - learning_paper_trader.py                                               ║
║     - fixed_notifier.py                                                      ║
║     - learning_integration.py (this file)                                    ║
║                                                                              ║
║  2. In master_v2.py, change the imports:                                     ║
║                                                                              ║
║     # Add at top of file:                                                    ║
║     from core.learning_integration import LearningTradingEngine              ║
║     from core.learning_integration import LearningNotifier                   ║
║                                                                              ║
║  3. In TradingSystem.__init__, replace:                                      ║
║                                                                              ║
║     # OLD:                                                                   ║
║     # self.paper_engine = PaperTradingEngine(...)                            ║
║                                                                              ║
║     # NEW:                                                                   ║
║     self.paper_engine = LearningTradingEngine(                               ║
║         db=self.db,                                                          ║
║         starting_balance=10.0                                                ║
║     )                                                                        ║
║                                                                              ║
║  4. Replace the notifier:                                                    ║
║                                                                              ║
║     # OLD:                                                                   ║
║     # self.notifier = Notifier()                                             ║
║                                                                              ║
║     # NEW:                                                                   ║
║     self.notifier = LearningNotifier()                                       ║
║                                                                              ║
║  5. In background_tasks(), add learning loop call:                           ║
║                                                                              ║
║     # In the background loop, after existing learning check:                 ║
║     if trading_system.paper_engine.should_run_learning():                    ║
║         result = trading_system.paper_engine.run_learning(                   ║
║             force=True,                                                      ║
║             notifier=trading_system.notifier                                 ║
║         )                                                                    ║
║         print(f"Learning complete: {result.get('status')}")                  ║
║                                                                              ║
║  6. In _close_position method, pass result to notifier:                      ║
║                                                                              ║
║     # Change send_exit_alert call to include result:                         ║
║     self.notifier.send_exit_alert(position, reason, pnl_pct, result)         ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)


# For direct testing
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'help':
        patch_master_for_learning()
    else:
        # Test initialization
        print("Testing Learning Trading Engine...")
        engine = LearningTradingEngine(db=None, starting_balance=10.0)
        engine.print_status()
        
        print("\n" + "=" * 60)
        print("For integration instructions, run: python learning_integration.py help")
        print("=" * 60)
