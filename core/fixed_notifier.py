"""
FIXED NOTIFIER
==============

Telegram notification system with FIXED hold time display.

The bug was that send_exit_alert was looking for 'hold_duration_minutes'
but the position dict coming from the paper trader had 'hold_minutes'.

This version:
1. Checks multiple field names for hold time
2. Calculates hold time from timestamps if field is missing
3. Displays comprehensive exit information
"""

import os
import requests
from datetime import datetime
from typing import Dict, Optional


class FixedNotifier:
    """
    Telegram notifier with fixed hold time display.
    """
    
    def __init__(self, token: str = None, chat_id: str = None):
        self.token = token or os.getenv('TELEGRAM_BOT_TOKEN')
        self.chat_id = chat_id or os.getenv('TELEGRAM_CHAT_ID')
        self.enabled = bool(self.token and self.chat_id)
        self._last_status_sent = None
    
    def send(self, message: str):
        """Send a message to Telegram"""
        if not self.enabled:
            return
        try:
            url = f"https://api.telegram.org/bot{self.token}/sendMessage"
            requests.post(
                url, 
                json={
                    'chat_id': self.chat_id, 
                    'text': message, 
                    'parse_mode': 'HTML'
                }, 
                timeout=10
            )
        except Exception as e:
            print(f"  ⚠️ Telegram error: {e}")
    
    def send_entry_alert(self, signal: Dict, decision: Dict):
        """Send entry notification"""
        current_hour = datetime.utcnow().hour
        
        msg = f"""🎯 <b>ENTRY SIGNAL</b>

Token: ${signal.get('token_symbol', 'UNKNOWN')}
Conviction: {decision.get('conviction_score', 0):.0f}/100
Wallets: {decision.get('wallet_count', 1)}
Regime: {decision.get('regime', 'UNKNOWN')}
Position: {decision.get('position_size_sol', 0):.3f} SOL
Stop: {decision.get('stop_loss', 0)*100:.0f}%
Target: {decision.get('take_profit', 0)*100:.0f}%
LLM: {'✅' if decision.get('llm_called') else '❌'}
Hour (UTC): {current_hour:02d}:00
Phase: {decision.get('learning_phase', 'N/A')}"""
        
        self.send(msg)
    
    def send_exit_alert(self, position: Dict, reason: str, pnl_pct: float, result: Dict = None):
        """
        Send exit notification with FIXED hold time.
        
        Args:
            position: The position dict (may have different field names)
            reason: Exit reason string
            pnl_pct: PnL percentage
            result: Optional result dict from close_position (has accurate hold_minutes)
        """
        emoji = "🟢" if pnl_pct > 0 else "🔴"
        
        # FIX: Get hold time from multiple possible sources
        hold_mins = self._get_hold_time(position, result)
        
        # Format hold time nicely
        if hold_mins >= 60:
            hold_str = f"{hold_mins/60:.1f} hours ({hold_mins:.0f} min)"
        else:
            hold_str = f"{hold_mins:.0f} min"
        
        # Get additional info if available
        pnl_sol = result.get('pnl_sol', position.get('pnl_sol', 0)) if result else position.get('pnl_sol', 0)
        entry_hour = position.get('entry_hour_utc', 'N/A')
        exit_hour = datetime.utcnow().hour
        
        msg = f"""{emoji} <b>EXIT</b>

Token: ${position.get('token_symbol', 'UNKNOWN')}
Reason: {reason}
P&L: {pnl_pct:+.1f}% ({pnl_sol:+.4f} SOL)
Hold: {hold_str}
Entry Hour: {entry_hour}:00 UTC
Exit Hour: {exit_hour:02d}:00 UTC"""
        
        self.send(msg)
    
    def _get_hold_time(self, position: Dict, result: Dict = None) -> float:
        """
        Get hold time from multiple possible sources.
        
        The fix for the "Hold: 0 minutes" bug.
        """
        # Priority 1: From result dict (most accurate)
        if result:
            if result.get('hold_minutes'):
                return result['hold_minutes']
            if result.get('hold_duration_minutes'):
                return result['hold_duration_minutes']
        
        # Priority 2: From position dict (check multiple field names)
        if position.get('hold_duration_minutes'):
            return position['hold_duration_minutes']
        if position.get('hold_minutes'):
            return position['hold_minutes']
        
        # Priority 3: Calculate from timestamps
        entry_time = position.get('entry_time')
        exit_time = position.get('exit_time') or datetime.utcnow()
        
        if entry_time:
            if isinstance(entry_time, str):
                try:
                    entry_time = datetime.fromisoformat(entry_time.replace('Z', ''))
                except:
                    return 0
            
            if isinstance(exit_time, str):
                try:
                    exit_time = datetime.fromisoformat(exit_time.replace('Z', ''))
                except:
                    exit_time = datetime.utcnow()
            
            return (exit_time - entry_time).total_seconds() / 60
        
        return 0
    
    def send_discovery_alert(self, wallet: str, performance: Dict):
        """Send wallet discovery notification"""
        msg = f"""🎯 <b>NEW WALLET DISCOVERED</b>

Address: <code>{wallet}</code>
Win Rate: {performance.get('win_rate', 0):.1%}
PnL (7d): {performance.get('pnl', 0):.2f} SOL
Completed Swings: {performance.get('completed_swings', 0)}
Avg Hold: {performance.get('avg_hold_hours', 0):.1f}h

✅ Automatically added to webhook!"""
        
        self.send(msg)
    
    def send_hourly_status(self, diagnostics: Dict, stats: Dict):
        """Send hourly status update"""
        now = datetime.utcnow()
        
        # Only send every hour
        if self._last_status_sent:
            if (now - self._last_status_sent).total_seconds() < 3500:
                return
        
        self._last_status_sent = now
        
        msg = f"""📊 <b>HOURLY STATUS</b>

⏰ {now.strftime('%Y-%m-%d %H:%M')} UTC

📈 Performance:
• Balance: {stats.get('balance', 0):.4f} SOL
• Return: {stats.get('return_pct', 0):+.1f}%
• Win Rate: {stats.get('win_rate', 0):.0%}
• Open Positions: {stats.get('open_positions', 0)}

🎓 Learning:
• Phase: {stats.get('phase', 'N/A')}
• Iteration: {stats.get('iteration', 0)}
• Blocked Hours: {stats.get('blocked_hours', [])}
• Preferred Hours: {stats.get('preferred_hours', [])}

🔧 System:
• Webhooks: {diagnostics.get('webhooks', {}).get('received', 0)} received
• Uptime: {diagnostics.get('uptime_hours', 0):.1f}h"""
        
        self.send(msg)
    
    def send_learning_alert(self, results: Dict):
        """Send learning iteration results"""
        perf = results.get('performance', {})
        
        msg = f"""🎓 <b>LEARNING ITERATION #{results.get('iteration', 0)}</b>

📊 Performance:
• Trades: {perf.get('total_trades', 0)}
• Win Rate: {perf.get('win_rate', 0):.1%}
• Total PnL: {perf.get('total_pnl', 0):+.4f} SOL

📈 Current Phase: {results.get('phase', 'unknown')}
"""
        
        # Add phase change if any
        if results.get('phase_changed'):
            msg += f"\n🎯 <b>PHASE CHANGE: {results.get('new_phase')}</b>\n"
        
        # Add blocked/preferred hours
        blocked = results.get('blocked_hours', [])
        preferred = results.get('preferred_hours', [])
        
        if blocked:
            msg += f"\n🚫 Blocked Hours: {blocked}"
        if preferred:
            msg += f"\n⭐ Preferred Hours: {preferred}"
        
        # Add recommendations (first 3)
        recs = results.get('recommendations', [])[:3]
        if recs:
            msg += "\n\n💡 Recommendations:"
            for rec in recs:
                msg += f"\n• {rec[:100]}"
        
        self.send(msg)
    
    def send_diurnal_summary(self, report: Dict):
        """Send diurnal (time-of-day) performance summary"""
        # Find best and worst hours
        best_hours = []
        worst_hours = []
        
        for hour, data in report.items():
            if data.get('trades', 0) >= 5:
                if data.get('win_rate', 0) >= 0.55:
                    best_hours.append((hour, data))
                elif data.get('win_rate', 0) < 0.35:
                    worst_hours.append((hour, data))
        
        best_hours.sort(key=lambda x: x[1].get('pnl_sol', 0), reverse=True)
        worst_hours.sort(key=lambda x: x[1].get('pnl_sol', 0))
        
        msg = "⏰ <b>DIURNAL PERFORMANCE SUMMARY</b>\n"
        
        if best_hours:
            msg += "\n✅ <b>Best Trading Hours:</b>"
            for hour, data in best_hours[:3]:
                msg += f"\n  • {hour:02d}:00 UTC: {data.get('win_rate', 0):.0%} WR, {data.get('pnl_sol', 0):+.4f} SOL"
        
        if worst_hours:
            msg += "\n\n❌ <b>Worst Trading Hours:</b>"
            for hour, data in worst_hours[:3]:
                msg += f"\n  • {hour:02d}:00 UTC: {data.get('win_rate', 0):.0%} WR, {data.get('pnl_sol', 0):+.4f} SOL"
        
        self.send(msg)


# For backwards compatibility
Notifier = FixedNotifier


if __name__ == "__main__":
    # Test the notifier
    notifier = FixedNotifier()
    
    if notifier.enabled:
        print("Notifier enabled, sending test message...")
        notifier.send("🧪 Test message from FixedNotifier")
    else:
        print("Notifier not enabled (missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID)")
