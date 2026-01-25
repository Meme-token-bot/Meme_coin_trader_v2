"""
MASTER V2 - Webhook-Based Trading System with Learning Paper Trader
Trading System V2 - Uses Helius webhooks + smart discovery + LEARNING MODE

UPDATED: Integrated Learning Paper Trader with:
1. Unlimited position opening for data collection
2. Comprehensive time-of-day tracking for diurnal analysis
3. Fixed hold time in Telegram notifications
4. Automatic learning loop every 6 hours
5. Progressive strategy refinement

COMPLETE VERSION - Ready to run
"""

import os
import sys
import time
import json
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from collections import deque

from flask import Flask, request, jsonify
from dotenv import load_dotenv

load_dotenv()

from core.database_v2 import DatabaseV2
from core.discovery_integration import HistorianV8 as Historian
from core.strategist_v2 import Strategist
from core.discovery_config import config as discovery_config
from infrastructure.helius_webhook_manager import HeliusWebhookManager

# ============================================================================
# CHANGE 1: Import Learning Paper Trader components
# ============================================================================
from core.learning_paper_trader import (
    LearningPaperTrader, 
    LearningConfig, 
    ExitReason
)


@dataclass
class MasterConfig:
    """Master configuration - now uses discovery_config for discovery settings"""
    webhook_host: str = '0.0.0.0'
    webhook_port: int = 5000
    position_check_interval: int = 300
    discovery_enabled: bool = True
    max_token_lookups_per_minute: int = 20
    max_api_calls_per_hour: int = 500
    paper_trading_enabled: bool = True
    paper_starting_balance: float = 10.0
    use_llm: bool = True
    max_open_positions: int = 999  # CHANGED: Unlimited for learning mode
    max_position_size_sol: float = 1.0
    
    @property
    def discovery_interval_hours(self) -> int:
        """Get discovery interval from discovery_config"""
        return discovery_config.discovery_interval_hours
    
    @property
    def discovery_api_budget(self) -> int:
        """Get discovery API budget from discovery_config"""
        return discovery_config.max_api_calls_per_discovery


CONFIG = MasterConfig()


class RateLimiter:
    def __init__(self, max_per_minute: int = 20, max_per_hour: int = 500):
        self.max_per_minute = max_per_minute
        self.max_per_hour = max_per_hour
        self.minute_calls = deque()
        self.hour_calls = deque()
    
    def can_call(self) -> bool:
        now = datetime.now()
        minute_ago = now - timedelta(minutes=1)
        hour_ago = now - timedelta(hours=1)
        
        while self.minute_calls and self.minute_calls[0] < minute_ago:
            self.minute_calls.popleft()
        while self.hour_calls and self.hour_calls[0] < hour_ago:
            self.hour_calls.popleft()
        
        if len(self.minute_calls) >= self.max_per_minute:
            return False
        if len(self.hour_calls) >= self.max_per_hour:
            return False
        return True
    
    def record_call(self):
        now = datetime.now()
        self.minute_calls.append(now)
        self.hour_calls.append(now)
    
    def get_stats(self) -> Dict:
        return {
            'calls_last_minute': len(self.minute_calls),
            'calls_last_hour': len(self.hour_calls),
            'limit_per_minute': self.max_per_minute,
            'limit_per_hour': self.max_per_hour
        }


@dataclass
class DiagnosticsTracker:
    start_time: datetime = field(default_factory=datetime.now)
    last_webhook_received: Optional[datetime] = None
    webhooks_received: int = 0
    webhooks_processed: int = 0
    webhooks_skipped: int = 0
    api_calls_made: int = 0
    api_errors: int = 0
    positions_opened: int = 0
    positions_closed: int = 0
    llm_calls: int = 0
    discoveries_run: int = 0
    wallets_discovered: int = 0
    learning_iterations: int = 0  # NEW: Track learning iterations
    recent_events: deque = field(default_factory=lambda: deque(maxlen=50))
    
    def log_event(self, event_type: str, details: str = ""):
        self.recent_events.append({
            'time': datetime.now().isoformat(),
            'type': event_type,
            'details': details
        })
    
    def to_dict(self) -> Dict:
        uptime = datetime.now() - self.start_time
        minutes_since_webhook = None
        if self.last_webhook_received:
            minutes_since_webhook = (datetime.now() - self.last_webhook_received).total_seconds() / 60
        
        return {
            'uptime_hours': uptime.total_seconds() / 3600,
            'last_webhook_received': self.last_webhook_received.isoformat() if self.last_webhook_received else None,
            'minutes_since_last_webhook': minutes_since_webhook,
            'webhooks': {'received': self.webhooks_received, 'processed': self.webhooks_processed, 'skipped': self.webhooks_skipped},
            'api': {'calls': self.api_calls_made, 'errors': self.api_errors},
            'positions': {'opened': self.positions_opened, 'closed': self.positions_closed},
            'discovery': {'runs': self.discoveries_run, 'wallets_found': self.wallets_discovered},
            'learning_iterations': self.learning_iterations,
            'llm_calls': self.llm_calls,
            'recent_events': list(self.recent_events)[-10:]
        }


# ============================================================================
# CHANGE 2: Fixed Notifier with accurate hold time
# ============================================================================
class Notifier:
    """Telegram notifier with FIXED hold time display"""
    
    def __init__(self, token: str = None, chat_id: str = None):
        self.token = token or os.getenv('TELEGRAM_BOT_TOKEN')
        self.chat_id = chat_id or os.getenv('TELEGRAM_CHAT_ID')
        self.enabled = bool(self.token and self.chat_id)
        self._last_status_sent = None
    
    def send(self, message: str):
        if not self.enabled:
            return
        try:
            import requests
            url = f"https://api.telegram.org/bot{self.token}/sendMessage"
            requests.post(url, json={'chat_id': self.chat_id, 'text': message, 'parse_mode': 'HTML'}, timeout=10)
        except Exception as e:
            print(f"  ⚠️ Telegram error: {e}")
    
    def send_entry_alert(self, signal: Dict, decision: Dict):
        """Send entry notification with learning context"""
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
Hour (UTC): {current_hour:02d}:00"""
        self.send(msg)
    
    def send_exit_alert(self, position: Dict, reason: str, pnl_pct: float, result: Dict = None):
        """
        Send exit notification with FIXED hold time.
        
        CHANGE 4: Now accepts result dict for accurate hold time
        """
        emoji = "🟢" if pnl_pct > 0 else "🔴"
        
        # FIX: Get hold time from multiple sources
        hold_mins = self._get_hold_time(position, result)
        
        # Format hold time nicely
        if hold_mins >= 60:
            hold_str = f"{hold_mins/60:.1f}h ({hold_mins:.0f}m)"
        else:
            hold_str = f"{hold_mins:.0f} min"
        
        # Get PnL in SOL
        pnl_sol = 0
        if result:
            pnl_sol = result.get('pnl_sol', 0)
        
        # Get entry/exit hours for diurnal tracking display
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
        This is THE FIX for the "Hold: 0 minutes" bug.
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
        msg = f"""🎯 <b>NEW WALLET DISCOVERED</b>

Address: <code>{wallet}</code>
Win Rate: {performance.get('win_rate', 0):.1%}
PnL (7d): {performance.get('pnl', 0):.2f} SOL
Completed Swings: {performance.get('completed_swings', 0)}
Avg Hold: {performance.get('avg_hold_hours', 0):.1f}h

✅ Automatically added to webhook!"""
        self.send(msg)
    
    def send_hourly_status(self, diagnostics: DiagnosticsTracker, paper_stats: Dict):
        now = datetime.now()
        if self._last_status_sent and (now - self._last_status_sent).total_seconds() < 3600:
            return
        self._last_status_sent = now
        
        diag = diagnostics.to_dict()
        last_webhook = f"{diag['minutes_since_last_webhook']:.0f}m ago" if diag['minutes_since_last_webhook'] else "Never"
        
        # Include learning info
        phase = paper_stats.get('phase', 'N/A')
        iteration = paper_stats.get('iteration', 0)
        blocked_hours = paper_stats.get('blocked_hours', [])
        
        msg = f"""📊 <b>Hourly Status</b>

Uptime: {diag['uptime_hours']:.1f}h
Webhooks: {diag['webhooks']['received']} received
Last webhook: {last_webhook}
Paper: {paper_stats.get('balance', 0):.2f} SOL ({paper_stats.get('return_pct', 0):+.1f}%)
Open positions: {paper_stats.get('open_positions', 0)}
Win rate: {paper_stats.get('win_rate', 0):.0%}

🎓 Learning:
Phase: {phase}
Iteration: {iteration}
Blocked hours: {blocked_hours if blocked_hours else 'None'}"""
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
        
        if results.get('phase_changed'):
            msg += f"\n🎯 <b>PHASE CHANGE: {results.get('new_phase')}</b>\n"
        
        blocked = results.get('blocked_hours', [])
        preferred = results.get('preferred_hours', [])
        
        if blocked:
            msg += f"\n🚫 Blocked Hours: {blocked}"
        if preferred:
            msg += f"\n⭐ Preferred Hours: {preferred}"
        
        recs = results.get('recommendations', [])[:3]
        if recs:
            msg += "\n\n💡 Recommendations:"
            for rec in recs:
                msg += f"\n• {rec[:80]}"
        
        self.send(msg)


# ============================================================================
# CHANGE 3: Learning Paper Trading Engine (drop-in replacement)
# ============================================================================
class LearningPaperTradingEngine:
    """
    Drop-in replacement for PaperTradingEngine with learning capabilities.
    
    Key differences:
    - No position limits (collects maximum data)
    - Comprehensive time tracking
    - Automatic learning every 6 hours
    - Progressive threshold tightening
    """
    
    def __init__(self, db, starting_balance: float = 10.0, max_positions: int = None):
        self.db = db
        
        config = LearningConfig(
            starting_balance_sol=starting_balance,
            max_open_positions=999,  # No limit for learning
            enable_auto_exits=True,
            learning_interval_hours=6.0
        )
        
        self._trader = LearningPaperTrader(
            db_path="learning_paper_trades.db",
            config=config
        )
        
        self._last_learning = datetime.utcnow() - timedelta(hours=5)
        
        print(f"🎓 Learning Paper Trading Engine initialized")
        print(f"   Phase: {self._trader.config.current_phase}")
        print(f"   Balance: {self._trader.balance:.4f} SOL")
        print(f"   Positions: {self._trader.open_position_count}")
        print(f"   Mode: LEARNING (unlimited positions)")
    
    @property
    def balance(self) -> float:
        return self._trader.balance
    
    @property
    def available_balance(self) -> float:
        return self._trader.balance - self._trader.reserved_balance
    
    def open_position(self, signal: Dict, decision: Dict, price: float) -> Optional[int]:
        """Open a paper position with learning tracking"""
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
        
        stop_loss = decision.get('stop_loss', -0.15)
        if stop_loss > 0:
            stop_loss = -stop_loss
        stop_loss_pct = stop_loss * 100
        
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
    
    def check_exit_conditions(self, position: Dict, current_price: float) -> Optional[str]:
        """Check if position should exit (handled by background monitor)"""
        entry_price = position.get('entry_price', 0)
        if entry_price <= 0 or current_price <= 0:
            return None
        
        pnl_pct = ((current_price / entry_price) - 1) * 100
        
        if pnl_pct <= position.get('stop_loss_pct', -15):
            return 'STOP_LOSS'
        if pnl_pct >= position.get('take_profit_pct', 30):
            return 'TAKE_PROFIT'
        
        peak_price = position.get('peak_price', entry_price)
        trailing_stop = position.get('trailing_stop_pct', 10)
        
        if peak_price > entry_price:
            peak_pnl_pct = ((peak_price / entry_price) - 1) * 100
            if peak_pnl_pct >= 15:
                from_peak_pct = ((current_price / peak_price) - 1) * 100
                if from_peak_pct <= -trailing_stop:
                    return 'TRAILING_STOP'
        
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
        """Close a paper position with accurate hold time tracking"""
        reason_map = {
            'STOP_LOSS': ExitReason.STOP_LOSS,
            'TAKE_PROFIT': ExitReason.TAKE_PROFIT,
            'TRAILING_STOP': ExitReason.TRAILING_STOP,
            'TIME_STOP': ExitReason.TIME_STOP,
            'SMART_EXIT': ExitReason.SMART_EXIT,
            'MANUAL': ExitReason.MANUAL,
        }
        
        # Handle exit reasons that might have extra text
        clean_reason = exit_reason.split(':')[0] if ':' in exit_reason else exit_reason
        reason_enum = reason_map.get(clean_reason, ExitReason.MANUAL)
        
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
                'hold_duration_minutes': result['hold_minutes'],
                'entry_hour_utc': result.get('entry_hour_utc'),
                'exit_hour_utc': result.get('exit_hour_utc'),
                'is_win': result['is_win']
            }
        return None
    
    def get_open_positions(self) -> List[Dict]:
        return self._trader.get_open_positions()
    
    def get_stats(self) -> Dict:
        """Get trading statistics including learning info"""
        summary = self._trader.get_performance_summary()
        return {
            'balance': summary.get('balance', 0),
            'starting_balance': summary.get('starting_balance', 0),
            'total_pnl': summary.get('total_pnl_sol', 0),
            'return_pct': summary.get('return_pct', 0),
            'open_positions': summary.get('open_positions', 0),
            'total_trades': summary.get('total_trades', 0),
            'win_rate': summary.get('win_rate', 0),
            'profit_factor': 0,
            'phase': summary.get('phase', 'exploration'),
            'iteration': summary.get('iteration', 0),
            'blocked_hours': summary.get('blocked_hours', []),
            'preferred_hours': summary.get('preferred_hours', []),
            'min_conviction': summary.get('min_conviction', 30),
            'min_wallet_wr': summary.get('min_wallet_wr', 0.40)
        }
    
    def should_run_learning(self) -> bool:
        """Check if it's time to run learning"""
        hours_since = (datetime.utcnow() - self._last_learning).total_seconds() / 3600
        return hours_since >= self._trader.config.learning_interval_hours
    
    def run_learning(self, force: bool = False, notifier=None) -> Dict:
        """Run learning iteration"""
        if not force and not self.should_run_learning():
            return {'status': 'skipped', 'reason': 'not time yet'}
        
        self._last_learning = datetime.utcnow()
        results = self._trader.run_learning_iteration(force=True)
        
        if notifier and hasattr(notifier, 'send_learning_alert'):
            try:
                notifier.send_learning_alert(results)
            except Exception as e:
                print(f"  ⚠️ Learning notification error: {e}")
        
        return results
    
    def get_diurnal_report(self) -> Dict:
        """Get time-of-day performance report"""
        return self._trader.get_diurnal_report()
    
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
    
    def print_status(self):
        self._trader.print_status()
    
    def stop(self):
        self._trader.stop_monitor()


class TradingSystem:
    def __init__(self):
        print("\n" + "="*70)
        print("🚀 TRADING SYSTEM V2 (LEARNING MODE)")
        print("="*70)
        
        self.helius_key = os.getenv('HELIUS_KEY')
        self.anthropic_key = os.getenv('ANTHROPIC_API_KEY')
        self.webhook_id = os.getenv('HELIUS_WEBHOOK_ID')
        
        if not self.helius_key:
            raise ValueError("HELIUS_KEY not set!")
        
        print("\n📦 Loading Components...")
        
        self.db = DatabaseV2()
        
        self.strategist = Strategist(self.db, self.anthropic_key)
        
        self.notifier = Notifier()
        print(f"  {'✅' if self.notifier.enabled else '⚠️'} Telegram: {'enabled' if self.notifier.enabled else 'disabled'}")
        
        # ====================================================================
        # CHANGE 3: Use Learning Paper Trading Engine
        # ====================================================================
        if CONFIG.paper_trading_enabled:
            self.paper_engine = LearningPaperTradingEngine(
                self.db, 
                starting_balance=CONFIG.paper_starting_balance,
                max_positions=CONFIG.max_open_positions
            )
            print(f"  ✅ Learning Paper Trading (Balance: {self.paper_engine.balance:.2f} SOL)")
        else:
            self.paper_engine = None
        
        self.rate_limiter = RateLimiter(CONFIG.max_token_lookups_per_minute, CONFIG.max_api_calls_per_hour)
        print(f"  ✅ Rate Limiter ({CONFIG.max_token_lookups_per_minute}/min, {CONFIG.max_api_calls_per_hour}/hr)")
        
        self.diagnostics = DiagnosticsTracker()
        print("  ✅ Diagnostics Tracker")
        
        webhook_url = os.getenv('WEBHOOK_URL', f'http://0.0.0.0:{CONFIG.webhook_port}/webhook/helius')

        try:
            from infrastructure.multi_webhook_manager import MultiWebhookManager
            self.multi_webhook_manager = MultiWebhookManager(self.helius_key, webhook_url, self.db)
            print(f"  ✅ Multi-Webhook Manager (unlimited scaling)")
        except Exception as e:
            self.multi_webhook_manager = None
            print(f"  ⚠️  Multi-Webhook disabled: {e}")

        if CONFIG.discovery_enabled:
            discovery_key = os.getenv('HELIUS_DISCOVERY_KEY')
            self.historian = Historian(
                self.db, 
                self.helius_key, 
                discovery_key,
                multi_webhook_manager=self.multi_webhook_manager
            )
            print(f"  ✅ Discovery (every {CONFIG.discovery_interval_hours}h, budget: {CONFIG.discovery_api_budget:,})")
        else:
            self.historian = None
            print("  ⚠️ Discovery disabled")
        
        self.start_time = datetime.now()
        
        print("\n" + "="*70)
        print("✅ TRADING SYSTEM V2 (LEARNING MODE) READY")
        print("="*70)
        self._print_status()
    
    def _print_status(self):
        status = self.strategist.get_status()
        llm_cost = self.strategist.get_llm_cost_today()
        
        print(f"\n📊 Status:")
        print(f"  Wallets tracked: {self.db.get_wallet_count()}")
        print(f"  Regime: {status['regime']} ({status['confidence']:.0%})")
        print(f"  Champion: {status['champion']}")
        print(f"  LLM: {'✅' if status['llm_enabled'] else '❌'} | Today: {llm_cost['calls']} calls (${llm_cost['cost_usd']:.4f})")
        
        if self.paper_engine:
            stats = self.paper_engine.get_stats()
            print(f"  Paper: {stats['balance']:.2f} SOL ({stats['return_pct']:+.1f}%) | {stats['open_positions']} open")
            print(f"  Learning Phase: {stats.get('phase', 'N/A')} | Iteration: {stats.get('iteration', 0)}")
            blocked = stats.get('blocked_hours', [])
            if blocked:
                print(f"  Blocked Hours: {blocked}")
        
        rate_stats = self.rate_limiter.get_stats()
        print(f"  Rate limit: {rate_stats['calls_last_hour']}/{rate_stats['limit_per_hour']} calls/hr")
        
        print(f"\n📋 Discovery Config:")
        print(f"  Interval: {CONFIG.discovery_interval_hours}h")
        print(f"  Budget: {CONFIG.discovery_api_budget:,} credits")
        print(f"  Max wallets/day: {discovery_config.max_new_wallets_per_day}")
        print(f"  Max total wallets: {discovery_config.max_total_wallets}")
    
    def process_webhook(self, tx: Dict) -> Dict:
        signature = tx.get('signature', '')
        tx_type = tx.get('type', '')
        fee_payer = tx.get('feePayer', '')
        
        self.diagnostics.webhooks_received += 1
        self.diagnostics.last_webhook_received = datetime.now()
        
        result = {'processed': False, 'action': None, 'reason': ''}
        
        if not self.db.is_wallet_tracked(fee_payer):
            self.diagnostics.webhooks_skipped += 1
            result['reason'] = 'Wallet not tracked'
            return result
        
        if self.db.is_signature_processed(signature):
            self.diagnostics.webhooks_skipped += 1
            result['reason'] = 'Already processed'
            return result
        
        if tx_type != 'SWAP':
            self.diagnostics.webhooks_skipped += 1
            result['reason'] = f'Not a swap: {tx_type}'
            return result
        
        trade = self._parse_webhook_swap(tx, fee_payer)
        
        if not trade:
            self.diagnostics.webhooks_skipped += 1
            result['reason'] = 'Could not parse trade'
            return result
        
        token_addr = trade['token_address']
        
        if self.historian.scanner.is_ignored_token(token_addr):
            self.db.mark_signature_processed(signature, fee_payer, trade['type'], token_addr)
            self.diagnostics.webhooks_skipped += 1
            result['reason'] = 'Ignored token'
            return result
        
        self.db.mark_signature_processed(signature, fee_payer, trade['type'], token_addr)
        self.diagnostics.webhooks_processed += 1
        
        wallet_data = self.db.get_wallet(fee_payer)
        if not wallet_data:
            result['reason'] = 'Wallet data not found'
            return result
        
        self.diagnostics.log_event('TRADE_DETECTED', f"{trade['type']} from {fee_payer[:8]}...")
        
        if trade['type'] == 'BUY':
            return self._process_buy(trade, wallet_data, token_addr, signature)
        elif trade['type'] == 'SELL':
            return self._process_sell(trade, wallet_data, token_addr, fee_payer)
        
        return result
    
    def _process_buy(self, trade: Dict, wallet_data: Dict, token_addr: str, signature: str) -> Dict:
        result = {'processed': True, 'action': 'BUY_SIGNAL', 'reason': ''}
        
        if self.db.is_position_tracked(wallet_data['address'], token_addr):
            result['reason'] = 'Position already tracked'
            return result
        
        if not self.rate_limiter.can_call():
            self.diagnostics.log_event('RATE_LIMITED', 'Token lookup skipped')
            result['reason'] = 'Rate limited'
            return result
        
        self.rate_limiter.record_call()
        self.diagnostics.api_calls_made += 1
        
        token_info = self.historian.scanner.get_token_info(token_addr)
        
        if not token_info:
            self.diagnostics.api_errors += 1
            result['reason'] = 'Could not get token info'
            return result
        
        signal_data = {
            'token_address': token_addr,
            'token_symbol': token_info.get('symbol', 'UNKNOWN'),
            'price': token_info.get('price_usd', 0),
            'liquidity': token_info.get('liquidity_usd', 0),
            'volume_24h': token_info.get('volume_24h', 0),
            'market_cap': token_info.get('market_cap', 0),
            'token_age_hours': token_info.get('age_hours', 0),
            'holder_count': token_info.get('holder_count', 0),
            'wallet': wallet_data['address'],
            'wallet_win_rate': wallet_data.get('win_rate', 0.5),
            'wallet_cluster': wallet_data.get('cluster', 'UNKNOWN')
        }
        
        decision = self.strategist.analyze_signal(signal_data, wallet_data, use_llm=CONFIG.use_llm)
        
        if decision.get('llm_called'):
            self.diagnostics.llm_calls += 1
        
        if decision.get('should_enter') and self.paper_engine:
            price = token_info.get('price_usd', 0)
            
            if price > 0:
                pos_id = self.paper_engine.open_position(signal_data, decision, price)
                
                if pos_id:
                    self.diagnostics.positions_opened += 1
                    print(f"  📥 Position opened: ${token_info.get('symbol', '?')} (ID: {pos_id})")
                    
                    if self.notifier:
                        self.notifier.send_entry_alert(signal_data, decision)
                
                result['action'] = 'POSITION_OPENED'
                result['position_id'] = pos_id
        else:
            result['action'] = 'SKIPPED'
            result['reason'] = decision.get('reason', 'Did not meet criteria')
        
        return result
    
    def _process_sell(self, trade: Dict, wallet_data: Dict, token_addr: str, wallet: str) -> Dict:
        result = {'processed': True, 'action': 'SELL_SIGNAL', 'reason': ''}
        
        if not self.rate_limiter.can_call():
            return result
        
        self.rate_limiter.record_call()
        self.diagnostics.api_calls_made += 1
        
        token_info = self.historian.scanner.get_token_info(token_addr)
        token_symbol = token_info.get('symbol', 'UNKNOWN') if token_info else 'UNKNOWN'
        price = token_info.get('price_usd', 0) if token_info else 0
        
        exit_recommendation = self.strategist.process_exit_signal(token_addr, token_symbol, wallet, price)
        
        if exit_recommendation and exit_recommendation.get('action') in ['FULL_EXIT', 'PARTIAL_EXIT']:
            print(f"\n  ⚠️ EXIT SIGNAL: {exit_recommendation.get('reason')}")
            
            if self.paper_engine:
                positions = self.paper_engine.get_open_positions()
                token_positions = [p for p in positions if p.get('token_address') == token_addr]
                
                for pos in token_positions:
                    if exit_recommendation.get('action') == 'FULL_EXIT':
                        self._close_position(pos, f"EXIT_SIGNAL: {exit_recommendation.get('reason')}", price)
        
        return result
    
    def _close_position(self, position: Dict, reason: str, price: float):
        """Close position with FIXED notification (passes result for accurate hold time)"""
        if self.paper_engine:
            result = self.paper_engine.close_position(position['id'], reason, price)
            
            if result:
                self.diagnostics.positions_closed += 1
                print(f"  📤 CLOSED: ${position.get('token_symbol', '?')} | {result['pnl_pct']:+.1f}% | Hold: {result['hold_minutes']:.0f}m")
                
                # ============================================================
                # CHANGE 4: Pass result to send_exit_alert for accurate hold time
                # ============================================================
                if self.notifier:
                    self.notifier.send_exit_alert(position, reason, result['pnl_pct'], result)
    
    def _parse_webhook_swap(self, tx: Dict, wallet: str) -> Optional[Dict]:
        try:
            token_transfers = tx.get('tokenTransfers', [])
            native_transfers = tx.get('nativeTransfers', [])
            
            tokens_in = {}
            tokens_out = {}
            sol_in = 0
            sol_out = 0
            
            SOL_MINT = "So11111111111111111111111111111111111111112"
            
            for t in token_transfers:
                mint = t.get('mint', '')
                if mint == SOL_MINT:
                    continue
                amount = t.get('tokenAmount', 0)
                if t.get('toUserAccount') == wallet:
                    tokens_in[mint] = tokens_in.get(mint, 0) + amount
                elif t.get('fromUserAccount') == wallet:
                    tokens_out[mint] = tokens_out.get(mint, 0) + amount
            
            for t in native_transfers:
                amount = t.get('amount', 0) / 1e9
                if t.get('toUserAccount') == wallet:
                    sol_in += amount
                elif t.get('fromUserAccount') == wallet:
                    sol_out += amount
            
            if len(tokens_in) == 1 and sol_out > 0:
                token_addr = list(tokens_in.keys())[0]
                return {'type': 'BUY', 'token_address': token_addr, 'amount': tokens_in[token_addr], 'sol_amount': sol_out}
            
            if len(tokens_out) == 1 and sol_in > 0:
                token_addr = list(tokens_out.keys())[0]
                return {'type': 'SELL', 'token_address': token_addr, 'amount': tokens_out[token_addr], 'sol_amount': sol_in}
            
            return None
        except Exception:
            return None
    
    def check_open_positions(self):
        if not self.paper_engine:
            return
        
        positions = self.paper_engine.get_open_positions()
        if not positions:
            return
        
        print(f"\n🔍 Checking {len(positions)} open position(s)...")
        
        for position in positions:
            if not self.rate_limiter.can_call():
                print("  ⚠️ Rate limited - will check remaining next cycle")
                break
            
            self.rate_limiter.record_call()
            self.diagnostics.api_calls_made += 1
            
            token_info = self.historian.scanner.get_token_info(position.get('token_address', ''))
            if not token_info:
                continue
            
            current_price = token_info.get('price_usd', 0)
            if current_price <= 0:
                continue
            
            exit_reason = self.paper_engine.check_exit_conditions(position, current_price)
            if exit_reason:
                self._close_position(position, exit_reason, current_price)
    
    def run_discovery(self) -> Dict:
        """Run discovery with proper budget from config"""
        if not self.historian:
            return {'error': 'Discovery not enabled'}
        
        self.diagnostics.discoveries_run += 1
        
        current_count = self.db.get_wallet_count()
        max_wallets = discovery_config.get_max_wallets_this_cycle(current_count)
        api_budget = discovery_config.max_api_calls_per_discovery
        
        print(f"\n   📋 Discovery parameters:")
        print(f"      API Budget: {api_budget:,} credits")
        print(f"      Max wallets this cycle: {max_wallets}")
        print(f"      Current wallet count: {current_count}/{discovery_config.max_total_wallets}")
        
        stats = self.historian.run_discovery(
            max_wallets=max_wallets,
            api_budget=api_budget
        )
        
        if stats.get('wallets_verified', 0) > 0:
            self.diagnostics.wallets_discovered += stats['wallets_verified']
        
        return stats
    
    def get_diagnostics(self) -> Dict:
        diag = self.diagnostics.to_dict()
        diag['rate_limiter'] = self.rate_limiter.get_stats()
        diag['strategist'] = self.strategist.get_status()
        diag['llm_cost_today'] = self.strategist.get_llm_cost_today()
        diag['discovery_config'] = {
            'interval_hours': CONFIG.discovery_interval_hours,
            'api_budget': CONFIG.discovery_api_budget,
            'max_wallets_per_day': discovery_config.max_new_wallets_per_day,
            'max_total_wallets': discovery_config.max_total_wallets
        }
        
        if self.paper_engine:
            diag['paper_trading'] = self.paper_engine.get_stats()
            diag['learning'] = {
                'phase': diag['paper_trading'].get('phase'),
                'iteration': diag['paper_trading'].get('iteration'),
                'blocked_hours': diag['paper_trading'].get('blocked_hours'),
                'preferred_hours': diag['paper_trading'].get('preferred_hours')
            }
        
        return diag


# Global instance
trading_system: Optional[TradingSystem] = None
app = Flask(__name__)


@app.route('/webhook/helius', methods=['POST'])
def helius_webhook():
    global trading_system
    
    if not trading_system:
        return jsonify({"status": "error", "message": "System not initialized"}), 500
    
    try:
        data = request.json
        transactions = data if isinstance(data, list) else [data]
        results = []
        
        for tx in transactions:
            signature = tx.get('signature', '')[:16]
            tx_type = tx.get('type', 'UNKNOWN')
            fee_payer = tx.get('feePayer', '')[:8]
            
            print(f"\n📩 WEBHOOK: {tx_type} from {fee_payer}... (sig: {signature}...)")
            
            result = trading_system.process_webhook(tx)
            results.append(result)
            
            if result.get('processed'):
                print(f"   ✅ {result.get('action', 'PROCESSED')}")
            else:
                print(f"   ⭕ Skipped: {result.get('reason', 'Unknown')}")
        
        return jsonify({"status": "success", "results": results}), 200
    except Exception as e:
        print(f"❌ Webhook error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500


# ============================================================================
# CHANGE 6: Updated /status endpoint with learning info
# ============================================================================
@app.route('/status', methods=['GET'])
def status():
    global trading_system
    if not trading_system:
        return jsonify({"status": "not_initialized"}), 500
    
    stats = trading_system.paper_engine.get_stats() if trading_system.paper_engine else {}
    
    return jsonify({
        "status": "running",
        "uptime_hours": trading_system.diagnostics.to_dict()['uptime_hours'],
        "paper_trading": {
            "balance": stats.get('balance', 0),
            "return_pct": stats.get('return_pct', 0),
            "open_positions": stats.get('open_positions', 0),
            "total_trades": stats.get('total_trades', 0),
            "win_rate": stats.get('win_rate', 0)
        },
        "learning": {
            "phase": stats.get('phase'),
            "iteration": stats.get('iteration'),
            "blocked_hours": stats.get('blocked_hours'),
            "preferred_hours": stats.get('preferred_hours')
        }
    }), 200


@app.route('/diagnostics', methods=['GET'])
def diagnostics():
    global trading_system
    if not trading_system:
        return jsonify({"status": "not_initialized"}), 500
    return jsonify(trading_system.get_diagnostics()), 200


@app.route('/positions', methods=['GET'])
def positions():
    global trading_system
    if not trading_system or not trading_system.paper_engine:
        return jsonify([]), 200
    return jsonify(trading_system.paper_engine.get_open_positions()), 200


@app.route('/new_wallets', methods=['GET'])
def new_wallets():
    global trading_system
    if not trading_system:
        return jsonify([]), 200
    
    with trading_system.db.connection() as conn:
        rows = conn.execute("""
            SELECT address, discovered_at, win_rate, pnl_7d
            FROM verified_wallets
            WHERE discovered_at >= datetime('now', '-24 hours')
            ORDER BY discovered_at DESC
        """).fetchall()
        return jsonify([dict(r) for r in rows]), 200


@app.route('/test', methods=['GET', 'POST'])
def test_endpoint():
    return jsonify({"status": "ok", "message": "Server is running", "timestamp": datetime.now().isoformat()}), 200


@app.route('/discovery/run', methods=['POST'])
def trigger_discovery():
    """Manually trigger a discovery run"""
    global trading_system
    if not trading_system:
        return jsonify({"status": "error", "message": "System not initialized"}), 500
    
    try:
        stats = trading_system.run_discovery()
        return jsonify({"status": "success", "stats": stats}), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/learning/run', methods=['POST'])
def trigger_learning():
    """Manually trigger the learning loop"""
    global trading_system
    if not trading_system:
        return jsonify({"status": "error", "message": "System not initialized"}), 500
    
    try:
        if trading_system.paper_engine:
            result = trading_system.paper_engine.run_learning(force=True, notifier=trading_system.notifier)
            trading_system.diagnostics.learning_iterations += 1
            return jsonify({"status": "success", "result": result}), 200
        else:
            return jsonify({"status": "error", "message": "Paper trading not enabled"}), 400
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/learning/insights', methods=['GET'])
def learning_insights():
    """Get learning insights and strategy performance"""
    global trading_system
    if not trading_system:
        return jsonify({"status": "error", "message": "System not initialized"}), 500
    
    try:
        insights = trading_system.strategist.get_learning_insights()
        
        if trading_system.paper_engine:
            insights['paper_trading_feedback'] = trading_system.paper_engine.get_strategy_feedback()
        
        return jsonify({"status": "success", "insights": insights}), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


# ============================================================================
# CHANGE 7: New /learning/diurnal endpoint
# ============================================================================
@app.route('/learning/diurnal', methods=['GET'])
def diurnal_report():
    """Get diurnal (time-of-day) performance report"""
    global trading_system
    if not trading_system:
        return jsonify({"status": "error", "message": "System not initialized"}), 500
    
    try:
        if trading_system.paper_engine:
            report = trading_system.paper_engine.get_diurnal_report()
            stats = trading_system.paper_engine.get_stats()
            
            return jsonify({
                "status": "success",
                "phase": stats.get('phase'),
                "iteration": stats.get('iteration'),
                "blocked_hours": stats.get('blocked_hours'),
                "preferred_hours": stats.get('preferred_hours'),
                "hourly_performance": report
            }), 200
        else:
            return jsonify({"status": "error", "message": "Paper trading not enabled"}), 400
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


# ============================================================================
# CHANGE 5: Updated background_tasks with learning loop
# ============================================================================
def background_tasks():
    global trading_system
    
    last_position_check = time.time()
    last_status_print = time.time()
    last_discovery = time.time() - 82800
    last_learning = time.time() - 18000  # Start ~5h ago
    
    while True:
        time.sleep(60)
        
        if not trading_system:
            continue
        
        now = time.time()
        
        if now - last_position_check >= CONFIG.position_check_interval:
            last_position_check = now
            try:
                trading_system.check_open_positions()
            except Exception as e:
                print(f"Position check error: {e}")
        
        discovery_interval = CONFIG.discovery_interval_hours * 3600
        if now - last_discovery >= discovery_interval:
            last_discovery = now
            try:
                stats = trading_system.run_discovery()
                
                wallets_added = stats.get('wallets_added_to_webhook', 0)
                wallets_failed = stats.get('wallets_failed_webhook', 0)
                
                if wallets_added > 0:
                    print(f"✅ {wallets_added} new wallet(s) discovered and added!")
                if wallets_failed > 0:
                    print(f"⚠️  {wallets_failed} wallet(s) failed to add - run 'python multi_webhook_manager.py sync' to retry")
            except Exception as e:
                print(f"Discovery error: {e}")
                import traceback
                traceback.print_exc()
        
        if now - last_status_print >= 1800:
            last_status_print = now
            trading_system._print_status()
            
            if trading_system.paper_engine and trading_system.notifier:
                trading_system.notifier.send_hourly_status(trading_system.diagnostics, trading_system.paper_engine.get_stats())
        
        # ====================================================================
        # LEARNING LOOP - runs every 6 hours using paper engine's learning
        # ====================================================================
        learning_interval = 6 * 3600  # 6 hours
        if now - last_learning >= learning_interval:
            last_learning = now
            try:
                print("\n" + "="*70)
                print("🎓 RUNNING LEARNING LOOP")
                print("="*70)
                
                if trading_system.paper_engine:
                    learning_result = trading_system.paper_engine.run_learning(
                        force=True,
                        notifier=trading_system.notifier
                    )
                    
                    if learning_result.get('status') == 'completed':
                        trading_system.diagnostics.learning_iterations += 1
                        print("✅ Learning loop completed")
                        
                        phase = learning_result.get('phase', 'unknown')
                        iteration = learning_result.get('iteration', 0)
                        print(f"   Phase: {phase}")
                        print(f"   Iteration: {iteration}")
                        
                        blocked = learning_result.get('blocked_hours', [])
                        if blocked:
                            print(f"   Blocked hours: {blocked}")
                        
                        if learning_result.get('phase_changed'):
                            new_phase = learning_result.get('new_phase')
                            print(f"   🎯 PHASE CHANGE: → {new_phase}")
                            
                            if trading_system.notifier and trading_system.notifier.enabled:
                                trading_system.notifier.send(
                                    f"🎓 <b>Learning Phase Change!</b>\n\n"
                                    f"New phase: {new_phase}\n"
                                    f"Iteration: {iteration}"
                                )
                    else:
                        print(f"   Status: {learning_result.get('status', 'unknown')}")
                
                # Also run strategist learning
                strategist_result = trading_system.strategist.run_learning_loop()
                
                if strategist_result.get('status') == 'completed':
                    if 'promotion' in strategist_result:
                        promo = strategist_result['promotion']
                        print(f"🏆 Strategy promoted: {promo.get('new_champion', 'Unknown')}")
                        
                        if trading_system.notifier and trading_system.notifier.enabled:
                            trading_system.notifier.send(
                                f"🏆 <b>Strategy Promotion!</b>\n\n"
                                f"New champion: {promo.get('new_champion', 'Unknown')}\n"
                                f"Reason: {promo.get('reason', 'Outperformed')}"
                            )
            except Exception as e:
                print(f"Learning loop error: {e}")
                import traceback
                traceback.print_exc()


def main():
    global trading_system
    
    try:
        trading_system = TradingSystem()
        
        bg_thread = threading.Thread(target=background_tasks, daemon=True)
        bg_thread.start()
        print(f"\n🔄 Background tasks running")
        
        print(f"\n🎧 WEBHOOK SERVER STARTING (LEARNING MODE)")
        print(f"   Webhook:     http://{CONFIG.webhook_host}:{CONFIG.webhook_port}/webhook/helius")
        print(f"   Status:      http://{CONFIG.webhook_host}:{CONFIG.webhook_port}/status")
        print(f"   Diagnostics: http://{CONFIG.webhook_host}:{CONFIG.webhook_port}/diagnostics")
        print(f"   Positions:   http://{CONFIG.webhook_host}:{CONFIG.webhook_port}/positions")
        print(f"   New Wallets: http://{CONFIG.webhook_host}:{CONFIG.webhook_port}/new_wallets")
        print(f"   Discovery:   POST http://{CONFIG.webhook_host}:{CONFIG.webhook_port}/discovery/run")
        print(f"   Learning:    POST http://{CONFIG.webhook_host}:{CONFIG.webhook_port}/learning/run")
        print(f"   Insights:    GET  http://{CONFIG.webhook_host}:{CONFIG.webhook_port}/learning/insights")
        print(f"   Diurnal:     GET  http://{CONFIG.webhook_host}:{CONFIG.webhook_port}/learning/diurnal")
        print(f"   Test:        http://{CONFIG.webhook_host}:{CONFIG.webhook_port}/test")
        print(f"\n   Press Ctrl+C to stop\n")
        
        app.run(host=CONFIG.webhook_host, port=CONFIG.webhook_port, debug=False, use_reloader=False)
        
    except KeyboardInterrupt:
        print("\n\n👋 Shutting down...")
        if trading_system:
            trading_system._print_status()
            if trading_system.paper_engine:
                trading_system.paper_engine.stop()
    except Exception as e:
        print(f"\n💥 FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
