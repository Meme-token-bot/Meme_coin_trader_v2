"""
MASTER V2 - Webhook-Based Trading System with Automated Discovery
Trading System V2 - Uses Helius webhooks + smart discovery + auto webhook management

UPDATED: Now properly passes budget parameters to discovery
         and uses discovery_config for intervals

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

from database_v2 import DatabaseV2
from historian import Historian
from strategist_v2 import Strategist
from discovery_config import config as discovery_config
from helius_webhook_manager import HeliusWebhookManager


@dataclass
class MasterConfig:
    """Master configuration - now uses discovery_config for discovery settings"""
    webhook_host: str = '0.0.0.0'
    webhook_port: int = 5000
    position_check_interval: int = 300
    discovery_enabled: bool = True
    # UPDATED: Use discovery_config values instead of hardcoded
    # discovery_interval_hours is now read from discovery_config
    max_token_lookups_per_minute: int = 20
    max_api_calls_per_hour: int = 500
    paper_trading_enabled: bool = True
    paper_starting_balance: float = 10.0
    use_llm: bool = True
    max_open_positions: int = 5
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
            'llm_calls': self.llm_calls,
            'recent_events': list(self.recent_events)[-10:]
        }


class Notifier:
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
        msg = f"""🎯 <b>ENTRY SIGNAL</b>

Token: ${signal.get('token_symbol', 'UNKNOWN')}
Conviction: {decision.get('conviction_score', 0):.0f}/100
Wallets: {decision.get('wallet_count', 1)}
Regime: {decision.get('regime', 'UNKNOWN')}
Position: {decision.get('position_size_sol', 0):.3f} SOL
Stop: {decision.get('stop_loss', 0)*100:.0f}%
Target: {decision.get('take_profit', 0)*100:.0f}%
LLM: {'✅' if decision.get('llm_called') else '❌'}"""
        self.send(msg)
    
    def send_exit_alert(self, position: Dict, reason: str, pnl_pct: float):
        emoji = "🟢" if pnl_pct > 0 else "🔴"
        hold_mins = position.get('hold_duration_minutes') or 0
        msg = f"""{emoji} <b>EXIT</b>

Token: ${position.get('token_symbol', 'UNKNOWN')}
Reason: {reason}
P&L: {pnl_pct:+.1f}%
Hold: {hold_mins:.0f} min"""
        self.send(msg)
    
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
        
        msg = f"""📊 <b>Hourly Status</b>

Uptime: {diag['uptime_hours']:.1f}h
Webhooks: {diag['webhooks']['received']} received
Last webhook: {last_webhook}
Paper: {paper_stats['balance']:.2f} SOL ({paper_stats['return_pct']:+.1f}%)
Open positions: {paper_stats['open_positions']}
API calls: {diag['api']['calls']} | Errors: {diag['api']['errors']}
LLM calls: {diag['llm_calls']}
Discoveries: {diag['discovery']['runs']} | Wallets found: {diag['discovery']['wallets_found']}"""
        self.send(msg)


class PaperTradingEngine:
    def __init__(self, db: DatabaseV2, starting_balance: float = 10.0):
        self.db = db
        self.starting_balance = starting_balance
        self.balance = starting_balance
        self.reserved = 0.0
        self._load_positions()
    
    def _load_positions(self):
        positions = self.db.get_open_positions()
        paper_positions = [p for p in positions if p.get('source') == 'paper']
        self.reserved = sum(p.get('position_size_sol', 0) or 0 for p in paper_positions)
    
    @property
    def available_balance(self) -> float:
        return self.balance - self.reserved
    
    def open_position(self, signal: Dict, decision: Dict, price: float) -> Optional[int]:
        size = min(decision.get('position_size_sol', 0.5), self.available_balance * 0.3, CONFIG.max_position_size_sol)
        if size < 0.1:
            return None
        
        pos_id = self.db.add_tracked_position(
            wallet=decision.get('wallets', ['paper_trade'])[0] if decision.get('wallets') else 'paper_trade',
            token_address=signal.get('token_address', ''),
            entry_time=datetime.now(),
            entry_price=price,
            amount=size / price if price > 0 else 0,
            token_symbol=signal.get('token_symbol', 'UNKNOWN'),
            market_data={
                'liquidity': signal.get('liquidity', 0),
                'volume_24h': signal.get('volume_24h', 0),
                'token_age_hours': signal.get('token_age_hours', 0),
                'position_size_sol': size,
                'conviction_score': decision.get('conviction_score', 0),
                'stop_loss': decision.get('stop_loss', -0.12),
                'take_profit': decision.get('take_profit', 0.25),
                'trailing_stop': decision.get('trailing_stop', 0.08),
                'max_hold_hours': decision.get('max_hold_hours', 12),
                'strategy': decision.get('strategy', 'unknown'),
                'wallet_count': decision.get('wallet_count', 1),
                'regime': decision.get('regime', 'UNKNOWN'),
                'source': 'paper'
            }
        )
        self.reserved += size
        return pos_id
    
    def check_exit_conditions(self, position: Dict, current_price: float) -> Optional[str]:
        entry_price = position.get('entry_price', 0)
        if entry_price <= 0:
            return None
        
        pnl_pct = (current_price - entry_price) / entry_price
        peak_price = position.get('peak_price') or entry_price
        
        if current_price > peak_price:
            peak_price = current_price
            self.db.update_position_peak(position['id'], peak_price, (peak_price - entry_price) / entry_price * 100)
        
        stop_loss = position.get('stop_loss_pct') or -0.12
        take_profit = position.get('take_profit_pct') or 0.25
        trailing_stop = position.get('trailing_stop_pct') or 0.08
        max_hold = position.get('max_hold_hours') or 12
        
        if pnl_pct <= stop_loss:
            return 'STOP_LOSS'
        if pnl_pct >= take_profit:
            return 'TAKE_PROFIT'
        
        if peak_price > entry_price:
            peak_pnl = (peak_price - entry_price) / entry_price
            if peak_pnl >= 0.15:
                drop_from_peak = (peak_price - current_price) / peak_price
                if drop_from_peak >= trailing_stop:
                    return 'TRAILING_STOP'
        
        entry_time = position.get('entry_time')
        if entry_time:
            if isinstance(entry_time, str):
                entry_time = datetime.fromisoformat(entry_time.replace('Z', '+00:00'))
            hold_hours = (datetime.now() - entry_time).total_seconds() / 3600
            if hold_hours >= max_hold:
                return 'TIME_STOP'
        return None
    
    def close_position(self, position_id: int, exit_reason: str, exit_price: float) -> Optional[Dict]:
        position = self.db.get_position(position_id=position_id)
        if not position:
            return None
        
        entry_price = position.get('entry_price', 0)
        entry_time = position.get('entry_time')
        size = position.get('position_size_sol') or 0
        
        if isinstance(entry_time, str):
            entry_time = datetime.fromisoformat(entry_time.replace('Z', '+00:00'))
        
        pnl_pct = ((exit_price - entry_price) / entry_price * 100) if entry_price > 0 else 0
        pnl_sol = size * (pnl_pct / 100)
        hold_minutes = (datetime.now() - entry_time).total_seconds() / 60 if entry_time else 0
        
        self.db.close_position(position_id, {
            'exit_time': datetime.now(),
            'exit_price': exit_price,
            'exit_reason': exit_reason,
            'profit_pct': pnl_pct,
            'profit_sol': pnl_sol,
            'hold_duration_minutes': hold_minutes,
            'peak_price': position.get('peak_price'),
            'peak_unrealized_pct': position.get('peak_unrealized_pct')
        })
        
        self.balance += pnl_sol
        self.reserved = max(0, self.reserved - size)
        return {'pnl_pct': pnl_pct, 'pnl_sol': pnl_sol, 'hold_minutes': hold_minutes}
    
    def get_open_positions(self) -> List[Dict]:
        positions = self.db.get_open_positions()
        return [p for p in positions if p.get('source') == 'paper']
    
    def get_stats(self) -> Dict:
        return {
            'balance': self.balance,
            'starting_balance': self.starting_balance,
            'total_pnl': self.balance - self.starting_balance,
            'return_pct': ((self.balance / self.starting_balance) - 1) * 100,
            'open_positions': len(self.get_open_positions())
        }


class TradingSystem:
    def __init__(self):
        print("\n" + "="*70)
        print("🚀 TRADING SYSTEM V2 (WEBHOOK + AUTO-DISCOVERY)")
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
        
        if CONFIG.paper_trading_enabled:
            self.paper_engine = PaperTradingEngine(self.db, CONFIG.paper_starting_balance)
            print(f"  ✅ Paper Trading (Balance: {self.paper_engine.balance:.2f} SOL)")
        else:
            self.paper_engine = None
        
        self.rate_limiter = RateLimiter(CONFIG.max_token_lookups_per_minute, CONFIG.max_api_calls_per_hour)
        print(f"  ✅ Rate Limiter ({CONFIG.max_token_lookups_per_minute}/min, {CONFIG.max_api_calls_per_hour}/hr)")
        
        self.diagnostics = DiagnosticsTracker()
        print("  ✅ Diagnostics Tracker")
        
        # Initialize multi-webhook manager (scales past 25 wallets)
        webhook_url = os.getenv('WEBHOOK_URL', f'http://0.0.0.0:{CONFIG.webhook_port}/webhook/helius')

        try:
            from multi_webhook_manager import MultiWebhookManager
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
        
        print("\n" + "="*70)
        print("✅ TRADING SYSTEM V2 READY")
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
        
        rate_stats = self.rate_limiter.get_stats()
        print(f"  Rate limit: {rate_stats['calls_last_hour']}/{rate_stats['limit_per_hour']} calls/hr")
        
        # Show discovery config
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
        
        if not token_info or token_info.get('price_usd', 0) <= 0:
            self.diagnostics.api_errors += 1
            result['reason'] = 'Could not get token info'
            return result
        
        signal_data = {
            'token_symbol': token_info.get('symbol', 'UNKNOWN'),
            'token_address': token_addr,
            'price': token_info.get('price_usd', 0),
            'liquidity': token_info.get('liquidity', 0),
            'volume_24h': token_info.get('volume_24h', 0),
            'token_age_hours': token_info.get('age_hours', 0),
            'wallet_count': 1,
            'avg_wallet_win_rate': wallet_data.get('win_rate', 0.5)
        }
        
        print(f"\n  📊 Analyzing ${signal_data['token_symbol']}...")
        self.diagnostics.log_event('ANALYZING', f"${signal_data['token_symbol']}")
        
        decision = self.strategist.analyze_signal(signal_data, wallet_data, use_llm=CONFIG.use_llm)
        
        if decision.get('llm_called'):
            self.diagnostics.llm_calls += 1
        
        print(f"     Base: {decision.get('base_score', 0):.0f} | LLM: {decision.get('llm_adjustment', 0):+.0f} | Final: {decision.get('conviction_score', 0):.0f}")
        
        if decision.get('should_enter') and self.paper_engine:
            pos_id = self.paper_engine.open_position(signal_data, decision, token_info.get('price_usd', 0))
            
            if pos_id:
                self.diagnostics.positions_opened += 1
                self.diagnostics.log_event('POSITION_OPENED', f"${signal_data['token_symbol']} ID:{pos_id}")
                print(f"     ✅ POSITION OPENED (ID: {pos_id})")
                
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
        if self.paper_engine:
            result = self.paper_engine.close_position(position['id'], reason, price)
            
            if result:
                self.diagnostics.positions_closed += 1
                print(f"  📤 CLOSED: ${position.get('token_symbol', '?')} | {reason} | {result.get('pnl_pct', 0):+.1f}%")
                
                if self.notifier:
                    self.notifier.send_exit_alert(position, reason, result.get('pnl_pct', 0))
    
    def _parse_webhook_swap(self, tx: Dict, wallet_address: str) -> Optional[Dict]:
        try:
            token_transfers = tx.get('tokenTransfers', [])
            native_transfers = tx.get('nativeTransfers', [])
            
            sol_in, sol_out = 0, 0
            tokens_in, tokens_out = {}, {}
            WSOL = "So11111111111111111111111111111111111111112"
            
            for transfer in token_transfers:
                mint = transfer.get('mint', '')
                amount = float(transfer.get('tokenAmount', 0))
                from_addr = transfer.get('fromUserAccount', '')
                to_addr = transfer.get('toUserAccount', '')
                
                if from_addr == wallet_address:
                    if mint == WSOL:
                        sol_out += amount
                    else:
                        tokens_out[mint] = tokens_out.get(mint, 0) + amount
                elif to_addr == wallet_address:
                    if mint == WSOL:
                        sol_in += amount
                    else:
                        tokens_in[mint] = tokens_in.get(mint, 0) + amount
            
            for transfer in native_transfers:
                amount = float(transfer.get('amount', 0)) / 1e9
                from_addr = transfer.get('fromUserAccount', '')
                to_addr = transfer.get('toUserAccount', '')
                
                if from_addr == wallet_address:
                    sol_out += amount
                elif to_addr == wallet_address:
                    sol_in += amount
            
            if len(tokens_in) == 1 and sol_out > 0:
                token_addr = list(tokens_in.keys())[0]
                return {'type': 'BUY', 'token_address': token_addr, 'amount': tokens_in[token_addr], 'sol_amount': sol_out}
            elif len(tokens_out) == 1 and sol_in > 0:
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
        
        # UPDATED: Pass the budget parameters from config
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


@app.route('/status', methods=['GET'])
def status():
    global trading_system
    if not trading_system:
        return jsonify({"status": "not_initialized"}), 500
    return jsonify({"status": "running", "uptime_hours": trading_system.diagnostics.to_dict()['uptime_hours']}), 200


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
        result = trading_system.strategist.run_learning_loop(force=True)
        return jsonify({"status": "success", "result": result}), 200
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
        return jsonify({"status": "success", "insights": insights}), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


def background_tasks():
    global trading_system
    
    last_position_check = time.time()
    last_status_print = time.time()
    last_discovery = time.time() - 82800  # Start ~23h ago to trigger soon
    last_learning = time.time() - 18000   # Start ~5h ago to trigger within first hour
    
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
        
        # UPDATED: Use CONFIG property which reads from discovery_config
        discovery_interval_seconds = CONFIG.discovery_interval_hours * 3600
        
        if CONFIG.discovery_enabled and (now - last_discovery >= discovery_interval_seconds):
            last_discovery = now
            
            wallet_count = trading_system.db.get_wallet_count()
            
            if wallet_count >= discovery_config.max_total_wallets:
                print(f"\n⚠️ Already have {wallet_count} wallets - skipping discovery")
                continue
            
            try:
                print("\n" + "="*70)
                print(f"🔍 RUNNING WALLET DISCOVERY ({CONFIG.discovery_interval_hours}h cycle)")
                print(f"   Budget: {CONFIG.discovery_api_budget:,} credits")
                print("="*70)
                
                discovery_stats = trading_system.run_discovery()
                
                print(f"\n✅ Discovery complete:")
                print(f"   Tokens scanned: {discovery_stats.get('tokens_discovered', 0)}")
                print(f"   Candidates found: {discovery_stats.get('wallet_candidates_found', 0)}")
                print(f"   Pre-filtered out: {discovery_stats.get('wallets_prefiltered_out', 0)}")
                print(f"   Wallets profiled: {discovery_stats.get('wallets_profiled', 0)}")
                print(f"   Wallets verified: {discovery_stats.get('wallets_verified', 0)}")
                print(f"   API calls used: {discovery_stats.get('helius_api_calls', 0):,}")
                
                # Report webhook additions (note: may differ from verified count if rate limited)
                wallets_added = discovery_stats.get('wallets_added_to_webhook', discovery_stats.get('wallets_verified', 0))
                wallets_failed = discovery_stats.get('wallets_failed_webhook', 0)
                
                if wallets_added > 0:
                    print(f"\n✅ {wallets_added} wallet(s) added to webhook!")
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
        
        # LEARNING LOOP - runs every 6 hours
        learning_interval = 6 * 3600  # 6 hours
        if now - last_learning >= learning_interval:
            last_learning = now
            try:
                print("\n" + "="*70)
                print("🧠 RUNNING LEARNING LOOP")
                print("="*70)
                
                learning_result = trading_system.strategist.run_learning_loop()
                
                if learning_result.get('status') == 'completed':
                    print("✅ Learning loop completed")
                    
                    # Log any promotions
                    if 'promotion' in learning_result:
                        promo = learning_result['promotion']
                        print(f"🏆 Strategy promoted: {promo.get('new_champion', 'Unknown')}")
                        
                        # Notify via Telegram if available
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
        
        print(f"\n🎧 WEBHOOK SERVER STARTING")
        print(f"   Webhook:     http://{CONFIG.webhook_host}:{CONFIG.webhook_port}/webhook/helius")
        print(f"   Status:      http://{CONFIG.webhook_host}:{CONFIG.webhook_port}/status")
        print(f"   Diagnostics: http://{CONFIG.webhook_host}:{CONFIG.webhook_port}/diagnostics")
        print(f"   New Wallets: http://{CONFIG.webhook_host}:{CONFIG.webhook_port}/new_wallets")
        print(f"   Discovery:   POST http://{CONFIG.webhook_host}:{CONFIG.webhook_port}/discovery/run")
        print(f"   Learning:    POST http://{CONFIG.webhook_host}:{CONFIG.webhook_port}/learning/run")
        print(f"   Insights:    GET  http://{CONFIG.webhook_host}:{CONFIG.webhook_port}/learning/insights")
        print(f"   Test:        http://{CONFIG.webhook_host}:{CONFIG.webhook_port}/test")
        print(f"\n   Press Ctrl+C to stop\n")
        
        app.run(host=CONFIG.webhook_host, port=CONFIG.webhook_port, debug=False, use_reloader=False)
        
    except KeyboardInterrupt:
        print("\n\n👋 Shutting down...")
        if trading_system:
            trading_system._print_status()
    except Exception as e:
        print(f"\n💥 FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
