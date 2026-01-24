"""
PAPER TRADER INTEGRATION
========================

Integrates the Effective Paper Trader with:
- Discovery system (receives buy signals)
- Strategist (provides entry context and exit parameters)
- Position monitoring loop

This is the glue code that makes everything work together.
"""

import os
import json
import asyncio
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass

from effective_paper_trader import (
    EffectivePaperTrader, 
    PaperTraderConfig, 
    EntryContext,
    ExitReason
)


@dataclass 
class IntegrationConfig:
    """Configuration for the integration layer"""
    # Paper trading
    starting_balance_sol: float = 10.0
    max_open_positions: int = 5
    default_position_size_sol: float = 0.3
    
    # Signal filtering
    min_conviction_to_trade: int = 60
    min_liquidity_to_trade: float = 30000
    min_volume_24h_to_trade: float = 15000
    max_token_age_hours: float = 72
    
    # Exit parameters
    default_stop_loss_pct: float = -12.0
    default_take_profit_pct: float = 30.0
    default_trailing_stop_pct: float = 8.0
    max_hold_hours: int = 12
    
    # Enable/disable features
    enable_auto_monitoring: bool = True
    log_all_signals: bool = True


class PaperTraderIntegration:
    """
    Bridges the discovery/strategist system with paper trading.
    
    Usage:
        integration = PaperTraderIntegration()
        
        # When discovery system finds a signal:
        result = integration.process_buy_signal(signal_data)
        
        # Periodically check positions:
        integration.check_positions()
    """
    
    def __init__(self, config: IntegrationConfig = None, db_path: str = "paper_trades_v3.db"):
        self.config = config or IntegrationConfig()
        
        # Initialize paper trader with config
        trader_config = PaperTraderConfig(
            starting_balance_sol=self.config.starting_balance_sol,
            max_open_positions=self.config.max_open_positions,
            default_position_size_sol=self.config.default_position_size_sol,
            default_stop_loss_pct=self.config.default_stop_loss_pct,
            default_take_profit_pct=self.config.default_take_profit_pct,
            default_trailing_stop_pct=self.config.default_trailing_stop_pct,
            max_hold_hours=self.config.max_hold_hours,
            enable_auto_exits=self.config.enable_auto_monitoring
        )
        
        self.trader = EffectivePaperTrader(db_path=db_path, config=trader_config)
        
        # Track signals for analysis
        self._signal_log = []
        self._rejected_signals = []
        
        print(f"📊 Paper Trader Integration initialized")
        print(f"   Max positions: {self.config.max_open_positions}")
        print(f"   Min conviction: {self.config.min_conviction_to_trade}")
    
    def process_buy_signal(self, signal: Dict) -> Optional[Dict]:
        """
        Process a buy signal from the discovery/strategist system.
        
        Expected signal format:
        {
            'token_address': str,
            'token_symbol': str,
            'entry_price': float,
            'conviction_score': float,
            'wallet_source': str,
            'wallet_cluster': str,
            'wallet_win_rate': float,
            'wallet_roi_7d': float,
            'liquidity_usd': float,
            'volume_24h_usd': float,
            'market_cap_usd': float,
            'token_age_hours': float,
            'strategy_name': str,
            'market_regime': str,
            # Optional exit overrides
            'stop_loss_pct': float,
            'take_profit_pct': float,
            'trailing_stop_pct': float,
            'position_size_sol': float,
        }
        
        Returns trade result if opened, None if rejected.
        """
        # Log signal
        if self.config.log_all_signals:
            self._signal_log.append({
                'timestamp': datetime.now().isoformat(),
                'signal': signal
            })
        
        # Validate required fields
        required = ['token_address', 'token_symbol', 'entry_price']
        for field in required:
            if field not in signal:
                self._reject_signal(signal, f"Missing required field: {field}")
                return None
        
        token_address = signal['token_address']
        token_symbol = signal['token_symbol']
        entry_price = signal['entry_price']
        
        if entry_price <= 0:
            self._reject_signal(signal, "Invalid entry price")
            return None
        
        # Apply filters
        conviction = signal.get('conviction_score', 50)
        if conviction < self.config.min_conviction_to_trade:
            self._reject_signal(signal, f"Conviction too low ({conviction} < {self.config.min_conviction_to_trade})")
            return None
        
        liquidity = signal.get('liquidity_usd', 0)
        if liquidity < self.config.min_liquidity_to_trade:
            self._reject_signal(signal, f"Liquidity too low (${liquidity:,.0f} < ${self.config.min_liquidity_to_trade:,.0f})")
            return None
        
        volume = signal.get('volume_24h_usd', 0)
        if volume < self.config.min_volume_24h_to_trade:
            self._reject_signal(signal, f"Volume too low (${volume:,.0f} < ${self.config.min_volume_24h_to_trade:,.0f})")
            return None
        
        token_age = signal.get('token_age_hours', 0)
        if token_age > self.config.max_token_age_hours:
            self._reject_signal(signal, f"Token too old ({token_age:.1f}h > {self.config.max_token_age_hours}h)")
            return None
        
        # Check if we can open
        can_open, reason = self.trader.can_open_position(token_address)
        if not can_open:
            self._reject_signal(signal, reason)
            return None
        
        # Build entry context
        context = EntryContext(
            wallet_source=signal.get('wallet_source', 'unknown'),
            wallet_cluster=signal.get('wallet_cluster', 'UNKNOWN'),
            wallet_win_rate=signal.get('wallet_win_rate', 0),
            wallet_roi_7d=signal.get('wallet_roi_7d', 0),
            liquidity_usd=liquidity,
            volume_24h_usd=volume,
            market_cap_usd=signal.get('market_cap_usd', 0),
            token_age_hours=token_age,
            holder_count=signal.get('holder_count', 0),
            conviction_score=conviction,
            signal_wallets_count=signal.get('wallet_count', 1),
            clusters_detected=signal.get('clusters', []),
            aggregated_signal=signal.get('aggregated', False),
            sol_price_usd=signal.get('sol_price_usd', 0),
            market_regime=signal.get('market_regime', 'NEUTRAL'),
            strategy_name=signal.get('strategy_name', 'default'),
            strategy_version=signal.get('strategy_version', '1.0'),
            entry_reason=signal.get('reason', '')
        )
        
        # Open position
        position_id = self.trader.open_position(
            token_address=token_address,
            token_symbol=token_symbol,
            entry_price=entry_price,
            size_sol=signal.get('position_size_sol'),
            context=context,
            stop_loss_pct=signal.get('stop_loss_pct'),
            take_profit_pct=signal.get('take_profit_pct'),
            trailing_stop_pct=signal.get('trailing_stop_pct'),
            max_hold_hours=signal.get('max_hold_hours'),
            notes=signal.get('notes', '')
        )
        
        if position_id:
            print(f"✅ Paper trade opened: {token_symbol} (ID: {position_id})")
            return {
                'success': True,
                'position_id': position_id,
                'token_symbol': token_symbol,
                'entry_price': entry_price,
                'size_sol': self.trader.config.default_position_size_sol
            }
        else:
            self._reject_signal(signal, "Position open failed")
            return None
    
    def _reject_signal(self, signal: Dict, reason: str):
        """Log a rejected signal"""
        self._rejected_signals.append({
            'timestamp': datetime.now().isoformat(),
            'token': signal.get('token_symbol', 'UNKNOWN'),
            'reason': reason,
            'signal': signal
        })
        print(f"⏭️ Signal rejected: {signal.get('token_symbol', 'UNKNOWN')} - {reason}")
    
    def process_exit_signal(self, signal: Dict) -> Optional[Dict]:
        """
        Process an exit signal (e.g., multiple wallets exiting).
        
        Expected signal format:
        {
            'token_address': str,
            'exit_price': float,
            'reason': str,  # 'SNIPER_EXIT', 'MULTI_WALLET_EXIT', etc.
            'urgency': str  # 'HIGH', 'MEDIUM', 'LOW'
        }
        """
        token_address = signal.get('token_address')
        if not token_address:
            return None
        
        # Find open position for this token
        positions = self.trader.get_open_positions()
        position = next((p for p in positions if p['token_address'] == token_address), None)
        
        if not position:
            return None
        
        exit_price = signal.get('exit_price', 0)
        if exit_price <= 0:
            exit_price = self.trader._get_token_price(token_address)
        
        if exit_price <= 0:
            return None
        
        reason_map = {
            'SNIPER_EXIT': ExitReason.SMART_EXIT,
            'MULTI_WALLET_EXIT': ExitReason.SMART_EXIT,
            'RUG_DETECTED': ExitReason.RUG_DETECTED,
            'LIQUIDITY_DRIED': ExitReason.LIQUIDITY_DRIED
        }
        
        exit_reason = reason_map.get(signal.get('reason', ''), ExitReason.SMART_EXIT)
        
        result = self.trader.close_position(
            position_id=position['id'],
            exit_price=exit_price,
            exit_reason=exit_reason,
            notes=signal.get('notes', '')
        )
        
        return result
    
    def get_stats(self) -> Dict:
        """Get comprehensive stats for display"""
        summary = self.trader.get_performance_summary()
        
        summary['signals_received'] = len(self._signal_log)
        summary['signals_rejected'] = len(self._rejected_signals)
        summary['conversion_rate'] = (
            summary['total_trades'] / max(1, len(self._signal_log))
        ) if self._signal_log else 0
        
        # Recent rejection reasons
        if self._rejected_signals:
            recent = self._rejected_signals[-20:]
            reasons = {}
            for r in recent:
                reason = r['reason'].split('(')[0].strip()
                reasons[reason] = reasons.get(reason, 0) + 1
            summary['recent_rejection_reasons'] = reasons
        
        return summary
    
    def get_strategy_feedback(self) -> Dict:
        """
        Get feedback for the strategist to improve.
        This is the key data that informs strategy changes.
        """
        analysis = self.trader.get_strategy_analysis(days=14)
        
        # Add signal conversion analysis
        if self._rejected_signals:
            rejection_analysis = {}
            for r in self._rejected_signals:
                reason = r['reason'].split('(')[0].strip()
                if reason not in rejection_analysis:
                    rejection_analysis[reason] = {
                        'count': 0,
                        'avg_conviction': 0,
                        'convictions': []
                    }
                rejection_analysis[reason]['count'] += 1
                conv = r['signal'].get('conviction_score', 0)
                rejection_analysis[reason]['convictions'].append(conv)
            
            for reason in rejection_analysis:
                convs = rejection_analysis[reason]['convictions']
                rejection_analysis[reason]['avg_conviction'] = sum(convs) / len(convs) if convs else 0
                del rejection_analysis[reason]['convictions']
            
            analysis['rejection_analysis'] = rejection_analysis
        
        return analysis
    
    def print_dashboard(self):
        """Print a nice dashboard"""
        stats = self.get_stats()
        
        print("\n" + "="*70)
        print("📊 PAPER TRADING DASHBOARD")
        print("="*70)
        
        # Account section
        print(f"\n💰 ACCOUNT")
        print(f"   Balance: {stats['balance']:.4f} SOL ({stats['return_pct']:+.1f}%)")
        print(f"   Total PnL: {stats['total_pnl_sol']:+.4f} SOL")
        print(f"   Max Drawdown: {stats['max_drawdown_pct']:.1f}%")
        
        # Performance section
        print(f"\n📈 PERFORMANCE")
        print(f"   Trades: {stats['total_trades']} ({stats['win_rate']:.1%} win rate)")
        print(f"   Profit Factor: {stats['profit_factor']:.2f}")
        print(f"   Avg Win: {stats['avg_win_pct']:+.1f}% | Avg Loss: {stats['avg_loss_pct']:+.1f}%")
        print(f"   Best: {stats['best_trade_pct']:+.1f}% | Worst: {stats['worst_trade_pct']:+.1f}%")
        
        # Signals section
        print(f"\n📡 SIGNALS")
        print(f"   Received: {stats['signals_received']}")
        print(f"   Rejected: {stats['signals_rejected']}")
        print(f"   Conversion: {stats['conversion_rate']:.1%}")
        
        if stats.get('recent_rejection_reasons'):
            print(f"   Rejection reasons:")
            for reason, count in sorted(stats['recent_rejection_reasons'].items(), 
                                        key=lambda x: -x[1])[:5]:
                print(f"      {reason}: {count}")
        
        # Positions section
        print(f"\n📦 OPEN POSITIONS: {stats['open_positions']}/{self.trader.config.max_open_positions}")
        positions = self.trader.get_open_positions()
        if positions:
            for pos in positions:
                entry_time = datetime.fromisoformat(pos['entry_time'])
                hold_hours = (datetime.now() - entry_time).total_seconds() / 3600
                current = pos['current_price'] or pos['entry_price']
                pnl_pct = ((current / pos['entry_price']) - 1) * 100 if pos['entry_price'] > 0 else 0
                emoji = "✅" if pnl_pct > 0 else "❌"
                print(f"   {emoji} {pos['token_symbol']}: {pnl_pct:+.1f}% | {hold_hours:.1f}h | SL:{pos['stop_loss_pct']}% TP:{pos['take_profit_pct']}%")
        
        # Exit breakdown
        if stats.get('by_exit_reason'):
            print(f"\n🚪 EXIT BREAKDOWN")
            for reason, data in sorted(stats['by_exit_reason'].items(), 
                                       key=lambda x: -x[1]['count']):
                wr = data['wins'] / data['count'] if data['count'] else 0
                print(f"   {reason}: {data['count']} ({wr:.0%} win, {data['pnl']:+.4f} SOL)")
        
        print("\n" + "="*70)


class SignalAdapter:
    """
    Adapts signals from your existing discovery system to the paper trader format.
    Customize this based on your actual signal format.
    """
    
    @staticmethod
    def from_discovery_signal(raw_signal: Dict) -> Dict:
        """
        Convert a discovery system signal to paper trader format.
        
        Customize this method based on your actual signal structure.
        """
        return {
            'token_address': raw_signal.get('token_address') or raw_signal.get('mint'),
            'token_symbol': raw_signal.get('token_symbol') or raw_signal.get('symbol', 'UNKNOWN'),
            'entry_price': raw_signal.get('entry_price') or raw_signal.get('price', 0),
            'conviction_score': raw_signal.get('conviction_score') or raw_signal.get('conviction', 50),
            
            # Wallet info
            'wallet_source': raw_signal.get('wallet_source') or raw_signal.get('wallet', ''),
            'wallet_cluster': raw_signal.get('wallet_cluster') or raw_signal.get('cluster', 'UNKNOWN'),
            'wallet_win_rate': raw_signal.get('wallet_win_rate') or raw_signal.get('win_rate', 0),
            'wallet_roi_7d': raw_signal.get('wallet_roi_7d') or raw_signal.get('roi_7d', 0),
            
            # Token metrics
            'liquidity_usd': raw_signal.get('liquidity_usd') or raw_signal.get('liquidity', 0),
            'volume_24h_usd': raw_signal.get('volume_24h_usd') or raw_signal.get('volume_24h', 0),
            'market_cap_usd': raw_signal.get('market_cap_usd') or raw_signal.get('market_cap', 0),
            'token_age_hours': raw_signal.get('token_age_hours') or raw_signal.get('age_hours', 0),
            
            # Strategy
            'strategy_name': raw_signal.get('strategy_name', 'default'),
            'market_regime': raw_signal.get('market_regime') or raw_signal.get('regime', 'NEUTRAL'),
            
            # Optional overrides
            'stop_loss_pct': raw_signal.get('stop_loss_pct') or raw_signal.get('stop_loss'),
            'take_profit_pct': raw_signal.get('take_profit_pct') or raw_signal.get('take_profit'),
            'trailing_stop_pct': raw_signal.get('trailing_stop_pct') or raw_signal.get('trailing_stop'),
            'position_size_sol': raw_signal.get('position_size_sol') or raw_signal.get('size_sol'),
            
            # Multi-wallet aggregation
            'wallet_count': raw_signal.get('wallet_count', 1),
            'clusters': raw_signal.get('clusters', []),
            'aggregated': raw_signal.get('aggregated', False),
            
            # Notes
            'reason': raw_signal.get('reason', ''),
            'notes': raw_signal.get('notes', '')
        }
    
    @staticmethod
    def from_strategist_decision(decision: Dict) -> Dict:
        """
        Convert a strategist decision to paper trader format.
        """
        return {
            'token_address': decision.get('token_address'),
            'token_symbol': decision.get('token_symbol'),
            'entry_price': decision.get('best_entry_price') or decision.get('entry_price', 0),
            'conviction_score': decision.get('conviction_score', 50),
            
            'wallet_source': decision.get('wallets', ['unknown'])[0] if decision.get('wallets') else 'unknown',
            'wallet_count': decision.get('wallet_count', 1),
            'clusters': list(decision.get('clusters', {}).keys()),
            'aggregated': decision.get('wallet_count', 1) > 1,
            
            'strategy_name': decision.get('strategy_name', 'aggregated'),
            'reason': decision.get('reason', ''),
            
            # These should come from your market data
            'liquidity_usd': decision.get('liquidity_usd', 50000),
            'volume_24h_usd': decision.get('volume_24h_usd', 20000),
            'token_age_hours': decision.get('token_age_hours', 1),
        }


# Example usage and testing
if __name__ == "__main__":
    import sys
    
    # Initialize integration
    integration = PaperTraderIntegration()
    
    if len(sys.argv) < 2:
        integration.print_dashboard()
        sys.exit(0)
    
    command = sys.argv[1].lower()
    
    if command == 'dashboard':
        integration.print_dashboard()
    
    elif command == 'feedback':
        feedback = integration.get_strategy_feedback()
        print("\n📊 STRATEGY FEEDBACK")
        print("="*60)
        print(json.dumps(feedback, indent=2, default=str))
    
    elif command == 'test':
        # Test with a sample signal
        test_signal = {
            'token_address': 'TEST123',
            'token_symbol': 'TEST',
            'entry_price': 0.001,
            'conviction_score': 75,
            'wallet_source': 'test_wallet',
            'wallet_cluster': 'SNIPER',
            'liquidity_usd': 50000,
            'volume_24h_usd': 25000,
            'token_age_hours': 2
        }
        
        result = integration.process_buy_signal(test_signal)
        print(f"Test result: {result}")
    
    elif command == 'help':
        print("""
Paper Trader Integration Commands:
  dashboard    - Show trading dashboard
  feedback     - Get strategy feedback
  test         - Test with sample signal
  help         - Show this help
        """)
    
    else:
        print(f"Unknown command: {command}")
