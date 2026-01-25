"""
profiler.py - FIXED Wallet Profiler
Standalone profiler using Helius parsed transactions API

FIX: Changed swap detection from len(tokens) == 1 to len(tokens) >= 1
This matches the logic in ActiveTraderDiscovery._analyze_swap and handles
real-world swaps that involve multiple token transfers (routing, fees, etc.)

USAGE:
  from profiler import WalletProfiler
  
  profiler = WalletProfiler(helius_key)
  stats = profiler.calculate_performance(wallet_address, days=7)
"""

import requests
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from collections import defaultdict


class WalletProfiler:
    """
    Profiles wallet trading performance using Helius Enhanced API
    
    Cost: ~20-30 credits per wallet
    Much more accurate than parsing raw transactions
    """
    
    def __init__(self, helius_key: str):
        self.helius_key = helius_key
        self.helius_api = f"https://api.helius.xyz/v0"
        self._last_call = 0
        self._delay = 0.1
        print("  ✅ Wallet Profiler (Helius Parsed API) - FIXED")
    
    def _rate_limit(self):
        elapsed = time.time() - self._last_call
        if elapsed < self._delay:
            time.sleep(self._delay - elapsed)
        self._last_call = time.time()
    
    def calculate_performance(self, wallet: str, days: int = 7, usage_tracker=None) -> Dict:
        """Calculate wallet performance"""
        
        if usage_tracker:
            if not usage_tracker.can_make_call(5):
                raise Exception("Budget exceeded")
            usage_tracker.record_call('profile_wallet', 5)
        
        swaps = self._get_wallet_swaps(wallet, days)
        
        if not swaps:
            return self._empty_stats()
        
        positions = self._match_positions(swaps)
        
        if not positions:
            return self._empty_stats()
        
        return self._calculate_stats(positions)
    
    def _get_wallet_swaps(self, wallet: str, days: int) -> List[Dict]:
        """Get all SWAP transactions using Helius API"""
        
        swaps = []
        before_sig = None
        max_requests = 5
        
        for i in range(max_requests):
            self._rate_limit()
            
            url = f"{self.helius_api}/addresses/{wallet}/transactions"
            params = {
                "api-key": self.helius_key,
                "type": "SWAP",
                "limit": 100
            }
            
            if before_sig:
                params["before"] = before_sig
            
            try:
                res = requests.get(url, params=params, timeout=15)
                
                if res.status_code != 200:
                    break
                
                data = res.json()
                
                if not data:
                    break
                
                cutoff = datetime.now() - timedelta(days=days)
                
                for tx in data:
                    timestamp = tx.get('timestamp', 0)
                    
                    if timestamp == 0:
                        continue
                    
                    tx_time = datetime.fromtimestamp(timestamp)
                    
                    if tx_time < cutoff:
                        return swaps
                    
                    swap = self._parse_swap(tx, wallet)
                    if swap:
                        swaps.append(swap)
                
                if len(data) < 100:
                    break
                
                before_sig = data[-1].get('signature')
                
            except Exception as e:
                break
        
        return swaps
    
    def _parse_swap(self, tx: Dict, wallet: str) -> Optional[Dict]:
        """
        Parse Helius swap transaction
        
        FIX: Changed from len(tokens) == 1 to len(tokens) >= 1
        Real swaps often have multiple token transfers due to:
        - Multi-hop routing (e.g., SOL -> USDC -> MEME)
        - Fee tokens
        - Intermediate tokens
        - LP token interactions
        
        When multiple tokens are involved, we take the one with the largest amount
        as the "primary" token for the trade.
        """
        
        signature = tx.get('signature', '')
        timestamp = tx.get('timestamp', 0)
        token_transfers = tx.get('tokenTransfers', [])
        native_transfers = tx.get('nativeTransfers', [])
        
        if not token_transfers:
            return None
        
        WSOL = "So11111111111111111111111111111111111111112"
        USDC = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"
        USDT = "Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB"
        
        stables = {WSOL, USDC, USDT}
        
        sol_in = sol_out = 0
        tokens_in = {}
        tokens_out = {}
        
        for transfer in token_transfers:
            mint = transfer.get('mint', '')
            amount = float(transfer.get('tokenAmount', 0) or 0)
            from_addr = transfer.get('fromUserAccount', '')
            to_addr = transfer.get('toUserAccount', '')
            
            if from_addr == wallet:
                if mint in stables:
                    sol_out += amount
                else:
                    tokens_out[mint] = tokens_out.get(mint, 0) + amount
            elif to_addr == wallet:
                if mint in stables:
                    sol_in += amount
                else:
                    tokens_in[mint] = tokens_in.get(mint, 0) + amount
        
        for transfer in native_transfers:
            amount = float(transfer.get('amount', 0) or 0) / 1e9
            from_addr = transfer.get('fromUserAccount', '')
            to_addr = transfer.get('toUserAccount', '')
            
            if from_addr == wallet:
                sol_out += amount
            elif to_addr == wallet:
                sol_in += amount
        
        # FIX: Changed from == 1 to >= 1
        # When multiple tokens, pick the one with largest amount
        if len(tokens_in) >= 1 and sol_out > 0:
            # BUY: Wallet spent SOL/stables and received token(s)
            # Pick the token with the largest amount as the primary
            token = max(tokens_in.keys(), key=lambda t: tokens_in[t])
            return {
                'type': 'BUY',
                'token': token,
                'token_amount': tokens_in[token],
                'sol_amount': sol_out,
                'timestamp': timestamp,
                'signature': signature
            }
        elif len(tokens_out) >= 1 and sol_in > 0:
            # SELL: Wallet spent token(s) and received SOL/stables
            # Pick the token with the largest amount as the primary
            token = max(tokens_out.keys(), key=lambda t: tokens_out[t])
            return {
                'type': 'SELL',
                'token': token,
                'token_amount': tokens_out[token],
                'sol_amount': sol_in,
                'timestamp': timestamp,
                'signature': signature
            }
        
        return None
    
    def _match_positions(self, swaps: List[Dict]) -> List[Dict]:
        """Match BUY→SELL pairs"""
        
        positions = []
        open_positions = defaultdict(lambda: {
            'buys': [],
            'total_tokens': 0,
            'total_sol_spent': 0
        })
        
        swaps.sort(key=lambda x: x.get('timestamp', 0))
        
        for swap in swaps:
            token = swap['token']
            swap_type = swap['type']
            
            if swap_type == 'BUY':
                open_positions[token]['buys'].append(swap)
                open_positions[token]['total_tokens'] += swap['token_amount']
                open_positions[token]['total_sol_spent'] += swap['sol_amount']
            
            elif swap_type == 'SELL':
                if token not in open_positions or not open_positions[token]['buys']:
                    continue
                
                sell_amount = swap['token_amount']
                sell_sol = swap['sol_amount']
                sell_time = swap['timestamp']
                
                avg_buy_price = (
                    open_positions[token]['total_sol_spent'] / 
                    max(0.0001, open_positions[token]['total_tokens'])
                )
                
                avg_sell_price = sell_sol / max(0.0001, sell_amount)
                
                sol_spent = sell_amount * avg_buy_price
                profit_sol = sell_sol - sol_spent
                profit_pct = (profit_sol / sol_spent * 100) if sol_spent > 0 else 0
                
                first_buy = open_positions[token]['buys'][0]
                buy_time = first_buy['timestamp']
                hold_hours = (sell_time - buy_time) / 3600 if sell_time > buy_time else 0
                
                positions.append({
                    'token': token,
                    'profit_pct': profit_pct,
                    'profit_sol': profit_sol,
                    'hold_hours': hold_hours,
                    'sol_spent': sol_spent,
                    'buy_time': buy_time,
                    'sell_time': sell_time
                })
                
                open_positions[token]['total_tokens'] -= sell_amount
                open_positions[token]['total_sol_spent'] -= sol_spent
                
                if open_positions[token]['total_tokens'] <= 0.001:
                    open_positions[token]['buys'] = []
                    open_positions[token]['total_tokens'] = 0
                    open_positions[token]['total_sol_spent'] = 0
        
        return positions
    
    def _calculate_stats(self, positions: List[Dict]) -> Dict:
        """Calculate statistics"""
        
        if not positions:
            return self._empty_stats()
        
        wins = [p for p in positions if p['profit_pct'] > 0]
        losses = [p for p in positions if p['profit_pct'] <= 0]
        
        total_pnl = sum(p['profit_sol'] for p in positions)
        total_roi = sum(p['profit_pct'] for p in positions)
        
        avg_win = sum(p['profit_pct'] for p in wins) / len(wins) if wins else 0
        avg_loss = abs(sum(p['profit_pct'] for p in losses) / len(losses)) if losses else 0
        
        return {
            'win_rate': len(wins) / len(positions),
            'pnl': total_pnl,
            'roi_7d': total_roi,
            'completed_swings': len(positions),
            'avg_hold_hours': sum(p['hold_hours'] for p in positions) / len(positions),
            'risk_reward_ratio': avg_win / avg_loss if avg_loss > 0 else 0,
            'best_trade_pct': max((p['profit_pct'] for p in positions), default=0),
            'worst_trade_pct': min((p['profit_pct'] for p in positions), default=0),
            'total_volume_sol': sum(p['sol_spent'] for p in positions),
            'avg_position_size_sol': sum(p['sol_spent'] for p in positions) / len(positions)
        }
    
    def _empty_stats(self) -> Dict:
        return {
            'win_rate': 0, 'pnl': 0, 'roi_7d': 0, 'completed_swings': 0,
            'avg_hold_hours': 0, 'risk_reward_ratio': 0, 'best_trade_pct': 0,
            'worst_trade_pct': 0, 'total_volume_sol': 0, 'avg_position_size_sol': 0
        }


if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    load_dotenv()
    
    helius_key = os.getenv('HELIUS_KEY')
    
    if not helius_key:
        print("❌ HELIUS_KEY not set")
        exit(1)
    
    profiler = WalletProfiler(helius_key)
    
    # Test wallet - use one from your discovery_debug.json
    test_wallet = input("Enter wallet address to profile: ").strip()
    
    if test_wallet:
        print(f"\n🔍 Profiling {test_wallet[:12]}...")
        stats = profiler.calculate_performance(test_wallet, days=7)
        
        print(f"\n📊 Results:")
        print(f"   Win Rate: {stats['win_rate']:.1%}")
        print(f"   PnL (7d): {stats['pnl']:.2f} SOL")
        print(f"   Completed Swings: {stats['completed_swings']}")
        print(f"   Avg Hold: {stats['avg_hold_hours']:.1f}h")
        print(f"   Best Trade: {stats['best_trade_pct']:+.1f}%")
        print(f"   Worst Trade: {stats['worst_trade_pct']:+.1f}%")
