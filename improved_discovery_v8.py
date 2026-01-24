"""
IMPROVED DISCOVERY SYSTEM v8
============================

Incorporates 4 discovery strategies:
1. LOWER THRESHOLDS - More lenient verification for new wallets
2. BIRDEYE LEADERBOARD - Pull proven profitable traders
3. NEW TOKEN FOCUS - Scan tokens < 24h old for active swing traders
4. REVERSE DISCOVERY - Find profitable trades, then track those wallets

Budget: ~5,000 credits per cycle (2 cycles/day)
"""

import os
import sys
import time
import json
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from dataclasses import dataclass

# Load environment
from dotenv import load_dotenv
load_dotenv()

HELIUS_KEY = os.getenv('HELIUS_KEY')
BIRDEYE_KEY = os.getenv('BIRDEYE_API_KEY')
HELIUS_API = f"https://api.helius.xyz/v0"
HELIUS_RPC = f"https://mainnet.helius-rpc.com/?api-key={HELIUS_KEY}"


# =============================================================================
# CONFIGURATION - LOWERED THRESHOLDS (Option 1)
# =============================================================================

@dataclass
class DiscoveryConfig:
    """Discovery configuration with LOWERED thresholds"""
    
    # Verification thresholds (LOWERED from v7)
    min_win_rate: float = 0.50       # Was 0.60 (50% instead of 60%)
    min_pnl_sol: float = 0.5         # Was 2.0 (0.5 SOL instead of 2 SOL)
    min_completed_swings: int = 1    # Was 3 (1 swing instead of 3)
    
    # Pre-filter thresholds
    min_buys: int = 1
    min_sells: int = 1
    min_total_trades: int = 2
    min_volume_sol: float = 0.1
    
    # Budget allocation per cycle
    total_budget: int = 5000
    extraction_budget_pct: float = 0.20      # 20% for token extraction
    validation_budget_pct: float = 0.15      # 15% for wallet validation
    profiling_budget_pct: float = 0.35       # 35% for profiling
    birdeye_budget_pct: float = 0.15         # 15% for Birdeye leaderboard
    reverse_discovery_pct: float = 0.15      # 15% for reverse discovery
    
    # Discovery sources
    max_tokens_to_scan: int = 15
    max_wallets_per_cycle: int = 20
    new_token_max_age_hours: int = 24        # Focus on tokens < 24h old
    
    # Scoring weights
    score_weight_volume: float = 1.0
    score_weight_trades: float = 1.5
    score_weight_balance: float = 2.0        # Balanced buy/sell ratio


config = DiscoveryConfig()


# =============================================================================
# UTILITY CLASSES
# =============================================================================

class APIBudgetTracker:
    """Track API credit usage"""
    
    def __init__(self, budget: int):
        self.budget = budget
        self.used = 0
        self.calls = defaultdict(int)
    
    def use(self, credits: int, operation: str = "unknown"):
        self.used += credits
        self.calls[operation] += credits
    
    def remaining(self) -> int:
        return self.budget - self.used
    
    def can_afford(self, credits: int) -> bool:
        return self.remaining() >= credits
    
    def summary(self) -> str:
        return f"{self.used}/{self.budget} credits used ({100*self.used/self.budget:.0f}%)"


class RateLimiter:
    """Rate limiter for API calls"""
    
    def __init__(self, calls_per_second: float = 5):
        self.min_interval = 1.0 / calls_per_second
        self.last_call = 0
    
    def wait(self):
        elapsed = time.time() - self.last_call
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self.last_call = time.time()


# =============================================================================
# OPTION 2: BIRDEYE LEADERBOARD DISCOVERY
# =============================================================================

class BirdeyeLeaderboardDiscovery:
    """
    Pull top traders from Birdeye.
    These are PROVEN profitable wallets.
    
    Cost: ~0 Helius credits (uses Birdeye API)
    """
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://public-api.birdeye.so"
        self.rate_limiter = RateLimiter(calls_per_second=2)
    
    def get_top_traders(self, timeframe: str = "24h", limit: int = 50) -> List[Dict]:
        """
        Get top traders by finding top traders from trending tokens.
        
        Strategy:
        1. Get trending tokens
        2. For each trending token, get top traders
        3. Return traders with positive PnL
        """
        if not self.api_key:
            print("   ⚠️ BIRDEYE_API_KEY not set - skipping Birdeye discovery")
            return []
        
        all_traders = []
        
        # Get traders from trending tokens
        traders = self._get_traders_from_trending_tokens()
        if traders:
            all_traders.extend(traders)
        
        # Deduplicate by address
        seen = set()
        unique_traders = []
        for t in all_traders:
            if t['address'] not in seen:
                seen.add(t['address'])
                unique_traders.append(t)
        
        return unique_traders[:limit]
    
    def _get_traders_from_trending_tokens(self) -> List[Dict]:
        """Get top traders from trending tokens"""
        if not self.api_key:
            return []
        
        traders = []
        
        try:
            # Get trending tokens first
            self.rate_limiter.wait()
            
            url = f"{self.base_url}/defi/token_trending"
            headers = {
                "X-API-KEY": self.api_key,
                "x-chain": "solana"
            }
            params = {"sort_by": "rank", "sort_type": "asc", "offset": 0, "limit": 20}
            
            response = requests.get(url, headers=headers, params=params, timeout=15)
            
            if response.status_code != 200:
                print(f"   ⚠️ Birdeye trending API returned {response.status_code}")
                return []
            
            data = response.json()
            
            # Debug: show raw response structure
            if isinstance(data, dict):
                print(f"   📦 Response keys: {list(data.keys())[:5]}")
            
            # Handle different response formats
            tokens = []
            if isinstance(data, dict):
                # Format 1: {success: true, data: {tokens: [...]}}
                if data.get('success') and isinstance(data.get('data'), dict):
                    tokens = data['data'].get('tokens', [])
                # Format 2: {data: {tokens: [...]}}
                elif isinstance(data.get('data'), dict):
                    tokens = data['data'].get('tokens', [])
                # Format 3: {data: [...]}
                elif isinstance(data.get('data'), list):
                    tokens = data['data']
                # Format 4: {tokens: [...]}
                elif 'tokens' in data:
                    tokens = data['tokens']
            
            if not tokens:
                print(f"   ⚠️ No trending tokens found in response")
                return []
            
            print(f"   ✅ Found {len(tokens)} trending tokens from Birdeye")
            
            # Get top traders for each trending token
            for token in tokens[:8]:  # Scan up to 8 trending tokens
                token_addr = token.get('address', '')
                token_symbol = token.get('symbol', '?')
                
                if not token_addr:
                    continue
                
                token_traders = self.get_token_top_traders(token_addr, limit=10)
                
                for t in token_traders:
                    # Only add if profitable
                    if t.get('pnl', 0) > 0:
                        t['source'] = 'birdeye_trending'
                        t['token_symbol'] = token_symbol
                        traders.append(t)
            
            if traders:
                print(f"   ✅ Found {len(traders)} profitable traders from trending tokens")
                
        except Exception as e:
            print(f"   ⚠️ Birdeye trending error: {e}")
        
        return traders
    
    def get_token_top_traders(self, token_address: str, limit: int = 20) -> List[Dict]:
        """Get top traders for a specific token"""
        if not self.api_key:
            return []
        
        traders = []
        
        try:
            self.rate_limiter.wait()
            
            # Correct endpoint for token top traders
            url = f"{self.base_url}/defi/v2/tokens/top_traders"
            
            headers = {
                "X-API-KEY": self.api_key,
                "x-chain": "solana"
            }
            
            params = {
                "address": token_address,
                "time_frame": "24h",
                "sort_by": "volume",  # Valid values: volume (not pnl!)
                "sort_type": "desc",
                "offset": 0,
                "limit": limit
            }
            
            response = requests.get(url, headers=headers, params=params, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                
                # Handle different response formats
                items = []
                if isinstance(data, dict):
                    # Format 1: {success: true, data: {items: [...]}}
                    if data.get('success') and isinstance(data.get('data'), dict):
                        items = data['data'].get('items', [])
                    # Format 2: {data: {items: [...]}}
                    elif isinstance(data.get('data'), dict):
                        items = data['data'].get('items', [])
                    # Format 3: {data: [...]}
                    elif isinstance(data.get('data'), list):
                        items = data['data']
                    # Format 4: {items: [...]}
                    elif 'items' in data:
                        items = data['items']
                
                for item in items:
                    if not item:
                        continue
                    
                    trader = {
                        'address': item.get('address', '') or item.get('owner', '') or item.get('wallet', ''),
                        'pnl': float(item.get('pnl', 0) or item.get('realizedPnl', 0) or 0),
                        'bought': float(item.get('bought', 0) or item.get('totalBuy', 0) or 0),
                        'sold': float(item.get('sold', 0) or item.get('totalSell', 0) or 0),
                        'source': 'birdeye_token_traders'
                    }
                    
                    if trader['address'] and len(trader['address']) > 30:
                        traders.append(trader)
            else:
                # Silently skip - this API might require premium
                pass
                        
        except Exception as e:
            # Silently skip errors for individual tokens
            pass
        
        return traders


# =============================================================================
# OPTION 3: NEW TOKEN DISCOVERY (< 24h old)
# =============================================================================

class NewTokenDiscovery:
    """
    Focus on NEW tokens (< 24h old) where swing trading is most active.
    
    Cost: 0 Helius credits (uses DexScreener)
    """
    
    def __init__(self):
        self.rate_limiter = RateLimiter(calls_per_second=2)
    
    def get_new_tokens(self, max_age_hours: int = 24, min_volume: float = 10000, 
                       min_liquidity: float = 5000, limit: int = 30) -> List[Dict]:
        """
        Find newly created tokens with trading activity.
        
        New tokens have more swing trading because:
        - Price volatility is higher
        - Entry/exit opportunities are clearer
        - Traders are more active
        """
        tokens = []
        
        try:
            self.rate_limiter.wait()
            
            # Use DexScreener token profiles/latest which shows recent tokens
            url = "https://api.dexscreener.com/token-profiles/latest/v1"
            response = requests.get(url, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                
                for item in data[:100]:
                    if item.get('chainId') != 'solana':
                        continue
                    
                    token_addr = item.get('tokenAddress', '')
                    if not token_addr or len(token_addr) < 30:
                        continue
                    
                    tokens.append({
                        'address': token_addr,
                        'symbol': item.get('symbol', '?'),
                        'description': item.get('description', '')[:50] if item.get('description') else '',
                        'source': 'dexscreener_latest'
                    })
                
                if tokens:
                    print(f"   ✅ Found {len(tokens)} latest token profiles")
            
            # Also try the boosted tokens (popular new tokens)
            self.rate_limiter.wait()
            boost_url = "https://api.dexscreener.com/token-boosts/latest/v1"
            boost_res = requests.get(boost_url, timeout=15)
            
            if boost_res.status_code == 200:
                boost_data = boost_res.json()
                
                existing_addrs = {t['address'] for t in tokens}
                
                for item in boost_data[:50]:
                    if item.get('chainId') != 'solana':
                        continue
                    
                    token_addr = item.get('tokenAddress', '')
                    if not token_addr or len(token_addr) < 30:
                        continue
                    
                    if token_addr in existing_addrs:
                        continue
                    
                    tokens.append({
                        'address': token_addr,
                        'symbol': '?',  # Will get from pair data
                        'source': 'dexscreener_boosted'
                    })
                
            # Get pair details for tokens (to get volume, age, etc.)
            enriched_tokens = []
            for token in tokens[:20]:  # Limit API calls
                self.rate_limiter.wait()
                
                pair_url = f"https://api.dexscreener.com/latest/dex/tokens/{token['address']}"
                try:
                    pair_res = requests.get(pair_url, timeout=10)
                    
                    if pair_res.status_code == 200:
                        pair_data = pair_res.json()
                        pairs = pair_data.get('pairs', [])
                        
                        if pairs:
                            pair = pairs[0]  # Take the main pair
                            
                            # Get creation time
                            created_at = pair.get('pairCreatedAt')
                            age_hours = 999
                            if created_at:
                                pair_time = datetime.fromtimestamp(created_at / 1000)
                                age_hours = (datetime.now() - pair_time).total_seconds() / 3600
                            
                            # Skip if too old
                            if age_hours > max_age_hours:
                                continue
                            
                            volume = float(pair.get('volume', {}).get('h24', 0) or 0)
                            liquidity = float(pair.get('liquidity', {}).get('usd', 0) or 0)
                            
                            # Skip if too low volume/liquidity
                            if volume < min_volume or liquidity < min_liquidity:
                                continue
                            
                            token['symbol'] = pair.get('baseToken', {}).get('symbol', token.get('symbol', '?'))
                            token['name'] = pair.get('baseToken', {}).get('name', '')
                            token['volume_24h'] = volume
                            token['liquidity'] = liquidity
                            token['price_change_24h'] = float(pair.get('priceChange', {}).get('h24', 0) or 0)
                            token['age_hours'] = age_hours
                            token['txns_24h'] = pair.get('txns', {}).get('h24', {})
                            
                            enriched_tokens.append(token)
                            
                except Exception as e:
                    continue
            
            # Sort by volume
            enriched_tokens.sort(key=lambda x: x.get('volume_24h', 0), reverse=True)
            tokens = enriched_tokens[:limit]
            
            if tokens:
                print(f"   ✅ Found {len(tokens)} new tokens (< {max_age_hours}h old) with volume > ${min_volume:,.0f}")
            else:
                print(f"   ⚠️ No new tokens found meeting criteria")
            
        except Exception as e:
            print(f"   ⚠️ New token discovery error: {e}")
        
        return tokens
    
    def get_pumping_new_tokens(self, min_gain: float = 30, max_age_hours: int = 48) -> List[Dict]:
        """Find tokens that are pumping (high gains)"""
        tokens = []
        
        try:
            self.rate_limiter.wait()
            
            # Get top boosted tokens
            url = "https://api.dexscreener.com/token-boosts/top/v1"
            response = requests.get(url, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                
                for item in data[:30]:
                    if item.get('chainId') != 'solana':
                        continue
                    
                    token_addr = item.get('tokenAddress', '')
                    if not token_addr:
                        continue
                    
                    # Get pair details
                    self.rate_limiter.wait()
                    pair_url = f"https://api.dexscreener.com/latest/dex/tokens/{token_addr}"
                    pair_res = requests.get(pair_url, timeout=10)
                    
                    if pair_res.status_code == 200:
                        pair_data = pair_res.json()
                        pairs = pair_data.get('pairs', [])
                        
                        if pairs:
                            pair = pairs[0]
                            
                            # Check age
                            created_at = pair.get('pairCreatedAt')
                            age_hours = 999
                            if created_at:
                                pair_time = datetime.fromtimestamp(created_at / 1000)
                                age_hours = (datetime.now() - pair_time).total_seconds() / 3600
                            
                            if age_hours > max_age_hours:
                                continue
                            
                            price_change = float(pair.get('priceChange', {}).get('h24', 0) or 0)
                            
                            if price_change >= min_gain:
                                tokens.append({
                                    'address': token_addr,
                                    'symbol': pair.get('baseToken', {}).get('symbol', '?'),
                                    'price_change_24h': price_change,
                                    'volume_24h': float(pair.get('volume', {}).get('h24', 0) or 0),
                                    'age_hours': age_hours,
                                    'source': 'pumping_new'
                                })
                
                if tokens:
                    print(f"   ✅ Found {len(tokens)} pumping tokens (>{min_gain}% gain, <{max_age_hours}h old)")
                else:
                    print(f"   ⚠️ No pumping tokens found")
                
        except Exception as e:
            print(f"   ⚠️ Pumping tokens error: {e}")
        
        return tokens


# =============================================================================
# OPTION 4: REVERSE DISCOVERY - Find Profitable Trades First
# =============================================================================

class ReverseDiscovery:
    """
    Reverse discovery: Find PROFITABLE TRADES first, then track those wallets.
    
    Instead of: Find wallets → Check if profitable
    We do: Find profitable token moves → Find who made money → Track them
    
    Cost: ~50-100 Helius credits per token analyzed
    """
    
    def __init__(self, helius_key: str):
        self.helius_key = helius_key
        self.helius_api = HELIUS_API
        self.helius_rpc = HELIUS_RPC
        self.rate_limiter = RateLimiter(calls_per_second=5)
        
        self.stables = {
            "So11111111111111111111111111111111111111112",   # WSOL
            "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v", # USDC
            "Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB", # USDT
        }
        
        self.excluded = {
            '675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8',  # Raydium
            'JUP6LkbZbjS1jKKwapdHNy74zcZ3tLUZoi5QNyVTaV4',  # Jupiter
            '5Q544fKrFoe6tsEbD7S8EmxGTJYAKtTVhAW5Q5pge4j1',  # Raydium
        }
    
    def find_profitable_traders(self, token_address: str, budget: APIBudgetTracker,
                                min_profit_sol: float = 0.5) -> List[Dict]:
        """
        Find wallets that made profit trading a specific token.
        
        Strategy:
        1. Get recent transactions for the token
        2. Track buys and sells per wallet
        3. Calculate realized profit
        4. Return wallets with profit > threshold
        """
        if not budget.can_afford(50):
            return []
        
        profitable_wallets = []
        wallet_trades = defaultdict(lambda: {'buys': [], 'sells': [], 'total_bought_sol': 0, 'total_sold_sol': 0})
        
        try:
            # Get token transactions
            self.rate_limiter.wait()
            
            payload = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "getSignaturesForAddress",
                "params": [token_address, {"limit": 200}]
            }
            
            res = requests.post(self.helius_rpc, json=payload, timeout=15)
            signatures = res.json().get('result', [])
            budget.use(2, "reverse_signatures")
            
            if not signatures:
                return []
            
            # Parse transactions in batches
            sig_list = [s.get('signature') for s in signatures[:100] if s.get('signature')]
            
            self.rate_limiter.wait()
            parse_url = f"{self.helius_api}/transactions?api-key={self.helius_key}"
            parse_res = requests.post(parse_url, json={"transactions": sig_list}, timeout=20)
            txs = parse_res.json()
            budget.use(20, "reverse_parse")
            
            if not isinstance(txs, list):
                return []
            
            # Analyze each transaction
            for tx in txs:
                if tx.get('type') != 'SWAP':
                    continue
                
                fee_payer = tx.get('feePayer', '')
                if not fee_payer or fee_payer in self.excluded:
                    continue
                
                # Analyze the swap
                swap = self._analyze_swap(tx, fee_payer)
                if not swap:
                    continue
                
                if swap['type'] == 'BUY':
                    wallet_trades[fee_payer]['buys'].append(swap)
                    wallet_trades[fee_payer]['total_bought_sol'] += swap['sol_amount']
                elif swap['type'] == 'SELL':
                    wallet_trades[fee_payer]['sells'].append(swap)
                    wallet_trades[fee_payer]['total_sold_sol'] += swap['sol_amount']
            
            # Calculate profits
            for wallet, trades in wallet_trades.items():
                if not trades['buys'] or not trades['sells']:
                    continue
                
                # Simple profit calculation: sold - bought
                profit = trades['total_sold_sol'] - trades['total_bought_sol']
                
                if profit >= min_profit_sol:
                    profitable_wallets.append({
                        'address': wallet,
                        'profit_sol': profit,
                        'buy_count': len(trades['buys']),
                        'sell_count': len(trades['sells']),
                        'total_bought': trades['total_bought_sol'],
                        'total_sold': trades['total_sold_sol'],
                        'roi_pct': (profit / trades['total_bought_sol'] * 100) if trades['total_bought_sol'] > 0 else 0,
                        'source': 'reverse_discovery',
                        'token': token_address[:12]
                    })
            
            # Sort by profit
            profitable_wallets.sort(key=lambda x: x['profit_sol'], reverse=True)
            
        except Exception as e:
            print(f"      ⚠️ Reverse discovery error: {e}")
        
        return profitable_wallets
    
    def _analyze_swap(self, tx: Dict, wallet: str) -> Optional[Dict]:
        """Analyze a swap transaction"""
        token_transfers = tx.get('tokenTransfers', [])
        native_transfers = tx.get('nativeTransfers', [])
        
        sol_in = sol_out = 0
        tokens_in = {}
        tokens_out = {}
        
        for transfer in token_transfers:
            mint = transfer.get('mint', '')
            amount = float(transfer.get('tokenAmount', 0) or 0)
            from_addr = transfer.get('fromUserAccount', '')
            to_addr = transfer.get('toUserAccount', '')
            
            if from_addr == wallet:
                if mint in self.stables:
                    sol_out += amount
                else:
                    tokens_out[mint] = tokens_out.get(mint, 0) + amount
            elif to_addr == wallet:
                if mint in self.stables:
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
        
        if len(tokens_in) >= 1 and sol_out > 0:
            return {'type': 'BUY', 'sol_amount': sol_out}
        elif len(tokens_out) >= 1 and sol_in > 0:
            return {'type': 'SELL', 'sol_amount': sol_in}
        
        return None


# =============================================================================
# QUICK WALLET VALIDATOR
# =============================================================================

class QuickWalletValidator:
    """
    Quick validation of wallet trading activity.
    Checks if wallet has actual swing trading behavior.
    
    Cost: ~2 Helius credits per wallet
    """
    
    def __init__(self, helius_key: str):
        self.helius_key = helius_key
        self.helius_api = HELIUS_API
        self.helius_rpc = HELIUS_RPC
        self.rate_limiter = RateLimiter(calls_per_second=5)
        
        self.stables = {
            "So11111111111111111111111111111111111111112",
            "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
            "Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB",
        }
    
    def validate(self, wallet: str, budget: APIBudgetTracker) -> Dict:
        """
        Quick validation of wallet activity.
        
        Returns:
            {
                'valid': bool,
                'buy_count': int,
                'sell_count': int,
                'unique_tokens': int,
                'has_completed_swings': bool,
                'api_calls': int
            }
        """
        result = {
            'valid': False,
            'buy_count': 0,
            'sell_count': 0,
            'unique_tokens': 0,
            'total_volume': 0,
            'has_completed_swings': False,
            'api_calls': 0
        }
        
        if not budget.can_afford(4):
            return result
        
        try:
            # Get recent signatures
            self.rate_limiter.wait()
            
            payload = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "getSignaturesForAddress",
                "params": [wallet, {"limit": 50}]
            }
            
            res = requests.post(self.helius_rpc, json=payload, timeout=15)
            signatures = res.json().get('result', [])
            result['api_calls'] += 1
            budget.use(1, "validate_signatures")
            
            if not signatures:
                return result
            
            # Parse transactions
            sig_list = [s.get('signature') for s in signatures if s.get('signature')]
            
            self.rate_limiter.wait()
            parse_url = f"{self.helius_api}/transactions?api-key={self.helius_key}"
            parse_res = requests.post(parse_url, json={"transactions": sig_list}, timeout=20)
            txs = parse_res.json()
            result['api_calls'] += 2
            budget.use(2, "validate_parse")
            
            if not isinstance(txs, list):
                return result
            
            # Analyze swaps
            tokens_bought = set()
            tokens_sold = set()
            
            for tx in txs:
                if tx.get('type') != 'SWAP':
                    continue
                
                swap = self._analyze_swap(tx, wallet)
                if swap:
                    result['total_volume'] += swap.get('sol_amount', 0)
                    
                    if swap['type'] == 'BUY':
                        result['buy_count'] += 1
                        tokens_bought.add(swap.get('token', ''))
                    elif swap['type'] == 'SELL':
                        result['sell_count'] += 1
                        tokens_sold.add(swap.get('token', ''))
            
            result['unique_tokens'] = len(tokens_bought | tokens_sold)
            result['has_completed_swings'] = len(tokens_bought & tokens_sold) > 0
            
            # Determine validity
            result['valid'] = (
                result['buy_count'] >= config.min_buys and
                result['sell_count'] >= config.min_sells and
                result['total_volume'] >= config.min_volume_sol
            )
            
        except Exception as e:
            print(f"      ⚠️ Validation error: {e}")
        
        return result
    
    def _analyze_swap(self, tx: Dict, wallet: str) -> Optional[Dict]:
        """Quick swap analysis"""
        token_transfers = tx.get('tokenTransfers', [])
        native_transfers = tx.get('nativeTransfers', [])
        
        sol_in = sol_out = 0
        tokens_in = {}
        tokens_out = {}
        
        for transfer in token_transfers:
            mint = transfer.get('mint', '')
            amount = float(transfer.get('tokenAmount', 0) or 0)
            from_addr = transfer.get('fromUserAccount', '')
            to_addr = transfer.get('toUserAccount', '')
            
            if from_addr == wallet:
                if mint in self.stables:
                    sol_out += amount
                else:
                    tokens_out[mint] = tokens_out.get(mint, 0) + amount
            elif to_addr == wallet:
                if mint in self.stables:
                    sol_in += amount
                else:
                    tokens_in[mint] = tokens_in.get(mint, 0) + amount
        
        for transfer in native_transfers:
            amount = float(transfer.get('amount', 0) or 0) / 1e9
            if transfer.get('fromUserAccount') == wallet:
                sol_out += amount
            elif transfer.get('toUserAccount') == wallet:
                sol_in += amount
        
        if len(tokens_in) >= 1 and sol_out > 0:
            token = max(tokens_in.keys(), key=lambda t: tokens_in[t])
            return {'type': 'BUY', 'token': token, 'sol_amount': sol_out}
        elif len(tokens_out) >= 1 and sol_in > 0:
            token = max(tokens_out.keys(), key=lambda t: tokens_out[t])
            return {'type': 'SELL', 'token': token, 'sol_amount': sol_in}
        
        return None


# =============================================================================
# WALLET PROFILER (LOWERED THRESHOLDS)
# =============================================================================

class WalletProfiler:
    """
    Profile wallet trading performance.
    Uses LOWERED thresholds for verification.
    
    Cost: ~10-20 Helius credits per wallet
    """
    
    def __init__(self, helius_key: str):
        self.helius_key = helius_key
        self.helius_api = HELIUS_API
        self.rate_limiter = RateLimiter(calls_per_second=5)
        
        self.stables = {
            "So11111111111111111111111111111111111111112",
            "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
            "Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB",
        }
    
    def profile(self, wallet: str, budget: APIBudgetTracker, days: int = 7) -> Dict:
        """
        Profile wallet and return performance stats.
        
        Returns:
            {
                'win_rate': float,
                'pnl_sol': float,
                'completed_swings': int,
                'avg_hold_hours': float,
                'meets_threshold': bool,
                'api_calls': int
            }
        """
        result = self._empty_result()
        
        if not budget.can_afford(15):
            return result
        
        try:
            swaps = self._get_swaps(wallet, budget, days)
            
            if not swaps:
                return result
            
            positions = self._match_positions(swaps)
            
            if not positions:
                return result
            
            result = self._calculate_stats(positions)
            
        except Exception as e:
            print(f"      ⚠️ Profile error: {e}")
        
        return result
    
    def _get_swaps(self, wallet: str, budget: APIBudgetTracker, days: int) -> List[Dict]:
        """Get swap transactions for wallet"""
        swaps = []
        
        self.rate_limiter.wait()
        
        url = f"{self.helius_api}/addresses/{wallet}/transactions"
        params = {
            "api-key": self.helius_key,
            "type": "SWAP",
            "limit": 100
        }
        
        try:
            res = requests.get(url, params=params, timeout=15)
            budget.use(5, "profile_get_swaps")
            
            if res.status_code != 200:
                return []
            
            data = res.json()
            cutoff = datetime.now() - timedelta(days=days)
            
            for tx in data:
                timestamp = tx.get('timestamp', 0)
                if timestamp == 0:
                    continue
                
                tx_time = datetime.fromtimestamp(timestamp)
                if tx_time < cutoff:
                    break
                
                swap = self._parse_swap(tx, wallet, timestamp)
                if swap:
                    swaps.append(swap)
                    
        except Exception as e:
            print(f"      ⚠️ Get swaps error: {e}")
        
        return swaps
    
    def _parse_swap(self, tx: Dict, wallet: str, timestamp: int) -> Optional[Dict]:
        """Parse a swap transaction"""
        token_transfers = tx.get('tokenTransfers', [])
        native_transfers = tx.get('nativeTransfers', [])
        
        sol_in = sol_out = 0
        tokens_in = {}
        tokens_out = {}
        
        for transfer in token_transfers:
            mint = transfer.get('mint', '')
            amount = float(transfer.get('tokenAmount', 0) or 0)
            from_addr = transfer.get('fromUserAccount', '')
            to_addr = transfer.get('toUserAccount', '')
            
            if from_addr == wallet:
                if mint in self.stables:
                    sol_out += amount
                else:
                    tokens_out[mint] = tokens_out.get(mint, 0) + amount
            elif to_addr == wallet:
                if mint in self.stables:
                    sol_in += amount
                else:
                    tokens_in[mint] = tokens_in.get(mint, 0) + amount
        
        for transfer in native_transfers:
            amount = float(transfer.get('amount', 0) or 0) / 1e9
            if transfer.get('fromUserAccount') == wallet:
                sol_out += amount
            elif transfer.get('toUserAccount') == wallet:
                sol_in += amount
        
        if len(tokens_in) >= 1 and sol_out > 0:
            token = max(tokens_in.keys(), key=lambda t: tokens_in[t])
            return {
                'type': 'BUY',
                'token': token,
                'token_amount': tokens_in[token],
                'sol_amount': sol_out,
                'timestamp': timestamp
            }
        elif len(tokens_out) >= 1 and sol_in > 0:
            token = max(tokens_out.keys(), key=lambda t: tokens_out[t])
            return {
                'type': 'SELL',
                'token': token,
                'token_amount': tokens_out[token],
                'sol_amount': sol_in,
                'timestamp': timestamp
            }
        
        return None
    
    def _match_positions(self, swaps: List[Dict]) -> List[Dict]:
        """Match buy→sell pairs into completed positions"""
        positions = []
        open_positions = defaultdict(lambda: {
            'buys': [],
            'total_tokens': 0,
            'total_sol_spent': 0
        })
        
        swaps.sort(key=lambda x: x.get('timestamp', 0))
        
        for swap in swaps:
            token = swap['token']
            
            if swap['type'] == 'BUY':
                open_positions[token]['buys'].append(swap)
                open_positions[token]['total_tokens'] += swap['token_amount']
                open_positions[token]['total_sol_spent'] += swap['sol_amount']
            
            elif swap['type'] == 'SELL':
                if token not in open_positions or not open_positions[token]['buys']:
                    continue
                
                sell_sol = swap['sol_amount']
                sell_time = swap['timestamp']
                
                # Calculate average buy price
                total_spent = open_positions[token]['total_sol_spent']
                total_tokens = open_positions[token]['total_tokens']
                
                if total_tokens <= 0:
                    continue
                
                avg_buy_price = total_spent / total_tokens
                sell_tokens = swap['token_amount']
                
                # Calculate profit
                sol_cost = sell_tokens * avg_buy_price
                profit_sol = sell_sol - sol_cost
                profit_pct = (profit_sol / sol_cost * 100) if sol_cost > 0 else 0
                
                # Calculate hold time
                first_buy = open_positions[token]['buys'][0]
                hold_seconds = sell_time - first_buy['timestamp']
                hold_hours = hold_seconds / 3600
                
                positions.append({
                    'token': token,
                    'entry_sol': sol_cost,
                    'exit_sol': sell_sol,
                    'profit_sol': profit_sol,
                    'profit_pct': profit_pct,
                    'hold_hours': hold_hours,
                    'is_win': profit_sol > 0
                })
                
                # Reduce open position
                open_positions[token]['total_tokens'] -= sell_tokens
                open_positions[token]['total_sol_spent'] -= sol_cost
        
        return positions
    
    def _calculate_stats(self, positions: List[Dict]) -> Dict:
        """Calculate performance statistics"""
        if not positions:
            return self._empty_result()
        
        wins = sum(1 for p in positions if p['is_win'])
        total_pnl = sum(p['profit_sol'] for p in positions)
        avg_hold = sum(p['hold_hours'] for p in positions) / len(positions)
        
        win_rate = wins / len(positions) if positions else 0
        
        result = {
            'win_rate': win_rate,
            'pnl_sol': total_pnl,
            'completed_swings': len(positions),
            'avg_hold_hours': avg_hold,
            'wins': wins,
            'losses': len(positions) - wins,
            'api_calls': 5
        }
        
        # Check against LOWERED thresholds
        result['meets_threshold'] = (
            win_rate >= config.min_win_rate and
            total_pnl >= config.min_pnl_sol and
            len(positions) >= config.min_completed_swings
        )
        
        return result
    
    def _empty_result(self) -> Dict:
        return {
            'win_rate': 0,
            'pnl_sol': 0,
            'completed_swings': 0,
            'avg_hold_hours': 0,
            'wins': 0,
            'losses': 0,
            'meets_threshold': False,
            'api_calls': 0
        }


# =============================================================================
# MAIN DISCOVERY ENGINE
# =============================================================================

class ImprovedDiscoveryV8:
    """
    Improved Discovery System v8
    
    Combines all 4 strategies:
    1. Lower thresholds for verification
    2. Birdeye leaderboard for proven traders
    3. New token focus for active swing trading
    4. Reverse discovery for profitable trades
    """
    
    def __init__(self, db, helius_key: str = None, birdeye_key: str = None):
        self.db = db
        self.helius_key = helius_key or HELIUS_KEY
        self.birdeye_key = birdeye_key or BIRDEYE_KEY
        
        # Discovery sources
        self.birdeye_discovery = BirdeyeLeaderboardDiscovery(self.birdeye_key)
        self.new_token_discovery = NewTokenDiscovery()
        self.reverse_discovery = ReverseDiscovery(self.helius_key)
        
        # Validation and profiling
        self.validator = QuickWalletValidator(self.helius_key)
        self.profiler = WalletProfiler(self.helius_key)
        
        print("✅ Improved Discovery v8 initialized")
        print(f"   Thresholds: WR≥{config.min_win_rate:.0%} | PnL≥{config.min_pnl_sol} SOL | Swings≥{config.min_completed_swings}")
        print(f"   Sources: Birdeye Leaderboard, New Tokens, Reverse Discovery")
    
    def run_discovery(self, api_budget: int = 5000, max_wallets: int = 20) -> Dict:
        """
        Run multi-strategy discovery.
        
        Budget allocation:
        - 20% Token extraction
        - 15% Wallet validation
        - 35% Profiling
        - 15% Birdeye leaderboard
        - 15% Reverse discovery
        """
        budget = APIBudgetTracker(api_budget)
        
        stats = {
            'candidates_found': 0,
            'from_birdeye': 0,
            'from_new_tokens': 0,
            'from_reverse': 0,
            'validated': 0,
            'profiled': 0,
            'verified': 0,
            'verified_wallets': [],
            'api_credits_used': 0
        }
        
        print(f"\n{'='*70}")
        print(f"🎯 IMPROVED DISCOVERY v8 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"   Budget: {api_budget} credits | Target: {max_wallets} wallets")
        print(f"   Thresholds: WR≥{config.min_win_rate:.0%} | PnL≥{config.min_pnl_sol} SOL | Swings≥{config.min_completed_swings}")
        print(f"{'='*70}")
        
        all_candidates = {}
        
        # =================================================================
        # SOURCE 1: BIRDEYE LEADERBOARD (Option 2)
        # =================================================================
        print(f"\n{'='*70}")
        print("SOURCE 1: BIRDEYE LEADERBOARD (Proven Profitable Traders)")
        print(f"{'='*70}")
        
        birdeye_traders = self.birdeye_discovery.get_top_traders(timeframe="24h", limit=30)
        
        for trader in birdeye_traders:
            addr = trader['address']
            if self._is_already_tracked(addr):
                continue
            
            if addr not in all_candidates:
                all_candidates[addr] = {
                    'address': addr,
                    'source': 'birdeye_leaderboard',
                    'birdeye_pnl': trader.get('pnl', 0),
                    'birdeye_win_rate': trader.get('win_rate', 0),
                    'score': 100  # High priority for proven traders
                }
                stats['from_birdeye'] += 1
        
        print(f"   Added {stats['from_birdeye']} candidates from Birdeye")
        
        # =================================================================
        # SOURCE 2: NEW TOKENS (Option 3)
        # =================================================================
        print(f"\n{'='*70}")
        print(f"SOURCE 2: NEW TOKENS (< {config.new_token_max_age_hours}h old)")
        print(f"{'='*70}")
        
        new_tokens = self.new_token_discovery.get_new_tokens(
            max_age_hours=config.new_token_max_age_hours,
            limit=10
        )
        
        extraction_budget = int(api_budget * config.extraction_budget_pct)
        
        for token in new_tokens[:5]:
            if not budget.can_afford(20):
                break
            
            print(f"\n   ${token['symbol']} (age: {token['age_hours']:.1f}h, vol: ${token['volume_24h']:,.0f})")
            
            # Get traders from this token using reverse discovery
            traders = self.reverse_discovery.find_profitable_traders(
                token['address'], 
                budget,
                min_profit_sol=0.1  # Lower threshold for new tokens
            )
            
            for trader in traders[:10]:
                addr = trader['address']
                if self._is_already_tracked(addr):
                    continue
                
                if addr not in all_candidates:
                    all_candidates[addr] = {
                        'address': addr,
                        'source': 'new_token',
                        'token': token['symbol'],
                        'profit_sol': trader.get('profit_sol', 0),
                        'score': 80 + min(20, trader.get('profit_sol', 0) * 10)
                    }
                    stats['from_new_tokens'] += 1
            
            print(f"      Found {len(traders)} profitable traders")
        
        print(f"\n   Added {stats['from_new_tokens']} candidates from new tokens")
        
        # =================================================================
        # SOURCE 3: REVERSE DISCOVERY ON TRENDING TOKENS (Option 4)
        # =================================================================
        print(f"\n{'='*70}")
        print("SOURCE 3: REVERSE DISCOVERY (Find Profitable Trades)")
        print(f"{'='*70}")
        
        # Get trending tokens
        pumping_tokens = self.new_token_discovery.get_pumping_new_tokens(min_gain=30)
        
        reverse_budget = int(api_budget * config.reverse_discovery_pct)
        
        for token in pumping_tokens[:5]:
            if budget.used >= reverse_budget + extraction_budget:
                break
            
            print(f"\n   ${token['symbol']} (+{token['price_change_24h']:.0f}%)")
            
            traders = self.reverse_discovery.find_profitable_traders(
                token['address'],
                budget,
                min_profit_sol=0.3
            )
            
            for trader in traders[:10]:
                addr = trader['address']
                if self._is_already_tracked(addr):
                    continue
                
                if addr not in all_candidates:
                    all_candidates[addr] = {
                        'address': addr,
                        'source': 'reverse_discovery',
                        'token': token['symbol'],
                        'profit_sol': trader.get('profit_sol', 0),
                        'roi_pct': trader.get('roi_pct', 0),
                        'score': 70 + min(30, trader.get('roi_pct', 0) / 10)
                    }
                    stats['from_reverse'] += 1
            
            print(f"      Found {len(traders)} profitable traders")
        
        print(f"\n   Added {stats['from_reverse']} candidates from reverse discovery")
        
        # =================================================================
        # SOURCE 4: HIGH-VOLUME TOKEN TRADERS (Fallback)
        # =================================================================
        # If we have few candidates, scan high-volume tokens for active traders
        if len(all_candidates) < 10 and budget.remaining() > 200:
            print(f"\n{'='*70}")
            print("SOURCE 4: HIGH-VOLUME TOKEN TRADERS (Fallback)")
            print(f"{'='*70}")
            
            high_volume_tokens = self._get_high_volume_tokens()
            stats['from_high_volume'] = 0
            
            for token in high_volume_tokens[:5]:
                if budget.remaining() < 50:
                    break
                
                print(f"\n   ${token['symbol']} (vol: ${token['volume_24h']:,.0f})")
                
                traders = self.reverse_discovery.find_profitable_traders(
                    token['address'],
                    budget,
                    min_profit_sol=0.1  # Lower threshold for fallback
                )
                
                for trader in traders[:15]:
                    addr = trader['address']
                    if self._is_already_tracked(addr):
                        continue
                    
                    if addr not in all_candidates:
                        all_candidates[addr] = {
                            'address': addr,
                            'source': 'high_volume_token',
                            'token': token['symbol'],
                            'profit_sol': trader.get('profit_sol', 0),
                            'score': 60 + min(20, trader.get('profit_sol', 0) * 10)
                        }
                        stats['from_high_volume'] += 1
                
                print(f"      Found {len(traders)} profitable traders")
            
            print(f"\n   Added {stats.get('from_high_volume', 0)} candidates from high-volume tokens")
        
        stats['candidates_found'] = len(all_candidates)
        
        # =================================================================
        # VALIDATION & PROFILING
        # =================================================================
        print(f"\n{'='*70}")
        print("VALIDATION & PROFILING")
        print(f"   Candidates to process: {len(all_candidates)}")
        print(f"   Budget remaining: {budget.remaining()}")
        print(f"{'='*70}")
        
        # Sort candidates by score
        sorted_candidates = sorted(
            all_candidates.values(),
            key=lambda x: x.get('score', 0),
            reverse=True
        )
        
        verified_count = 0
        
        for candidate in sorted_candidates:
            if verified_count >= max_wallets:
                print(f"\n   ✅ Reached target: {max_wallets} wallets")
                break
            
            if not budget.can_afford(20):
                print(f"\n   ⚠️ Budget exhausted")
                break
            
            wallet = candidate['address']
            source = candidate.get('source', 'unknown')
            
            print(f"\n   💎 {wallet[:16]}... ({source})")
            
            # Quick validation
            validation = self.validator.validate(wallet, budget)
            stats['validated'] += 1
            
            if not validation['valid']:
                print(f"      ❌ Invalid: {validation['buy_count']}B/{validation['sell_count']}S")
                continue
            
            print(f"      ✓ Valid: {validation['buy_count']}B/{validation['sell_count']}S")
            
            # Full profiling
            profile = self.profiler.profile(wallet, budget)
            stats['profiled'] += 1
            
            wr_emoji = "✅" if profile['win_rate'] >= config.min_win_rate else "❌"
            pnl_emoji = "✅" if profile['pnl_sol'] >= config.min_pnl_sol else "❌"
            swings_emoji = "✅" if profile['completed_swings'] >= config.min_completed_swings else "❌"
            
            print(f"      WR: {profile['win_rate']:.0%} {wr_emoji}")
            print(f"      PnL: {profile['pnl_sol']:.2f} SOL {pnl_emoji}")
            print(f"      Swings: {profile['completed_swings']} {swings_emoji}")
            
            if profile['meets_threshold']:
                # VERIFIED! Add to database
                print(f"      🎉 VERIFIED!")
                
                self._add_wallet_to_db(wallet, profile, candidate)
                
                verified_count += 1
                stats['verified'] += 1
                stats['verified_wallets'].append({
                    'address': wallet,
                    'win_rate': profile['win_rate'],
                    'pnl_sol': profile['pnl_sol'],
                    'completed_swings': profile['completed_swings'],
                    'source': source
                })
        
        stats['api_credits_used'] = budget.used
        
        # =================================================================
        # SUMMARY
        # =================================================================
        print(f"\n{'='*70}")
        print("✅ DISCOVERY v8 COMPLETE")
        print(f"{'='*70}")
        print(f"   Sources:")
        print(f"      Birdeye Leaderboard: {stats['from_birdeye']} candidates")
        print(f"      New Tokens: {stats['from_new_tokens']} candidates")
        print(f"      Reverse Discovery: {stats['from_reverse']} candidates")
        print(f"      High Volume Fallback: {stats.get('from_high_volume', 0)} candidates")
        print(f"   Total candidates: {stats['candidates_found']}")
        print(f"   Validated: {stats['validated']}")
        print(f"   Profiled: {stats['profiled']}")
        print(f"   VERIFIED: {stats['verified']} ✅")
        print(f"   API Credits: {budget.summary()}")
        
        if stats['verified_wallets']:
            print(f"\n   ✨ New verified wallets:")
            for w in stats['verified_wallets']:
                print(f"      {w['address'][:20]}...")
                print(f"         WR: {w['win_rate']:.0%} | PnL: {w['pnl_sol']:.2f} SOL | Swings: {w['completed_swings']} | Source: {w['source']}")
        
        print(f"{'='*70}")
        
        # Save debug info
        self._save_debug(stats, budget)
        
        return stats
    
    def _is_already_tracked(self, wallet: str) -> bool:
        """Check if wallet is already in database"""
        try:
            return self.db.is_wallet_tracked(wallet)
        except:
            return False
    
    def _add_wallet_to_db(self, wallet: str, profile: Dict, candidate: Dict):
        """Add verified wallet to database"""
        try:
            # Build stats dict matching database_v2.add_verified_wallet signature
            stats = {
                'pnl': profile.get('pnl_sol', 0),
                'win_rate': profile.get('win_rate', 0) * 100,  # Convert to percentage
                'completed_swings': profile.get('completed_swings', 0),
                'avg_hold_hours': profile.get('avg_hold_hours', 0),
                'roi_7d': 0,
                'risk_reward_ratio': 0,
                'best_trade_pct': 0,
                'worst_trade_pct': 0,
                'total_volume_sol': 0,
                'avg_position_size_sol': 0,
                'source': candidate.get('source', 'discovery_v8')
            }
            
            self.db.add_verified_wallet(wallet, stats)
        except Exception as e:
            print(f"      ⚠️ DB error: {e}")
    
    def _save_debug(self, stats: Dict, budget: APIBudgetTracker):
        """Save debug information"""
        try:
            debug = {
                'timestamp': datetime.now().isoformat(),
                'stats': stats,
                'budget': {
                    'total': budget.budget,
                    'used': budget.used,
                    'breakdown': dict(budget.calls)
                },
                'config': {
                    'min_win_rate': config.min_win_rate,
                    'min_pnl_sol': config.min_pnl_sol,
                    'min_completed_swings': config.min_completed_swings
                }
            }
            
            with open('discovery_debug.json', 'w') as f:
                json.dump(debug, f, indent=2, default=str)
                
        except Exception as e:
            print(f"   ⚠️ Debug save error: {e}")
    
    def _get_high_volume_tokens(self, limit: int = 10) -> List[Dict]:
        """Get high-volume tokens from DexScreener"""
        tokens = []
        
        try:
            # Use DexScreener search for high volume Solana tokens
            response = requests.get(
                "https://api.dexscreener.com/latest/dex/tokens/So11111111111111111111111111111111111111112",
                timeout=15
            )
            
            if response.status_code != 200:
                print(f"   ⚠️ DexScreener returned {response.status_code}")
                return tokens
            
            data = response.json()
            if not data:
                return tokens
            
            pairs = data.get('pairs', [])
            if not pairs:
                return tokens
            
            # Deduplicate by base token and sort by volume
            seen_tokens = set()
            
            for pair in pairs[:100]:
                if not pair:
                    continue
                
                base_token = pair.get('baseToken')
                if not base_token:
                    continue
                
                token_addr = base_token.get('address', '')
                
                if not token_addr or token_addr in seen_tokens:
                    continue
                
                # Skip SOL/WSOL
                if 'So11111111111111111111111111111111111111' in token_addr:
                    continue
                
                seen_tokens.add(token_addr)
                
                volume_data = pair.get('volume', {}) or {}
                liquidity_data = pair.get('liquidity', {}) or {}
                
                volume = float(volume_data.get('h24', 0) or 0)
                liquidity = float(liquidity_data.get('usd', 0) or 0)
                
                # Only include tokens with significant volume
                if volume >= 50000 and liquidity >= 10000:
                    tokens.append({
                        'address': token_addr,
                        'symbol': base_token.get('symbol', '?'),
                        'volume_24h': volume,
                        'liquidity': liquidity
                    })
            
            # Sort by volume
            tokens.sort(key=lambda x: x['volume_24h'], reverse=True)
            tokens = tokens[:limit]
            
            if tokens:
                print(f"   ✅ Found {len(tokens)} high-volume tokens")
        
        except Exception as e:
            print(f"   ⚠️ High volume tokens error: {e}")
        
        return tokens


# =============================================================================
# CLI INTERFACE
# =============================================================================

if __name__ == "__main__":
    from database_v2 import DatabaseV2
    
    print("\n" + "="*70)
    print("🚀 IMPROVED DISCOVERY SYSTEM v8")
    print("="*70)
    
    db = DatabaseV2()
    discovery = ImprovedDiscoveryV8(db)
    
    # Parse arguments
    budget = 5000
    max_wallets = 20
    
    if len(sys.argv) > 1:
        try:
            budget = int(sys.argv[1])
        except:
            pass
    
    if len(sys.argv) > 2:
        try:
            max_wallets = int(sys.argv[2])
        except:
            pass
    
    print(f"\nRunning with budget={budget}, max_wallets={max_wallets}")
    
    stats = discovery.run_discovery(api_budget=budget, max_wallets=max_wallets)
    
    print(f"\n✅ Discovery complete!")
    print(f"   Verified {stats['verified']} wallet(s)")
