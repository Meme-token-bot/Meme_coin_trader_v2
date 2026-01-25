"""
COMPLETE HYBRID DISCOVERY SYSTEM
Uses multiple FREE APIs to find profitable wallets efficiently

API Distribution:
- DexScreener (FREE): Token discovery
- Birdeye (FREE tier): Holder data + token metrics
- Helius (1M/month): Only for wallet profiling & webhooks

This approach uses <5% of Helius budget while maximizing discovery!
"""

import requests
import time
from datetime import datetime, timedelta
from typing import Dict, List, Set, Optional, Tuple
from collections import defaultdict, Counter
import os


class TokenDiscoveryEngine:
    """Discovers promising tokens using FREE APIs"""
    
    def __init__(self):
        self.dex_url = "https://api.dexscreener.com/latest/dex"
        self._last_call = 0
        self._call_delay = 1.0  # DexScreener rate limit
    
    def _rate_limit(self):
        elapsed = time.time() - self._last_call
        if elapsed < self._call_delay:
            time.sleep(self._call_delay - elapsed)
        self._last_call = time.time()
    
    def find_pumping_tokens(self, min_gain: float = 100, limit: int = 15) -> List[Dict]:
        """Find tokens with big 24h price increases"""
        self._rate_limit()
        try:
            res = requests.get(f"{self.dex_url}/tokens/trending", timeout=15)
            data = res.json()
            
            tokens = []
            for pair in data.get('pairs', [])[:100]:
                if pair.get('chainId') != 'solana':
                    continue
                
                price_change = float(pair.get('priceChange', {}).get('h24', 0) or 0)
                liquidity = float(pair.get('liquidity', {}).get('usd', 0) or 0)
                volume = float(pair.get('volume', {}).get('h24', 0) or 0)
                
                if price_change >= min_gain and liquidity >= 15000 and volume >= 15000:
                    tokens.append({
                        'address': pair.get('baseToken', {}).get('address'),
                        'symbol': pair.get('baseToken', {}).get('symbol', 'UNKNOWN'),
                        'name': pair.get('baseToken', {}).get('name', 'Unknown'),
                        'price_usd': float(pair.get('priceUsd', 0) or 0),
                        'price_change_24h': price_change,
                        'liquidity': liquidity,
                        'volume_24h': volume,
                        'fdv': float(pair.get('fdv', 0) or 0),
                        'created_at': pair.get('pairCreatedAt'),
                        'source': 'pumping'
                    })
            
            result = sorted(tokens, key=lambda x: x['price_change_24h'], reverse=True)[:limit]
            print(f"   📈 Found {len(result)} pumping tokens")
            return result
        
        except Exception as e:
            print(f"   ❌ Error finding pumping tokens: {e}")
            return []
    
    def find_trending_tokens(self, min_volume: float = 50000, limit: int = 15) -> List[Dict]:
        """Find high-volume trending tokens"""
        self._rate_limit()
        try:
            res = requests.get(f"{self.dex_url}/tokens/solana", timeout=15)
            data = res.json()
            
            tokens = []
            for pair in data.get('pairs', [])[:50]:
                volume = float(pair.get('volume', {}).get('h24', 0) or 0)
                liquidity = float(pair.get('liquidity', {}).get('usd', 0) or 0)
                
                if volume >= min_volume and liquidity >= 20000:
                    tokens.append({
                        'address': pair.get('baseToken', {}).get('address'),
                        'symbol': pair.get('baseToken', {}).get('symbol', 'UNKNOWN'),
                        'name': pair.get('baseToken', {}).get('name', 'Unknown'),
                        'price_usd': float(pair.get('priceUsd', 0) or 0),
                        'price_change_24h': float(pair.get('priceChange', {}).get('h24', 0) or 0),
                        'liquidity': liquidity,
                        'volume_24h': volume,
                        'fdv': float(pair.get('fdv', 0) or 0),
                        'created_at': pair.get('pairCreatedAt'),
                        'source': 'trending'
                    })
            
            result = sorted(tokens, key=lambda x: x['volume_24h'], reverse=True)[:limit]
            print(f"   🔥 Found {len(result)} trending tokens")
            return result
        
        except Exception as e:
            print(f"   ❌ Error finding trending tokens: {e}")
            return []
    
    def find_new_launches(self, max_age_hours: int = 6, limit: int = 10) -> List[Dict]:
        """Find newly launched tokens"""
        self._rate_limit()
        try:
            res = requests.get(f"{self.dex_url}/search", params={"q": "solana"}, timeout=15)
            data = res.json()
            
            cutoff = datetime.now() - timedelta(hours=max_age_hours)
            tokens = []
            
            for pair in data.get('pairs', [])[:100]:
                if pair.get('chainId') != 'solana':
                    continue
                
                created_at = pair.get('pairCreatedAt')
                if not created_at:
                    continue
                
                created_dt = datetime.fromtimestamp(created_at / 1000)
                liquidity = float(pair.get('liquidity', {}).get('usd', 0) or 0)
                
                if created_dt >= cutoff and liquidity >= 5000:
                    tokens.append({
                        'address': pair.get('baseToken', {}).get('address'),
                        'symbol': pair.get('baseToken', {}).get('symbol', 'UNKNOWN'),
                        'name': pair.get('baseToken', {}).get('name', 'Unknown'),
                        'price_usd': float(pair.get('priceUsd', 0) or 0),
                        'price_change_24h': float(pair.get('priceChange', {}).get('h24', 0) or 0),
                        'liquidity': liquidity,
                        'volume_24h': float(pair.get('volume', {}).get('h24', 0) or 0),
                        'fdv': float(pair.get('fdv', 0) or 0),
                        'created_at': created_at,
                        'age_hours': (datetime.now() - created_dt).total_seconds() / 3600,
                        'source': 'new_launch'
                    })
            
            result = sorted(tokens, key=lambda x: x['created_at'], reverse=True)[:limit]
            print(f"   🆕 Found {len(result)} new launches")
            return result
        
        except Exception as e:
            print(f"   ❌ Error finding new launches: {e}")
            return []


class WalletDiscoveryEngine:
    """Discovers wallet addresses using Helius with efficient method"""
    
    def __init__(self, helius_key: str, birdeye_key: Optional[str] = None):
        self.helius_key = helius_key
        self.helius_url = f"https://mainnet.helius-rpc.com/?api-key={helius_key}"
        self.birdeye_url = "https://public-api.birdeye.so"
        self.birdeye_key = birdeye_key or os.getenv('BIRDEYE_API_KEY')
        
        self._last_helius_call = 0
        self._helius_delay = 0.1
        self._last_birdeye_call = 0
        self._birdeye_delay = 0.3
    
    def get_top_holders(self, token_address: str, limit: int = 5) -> List[str]:
        """
        Get top holder addresses using the 1-credit Helius method.
        
        Cost: ~6 credits per token (1 for largest accounts + 1 per holder to resolve)
        """
        wallets = []
        
        # Rate limit
        elapsed = time.time() - self._last_helius_call
        if elapsed < self._helius_delay:
            time.sleep(self._helius_delay - elapsed)
        self._last_helius_call = time.time()
        
        try:
            # Step 1: Get top 20 token accounts (1 credit)
            payload_largest = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "getTokenLargestAccounts",
                "params": [token_address]
            }
            res = requests.post(self.helius_url, json=payload_largest, timeout=10)
            account_data = res.json().get('result', {}).get('value', [])
            
            # Step 2: Resolve owner for top accounts (1 credit per account)
            for entry in account_data[:limit]:
                token_account_pubkey = entry.get('address')
                
                # Rate limit
                elapsed = time.time() - self._last_helius_call
                if elapsed < self._helius_delay:
                    time.sleep(self._helius_delay - elapsed)
                self._last_helius_call = time.time()
                
                # Use getAccountInfo to find the owner (1 credit)
                payload_info = {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "getAccountInfo",
                    "params": [token_account_pubkey, {"encoding": "jsonParsed"}]
                }
                info_res = requests.post(self.helius_url, json=payload_info, timeout=10)
                
                # Extract the owner from the parsed data
                parsed_info = info_res.json().get('result', {}).get('value', {}).get('data', {}).get('parsed', {}).get('info', {})
                owner_wallet = parsed_info.get('owner')
                
                if owner_wallet:
                    wallets.append(owner_wallet)
            
            return wallets
        
        except Exception as e:
            print(f"      Helius holder error: {e}")
            return []
    
    def get_holders_birdeye(self, token_address: str) -> List[str]:
        """Get holders from Birdeye (FREE tier - backup method)"""
        if not self.birdeye_key:
            return []
        
        elapsed = time.time() - self._last_birdeye_call
        if elapsed < self._birdeye_delay:
            time.sleep(self._birdeye_delay - elapsed)
        self._last_birdeye_call = time.time()
        
        try:
            url = f"{self.birdeye_url}/v1/token/holder"
            headers = {"X-API-KEY": self.birdeye_key}
            params = {"address": token_address}
            
            res = requests.get(url, params=params, headers=headers, timeout=10)
            data = res.json()
            
            if not data.get('success'):
                return []
            
            holders = []
            for holder in data.get('data', {}).get('items', [])[:15]:
                address = holder.get('address')
                if address:
                    holders.append(address)
            
            return holders
        
        except Exception as e:
            print(f"      Birdeye error: {e}")
            return []


class HybridDiscoverySystem:
    """
    Complete discovery system using all available APIs efficiently.
    """
    
    def __init__(self, db, scanner, profiler, birdeye_key: Optional[str] = None):
        self.db = db
        self.scanner = scanner
        self.profiler = profiler
        
        # Discovery engines
        self.token_discovery = TokenDiscoveryEngine()
        self.wallet_discovery = WalletDiscoveryEngine(
            helius_key=scanner.helius_key,
            birdeye_key=birdeye_key
        )
        
        # Tracking
        self.session_api_calls = 0
        
        print("  ✅ Hybrid Discovery System initialized")
        if birdeye_key:
            print("     Birdeye API: enabled")
        else:
            print("     Birdeye API: disabled (set BIRDEYE_API_KEY to enable)")
    
    def run_discovery(self, api_budget: int = 300, max_wallets: int = 15) -> Dict:
        """
        Main discovery pipeline.
        
        Pipeline:
        1. Find promising tokens (DexScreener - FREE)
        2. Get holder addresses (Helius 1-credit method - EFFICIENT)
        3. Score candidates (local processing - FREE)
        4. Profile top candidates (Helius - COSTS CREDITS)
        5. Verify and add to database
        """
        self.session_api_calls = 0
        cycle_start_time = datetime.now()
        
        stats = {
            'tokens_discovered': 0,
            'wallet_candidates_found': 0,
            'wallets_profiled': 0,
            'wallets_verified': 0,
            'wallets_rejected': 0,
            'helius_api_calls': 0,
            'token_sources': {}
        }
        
        print(f"\n{'='*70}")
        print(f"🎯 HYBRID DISCOVERY - {datetime.now().strftime('%H:%M:%S')}")
        print(f"   Helius Budget: {api_budget} calls | Target: {max_wallets} wallets")
        print(f"{'='*70}")
        
        # STEP 1: Discover tokens (ALL FREE)
        print(f"\n{'='*70}")
        print("STEP 1: TOKEN DISCOVERY (FREE APIs)")
        print(f"{'='*70}")
        
        all_tokens = []
        
        # Find pumping tokens
        pumping = self.token_discovery.find_pumping_tokens(min_gain=100, limit=10)
        all_tokens.extend(pumping)
        stats['token_sources']['pumping'] = len(pumping)
        
        # Find trending tokens
        trending = self.token_discovery.find_trending_tokens(min_volume=50000, limit=10)
        all_tokens.extend(trending)
        stats['token_sources']['trending'] = len(trending)
        
        # Remove duplicates
        unique_tokens = {t['address']: t for t in all_tokens}.values()
        all_tokens = list(unique_tokens)
        
        stats['tokens_discovered'] = len(all_tokens)
        print(f"\n   ✅ Total unique tokens: {len(all_tokens)}")
        
        # STEP 2: Extract wallet candidates (USES HELIUS - EFFICIENT METHOD)
        print(f"\n{'='*70}")
        print("STEP 2: WALLET EXTRACTION (Helius 1-credit method)")
        print(f"{'='*70}")
        
        wallet_candidates = {}  # {wallet_address: metadata}
        
        for token in all_tokens[:8]:  # Limit to 8 tokens to stay in budget
            token_addr = token['address']
            symbol = token['symbol']
            
            print(f"\n   Extracting holders for ${symbol}...")
            
            # Get holders using 1-credit method (~6 credits per token)
            holders = self.wallet_discovery.get_top_holders(token_addr, limit=5)
            self.session_api_calls += 6  # Estimate: 1 + 5
            
            print(f"      Found {len(holders)} holder(s) (~6 credits)")
            
            # Track which tokens each wallet appears in
            for wallet in holders:
                if self.db.is_wallet_tracked(wallet):
                    continue
                
                if wallet not in wallet_candidates:
                    wallet_candidates[wallet] = {
                        'seen_in_tokens': [],
                        'seen_in_symbols': [],
                        'sources': set(),
                        'score': 0
                    }
                
                wallet_candidates[wallet]['seen_in_tokens'].append(token_addr)
                wallet_candidates[wallet]['seen_in_symbols'].append(symbol)
                wallet_candidates[wallet]['sources'].add(token['source'])
        
        stats['wallet_candidates_found'] = len(wallet_candidates)
        print(f"\n   ✅ Found {len(wallet_candidates)} unique wallet candidates")
        print(f"   ⚡ Helius credits used so far: {self.session_api_calls}")
        
        # STEP 3: Score candidates (FREE - local processing)
        print(f"\n{'='*70}")
        print("STEP 3: CANDIDATE SCORING")
        print(f"{'='*70}")
        
        scored_candidates = self._score_candidates(wallet_candidates)
        
        # Take top candidates
        top_candidates = sorted(
            scored_candidates.items(),
            key=lambda x: x[1]['score'],
            reverse=True
        )[:max_wallets * 2]  # Get 2x candidates to profile
        
        print(f"   ✅ Selected top {len(top_candidates)} candidates for profiling")
        
        # STEP 4: Profile candidates (USES HELIUS - COSTS CREDITS)
        print(f"\n{'='*70}")
        print(f"STEP 4: WALLET PROFILING (Helius API - Budget: {api_budget})")
        print(f"{'='*70}")
        
        verified_count = 0
        
        for wallet, metadata in top_candidates:
            if verified_count >= max_wallets:
                print(f"\n   ✅ Reached target of {max_wallets} wallets")
                break
            
            # Check budget
            estimated_cost = 25
            if self.session_api_calls + estimated_cost > api_budget:
                print(f"\n   ⚠️ Budget limit reached ({self.session_api_calls}/{api_budget})")
                break
            
            print(f"\n   🔍 {wallet[:8]}... (score: {metadata['score']:.0f})")
            print(f"      Seen in: {', '.join(metadata['seen_in_symbols'][:3])}")
            
            try:
                # Profile wallet (USES HELIUS API)
                calls_before = self.session_api_calls
                performance = self._profile_wallet(wallet)
                calls_used = self.session_api_calls - calls_before
                
                stats['wallets_profiled'] += 1
                
                # Verify
                if self._verify_wallet(performance):
                    self.db.add_verified_wallet(wallet, performance)
                    stats['wallets_verified'] += 1
                    verified_count += 1
                    
                    print(f"      ✅ VERIFIED!")
                    print(f"         WR: {performance['win_rate']:.1%} | PnL: {performance['pnl']:.2f} SOL")
                    print(f"         Swings: {performance['completed_swings']} | API: {calls_used}")
                else:
                    stats['wallets_rejected'] += 1
                    print(f"      ❌ Rejected: WR={performance['win_rate']:.1%}, PnL={performance['pnl']:.2f}")
            
            except Exception as e:
                print(f"      ❌ Error: {e}")
                stats['wallets_rejected'] += 1
        
        stats['helius_api_calls'] = self.session_api_calls
        
        # SUMMARY
        print(f"\n{'='*70}")
        print("✅ DISCOVERY COMPLETE")
        print(f"{'='*70}")
        print(f"   Tokens discovered: {stats['tokens_discovered']}")
        print(f"      Pumping: {stats['token_sources'].get('pumping', 0)}")
        print(f"      Trending: {stats['token_sources'].get('trending', 0)}")
        print(f"   Wallet candidates: {stats['wallet_candidates_found']}")
        print(f"   Wallets profiled: {stats['wallets_profiled']}")
        print(f"   Wallets VERIFIED: {stats['wallets_verified']} ✅")
        print(f"   Wallets rejected: {stats['wallets_rejected']}")
        print(f"   Helius API calls: {stats['helius_api_calls']}/{api_budget}")
        
        if stats['wallets_profiled'] > 0:
            efficiency = 100 * stats['wallets_verified'] / stats['wallets_profiled']
            print(f"   Verification rate: {efficiency:.0f}%")
        
        # Monthly projection
        daily_credits = stats['helius_api_calls']
        monthly_projection = daily_credits * 30
        print(f"\n   📊 Monthly projection: {monthly_projection:,.0f} credits ({monthly_projection/1000000*100:.1f}% of 1M)")
        
        print(f"{'='*70}\n")
        
        return stats
    
    def _score_candidates(self, candidates: Dict) -> Dict:
        """Score wallet candidates"""
        scored = {}
        
        for wallet, metadata in candidates.items():
            score = 0
            
            # Bonus for appearing in multiple tokens (indicates skill)
            num_tokens = len(metadata['seen_in_tokens'])
            if num_tokens >= 3:
                score += 50
            elif num_tokens >= 2:
                score += 30
            else:
                score += 10
            
            # Bonus for pumping tokens
            if 'pumping' in metadata['sources']:
                score += 30
            
            # Bonus for being in both pumping AND trending
            if len(metadata['sources']) >= 2:
                score += 20
            
            metadata['score'] = score
            scored[wallet] = metadata
        
        return scored
    
    def _profile_wallet(self, wallet: str) -> Dict:
        """Profile wallet performance (USES HELIUS)"""
        # Track API usage
        calls_before = self.session_api_calls
        
        # Get signatures (1 call)
        self.session_api_calls += 1
        signatures = self.scanner.get_recent_signatures(wallet, limit=100)
        
        if not signatures:
            return self._empty_performance()
        
        # Parse trades (estimate)
        estimated_parsing_calls = max(1, len(signatures) // 5)
        self.session_api_calls += estimated_parsing_calls
        
        # Calculate performance
        performance = self.profiler.calculate_performance(wallet, days=7)
        
        return performance
    
    def _verify_wallet(self, performance: Dict) -> bool:
        """Check if wallet meets standards"""
        return (
            performance['win_rate'] >= 0.55 and
            performance['pnl'] >= 3.0 and
            performance['completed_swings'] >= 5
        )
    
    def _empty_performance(self) -> Dict:
        return {
            'win_rate': 0, 'pnl': 0, 'roi_7d': 0, 'completed_swings': 0,
            'avg_hold_hours': 0, 'risk_reward_ratio': 0, 'best_trade_pct': 0,
            'worst_trade_pct': 0, 'total_volume_sol': 0, 'avg_position_size_sol': 0
        }
