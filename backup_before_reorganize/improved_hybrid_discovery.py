"""
IMPROVED HYBRID DISCOVERY SYSTEM
Fixes for low wallet discovery rates

Changes from original:
1. Lowered verification thresholds (configurable)
2. Scans more tokens per cycle
3. Better logging to understand what's happening
4. Respects discovery_config.py thresholds
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
        self._call_delay = 1.0
    
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
                
                # LOWERED thresholds for better coverage
                if price_change >= min_gain and liquidity >= 10000 and volume >= 10000:
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
            print(f"   📈 Found {len(result)} pumping tokens (min +{min_gain}%)")
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
                
                if volume >= min_volume and liquidity >= 15000:
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
            print(f"   🔥 Found {len(result)} trending tokens (min vol ${min_volume:,.0f})")
            return result
        
        except Exception as e:
            print(f"   ❌ Error finding trending tokens: {e}")
            return []
    
    def find_new_launches(self, max_age_hours: int = 12, limit: int = 10) -> List[Dict]:
        """Find newly launched tokens - EXPANDED time window"""
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
            print(f"   🆕 Found {len(result)} new launches (<{max_age_hours}h old)")
            return result
        
        except Exception as e:
            print(f"   ❌ Error finding new launches: {e}")
            return []


class WalletDiscoveryEngine:
    """Discovers wallet addresses using Helius"""
    
    def __init__(self, helius_key: str, birdeye_key: Optional[str] = None):
        self.helius_key = helius_key
        self.helius_url = f"https://mainnet.helius-rpc.com/?api-key={helius_key}"
        self.birdeye_url = "https://public-api.birdeye.so"
        self.birdeye_key = birdeye_key or os.getenv('BIRDEYE_API_KEY')
        
        self._last_helius_call = 0
        self._helius_delay = 0.1
        self._last_birdeye_call = 0
        self._birdeye_delay = 0.3
    
    def get_top_holders(self, token_address: str, limit: int = 10) -> List[str]:
        """Get top holder addresses - INCREASED limit"""
        wallets = []
        
        elapsed = time.time() - self._last_helius_call
        if elapsed < self._helius_delay:
            time.sleep(self._helius_delay - elapsed)
        self._last_helius_call = time.time()
        
        try:
            payload_largest = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "getTokenLargestAccounts",
                "params": [token_address]
            }
            res = requests.post(self.helius_url, json=payload_largest, timeout=10)
            account_data = res.json().get('result', {}).get('value', [])
            
            for entry in account_data[:limit]:
                token_account_pubkey = entry.get('address')
                
                elapsed = time.time() - self._last_helius_call
                if elapsed < self._helius_delay:
                    time.sleep(self._helius_delay - elapsed)
                self._last_helius_call = time.time()
                
                payload_info = {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "getAccountInfo",
                    "params": [token_account_pubkey, {"encoding": "jsonParsed"}]
                }
                info_res = requests.post(self.helius_url, json=payload_info, timeout=10)
                
                parsed_info = info_res.json().get('result', {}).get('value', {}).get('data', {}).get('parsed', {}).get('info', {})
                owner_wallet = parsed_info.get('owner')
                
                if owner_wallet:
                    wallets.append(owner_wallet)
            
            return wallets
        
        except Exception as e:
            print(f"      Helius holder error: {e}")
            return []


class ImprovedHybridDiscoverySystem:
    """
    Improved discovery system with:
    - Configurable thresholds from discovery_config.py
    - More tokens scanned
    - Better logging
    - Detailed stats
    """
    
    def __init__(self, db, scanner, profiler, birdeye_key: Optional[str] = None):
        self.db = db
        self.scanner = scanner
        self.profiler = profiler
        
        # Import config
        try:
            from discovery_config import config as discovery_config
            self.config = discovery_config
        except:
            self.config = None
        
        self.token_discovery = TokenDiscoveryEngine()
        self.wallet_discovery = WalletDiscoveryEngine(
            helius_key=scanner.helius_key,
            birdeye_key=birdeye_key
        )
        
        self.session_api_calls = 0
        
        # Discovery stats for logging
        self.discovery_log = []
        
        print("  ✅ Improved Hybrid Discovery System initialized")
    
    def run_discovery(self, api_budget: int = 500, max_wallets: int = 10) -> Dict:
        """Run discovery with improved settings"""
        
        self.session_api_calls = 0
        self.discovery_log = []
        cycle_start_time = datetime.now()
        
        # Get thresholds from config
        if self.config:
            min_win_rate = self.config.min_win_rate  # 0.50
            min_pnl = self.config.min_pnl            # 2.0
            min_swings = self.config.min_completed_swings  # 3
        else:
            # Fallback to reasonable defaults
            min_win_rate = 0.50
            min_pnl = 2.0
            min_swings = 3
        
        stats = {
            'tokens_discovered': 0,
            'wallet_candidates_found': 0,
            'wallets_profiled': 0,
            'wallets_verified': 0,
            'wallets_rejected': 0,
            'rejection_reasons': defaultdict(int),
            'helius_api_calls': 0,
            'token_sources': {},
            'verified_wallets': []
        }
        
        print(f"\n{'='*70}")
        print(f"🎯 IMPROVED DISCOVERY - {datetime.now().strftime('%H:%M:%S')}")
        print(f"   Helius Budget: {api_budget} calls | Target: {max_wallets} wallets")
        print(f"   Thresholds: WR≥{min_win_rate:.0%} | PnL≥{min_pnl} SOL | Swings≥{min_swings}")
        print(f"{'='*70}")
        
        # STEP 1: Discover tokens (ALL FREE)
        print(f"\n{'='*70}")
        print("STEP 1: TOKEN DISCOVERY (FREE APIs)")
        print(f"{'='*70}")
        
        all_tokens = []
        
        # Find MORE tokens with LOWER thresholds
        pumping = self.token_discovery.find_pumping_tokens(min_gain=50, limit=15)  # Lower from 100 to 50
        all_tokens.extend(pumping)
        stats['token_sources']['pumping'] = len(pumping)
        
        trending = self.token_discovery.find_trending_tokens(min_volume=30000, limit=15)  # Lower from 50k
        all_tokens.extend(trending)
        stats['token_sources']['trending'] = len(trending)
        
        new_launches = self.token_discovery.find_new_launches(max_age_hours=12, limit=10)
        all_tokens.extend(new_launches)
        stats['token_sources']['new_launches'] = len(new_launches)
        
        # Remove duplicates
        unique_tokens = {t['address']: t for t in all_tokens if t.get('address')}.values()
        all_tokens = list(unique_tokens)
        
        stats['tokens_discovered'] = len(all_tokens)
        print(f"\n   ✅ Total unique tokens: {len(all_tokens)}")
        
        if not all_tokens:
            print("   ⚠️  No tokens found! Check internet connection.")
            return stats
        
        # STEP 2: Extract wallet candidates (USES HELIUS)
        print(f"\n{'='*70}")
        print("STEP 2: WALLET EXTRACTION (Helius - ~11 credits per token)")
        print(f"{'='*70}")
        
        wallet_candidates = {}
        tokens_to_scan = min(15, len(all_tokens))  # INCREASED from 8
        
        for token in all_tokens[:tokens_to_scan]:
            token_addr = token['address']
            symbol = token['symbol']
            
            # Check budget
            if self.session_api_calls + 11 > api_budget * 0.4:  # Reserve 60% for profiling
                print(f"\n   ⚠️  Stopping extraction to save budget for profiling")
                break
            
            print(f"\n   Extracting holders for ${symbol}...")
            
            # Get MORE holders per token
            holders = self.wallet_discovery.get_top_holders(token_addr, limit=10)  # INCREASED from 5
            self.session_api_calls += 1 + len(holders)
            
            print(f"      Found {len(holders)} holder(s)")
            
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
        
        if not wallet_candidates:
            print("   ⚠️  No new wallet candidates found!")
            return stats
        
        # STEP 3: Score candidates (FREE)
        print(f"\n{'='*70}")
        print("STEP 3: CANDIDATE SCORING")
        print(f"{'='*70}")
        
        scored_candidates = self._score_candidates(wallet_candidates)
        
        # Take MORE candidates to profile
        top_candidates = sorted(
            scored_candidates.items(),
            key=lambda x: x[1]['score'],
            reverse=True
        )[:max_wallets * 3]  # Profile 3x target
        
        print(f"   ✅ Selected top {len(top_candidates)} candidates for profiling")
        
        # STEP 4: Profile candidates (USES HELIUS)
        print(f"\n{'='*70}")
        print(f"STEP 4: WALLET PROFILING")
        print(f"   Budget remaining: {api_budget - self.session_api_calls}")
        print(f"{'='*70}")
        
        verified_count = 0
        
        for wallet, metadata in top_candidates:
            if verified_count >= max_wallets:
                print(f"\n   ✅ Reached target of {max_wallets} wallets")
                break
            
            # Check budget
            estimated_cost = 25
            if self.session_api_calls + estimated_cost > api_budget:
                print(f"\n   ⚠️  Budget limit reached ({self.session_api_calls}/{api_budget})")
                break
            
            print(f"\n   🔍 {wallet[:12]}... (score: {metadata['score']:.0f})")
            print(f"      Seen in: {', '.join(metadata['seen_in_symbols'][:3])}")
            
            try:
                calls_before = self.session_api_calls
                performance = self._profile_wallet(wallet)
                calls_used = self.session_api_calls - calls_before
                
                stats['wallets_profiled'] += 1
                
                # Log detailed info
                self.discovery_log.append({
                    'wallet': wallet,
                    'win_rate': performance['win_rate'],
                    'pnl': performance['pnl'],
                    'swings': performance['completed_swings'],
                    'verified': False,
                    'reason': None
                })
                
                # Check against thresholds
                wr_ok = performance['win_rate'] >= min_win_rate
                pnl_ok = performance['pnl'] >= min_pnl
                swings_ok = performance['completed_swings'] >= min_swings
                
                print(f"      WR: {performance['win_rate']:.1%} {'✅' if wr_ok else '❌'}")
                print(f"      PnL: {performance['pnl']:.2f} SOL {'✅' if pnl_ok else '❌'}")
                print(f"      Swings: {performance['completed_swings']} {'✅' if swings_ok else '❌'}")
                
                if wr_ok and pnl_ok and swings_ok:
                    self.db.add_verified_wallet(wallet, performance)
                    stats['wallets_verified'] += 1
                    stats['verified_wallets'].append({
                        'address': wallet,
                        'win_rate': performance['win_rate'],
                        'pnl': performance['pnl']
                    })
                    verified_count += 1
                    
                    self.discovery_log[-1]['verified'] = True
                    
                    print(f"      ✅ VERIFIED! (API: {calls_used})")
                else:
                    stats['wallets_rejected'] += 1
                    
                    # Track rejection reasons
                    if not wr_ok:
                        stats['rejection_reasons']['low_win_rate'] += 1
                        self.discovery_log[-1]['reason'] = 'low_win_rate'
                    elif not pnl_ok:
                        stats['rejection_reasons']['low_pnl'] += 1
                        self.discovery_log[-1]['reason'] = 'low_pnl'
                    elif not swings_ok:
                        stats['rejection_reasons']['low_swings'] += 1
                        self.discovery_log[-1]['reason'] = 'low_swings'
            
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
        print(f"      New launches: {stats['token_sources'].get('new_launches', 0)}")
        print(f"   Wallet candidates: {stats['wallet_candidates_found']}")
        print(f"   Wallets profiled: {stats['wallets_profiled']}")
        print(f"   Wallets VERIFIED: {stats['wallets_verified']} ✅")
        print(f"   Wallets rejected: {stats['wallets_rejected']}")
        
        if stats['rejection_reasons']:
            print(f"\n   Rejection breakdown:")
            for reason, count in stats['rejection_reasons'].items():
                print(f"      {reason}: {count}")
        
        print(f"\n   Helius API calls: {stats['helius_api_calls']}/{api_budget}")
        
        if stats['wallets_profiled'] > 0:
            efficiency = 100 * stats['wallets_verified'] / stats['wallets_profiled']
            print(f"   Verification rate: {efficiency:.0f}%")
        
        if stats['verified_wallets']:
            print(f"\n   📋 Verified wallets:")
            for w in stats['verified_wallets']:
                print(f"      {w['address'][:12]}... | WR: {w['win_rate']:.1%} | PnL: {w['pnl']:.2f}")
        
        print(f"{'='*70}\n")
        
        return stats
    
    def _score_candidates(self, candidates: Dict) -> Dict:
        """Score wallet candidates"""
        scored = {}
        
        for wallet, metadata in candidates.items():
            score = 0
            
            num_tokens = len(metadata['seen_in_tokens'])
            if num_tokens >= 3:
                score += 50
            elif num_tokens >= 2:
                score += 30
            else:
                score += 10
            
            if 'pumping' in metadata['sources']:
                score += 30
            
            if 'new_launch' in metadata['sources']:
                score += 20
            
            if len(metadata['sources']) >= 2:
                score += 20
            
            metadata['score'] = score
            scored[wallet] = metadata
        
        return scored
    
    def _profile_wallet(self, wallet: str) -> Dict:
        """Profile wallet performance"""
        self.session_api_calls += 1
        signatures = self.scanner.get_recent_signatures(wallet, limit=100)
        
        if not signatures:
            return self._empty_performance()
        
        estimated_parsing_calls = max(1, len(signatures) // 5)
        self.session_api_calls += estimated_parsing_calls
        
        performance = self.profiler.calculate_performance(wallet, days=7)
        
        return performance
    
    def _empty_performance(self) -> Dict:
        return {
            'win_rate': 0, 'pnl': 0, 'roi_7d': 0, 'completed_swings': 0,
            'avg_hold_hours': 0, 'risk_reward_ratio': 0, 'best_trade_pct': 0,
            'worst_trade_pct': 0, 'total_volume_sol': 0, 'avg_position_size_sol': 0
        }
    
    def get_discovery_log(self) -> List[Dict]:
        """Get detailed log of last discovery run"""
        return self.discovery_log


# For drop-in replacement
HybridDiscoverySystem = ImprovedHybridDiscoverySystem
