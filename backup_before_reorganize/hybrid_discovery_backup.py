"""
FIXED HYBRID DISCOVERY SYSTEM v5
Only captures FEE PAYERS (the actual traders)

Problem with v4:
- Was capturing ANY wallet in swap transactions
- Many of these are recipients, intermediaries, or protocol addresses
- They don't have trading history because they didn't initiate trades

Solution:
- Only capture the FEE PAYER from each swap
- The fee payer is the wallet that initiated and paid for the transaction
- These are the actual traders

INSTALLATION:
  cp fixed_hybrid_discovery_v5.py ~/Documents/Meme_coin_trader_V2/hybrid_discovery.py
"""

import requests
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from collections import defaultdict
import os


# =============================================================================
# TOKEN DISCOVERY ENGINE (FREE - DexScreener)
# =============================================================================

class TokenDiscoveryEngine:
    """Token discovery using DexScreener search API"""
    
    def __init__(self):
        self.base_url = "https://api.dexscreener.com"
        self._last_call = 0
        self._call_delay = 0.5
    
    def _rate_limit(self):
        elapsed = time.time() - self._last_call
        if elapsed < self._call_delay:
            time.sleep(self._call_delay - elapsed)
        self._last_call = time.time()
    
    def _safe_request(self, url: str) -> Optional[Dict]:
        self._rate_limit()
        try:
            res = requests.get(url, timeout=15)
            if res.status_code != 200:
                return None
            return res.json()
        except:
            return None
    
    def _extract_solana_pairs(self, data: Optional[Dict]) -> List[Dict]:
        if not data:
            return []
        pairs = data.get('pairs')
        if not pairs or not isinstance(pairs, list):
            return []
        return [p for p in pairs if isinstance(p, dict) and str(p.get('chainId', '')).lower() == 'solana']
    
    def _pair_to_token(self, pair: Dict, source: str) -> Optional[Dict]:
        try:
            base = pair.get('baseToken', {})
            address = base.get('address')
            if not address:
                return None
            
            def safe_num(obj, *keys):
                for key in keys:
                    obj = obj.get(key, 0) if isinstance(obj, dict) else 0
                try:
                    return float(obj or 0)
                except:
                    return 0
            
            return {
                'address': address,
                'symbol': base.get('symbol', 'UNKNOWN'),
                'name': base.get('name', 'Unknown'),
                'price_usd': safe_num(pair, 'priceUsd'),
                'price_change_24h': safe_num(pair, 'priceChange', 'h24'),
                'liquidity': safe_num(pair, 'liquidity', 'usd'),
                'volume_24h': safe_num(pair, 'volume', 'h24'),
                'fdv': safe_num(pair, 'fdv'),
                'created_at': pair.get('pairCreatedAt'),
                'pair_address': pair.get('pairAddress'),
                'source': source
            }
        except:
            return None
    
    def search_tokens(self, query: str, source: str = None) -> List[Dict]:
        url = f"{self.base_url}/latest/dex/search?q={query}"
        data = self._safe_request(url)
        pairs = self._extract_solana_pairs(data)
        
        tokens = []
        for pair in pairs:
            token = self._pair_to_token(pair, source or f'search_{query}')
            if token:
                tokens.append(token)
        return tokens
    
    def find_pumping_tokens(self, min_gain: float = 30, limit: int = 20) -> List[Dict]:
        print(f"   📈 Finding pumping tokens (>{min_gain:.0f}% gain)...")
        
        all_tokens = []
        for query in ['pump', 'moon', 'sol', 'meme', 'pepe', 'doge', 'ai', 'cat', 'dog']:
            tokens = self.search_tokens(query, 'pumping')
            all_tokens.extend(tokens)
        
        seen = set()
        unique = [t for t in all_tokens if t['address'] not in seen and not seen.add(t['address'])]
        
        pumping = [t for t in unique if t['price_change_24h'] >= min_gain and t['liquidity'] >= 10000 and t['volume_24h'] >= 10000]
        pumping.sort(key=lambda x: x['price_change_24h'], reverse=True)
        
        result = pumping[:limit]
        print(f"      Found {len(result)} pumping tokens")
        return result
    
    def find_high_volume_tokens(self, min_volume: float = 50000, limit: int = 20) -> List[Dict]:
        print(f"   🔥 Finding high-volume tokens (>${min_volume:,.0f})...")
        
        all_tokens = []
        for query in ['sol', 'solana', 'jup', 'bonk', 'wif', 'popcat', 'wen']:
            tokens = self.search_tokens(query, 'volume')
            all_tokens.extend(tokens)
        
        seen = set()
        unique = [t for t in all_tokens if t['address'] not in seen and not seen.add(t['address'])]
        
        high_vol = [t for t in unique if t['volume_24h'] >= min_volume and t['liquidity'] >= 20000]
        high_vol.sort(key=lambda x: x['volume_24h'], reverse=True)
        
        result = high_vol[:limit]
        print(f"      Found {len(result)} high-volume tokens")
        return result
    
    def get_all_discovery_tokens(self) -> List[Dict]:
        all_tokens = []
        
        try:
            pumping = self.find_pumping_tokens(min_gain=20, limit=15)
            all_tokens.extend(pumping)
        except Exception as e:
            print(f"   ⚠️  Pumping error: {e}")
        
        try:
            high_vol = self.find_high_volume_tokens(min_volume=30000, limit=15)
            all_tokens.extend(high_vol)
        except Exception as e:
            print(f"   ⚠️  Volume error: {e}")
        
        seen = set()
        unique = [t for t in all_tokens if t['address'] not in seen and not seen.add(t['address'])]
        
        return unique


# =============================================================================
# ACTIVE TRADER DISCOVERY v2 (Only Fee Payers)
# =============================================================================

class ActiveTraderDiscovery:
    """
    Find ACTIVE TRADERS by capturing only FEE PAYERS from swap transactions.
    
    The fee payer is the wallet that initiated the transaction - 
    this is the actual trader, not an intermediary or recipient.
    """
    
    def __init__(self, helius_key: str):
        self.helius_key = helius_key
        self.helius_rpc = f"https://mainnet.helius-rpc.com/?api-key={helius_key}"
        self.helius_api = f"https://api.helius.xyz/v0"
        self._last_call = 0
        self._delay = 0.12
        
        # Known program/protocol addresses to exclude
        self.excluded_addresses = {
            # Common AMM/DEX programs
            '675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8',  # Raydium
            'JUP6LkbZbjS1jKKwapdHNy74zcZ3tLUZoi5QNyVTaV4',  # Jupiter
            'whirLbMiicVdio4qvUfM5KAg6Ct8VwpYzGff3uctyCc',  # Orca
            '9W959DqEETiGZocYWCQPaJ6sBmUzgfxXfqGeTEdp3aQP',  # Orca v2
        }
    
    def _rate_limit(self):
        elapsed = time.time() - self._last_call
        if elapsed < self._delay:
            time.sleep(self._delay - elapsed)
        self._last_call = time.time()
    
    def _is_likely_bot_or_protocol(self, address: str) -> bool:
        """Check if address is likely a bot or protocol"""
        if address in self.excluded_addresses:
            return True
        # Exclude very short addresses (likely system accounts)
        if len(address) < 32:
            return True
        return False
    
    def get_swap_fee_payers(self, token_address: str, limit: int = 30) -> List[Dict]:
        """
        Get fee payers (actual traders) from recent swap transactions.
        
        Returns list of dicts with wallet address and swap details.
        """
        traders = {}  # address -> trade info
        
        try:
            # Step 1: Get recent signatures for the token
            self._rate_limit()
            
            payload = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "getSignaturesForAddress",
                "params": [token_address, {"limit": 100}]
            }
            
            res = requests.post(self.helius_rpc, json=payload, timeout=15)
            signatures = res.json().get('result', [])
            
            if not signatures:
                return []
            
            # Step 2: Parse transactions with Helius
            sig_list = [s.get('signature') for s in signatures[:50] if s.get('signature')]
            
            if not sig_list:
                return []
            
            self._rate_limit()
            parse_url = f"{self.helius_api}/transactions?api-key={self.helius_key}"
            parse_res = requests.post(parse_url, json={"transactions": sig_list}, timeout=15)
            parsed_txs = parse_res.json()
            
            if not isinstance(parsed_txs, list):
                return []
            
            # Step 3: Extract ONLY fee payers from SWAP transactions
            for tx in parsed_txs:
                if not isinstance(tx, dict):
                    continue
                
                # Only interested in swaps
                if tx.get('type') != 'SWAP':
                    continue
                
                fee_payer = tx.get('feePayer')
                
                if not fee_payer:
                    continue
                
                # Skip known protocols/bots
                if self._is_likely_bot_or_protocol(fee_payer):
                    continue
                
                # Analyze the swap to understand direction
                swap_info = self._analyze_swap(tx, fee_payer)
                
                if swap_info and swap_info.get('sol_amount', 0) >= 0.01:  # Minimum 0.01 SOL trade
                    if fee_payer not in traders:
                        traders[fee_payer] = {
                            'address': fee_payer,
                            'swaps': [],
                            'total_sol_volume': 0,
                            'buy_count': 0,
                            'sell_count': 0
                        }
                    
                    traders[fee_payer]['swaps'].append(swap_info)
                    traders[fee_payer]['total_sol_volume'] += swap_info.get('sol_amount', 0)
                    
                    if swap_info.get('type') == 'BUY':
                        traders[fee_payer]['buy_count'] += 1
                    elif swap_info.get('type') == 'SELL':
                        traders[fee_payer]['sell_count'] += 1
                
                if len(traders) >= limit:
                    break
            
            return list(traders.values())
            
        except Exception as e:
            print(f"      ⚠️  Error: {e}")
            return []
    
    def _analyze_swap(self, tx: Dict, fee_payer: str) -> Optional[Dict]:
        """Analyze a swap transaction to determine type and amounts"""
        
        token_transfers = tx.get('tokenTransfers', [])
        native_transfers = tx.get('nativeTransfers', [])
        
        WSOL = "So11111111111111111111111111111111111111112"
        USDC = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"
        USDT = "Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB"
        
        stables = {WSOL, USDC, USDT}
        
        sol_in, sol_out = 0, 0
        tokens_in, tokens_out = {}, {}
        
        for transfer in token_transfers:
            mint = transfer.get('mint', '')
            amount = float(transfer.get('tokenAmount', 0) or 0)
            from_addr = transfer.get('fromUserAccount', '')
            to_addr = transfer.get('toUserAccount', '')
            
            if from_addr == fee_payer:
                if mint in stables:
                    sol_out += amount
                else:
                    tokens_out[mint] = tokens_out.get(mint, 0) + amount
            elif to_addr == fee_payer:
                if mint in stables:
                    sol_in += amount
                else:
                    tokens_in[mint] = tokens_in.get(mint, 0) + amount
        
        for transfer in native_transfers:
            amount = float(transfer.get('amount', 0) or 0) / 1e9
            from_addr = transfer.get('fromUserAccount', '')
            to_addr = transfer.get('toUserAccount', '')
            
            if from_addr == fee_payer:
                sol_out += amount
            elif to_addr == fee_payer:
                sol_in += amount
        
        # Determine trade type
        if len(tokens_in) >= 1 and sol_out > 0:
            # Spent SOL/stables, got tokens = BUY
            token = list(tokens_in.keys())[0]
            return {
                'type': 'BUY',
                'token': token,
                'token_amount': tokens_in[token],
                'sol_amount': sol_out
            }
        elif len(tokens_out) >= 1 and sol_in > 0:
            # Spent tokens, got SOL/stables = SELL
            token = list(tokens_out.keys())[0]
            return {
                'type': 'SELL',
                'token': token,
                'token_amount': tokens_out[token],
                'sol_amount': sol_in
            }
        
        return None


# =============================================================================
# HYBRID DISCOVERY SYSTEM v5
# =============================================================================

class HybridDiscoverySystem:
    """
    Wallet discovery that finds ACTUAL TRADERS (fee payers only).
    """
    
    def __init__(self, db, scanner, profiler, birdeye_key: Optional[str] = None):
        self.db = db
        self.scanner = scanner
        self.profiler = profiler
        
        # Discovery engines
        self.token_discovery = TokenDiscoveryEngine()
        self.trader_discovery = ActiveTraderDiscovery(scanner.helius_key)
        
        # Load thresholds
        try:
            from discovery_config import config
            self.min_win_rate = config.min_win_rate
            self.min_pnl = config.min_pnl
            self.min_swings = config.min_completed_swings
        except:
            self.min_win_rate = 0.50
            self.min_pnl = 2.0
            self.min_swings = 3
        
        self.session_api_calls = 0
        
        print("  ✅ Hybrid Discovery v5 (Fee Payers Only)")
        print(f"     Thresholds: WR≥{self.min_win_rate:.0%} | PnL≥{self.min_pnl} | Swings≥{self.min_swings}")
    
    def run_discovery(self, api_budget: int = 500, max_wallets: int = 10) -> Dict:
        """Run discovery targeting actual traders"""
        
        self.session_api_calls = 0
        
        stats = {
            'tokens_discovered': 0,
            'wallet_candidates_found': 0,
            'wallets_profiled': 0,
            'wallets_verified': 0,
            'wallets_rejected': 0,
            'rejection_reasons': {},
            'helius_api_calls': 0,
            'token_sources': {},
            'verified_wallets': []
        }
        
        print(f"\n{'='*70}")
        print(f"🎯 WALLET DISCOVERY v5 (Fee Payers) - {datetime.now().strftime('%H:%M:%S')}")
        print(f"   Budget: {api_budget} credits | Target: {max_wallets} wallets")
        print(f"{'='*70}")
        
        # =================================================================
        # STEP 1: Token Discovery (FREE)
        # =================================================================
        print(f"\n{'='*70}")
        print("STEP 1: TOKEN DISCOVERY (DexScreener - FREE)")
        print(f"{'='*70}")
        
        all_tokens = self.token_discovery.get_all_discovery_tokens()
        
        sources = defaultdict(int)
        for t in all_tokens:
            sources[t.get('source', 'unknown')] += 1
        stats['token_sources'] = dict(sources)
        stats['tokens_discovered'] = len(all_tokens)
        
        print(f"\n   ✅ Total tokens: {len(all_tokens)}")
        
        if not all_tokens:
            print("\n   ❌ No tokens found!")
            return stats
        
        # =================================================================
        # STEP 2: Extract FEE PAYERS (Helius)
        # =================================================================
        print(f"\n{'='*70}")
        print("STEP 2: FIND FEE PAYERS (Actual Traders)")
        print("   Only capturing wallets that INITIATED swaps")
        print(f"{'='*70}")
        
        wallet_candidates = {}
        extraction_budget = int(api_budget * 0.4)
        
        # Prioritize high-volume tokens
        all_tokens.sort(key=lambda x: x['volume_24h'], reverse=True)
        
        tokens_to_scan = min(10, len(all_tokens))
        print(f"\n   Scanning {tokens_to_scan} tokens for fee payers...")
        
        for token in all_tokens[:tokens_to_scan]:
            if self.session_api_calls >= extraction_budget:
                print(f"\n   ⚠️  Extraction budget reached")
                break
            
            symbol = token['symbol']
            vol = token['volume_24h']
            print(f"\n   ${symbol} (vol: ${vol:,.0f})")
            
            try:
                # Get fee payers from recent swaps
                traders = self.trader_discovery.get_swap_fee_payers(token['address'], limit=20)
                self.session_api_calls += 2
                
                new_count = 0
                for trader in traders:
                    wallet = trader['address']
                    
                    if self.db.is_wallet_tracked(wallet):
                        continue
                    
                    if wallet not in wallet_candidates:
                        wallet_candidates[wallet] = {
                            'tokens': [],
                            'symbols': [],
                            'sources': set(),
                            'total_volume': 0,
                            'buy_count': 0,
                            'sell_count': 0,
                            'score': 0
                        }
                        new_count += 1
                    
                    wallet_candidates[wallet]['tokens'].append(token['address'])
                    wallet_candidates[wallet]['symbols'].append(symbol)
                    wallet_candidates[wallet]['sources'].add(token['source'])
                    wallet_candidates[wallet]['total_volume'] += trader.get('total_sol_volume', 0)
                    wallet_candidates[wallet]['buy_count'] += trader.get('buy_count', 0)
                    wallet_candidates[wallet]['sell_count'] += trader.get('sell_count', 0)
                
                print(f"      Found {len(traders)} traders, {new_count} new candidates")
                
            except Exception as e:
                print(f"      ⚠️  Error: {e}")
        
        stats['wallet_candidates_found'] = len(wallet_candidates)
        print(f"\n   ✅ Total fee payer candidates: {len(wallet_candidates)}")
        print(f"   ⚡ Credits used: {self.session_api_calls}")
        
        if not wallet_candidates:
            stats['helius_api_calls'] = self.session_api_calls
            return stats
        
        # =================================================================
        # STEP 3: Score Candidates (FREE)
        # =================================================================
        print(f"\n{'='*70}")
        print("STEP 3: CANDIDATE SCORING")
        print(f"{'='*70}")
        
        for wallet, data in wallet_candidates.items():
            score = 10
            
            # Multi-token trading is good
            n_tokens = len(data['tokens'])
            if n_tokens >= 3:
                score += 40
            elif n_tokens >= 2:
                score += 25
            
            # Having both buys AND sells is great (indicates active trading)
            if data['buy_count'] > 0 and data['sell_count'] > 0:
                score += 30
                print(f"   💎 {wallet[:12]}... has {data['buy_count']} buys + {data['sell_count']} sells")
            elif data['buy_count'] > 0 or data['sell_count'] > 0:
                score += 15
            
            # Higher volume = more serious trader
            if data['total_volume'] >= 10:
                score += 20
            elif data['total_volume'] >= 1:
                score += 10
            
            # Pumping tokens
            if 'pumping' in data['sources']:
                score += 15
            
            data['score'] = score
        
        # Sort by score, prioritizing those with both buys and sells
        sorted_candidates = sorted(
            wallet_candidates.items(),
            key=lambda x: (x[1]['buy_count'] > 0 and x[1]['sell_count'] > 0, x[1]['score']),
            reverse=True
        )
        
        to_profile = sorted_candidates[:max_wallets * 3]
        print(f"\n   Selected {len(to_profile)} top candidates for profiling")
        
        # =================================================================
        # STEP 4: Profile Wallets (Helius)
        # =================================================================
        print(f"\n{'='*70}")
        print("STEP 4: WALLET PROFILING")
        print(f"   Budget remaining: {api_budget - self.session_api_calls}")
        print(f"{'='*70}")
        
        verified_count = 0
        rejection_reasons = defaultdict(int)
        
        for wallet, data in to_profile:
            if verified_count >= max_wallets:
                print(f"\n   ✅ Target reached: {max_wallets} wallets")
                break
            
            if self.session_api_calls + 25 > api_budget:
                print(f"\n   ⚠️  Budget limit reached")
                break
            
            has_both = data['buy_count'] > 0 and data['sell_count'] > 0
            indicator = "💎" if has_both else "🔍"
            
            print(f"\n   {indicator} {wallet[:16]}... (score: {data['score']})")
            print(f"      Tokens: {', '.join(data['symbols'][:3])}")
            print(f"      Activity: {data['buy_count']} buys, {data['sell_count']} sells, {data['total_volume']:.2f} SOL vol")
            
            try:
                perf = self._profile_wallet(wallet)
                stats['wallets_profiled'] += 1
                
                wr_ok = perf['win_rate'] >= self.min_win_rate
                pnl_ok = perf['pnl'] >= self.min_pnl
                swings_ok = perf['completed_swings'] >= self.min_swings
                
                print(f"      WR: {perf['win_rate']:.1%} {'✅' if wr_ok else '❌'}")
                print(f"      PnL: {perf['pnl']:.2f} SOL {'✅' if pnl_ok else '❌'}")
                print(f"      Swings: {perf['completed_swings']} {'✅' if swings_ok else '❌'}")
                
                if wr_ok and pnl_ok and swings_ok:
                    self.db.add_verified_wallet(wallet, perf)
                    stats['wallets_verified'] += 1
                    stats['verified_wallets'].append({
                        'address': wallet,
                        'win_rate': perf['win_rate'],
                        'pnl': perf['pnl'],
                        'swings': perf['completed_swings']
                    })
                    verified_count += 1
                    print(f"      ✅ VERIFIED!")
                else:
                    stats['wallets_rejected'] += 1
                    if not swings_ok:
                        rejection_reasons['low_swings'] += 1
                    elif not wr_ok:
                        rejection_reasons['low_win_rate'] += 1
                    else:
                        rejection_reasons['low_pnl'] += 1
                
            except Exception as e:
                print(f"      ❌ Error: {e}")
                stats['wallets_rejected'] += 1
                rejection_reasons['error'] += 1
        
        stats['rejection_reasons'] = dict(rejection_reasons)
        stats['helius_api_calls'] = self.session_api_calls
        
        # =================================================================
        # SUMMARY
        # =================================================================
        print(f"\n{'='*70}")
        print("✅ DISCOVERY COMPLETE")
        print(f"{'='*70}")
        print(f"   Tokens scanned: {stats['tokens_discovered']}")
        print(f"   Fee payer candidates: {stats['wallet_candidates_found']}")
        print(f"   Profiled: {stats['wallets_profiled']}")
        print(f"   VERIFIED: {stats['wallets_verified']} ✅")
        print(f"   Rejected: {stats['wallets_rejected']}")
        
        if rejection_reasons:
            print(f"\n   Rejection breakdown:")
            for reason, count in rejection_reasons.items():
                print(f"      {reason}: {count}")
        
        print(f"\n   Helius credits: {stats['helius_api_calls']}/{api_budget}")
        
        if stats['wallets_profiled'] > 0:
            rate = 100 * stats['wallets_verified'] / stats['wallets_profiled']
            print(f"   Verification rate: {rate:.0f}%")
        
        if stats['verified_wallets']:
            print(f"\n   ✨ New verified wallets:")
            for w in stats['verified_wallets']:
                print(f"      {w['address']}")
                print(f"         WR: {w['win_rate']:.1%} | PnL: {w['pnl']:.2f} SOL | Swings: {w['swings']}")
        
        print(f"{'='*70}\n")
        
        return stats
    
    def _profile_wallet(self, wallet: str) -> Dict:
        """Profile wallet trading performance"""
        self.session_api_calls += 1
        sigs = self.scanner.get_recent_signatures(wallet, limit=100)
        
        if not sigs:
            return self._empty_performance()
        
        self.session_api_calls += max(1, len(sigs) // 5)
        return self.profiler.calculate_performance(wallet, days=7)
    
    def _empty_performance(self) -> Dict:
        return {
            'win_rate': 0, 'pnl': 0, 'roi_7d': 0, 'completed_swings': 0,
            'avg_hold_hours': 0, 'risk_reward_ratio': 0, 'best_trade_pct': 0,
            'worst_trade_pct': 0, 'total_volume_sol': 0, 'avg_position_size_sol': 0
        }
