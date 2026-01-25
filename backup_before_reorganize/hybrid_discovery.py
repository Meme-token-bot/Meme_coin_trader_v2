"""
HYBRID DISCOVERY SYSTEM v7
Optimized for MAXIMUM wallet discovery with smart pre-filtering

KEY IMPROVEMENTS over v5:
1. STRICT PRE-FILTER: ONLY profile wallets with BOTH buys AND sells
   - Wallets with 0 sells = 0 possible swings = SKIP (saves ~20 credits each!)
2. INCREASED BUDGET: Uses more of your 1M credits/month allocation
3. MORE TOKENS: Scans more tokens to find more candidates
4. MINIMUM ACTIVITY: Requires minimum trade count before profiling
5. BETTER SCORING: Prioritizes wallets most likely to pass verification

BUDGET MATH (1M credits/month):
- Daily budget: 33,333 credits
- Webhook monitoring: ~5,000/day (100 wallets Ã— 50 trades)
- Discovery: ~10,000/day (2 runs Ã— 5,000 each)
- Buffer: ~18,000/day
- Monthly total: ~600,000 (60% utilization - safe margin)

INSTALLATION:
  cp hybrid_discovery_v7.py ~/Documents/Meme_coin_trader_V2/hybrid_discovery.py
"""

import requests
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import os


# =============================================================================
# TOKEN DISCOVERY ENGINE (FREE - DexScreener)
# =============================================================================

class TokenDiscoveryEngine:
    """Token discovery using DexScreener search API - completely FREE"""
    
    def __init__(self):
        self.base_url = "https://api.dexscreener.com"
        self._last_call = 0
        self._call_delay = 0.3  # Slightly faster for DexScreener
    
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
    
    def find_pumping_tokens(self, min_gain: float = 20, limit: int = 30) -> List[Dict]:
        """Find tokens with significant price gains - EXPANDED search"""
        print(f"   ðŸ“ˆ Finding pumping tokens (>{min_gain:.0f}% gain)...")
        
        all_tokens = []
        # Expanded query list for more token coverage
        queries = [
            'pump', 'moon', 'sol', 'meme', 'pepe', 'doge', 'ai', 'cat', 'dog',
            'trump', 'elon', 'wojak', 'chad', 'bonk', 'wif', 'popcat', 'wen',
            'jup', 'ray', 'orca', 'fartcoin', 'goat', 'pnut', 'act', 'virtual'
        ]
        
        for query in queries:
            tokens = self.search_tokens(query, 'pumping')
            all_tokens.extend(tokens)
        
        # Deduplicate
        seen = set()
        unique = [t for t in all_tokens if t['address'] not in seen and not seen.add(t['address'])]
        
        # Filter for quality
        pumping = [t for t in unique 
                   if t['price_change_24h'] >= min_gain 
                   and t['liquidity'] >= 10000 
                   and t['volume_24h'] >= 10000]
        pumping.sort(key=lambda x: x['price_change_24h'], reverse=True)
        
        result = pumping[:limit]
        print(f"      Found {len(result)} pumping tokens")
        return result
    
    def find_high_volume_tokens(self, min_volume: float = 30000, limit: int = 30) -> List[Dict]:
        """Find tokens with high trading volume"""
        print(f"   ðŸ”¥ Finding high-volume tokens (>${min_volume:,.0f})...")
        
        all_tokens = []
        queries = ['sol', 'solana', 'jup', 'bonk', 'wif', 'popcat', 'wen', 
                   'pepe', 'doge', 'trump', 'ai', 'meme']
        
        for query in queries:
            tokens = self.search_tokens(query, 'volume')
            all_tokens.extend(tokens)
        
        seen = set()
        unique = [t for t in all_tokens if t['address'] not in seen and not seen.add(t['address'])]
        
        high_vol = [t for t in unique 
                    if t['volume_24h'] >= min_volume 
                    and t['liquidity'] >= 15000]
        high_vol.sort(key=lambda x: x['volume_24h'], reverse=True)
        
        result = high_vol[:limit]
        print(f"      Found {len(result)} high-volume tokens")
        return result
    
    def find_new_tokens(self, max_age_hours: int = 48, limit: int = 20) -> List[Dict]:
        """Find recently created tokens - where early traders are found"""
        print(f"   ðŸ†• Finding new tokens (<{max_age_hours}h old)...")
        
        all_tokens = []
        queries = ['new', 'launch', 'fair', 'presale']
        
        for query in queries:
            tokens = self.search_tokens(query, 'new')
            all_tokens.extend(tokens)
        
        seen = set()
        unique = [t for t in all_tokens if t['address'] not in seen and not seen.add(t['address'])]
        
        now = datetime.now().timestamp() * 1000
        cutoff = now - (max_age_hours * 3600 * 1000)
        
        new_tokens = []
        for t in unique:
            created = t.get('created_at')
            if created and created > cutoff and t['liquidity'] >= 5000:
                new_tokens.append(t)
        
        new_tokens.sort(key=lambda x: x.get('created_at', 0), reverse=True)
        
        result = new_tokens[:limit]
        print(f"      Found {len(result)} new tokens")
        return result
    
    def get_all_discovery_tokens(self) -> List[Dict]:
        """Get comprehensive token list for discovery"""
        all_tokens = []
        
        try:
            pumping = self.find_pumping_tokens(min_gain=15, limit=30)
            all_tokens.extend(pumping)
        except Exception as e:
            print(f"   ⚠️  Pumping error: {e}")
        
        try:
            high_vol = self.find_high_volume_tokens(min_volume=25000, limit=30)
            all_tokens.extend(high_vol)
        except Exception as e:
            print(f"   ⚠️  Volume error: {e}")
        
        try:
            new = self.find_new_tokens(max_age_hours=72, limit=20)
            all_tokens.extend(new)
        except Exception as e:
            print(f"   ⚠️  New tokens error: {e}")
        
        # Deduplicate
        seen = set()
        unique = [t for t in all_tokens if t['address'] not in seen and not seen.add(t['address'])]
        
        print(f"\n   ✅ Total unique tokens: {len(unique)}")
        return unique


# =============================================================================
# ACTIVE TRADER DISCOVERY (Helius API)
# =============================================================================

class ActiveTraderDiscovery:
    """
    Find ACTIVE TRADERS by capturing FEE PAYERS from swap transactions.
    
    v7 IMPROVEMENT: Track both buys AND sells per wallet during extraction
    so we can pre-filter before expensive profiling.
    """
    
    def __init__(self, helius_key: str):
        self.helius_key = helius_key
        self.helius_rpc = f"https://mainnet.helius-rpc.com/?api-key={helius_key}"
        self.helius_api = f"https://api.helius.xyz/v0"
        self._last_call = 0
        self._delay = 0.1  # 10 calls/second to Helius
        
        # Known program/protocol addresses to exclude
        self.excluded_addresses = {
            '675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8',  # Raydium
            'JUP6LkbZbjS1jKKwapdHNy74zcZ3tLUZoi5QNyVTaV4',  # Jupiter
            'whirLbMiicVdio4qvUfM5KAg6Ct8VwpYzGff3uctyCc',  # Orca
            '9W959DqEETiGZocYWCQPaJ6sBmUzgfxXfqGeTEdp3aQP',  # Orca v2
            'TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA',  # SPL Token
            'ATokenGPvbdGVxr1b2hvZbsiqW5xWH25efTNsLJA8knL',  # Associated Token
        }
    
    def _rate_limit(self):
        elapsed = time.time() - self._last_call
        if elapsed < self._delay:
            time.sleep(self._delay - elapsed)
        self._last_call = time.time()
    
    def _is_likely_bot_or_protocol(self, address: str) -> bool:
        if address in self.excluded_addresses:
            return True
        if len(address) < 32:
            return True
        return False
    
    def get_swap_fee_payers(self, token_address: str, limit: int = 50) -> List[Dict]:
        """
        Get fee payers (actual traders) from recent swap transactions.
        Returns detailed trade info including buy/sell counts.
        """
        traders = {}
        
        try:
            # Step 1: Get recent signatures for the token
            self._rate_limit()
            
            payload = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "getSignaturesForAddress",
                "params": [token_address, {"limit": 150}]  # Get more signatures
            }
            
            res = requests.post(self.helius_rpc, json=payload, timeout=15)
            signatures = res.json().get('result', [])
            
            if not signatures:
                return []
            
            # Step 2: Parse transactions with Helius (batched)
            sig_list = [s.get('signature') for s in signatures[:100] if s.get('signature')]
            
            if not sig_list:
                return []
            
            self._rate_limit()
            parse_url = f"{self.helius_api}/transactions?api-key={self.helius_key}"
            parse_res = requests.post(parse_url, json={"transactions": sig_list}, timeout=20)
            parsed_txs = parse_res.json()
            
            if not isinstance(parsed_txs, list):
                return []
            
            # Step 3: Extract fee payers with detailed trade info
            for tx in parsed_txs:
                if not isinstance(tx, dict):
                    continue
                
                if tx.get('type') != 'SWAP':
                    continue
                
                fee_payer = tx.get('feePayer')
                
                if not fee_payer or self._is_likely_bot_or_protocol(fee_payer):
                    continue
                
                swap_info = self._analyze_swap(tx, fee_payer)
                
                if swap_info and swap_info.get('sol_amount', 0) >= 0.01:
                    if fee_payer not in traders:
                        traders[fee_payer] = {
                            'address': fee_payer,
                            'swaps': [],
                            'total_sol_volume': 0,
                            'buy_count': 0,
                            'sell_count': 0,
                            'unique_tokens_traded': set()
                        }
                    
                    traders[fee_payer]['swaps'].append(swap_info)
                    traders[fee_payer]['total_sol_volume'] += swap_info.get('sol_amount', 0)
                    traders[fee_payer]['unique_tokens_traded'].add(swap_info.get('token', ''))
                    
                    if swap_info.get('type') == 'BUY':
                        traders[fee_payer]['buy_count'] += 1
                    elif swap_info.get('type') == 'SELL':
                        traders[fee_payer]['sell_count'] += 1
                
                if len(traders) >= limit:
                    break
            
            # Convert sets to counts for JSON serialization
            for wallet in traders.values():
                wallet['unique_tokens'] = len(wallet['unique_tokens_traded'])
                del wallet['unique_tokens_traded']
            
            return list(traders.values())
            
        except Exception as e:
            print(f"      ⚠️  Error extracting traders: {e}")
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
            token = list(tokens_in.keys())[0]
            return {
                'type': 'BUY',
                'token': token,
                'token_amount': tokens_in[token],
                'sol_amount': sol_out
            }
        elif len(tokens_out) >= 1 and sol_in > 0:
            token = list(tokens_out.keys())[0]
            return {
                'type': 'SELL',
                'token': token,
                'token_amount': tokens_out[token],
                'sol_amount': sol_in
            }
        
        return None
    
    def validate_wallet_level(self, wallet_address: str, limit: int = 50) -> Dict:
        """
        Validate a wallet by checking their ACTUAL transaction history.
        
        This fixes the token-centric view problem: when scanning $MEME transactions,
        we only see trades FOR $MEME. But a wallet might buy $MEME and sell $PEPE.
        
        This method fetches the WALLET's transactions to see all their trades.
        
        Returns: {
            'buy_count': int,
            'sell_count': int,
            'unique_tokens': int,
            'total_volume': float,
            'has_completed_swings': bool
        }
        
        Cost: ~2 API calls (1 for signatures, 1 for parsing)
        """
        result = {
            'buy_count': 0,
            'sell_count': 0,
            'unique_tokens': 0,
            'total_volume': 0,
            'has_completed_swings': False,
            'api_calls': 0
        }
        
        try:
            # Step 1: Get recent signatures for the WALLET
            self._rate_limit()
            
            payload = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "getSignaturesForAddress",
                "params": [wallet_address, {"limit": limit}]
            }
            
            res = requests.post(self.helius_rpc, json=payload, timeout=15)
            signatures = res.json().get('result', [])
            result['api_calls'] += 1
            
            if not signatures:
                return result
            
            # Step 2: Parse transactions
            sig_list = [s.get('signature') for s in signatures if s.get('signature')]
            
            if not sig_list:
                return result
            
            self._rate_limit()
            parse_url = f"{self.helius_api}/transactions?api-key={self.helius_key}"
            parse_res = requests.post(parse_url, json={"transactions": sig_list}, timeout=20)
            parsed_txs = parse_res.json()
            result['api_calls'] += 1
            
            if not isinstance(parsed_txs, list):
                return result
            
            # Step 3: Analyze all SWAPs from this wallet's perspective
            tokens_bought = set()
            tokens_sold = set()
            total_volume = 0
            
            for tx in parsed_txs:
                if not isinstance(tx, dict):
                    continue
                
                if tx.get('type') != 'SWAP':
                    continue
                
                swap_info = self._analyze_swap(tx, wallet_address)
                
                if swap_info:
                    total_volume += swap_info.get('sol_amount', 0)
                    token = swap_info.get('token', '')
                    
                    if swap_info.get('type') == 'BUY':
                        result['buy_count'] += 1
                        if token:
                            tokens_bought.add(token)
                    elif swap_info.get('type') == 'SELL':
                        result['sell_count'] += 1
                        if token:
                            tokens_sold.add(token)
            
            result['total_volume'] = total_volume
            result['unique_tokens'] = len(tokens_bought | tokens_sold)
            
            # Check for completed swings (bought AND sold same token)
            completed_tokens = tokens_bought & tokens_sold
            result['has_completed_swings'] = len(completed_tokens) > 0
            
            return result
            
        except Exception as e:
            print(f"      ⚠️  Wallet validation error: {e}")
            return result


# =============================================================================
# HYBRID DISCOVERY SYSTEM v7
# =============================================================================

class HybridDiscoverySystem:
    """
    Wallet discovery system v7 - With WALLET-LEVEL VALIDATION
    
    KEY FEATURES:
    1. TOKEN EXTRACTION: Find candidates from token transactions
    2. WALLET VALIDATION: For candidates that fail token-level filter,
       check their ACTUAL wallet history for buys+sells
    3. SMART PROFILING: Only profile wallets that pass validation
    4. SCORING: Prioritize promising wallets
    
    This fixes the "token-centric view" problem where a wallet buying
    $MEME and selling $PEPE would be rejected as "no sells" when
    only scanning $MEME transactions.
    """
    
    # Minimum requirements to even consider profiling (STRICT)
    MIN_BUYS_FOR_PROFILE = 1       # Must have at least 1 buy
    MIN_SELLS_FOR_PROFILE = 1      # Must have at least 1 sell (CRITICAL!)
    MIN_TOTAL_TRADES = 3           # Must have 3+ total trades visible
    MIN_SOL_VOLUME = 0.1           # Must have traded at least 0.1 SOL
    
    def __init__(self, db, scanner, profiler, birdeye_key: Optional[str] = None):
        self.db = db
        self.scanner = scanner
        self.profiler = profiler
        
        # Discovery engines
        self.token_discovery = TokenDiscoveryEngine()
        self.trader_discovery = ActiveTraderDiscovery(scanner.helius_key)
        
        # Load thresholds from config
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
        
        print("  ✅ Hybrid Discovery v7 (Wallet-Level Validation)")
        print(f"     Pre-filter: Token check → Wallet validation → Filter")
        print(f"     Verify thresholds: WR≥{self.min_win_rate:.0%} | PnL≥{self.min_pnl} SOL | Swings≥{self.min_swings}")
    
    def _passes_prefilter(self, wallet_data: Dict) -> Tuple[bool, str]:
        """
        STRICT pre-filter to avoid wasting API credits on non-viable wallets.
        
        Returns: (passes, reason_if_failed)
        """
        buys = wallet_data.get('buy_count', 0)
        sells = wallet_data.get('sell_count', 0)
        total = buys + sells
        volume = wallet_data.get('total_volume', 0)
        
        # CRITICAL: Must have sells to have completed swings
        if sells < self.MIN_SELLS_FOR_PROFILE:
            return False, f"no_sells ({sells} sells)"
        
        if buys < self.MIN_BUYS_FOR_PROFILE:
            return False, f"no_buys ({buys} buys)"
        
        if total < self.MIN_TOTAL_TRADES:
            return False, f"too_few_trades ({total} total)"
        
        if volume < self.MIN_SOL_VOLUME:
            return False, f"low_volume ({volume:.2f} SOL)"
        
        return True, "passed"
    
    def run_discovery(self, api_budget: int = 5000, max_wallets: int = 15) -> Dict:
        """
        Run discovery with STRICT pre-filtering and higher budget.
        
        Args:
            api_budget: Helius API credits to use (default 5000, can go up to 10000)
            max_wallets: Maximum verified wallets to add this cycle
        """
        
        self.session_api_calls = 0
        
        stats = {
            'tokens_discovered': 0,
            'wallet_candidates_found': 0,
            'wallets_prefiltered_out': 0,
            'wallets_profiled': 0,
            'wallets_verified': 0,
            'wallets_rejected': 0,
            'prefilter_reasons': {},
            'rejection_reasons': {},
            'helius_api_calls': 0,
            'token_sources': {},
            'verified_wallets': [],
            'credits_saved_by_prefilter': 0
        }
        
        print(f"\n{'='*70}")
        print(f"🎯 WALLET DISCOVERY v7 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"   Budget: {api_budget} credits | Target: {max_wallets} wallets")
        print(f"   Wallet-level validation enabled for better accuracy")
        print(f"{'='*70}")
        
        # =================================================================
        # STEP 1: Token Discovery (FREE - DexScreener)
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
        
        if not all_tokens:
            print("\n   ❌ No tokens found!")
            return stats
        
        # =================================================================
        # STEP 2: Extract Traders (Helius API)
        # =================================================================
        print(f"\n{'='*70}")
        print("STEP 2: EXTRACT ACTIVE TRADERS (Helius)")
        print(f"{'='*70}")
        
        wallet_candidates = {}
        extraction_budget = int(api_budget * 0.5)  # Use 50% for extraction
        
        # Sort by volume for better quality candidates
        all_tokens.sort(key=lambda x: x['volume_24h'], reverse=True)
        
        # Scan more tokens for broader coverage
        tokens_to_scan = min(25, len(all_tokens))
        print(f"\n   Scanning {tokens_to_scan} tokens for traders...")
        
        for token in all_tokens[:tokens_to_scan]:
            if self.session_api_calls >= extraction_budget:
                print(f"\n   ⚠️  Extraction budget reached ({extraction_budget})")
                break
            
            symbol = token['symbol']
            vol = token['volume_24h']
            print(f"\n   ${symbol} (vol: ${vol:,.0f})")
            
            try:
                traders = self.trader_discovery.get_swap_fee_payers(token['address'], limit=50)
                self.session_api_calls += 2  # 1 for signatures, 1 for parsing
                
                new_count = 0
                has_both_count = 0
                
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
                    wallet_candidates[wallet]['sources'].add(token.get('source', 'unknown'))
                    wallet_candidates[wallet]['total_volume'] += trader.get('total_sol_volume', 0)
                    wallet_candidates[wallet]['buy_count'] += trader.get('buy_count', 0)
                    wallet_candidates[wallet]['sell_count'] += trader.get('sell_count', 0)
                    
                    # Count wallets with both buys and sells
                    if wallet_candidates[wallet]['buy_count'] > 0 and wallet_candidates[wallet]['sell_count'] > 0:
                        has_both_count += 1
                
                print(f"      Found {len(traders)} traders, {new_count} new, {has_both_count} with buys+sells")
                
            except Exception as e:
                print(f"      ⚠️  Error: {e}")
        
        stats['wallet_candidates_found'] = len(wallet_candidates)
        print(f"\n   ✅ Total candidates: {len(wallet_candidates)}")
        print(f"   ⚡ Credits used for extraction: {self.session_api_calls}")
        
        if not wallet_candidates:
            stats['helius_api_calls'] = self.session_api_calls
            return stats
        
        # =================================================================
        # =================================================================
        # STEP 3: WALLET-LEVEL VALIDATION + PRE-FILTER
        # =================================================================
        print(f"\n{'='*70}")
        print("STEP 3: WALLET-LEVEL VALIDATION + PRE-FILTER")
        print("   Token-level check → Wallet-level validation for failures → Filter")
        print(f"{'='*70}")
        
        qualified_candidates = []
        prefilter_reasons = defaultdict(int)
        
        # Budget for wallet-level validation (use 15% of remaining budget)
        validation_budget = int((api_budget - self.session_api_calls) * 0.15)
        validation_calls = 0
        wallets_rescued = 0
        
        # Prioritize candidates by volume for validation
        sorted_candidates = sorted(
            wallet_candidates.items(),
            key=lambda x: x[1].get('total_volume', 0),
            reverse=True
        )
        
        print(f"\n   Validating {len(sorted_candidates)} candidates...")
        print(f"   Wallet-level validation budget: {validation_budget} credits")
        
        for wallet, data in sorted_candidates:
            # First try: token-level pre-filter
            passes, reason = self._passes_prefilter(data)
            
            if passes:
                # Already passes based on token-level data
                score = self._score_candidate(data)
                data['score'] = score
                qualified_candidates.append((wallet, data))
                continue
            
            # Failed token-level check - try wallet-level validation
            # Only validate if reason is "no_sells" or "no_buys" (not "too_few_trades" or "low_volume")
            should_validate = reason.startswith('no_sells') or reason.startswith('no_buys')
            
            if should_validate and validation_calls + 2 <= validation_budget:
                # Run wallet-level validation (costs ~2 API calls)
                validation = self.trader_discovery.validate_wallet_level(wallet, limit=50)
                validation_calls += validation.get('api_calls', 2)
                self.session_api_calls += validation.get('api_calls', 2)
                
                # Update candidate data with wallet-level stats
                if validation['buy_count'] > 0 and validation['sell_count'] > 0:
                    # Wallet has both buys AND sells at wallet level!
                    data['buy_count'] = validation['buy_count']
                    data['sell_count'] = validation['sell_count']
                    data['total_volume'] = max(data['total_volume'], validation['total_volume'])
                    data['wallet_validated'] = True
                    
                    # Re-check pre-filter with updated data
                    passes_now, _ = self._passes_prefilter(data)
                    
                    if passes_now:
                        score = self._score_candidate(data)
                        data['score'] = score
                        qualified_candidates.append((wallet, data))
                        wallets_rescued += 1
                        print(f"      ✅ Rescued {wallet[:12]}... ({validation['buy_count']}B/{validation['sell_count']}S)")
                        continue
                
                # Still failed after wallet-level validation
                prefilter_reasons[f"{reason} (validated)"] += 1
                stats['wallets_prefiltered_out'] += 1
            else:
                # Not validated - record original reason
                prefilter_reasons[reason] += 1
                stats['wallets_prefiltered_out'] += 1
        
        stats['prefilter_reasons'] = dict(prefilter_reasons)
        stats['wallets_rescued_by_validation'] = wallets_rescued
        stats['validation_api_calls'] = validation_calls
        
        # Calculate credits saved
        credits_per_profile = 20  # Approximate
        stats['credits_saved_by_prefilter'] = stats['wallets_prefiltered_out'] * credits_per_profile
        
        print(f"\n   📊 Pre-filter results:")
        print(f"      Total candidates: {len(wallet_candidates)}")
        print(f"      Passed (token-level): {len(qualified_candidates) - wallets_rescued}")
        print(f"      Rescued (wallet-level): {wallets_rescued}")
        print(f"      Total qualified: {len(qualified_candidates)}")
        print(f"      Filtered out: {stats['wallets_prefiltered_out']}")
        print(f"      Validation credits used: {validation_calls}")
        print(f"      Profile credits saved: ~{stats['credits_saved_by_prefilter']}")
        
        if prefilter_reasons:
            print(f"\n   Filter breakdown:")
            for reason, count in sorted(prefilter_reasons.items(), key=lambda x: -x[1]):
                print(f"      {reason}: {count}")
        
        if not qualified_candidates:
            print(f"\n   ❌ No candidates passed pre-filter!")
            stats['helius_api_calls'] = self.session_api_calls
            return stats
        
        # Sort by score (best candidates first)
        qualified_candidates.sort(key=lambda x: x[1]['score'], reverse=True)
        
        # Show top candidates
        print(f"\n   🏆 Top candidates (by score):")
        for wallet, data in qualified_candidates[:5]:
            validated = " [V]" if data.get('wallet_validated') else ""
            print(f"      {wallet[:12]}...{validated} | Score: {data['score']} | "
                  f"{data['buy_count']}B/{data['sell_count']}S | {data['total_volume']:.1f} SOL")
        
        # STEP 4: Profile Qualified Wallets (Helius API)
        # =================================================================
        print(f"\n{'='*70}")
        print("STEP 4: PROFILE QUALIFIED WALLETS (Helius)")
        print(f"   Budget remaining: {api_budget - self.session_api_calls}")
        print(f"   Candidates to profile: {len(qualified_candidates)}")
        print(f"{'='*70}")
        
        verified_count = 0
        rejection_reasons = defaultdict(int)
        
        # Profile more candidates since they're pre-qualified
        max_to_profile = min(len(qualified_candidates), max_wallets * 4)
        
        for wallet, data in qualified_candidates[:max_to_profile]:
            if verified_count >= max_wallets:
                print(f"\n   ✅ Target reached: {max_wallets} verified wallets")
                break
            
            if self.session_api_calls + 25 > api_budget:
                print(f"\n   ⚠️  Budget limit reached")
                break
            
            print(f"\n   💎 {wallet[:16]}... (score: {data['score']})")
            print(f"      Tokens: {', '.join(data['symbols'][:3])}")
            print(f"      Activity: {data['buy_count']} buys, {data['sell_count']} sells, {data['total_volume']:.2f} SOL")
            
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
        print("✅ DISCOVERY v7 COMPLETE")
        print(f"{'='*70}")
        print(f"   Tokens scanned: {stats['tokens_discovered']}")
        print(f"   Total candidates: {stats['wallet_candidates_found']}")
        print(f"   Pre-filtered out: {stats['wallets_prefiltered_out']} (saved ~{stats['credits_saved_by_prefilter']} credits)")
        print(f"   Profiled: {stats['wallets_profiled']}")
        print(f"   VERIFIED: {stats['wallets_verified']} ✅")
        print(f"   Rejected: {stats['wallets_rejected']}")
        
        if rejection_reasons:
            print(f"\n   Rejection breakdown (profiled but didn't qualify):")
            for reason, count in rejection_reasons.items():
                print(f"      {reason}: {count}")
        
        print(f"\n   Helius credits: {stats['helius_api_calls']}/{api_budget}")
        
        if stats['wallets_profiled'] > 0:
            rate = 100 * stats['wallets_verified'] / stats['wallets_profiled']
            print(f"   Verification rate: {rate:.0f}%")
        
        if stats['verified_wallets']:
            print(f"\n   âœ¨ New verified wallets:")
            for w in stats['verified_wallets']:
                print(f"      {w['address']}")
                print(f"         WR: {w['win_rate']:.1%} | PnL: {w['pnl']:.2f} SOL | Swings: {w['swings']}")
        
        print(f"{'='*70}\n")
        
        return stats
    
    def _score_candidate(self, data: Dict) -> int:
        """Score a candidate wallet for profiling priority"""
        score = 0
        
        # Both buys and sells (required, but bonus for more)
        buys = data.get('buy_count', 0)
        sells = data.get('sell_count', 0)
        
        # More completed round-trips = higher score
        min_pairs = min(buys, sells)
        if min_pairs >= 5:
            score += 50
        elif min_pairs >= 3:
            score += 35
        elif min_pairs >= 2:
            score += 25
        else:
            score += 15
        
        # Multi-token trading
        n_tokens = len(data.get('tokens', []))
        if n_tokens >= 4:
            score += 30
        elif n_tokens >= 2:
            score += 15
        
        # Higher volume
        volume = data.get('total_volume', 0)
        if volume >= 10:
            score += 25
        elif volume >= 3:
            score += 15
        elif volume >= 1:
            score += 5
        
        # Bonus for pumping tokens
        if 'pumping' in data.get('sources', set()):
            score += 10
        
        return score
    
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


# =============================================================================
# QUICK TEST
# =============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("HYBRID DISCOVERY v7 - Module Test")
    print("="*70)
    
    # Test token discovery (FREE)
    engine = TokenDiscoveryEngine()
    tokens = engine.get_all_discovery_tokens()
    
    print(f"\nToken discovery test: Found {len(tokens)} tokens")
    
    if tokens:
        print("\nSample tokens:")
        for t in tokens[:5]:
            print(f"  ${t['symbol']}: Vol ${t['volume_24h']:,.0f} | Liq ${t['liquidity']:,.0f}")
    
    print("\n" + "="*70)
    print("Module loaded successfully!")
    print("="*70)
