"""
THE HISTORIAN V2 - Wallet Discovery & Monitoring
Trading System V2 - Optimized for efficiency

Responsibilities:
1. Discover new swing traders from token launches
2. Profile and verify wallet performance
3. Monitor verified wallets for trades
4. Track wallet exits for exit signals

Optimizations:
- Batched API calls where possible
- Caching of token info
- Rate limit handling
- Minimal redundant processing
"""

import requests
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Set
from collections import defaultdict
import threading


class TokenScanner:
    """
    Handles all external API interactions for token and wallet data.
    Optimized with caching and rate limit awareness.
    """
    
    def __init__(self, helius_key: str):
        self.helius_key = helius_key
        self.helius_url = f"https://mainnet.helius-rpc.com/?api-key={helius_key}"
        self.helius_api = f"https://api.helius.xyz/v0"
        self.dex_url = "https://api.dexscreener.com/latest/dex"
        
        # Caching
        self._token_cache: Dict[str, Tuple[Dict, datetime]] = {}
        self._cache_ttl = 300  # 5 minutes
        
        # Rate limiting
        self._last_helius_call = 0
        self._helius_delay = 0.1  # 100ms between calls
        self._last_dex_call = 0
        self._dex_delay = 0.5  # 500ms between calls
        
        # Ignored tokens (stablecoins, wrapped assets)
        self.ignored_tokens = {
            'EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v',  # USDC
            'Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB',  # USDT
            'So11111111111111111111111111111111111111112',   # WSOL
            'mSoLzYCxHdYgdzU16g5QSh3i5K3z3KZK7ytfqcJm7So',  # mSOL
            '7dHbWXmci3dT8UFYWYZweBLXgycu7Y3iL6trKn1Y7ARj',  # stSOL
        }
        
        print("  ✅ Token Scanner initialized")
    
    def _rate_limit_helius(self):
        """Enforce rate limiting for Helius API"""
        elapsed = time.time() - self._last_helius_call
        if elapsed < self._helius_delay:
            time.sleep(self._helius_delay - elapsed)
        self._last_helius_call = time.time()
    
    def _rate_limit_dex(self):
        """Enforce rate limiting for DexScreener API"""
        elapsed = time.time() - self._last_dex_call
        if elapsed < self._dex_delay:
            time.sleep(self._dex_delay - elapsed)
        self._last_dex_call = time.time()
    
    def is_ignored_token(self, token_address: str) -> bool:
        """Check if token should be ignored"""
        return token_address in self.ignored_tokens
    
    # =========================================================================
    # SIGNATURE & TRANSACTION METHODS
    # =========================================================================
    
    def get_recent_signatures(self, wallet: str, limit: int = 10) -> List[Dict]:
        """Get recent transaction signatures for a wallet"""
        self._rate_limit_helius()
        try:
            payload = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "getSignaturesForAddress",
                "params": [wallet, {"limit": limit}]
            }
            res = requests.post(self.helius_url, json=payload, timeout=10)
            return res.json().get('result', [])
        except Exception:
            return []
    
    def get_parsed_trade(self, signature: str) -> Optional[Dict]:
        """
        Parse a transaction to extract trade info.
        Returns: {'type': 'BUY'/'SELL', 'token_address': str, 'amount': float, 'sol_amount': float}
        """
        self._rate_limit_helius()
        try:
            url = f"{self.helius_api}/transactions?api-key={self.helius_key}"
            res = requests.post(url, json={"transactions": [signature]}, timeout=10)
            data = res.json()
            
            if not data or len(data) == 0:
                return None
            
            tx = data[0]
            if tx.get('type') != 'SWAP':
                return None
            
            fee_payer = tx.get('feePayer', '')
            token_transfers = tx.get('tokenTransfers', [])
            native_transfers = tx.get('nativeTransfers', [])
            
            WSOL = "So11111111111111111111111111111111111111112"
            
            sol_in, sol_out = 0, 0
            tokens_in, tokens_out = {}, {}
            
            # Process token transfers
            for transfer in token_transfers:
                mint = transfer.get('mint', '')
                amount = float(transfer.get('tokenAmount', 0))
                from_addr = transfer.get('fromUserAccount', '')
                to_addr = transfer.get('toUserAccount', '')
                
                if from_addr == fee_payer:
                    if mint == WSOL:
                        sol_out += amount
                    else:
                        tokens_out[mint] = tokens_out.get(mint, 0) + amount
                elif to_addr == fee_payer:
                    if mint == WSOL:
                        sol_in += amount
                    else:
                        tokens_in[mint] = tokens_in.get(mint, 0) + amount
            
            # Process native SOL transfers
            for transfer in native_transfers:
                amount = float(transfer.get('amount', 0)) / 1e9
                from_addr = transfer.get('fromUserAccount', '')
                to_addr = transfer.get('toUserAccount', '')
                
                if from_addr == fee_payer:
                    sol_out += amount
                elif to_addr == fee_payer:
                    sol_in += amount
            
            # Determine trade type
            if len(tokens_in) == 1 and sol_out > 0:
                token_addr = list(tokens_in.keys())[0]
                return {
                    'type': 'BUY',
                    'token_address': token_addr,
                    'amount': tokens_in[token_addr],
                    'sol_amount': sol_out
                }
            elif len(tokens_out) == 1 and sol_in > 0:
                token_addr = list(tokens_out.keys())[0]
                return {
                    'type': 'SELL',
                    'token_address': token_addr,
                    'amount': tokens_out[token_addr],
                    'sol_amount': sol_in
                }
            
            return None
            
        except Exception:
            return None
    
    def get_parsed_trades_batch(self, signatures: List[str]) -> Dict[str, Optional[Dict]]:
        """
        Parse multiple transactions in a single API call (more efficient).
        """
        if not signatures:
            return {}
        
        self._rate_limit_helius()
        results = {}
        
        try:
            # Helius supports batching up to 100 transactions
            url = f"{self.helius_api}/transactions?api-key={self.helius_key}"
            res = requests.post(url, json={"transactions": signatures[:100]}, timeout=30)
            data = res.json()
            
            for i, tx in enumerate(data):
                sig = signatures[i] if i < len(signatures) else None
                if not sig:
                    continue
                
                if not tx or tx.get('type') != 'SWAP':
                    results[sig] = None
                    continue
                
                # Parse trade (same logic as single)
                results[sig] = self._parse_single_tx(tx)
            
            return results
            
        except Exception:
            return {sig: None for sig in signatures}
    
    def _parse_single_tx(self, tx: Dict) -> Optional[Dict]:
        """Parse a single transaction dict"""
        fee_payer = tx.get('feePayer', '')
        token_transfers = tx.get('tokenTransfers', [])
        native_transfers = tx.get('nativeTransfers', [])
        
        WSOL = "So11111111111111111111111111111111111111112"
        
        sol_in, sol_out = 0, 0
        tokens_in, tokens_out = {}, {}
        
        for transfer in token_transfers:
            mint = transfer.get('mint', '')
            amount = float(transfer.get('tokenAmount', 0))
            from_addr = transfer.get('fromUserAccount', '')
            to_addr = transfer.get('toUserAccount', '')
            
            if from_addr == fee_payer:
                if mint == WSOL:
                    sol_out += amount
                else:
                    tokens_out[mint] = tokens_out.get(mint, 0) + amount
            elif to_addr == fee_payer:
                if mint == WSOL:
                    sol_in += amount
                else:
                    tokens_in[mint] = tokens_in.get(mint, 0) + amount
        
        for transfer in native_transfers:
            amount = float(transfer.get('amount', 0)) / 1e9
            from_addr = transfer.get('fromUserAccount', '')
            to_addr = transfer.get('toUserAccount', '')
            
            if from_addr == fee_payer:
                sol_out += amount
            elif to_addr == fee_payer:
                sol_in += amount
        
        if len(tokens_in) == 1 and sol_out > 0:
            token_addr = list(tokens_in.keys())[0]
            return {'type': 'BUY', 'token_address': token_addr, 'amount': tokens_in[token_addr], 'sol_amount': sol_out}
        elif len(tokens_out) == 1 and sol_in > 0:
            token_addr = list(tokens_out.keys())[0]
            return {'type': 'SELL', 'token_address': token_addr, 'amount': tokens_out[token_addr], 'sol_amount': sol_in}
        
        return None
    
    # =========================================================================
    # TOKEN INFO METHODS
    # =========================================================================
    
    def get_token_info(self, token_address: str, force_refresh: bool = False) -> Optional[Dict]:
        """Get token info with caching"""
        # Check cache
        if not force_refresh and token_address in self._token_cache:
            cached, timestamp = self._token_cache[token_address]
            if (datetime.now() - timestamp).total_seconds() < self._cache_ttl:
                return cached
        
        self._rate_limit_dex()
        try:
            res = requests.get(
                f"{self.dex_url}/tokens/{token_address}",
                timeout=10
            )
            data = res.json()
            pairs = data.get('pairs', [])
            
            if not pairs:
                return None
            
            # Use the most liquid pair
            pair = max(pairs, key=lambda p: float(p.get('liquidity', {}).get('usd', 0) or 0))
            
            token_info = {
                'address': token_address,
                'symbol': pair.get('baseToken', {}).get('symbol', 'UNKNOWN'),
                'name': pair.get('baseToken', {}).get('name', 'Unknown'),
                'price_usd': float(pair.get('priceUsd', 0) or 0),
                'price_native': float(pair.get('priceNative', 0) or 0),
                'liquidity': float(pair.get('liquidity', {}).get('usd', 0) or 0),
                'volume_24h': float(pair.get('volume', {}).get('h24', 0) or 0),
                'market_cap': float(pair.get('marketCap', 0) or 0) or float(pair.get('fdv', 0) or 0),
                'price_change_24h': float(pair.get('priceChange', {}).get('h24', 0) or 0),
                'created_at': pair.get('pairCreatedAt'),
                'pair_address': pair.get('pairAddress'),
            }
            
            # Calculate age
            if token_info['created_at']:
                created = datetime.fromtimestamp(token_info['created_at'] / 1000)
                token_info['age_hours'] = (datetime.now() - created).total_seconds() / 3600
            else:
                token_info['age_hours'] = 0
            
            # Cache result
            self._token_cache[token_address] = (token_info, datetime.now())
            
            return token_info
            
        except Exception:
            return None
    
    def get_market_conditions(self) -> Dict:
        """Get current market conditions (SOL price, etc.)"""
        try:
            # Get SOL price from CoinGecko
            res = requests.get(
                "https://api.coingecko.com/api/v3/simple/price",
                params={"ids": "solana", "vs_currencies": "usd", "include_24hr_change": "true"},
                timeout=10
            )
            data = res.json()
            sol_data = data.get('solana', {})
            
            return {
                'sol_price_usd': sol_data.get('usd', 0),
                'sol_24h_change_pct': sol_data.get('usd_24h_change', 0),
                'timestamp': datetime.now()
            }
        except Exception:
            return {
                'sol_price_usd': 0,
                'sol_24h_change_pct': 0,
                'timestamp': datetime.now()
            }
    
    # =========================================================================
    # DISCOVERY METHODS
    # =========================================================================
    
    def get_historical_launches(self, days_ago: int = 3, limit: int = 20) -> List[Dict]:
        """Get recent token launches for discovery"""
        self._rate_limit_dex()
        try:
            res = requests.get(f"{self.dex_url}/search", params={"q": "raydium"}, timeout=10)
            all_pairs = [p for p in res.json().get('pairs', []) if p.get('chainId') == 'solana']
            
            target_min = datetime.now() - timedelta(days=days_ago + 4)
            target_max = datetime.now() - timedelta(hours=3)
            
            filtered = []
            for p in all_pairs:
                created = p.get('pairCreatedAt')
                if created:
                    created_dt = datetime.fromtimestamp(created / 1000)
                    if target_min <= created_dt <= target_max:
                        filtered.append(p)
            
            return filtered[:limit]
        except Exception:
            return []
    
    def find_swing_candidates(self, pair_address: str) -> List[str]:
        """Find potential swing traders from a token pair"""
        self._rate_limit_dex()
        try:
            res = requests.get(f"{self.dex_url}/pairs/solana/{pair_address}", timeout=10)
            pair_data = res.json().get('pair', {})
            
            if not pair_data:
                return []
            
            txns = pair_data.get('txns', {}).get('h24', {})
            if txns.get('buys', 0) < 10:
                return []
            
            # Get recent transactions to find traders
            # This would need Helius transaction history
            return []  # Placeholder - actual implementation would use Helius
            
        except Exception:
            return []
    
    def clear_cache(self):
        """Clear all caches"""
        self._token_cache.clear()


class WalletProfiler:
    """
    Profiles wallet trading performance.
    Calculates win rate, PnL, hold times, etc.
    """
    
    def __init__(self, scanner: TokenScanner):
        self.scanner = scanner
        print("  ✅ Wallet Profiler initialized")
    
    def calculate_performance(self, wallet: str, days: int = 7) -> Dict:
        """
        Calculate wallet trading performance over a period.
        
        Returns:
            Dict with win_rate, pnl, completed_swings, avg_hold_hours, etc.
        """
        signatures = self.scanner.get_recent_signatures(wallet, limit=100)
        
        if not signatures:
            return self._empty_stats()
        
        # Parse all trades
        trades = []
        for sig_info in signatures:
            sig = sig_info.get('signature')
            trade = self.scanner.get_parsed_trade(sig)
            if trade:
                trade['signature'] = sig
                trade['timestamp'] = sig_info.get('blockTime', 0)
                trades.append(trade)
        
        if len(trades) < 2:
            return self._empty_stats()
        
        # Match buys to sells
        positions = self._match_positions(trades)
        
        if not positions:
            return self._empty_stats()
        
        # Calculate metrics
        wins = sum(1 for p in positions if p['profit_pct'] > 0)
        losses = len(positions) - wins
        
        profits = [p['profit_pct'] for p in positions]
        hold_hours = [p['hold_hours'] for p in positions]
        
        avg_win = sum(p for p in profits if p > 0) / max(1, wins)
        avg_loss = abs(sum(p for p in profits if p <= 0) / max(1, losses))
        
        return {
            'win_rate': wins / len(positions) if positions else 0,
            'pnl': sum(p['profit_sol'] for p in positions),
            'roi_7d': sum(profits),
            'completed_swings': len(positions),
            'avg_hold_hours': sum(hold_hours) / len(hold_hours) if hold_hours else 0,
            'risk_reward_ratio': avg_win / avg_loss if avg_loss > 0 else 0,
            'best_trade_pct': max(profits) if profits else 0,
            'worst_trade_pct': min(profits) if profits else 0,
            'total_volume_sol': sum(p.get('sol_amount', 0) for p in positions),
            'avg_position_size_sol': sum(p.get('sol_amount', 0) for p in positions) / len(positions) if positions else 0,
        }
    
    def _match_positions(self, trades: List[Dict]) -> List[Dict]:
        """Match buy trades to sell trades to calculate positions"""
        positions = []
        open_positions = {}  # token_address -> buy_trade
        
        # Sort by timestamp
        trades.sort(key=lambda t: t.get('timestamp', 0))
        
        for trade in trades:
            token = trade['token_address']
            
            if trade['type'] == 'BUY':
                # Open new position (or add to existing)
                if token not in open_positions:
                    open_positions[token] = trade
            
            elif trade['type'] == 'SELL' and token in open_positions:
                # Close position
                buy = open_positions.pop(token)
                
                buy_price = buy.get('sol_amount', 0) / max(0.0001, buy.get('amount', 1))
                sell_price = trade.get('sol_amount', 0) / max(0.0001, trade.get('amount', 1))
                
                profit_pct = ((sell_price - buy_price) / buy_price * 100) if buy_price > 0 else 0
                profit_sol = trade.get('sol_amount', 0) - buy.get('sol_amount', 0)
                
                hold_seconds = trade.get('timestamp', 0) - buy.get('timestamp', 0)
                hold_hours = hold_seconds / 3600 if hold_seconds > 0 else 0
                
                positions.append({
                    'token': token,
                    'profit_pct': profit_pct,
                    'profit_sol': profit_sol,
                    'hold_hours': hold_hours,
                    'sol_amount': buy.get('sol_amount', 0)
                })
        
        return positions
    
    def _empty_stats(self) -> Dict:
        """Return empty stats dict"""
        return {
            'win_rate': 0,
            'pnl': 0,
            'roi_7d': 0,
            'completed_swings': 0,
            'avg_hold_hours': 0,
            'risk_reward_ratio': 0,
            'best_trade_pct': 0,
            'worst_trade_pct': 0,
            'total_volume_sol': 0,
            'avg_position_size_sol': 0,
        }


class Historian:
    """
    THE HISTORIAN - Main entry point for Phase 1 functionality.
    
    Orchestrates:
    - Wallet discovery from new token launches
    - Wallet verification and profiling
    - Real-time wallet monitoring
    - Exit signal tracking
    """
    
    def __init__(self, db, helius_key: str):
        self.db = db
        self.scanner = TokenScanner(helius_key)
        self.profiler = WalletProfiler(self.scanner)
        
        # Tracking
        self._monitored_wallets: Set[str] = set()
        self._recent_signatures: Dict[str, Set[str]] = defaultdict(set)
        
        print("\n📚 THE HISTORIAN initialized")
    
    # =========================================================================
    # DISCOVERY
    # =========================================================================
    
    def run_discovery(self, notify_callback=None) -> Dict:
        """
        Run wallet discovery cycle.
        Finds new swing traders from recent token launches.
        
        Returns:
            Dict with discovery statistics
        """
        print(f"\n{'='*60}")
        print(f"🔍 DISCOVERY MODE - {datetime.now().strftime('%H:%M:%S')}")
        print(f"{'='*60}")
        
        stats = {'tokens_scanned': 0, 'candidates_found': 0, 'wallets_verified': 0, 'wallets_rejected': 0}
        
        # Get recent launches
        launches = self.scanner.get_historical_launches(days_ago=5, limit=30)
        stats['tokens_scanned'] = len(launches)
        
        print(f"  Found {len(launches)} recent token launches")
        
        for token in launches:
            symbol = token.get('baseToken', {}).get('symbol', 'UNKNOWN')
            pair_address = token.get('pairAddress')
            
            if not pair_address:
                continue
            
            # Find swing trader candidates
            candidates = self.scanner.find_swing_candidates(pair_address)
            
            for wallet in candidates:
                if self.db.is_wallet_tracked(wallet):
                    continue
                
                stats['candidates_found'] += 1
                
                # Profile wallet
                try:
                    performance = self.profiler.calculate_performance(wallet, days=7)
                    
                    # Verification criteria
                    if (performance['win_rate'] >= 0.45 and 
                        performance['pnl'] > 0 and 
                        performance['completed_swings'] >= 3):
                        
                        self.db.add_verified_wallet(wallet, performance)
                        stats['wallets_verified'] += 1
                        
                        if notify_callback:
                            notify_callback('discovery', wallet, performance)
                    else:
                        stats['wallets_rejected'] += 1
                        
                except Exception as e:
                    if "429" in str(e) or "rate" in str(e).lower():
                        print(f"  ⚠️ Rate limited - pausing discovery")
                        break
        
        print(f"\n  ✅ Discovery complete: {stats['wallets_verified']} new wallets verified")
        return stats
    
    # =========================================================================
    # MONITORING
    # =========================================================================
    
    def monitor_wallets(self, signal_callback) -> Dict:
        """
        Monitor verified wallets for new trades.
        
        Args:
            signal_callback: Function to call with new signals
                            (wallet_data, trade, token_info) -> None
        
        Returns:
            Dict with monitoring statistics
        """
        stats = {'wallets_checked': 0, 'signals_buy': 0, 'signals_sell': 0, 'errors': 0}
        
        wallets = self.db.get_all_verified_wallets()
        stats['wallets_checked'] = len(wallets)
        
        for wallet_data in wallets:
            wallet = wallet_data['address']
            
            try:
                # Get recent signatures
                signatures = self.scanner.get_recent_signatures(wallet, limit=5)
                
                for sig_info in signatures:
                    sig = sig_info.get('signature')
                    
                    # Check if already processed
                    if self.db.is_signature_processed(sig):
                        continue
                    
                    # Mark as processed immediately (prevents duplicates)
                    if not self.db.mark_signature_processed(sig, wallet):
                        continue
                    
                    # Parse trade
                    trade = self.scanner.get_parsed_trade(sig)
                    
                    if not trade:
                        continue
                    
                    token_addr = trade['token_address']
                    
                    # Skip ignored tokens
                    if self.scanner.is_ignored_token(token_addr):
                        continue
                    
                    # Get token info
                    token_info = self.scanner.get_token_info(token_addr)
                    
                    if not token_info or token_info.get('price_usd', 0) <= 0:
                        continue
                    
                    # Update signature with trade details
                    self.db.mark_signature_processed(sig, wallet, trade['type'], token_addr)
                    
                    # Track stats
                    if trade['type'] == 'BUY':
                        stats['signals_buy'] += 1
                    else:
                        stats['signals_sell'] += 1
                    
                    # Send signal to callback
                    signal_callback(wallet_data, trade, token_info, sig)
                    
            except Exception as e:
                stats['errors'] += 1
        
        return stats
    
    # =========================================================================
    # EXIT TRACKING
    # =========================================================================
    
    def track_wallet_exits(self, open_tokens: List[str]) -> List[Dict]:
        """
        Track when verified wallets exit tokens we're watching.
        
        Args:
            open_tokens: List of token addresses we have positions in
        
        Returns:
            List of exit signals
        """
        exit_signals = []
        
        if not open_tokens:
            return exit_signals
        
        wallets = self.db.get_all_verified_wallets()
        
        for wallet_data in wallets:
            wallet = wallet_data['address']
            cluster = wallet_data.get('cluster', 'BALANCED')
            
            signatures = self.scanner.get_recent_signatures(wallet, limit=5)
            
            for sig_info in signatures:
                sig = sig_info.get('signature')
                
                if self.db.is_signature_processed(sig):
                    continue
                
                trade = self.scanner.get_parsed_trade(sig)
                
                if not trade or trade['type'] != 'SELL':
                    continue
                
                token = trade['token_address']
                
                if token not in open_tokens:
                    continue
                
                # This wallet is selling a token we're watching!
                token_info = self.scanner.get_token_info(token)
                
                exit_signal = {
                    'token_address': token,
                    'token_symbol': token_info.get('symbol', 'UNKNOWN') if token_info else 'UNKNOWN',
                    'wallet_address': wallet,
                    'wallet_cluster': cluster,
                    'exit_price': token_info.get('price_usd', 0) if token_info else 0,
                    'signature': sig,
                    'timestamp': datetime.now()
                }
                
                exit_signals.append(exit_signal)
                
                # Record in database
                self.db.add_exit_signal(
                    token, exit_signal['token_symbol'], wallet, 
                    cluster, exit_signal['exit_price'], sig
                )
        
        return exit_signals
    
    # =========================================================================
    # UTILITIES
    # =========================================================================
    
    def get_token_info(self, token_address: str) -> Optional[Dict]:
        """Get token info (passthrough to scanner)"""
        return self.scanner.get_token_info(token_address)
    
    def get_market_conditions(self) -> Dict:
        """Get market conditions (passthrough to scanner)"""
        return self.scanner.get_market_conditions()
    
    def get_wallet_count(self) -> int:
        """Get count of verified wallets"""
        return self.db.get_wallet_count()
    
    def clear_caches(self):
        """Clear all caches"""
        self.scanner.clear_cache()


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("HISTORIAN V2 - Module Test")
    print("="*60)
    
    import os
    from database_v2 import DatabaseV2
    
    helius_key = os.getenv('HELIUS_KEY')
    if not helius_key:
        print("⚠️ HELIUS_KEY not set - using mock mode")
        helius_key = "test"
    
    db = DatabaseV2("test_historian.db")
    historian = Historian(db, helius_key)
    
    print("\n✅ Historian initialized successfully")
    
    # Test market conditions
    print("\n1. Testing market conditions...")
    conditions = historian.get_market_conditions()
    print(f"   SOL Price: ${conditions.get('sol_price_usd', 0):.2f}")
    
    # Cleanup
    import os
    os.remove("test_historian.db")
