"""
Discovery Integration v8
========================

Integrates the new discovery system with the existing historian and master.

Usage in master_v2.py:
    from core.discovery_integration import run_improved_discovery
    
    # In main loop:
    if should_run_discovery:
        stats = run_improved_discovery(db, webhook_manager, notifier)
"""

import os
from datetime import datetime
from typing import Dict, Optional

from core.improved_discovery_v8 import ImprovedDiscoveryV8, config as discovery_config


def run_improved_discovery(
    db,
    webhook_manager=None,
    notifier=None,
    api_budget: int = 5000,
    max_wallets: int = 20
) -> Dict:
    """
    Run improved discovery and integrate results.
    
    Args:
        db: Database instance
        webhook_manager: MultiWebhookManager for auto-adding wallets
        notifier: TelegramNotifier for alerts
        api_budget: Helius API credit budget for this cycle
        max_wallets: Maximum wallets to verify this cycle
    
    Returns:
        Discovery statistics
    """
    print(f"\n{'='*70}")
    print("🔍 RUNNING IMPROVED DISCOVERY v8")
    print(f"   Budget: {api_budget} credits")
    print(f"   Max wallets: {max_wallets}")
    print(f"   Thresholds: WR≥{discovery_config.min_win_rate:.0%} | PnL≥{discovery_config.min_pnl_sol} SOL | Swings≥{discovery_config.min_completed_swings}")
    print(f"{'='*70}")
    
    # Initialize discovery
    helius_key = os.getenv('HELIUS_KEY')
    birdeye_key = os.getenv('BIRDEYE_API_KEY')
    
    discovery = ImprovedDiscoveryV8(
        db=db,
        helius_key=helius_key,
        birdeye_key=birdeye_key
    )
    
    # Run discovery
    stats = discovery.run_discovery(
        api_budget=api_budget,
        max_wallets=max_wallets
    )
    
    # Auto-add to webhooks
    if webhook_manager and stats.get('verified_wallets'):
        new_addresses = [w['address'] for w in stats['verified_wallets']]
        
        print(f"\n🔄 Auto-adding {len(new_addresses)} wallet(s) to webhook system...")
        
        webhook_stats = webhook_manager.add_wallets_batch(new_addresses)
        
        stats['wallets_added_to_webhook'] = webhook_stats['added']
        stats['wallets_failed_webhook'] = webhook_stats['failed']
        
        if webhook_stats['added'] > 0:
            print(f"   ✅ {webhook_stats['added']} wallet(s) added!")
            
            status = webhook_manager.get_status()
            print(f"   📊 System: {status['active_webhooks']} webhook(s), "
                  f"{status['total_wallets_assigned']} total wallets")
        
        if webhook_stats['failed'] > 0:
            print(f"   ⚠️ {webhook_stats['failed']} wallet(s) failed - run sync to retry")
    
    # Send notification
    if notifier and stats.get('verified', 0) > 0:
        _send_discovery_notification(notifier, stats)
    
    return stats


def _send_discovery_notification(notifier, stats: Dict):
    """Send Telegram notification about discovery results"""
    verified = stats.get('verified_wallets', [])
    
    if not verified:
        return
    
    message = f"""🎯 <b>Discovery v8 Complete</b>

📊 <b>Results:</b>
• Candidates: {stats.get('candidates_found', 0)}
• From Birdeye: {stats.get('from_birdeye', 0)}
• From New Tokens: {stats.get('from_new_tokens', 0)}
• From Reverse Discovery: {stats.get('from_reverse', 0)}
• Verified: {stats.get('verified', 0)} ✅

💰 <b>New Wallets:</b>"""
    
    for w in verified[:5]:
        message += f"""
• <code>{w['address'][:16]}...</code>
  WR: {w['win_rate']:.0%} | PnL: {w['pnl_sol']:.2f} SOL | Source: {w['source']}"""
    
    if len(verified) > 5:
        message += f"\n... and {len(verified) - 5} more"
    
    message += f"""

🔧 API Credits: {stats.get('api_credits_used', 0)}/{5000}"""
    
    try:
        notifier.send(message)
    except Exception as e:
        print(f"   ⚠️ Notification error: {e}")


# =============================================================================
# REPLACEMENT FOR HISTORIAN.run_discovery()
# =============================================================================

class TokenScannerStub:
    """
    Stub for TokenScanner compatibility.
    Provides is_ignored_token() and get_token_info() methods that master_v2.py expects.
    """
    
    def __init__(self, db, helius_key: str):
        self.db = db
        self.helius_key = helius_key
        self.ignored_tokens = set()
        self.dex_url = "https://api.dexscreener.com/latest/dex"
        self._token_cache = {}
        self._cache_ttl = 300  # 5 minutes
        self._last_dex_call = 0
        self._dex_delay = 0.5
        
        # Load ignored tokens from database if available
        try:
            # Common tokens to ignore (stables, wrapped SOL, etc.)
            self.ignored_tokens = {
                "So11111111111111111111111111111111111111112",   # WSOL
                "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v", # USDC
                "Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB", # USDT
            }
        except:
            pass
    
    def _rate_limit_dex(self):
        import time
        elapsed = time.time() - self._last_dex_call
        if elapsed < self._dex_delay:
            time.sleep(self._dex_delay - elapsed)
        self._last_dex_call = time.time()
    
    def is_ignored_token(self, token_address: str) -> bool:
        """Check if token should be ignored"""
        return token_address in self.ignored_tokens
    
    def add_ignored_token(self, token_address: str):
        """Add token to ignore list"""
        self.ignored_tokens.add(token_address)
    
    def get_token_info(self, token_address: str, force_refresh: bool = False) -> Optional[Dict]:
        """Get token info from DexScreener"""
        import requests
        
        if not force_refresh and token_address in self._token_cache:
            cached, timestamp = self._token_cache[token_address]
            if (datetime.now() - timestamp).total_seconds() < self._cache_ttl:
                return cached
        
        self._rate_limit_dex()
        try:
            res = requests.get(f"{self.dex_url}/tokens/{token_address}", timeout=10)
            data = res.json()
            pairs = data.get('pairs', [])
            
            if not pairs:
                return None
            
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
            
            if token_info['created_at']:
                created = datetime.fromtimestamp(token_info['created_at'] / 1000)
                token_info['age_hours'] = (datetime.now() - created).total_seconds() / 3600
            else:
                token_info['age_hours'] = 0
            
            self._token_cache[token_address] = (token_info, datetime.now())
            return token_info
        except:
            return None


class HistorianV8:
    """
    Updated Historian with v8 discovery.
    
    Drop-in replacement for the Historian class.
    """
    
    def __init__(self, db, helius_key: str, discovery_key: str = None, 
                 multi_webhook_manager=None):
        self.db = db
        self.helius_key = helius_key
        self.multi_webhook_manager = multi_webhook_manager
        
        # Scanner for is_ignored_token() compatibility
        self.scanner = TokenScannerStub(db, helius_key)
        
        # Initialize v8 discovery
        birdeye_key = os.getenv('BIRDEYE_API_KEY')
        self.discovery = ImprovedDiscoveryV8(
            db=db,
            helius_key=discovery_key or helius_key,
            birdeye_key=birdeye_key
        )
        
        print("📚 HISTORIAN v8 initialized (with improved discovery)")
    
    def run_discovery(self, notify_callback=None, max_wallets: int = 20, 
                      api_budget: int = 5000) -> Dict:
        """
        Run v8 discovery.
        
        Compatible with existing Historian interface.
        """
        stats = self.discovery.run_discovery(
            api_budget=api_budget,
            max_wallets=max_wallets
        )
        
        # Auto-add to webhooks
        if self.multi_webhook_manager and stats.get('verified_wallets'):
            new_addresses = [w['address'] for w in stats['verified_wallets']]
            
            print(f"\n🔄 Auto-adding {len(new_addresses)} wallet(s) to webhook system...")
            
            webhook_stats = self.multi_webhook_manager.add_wallets_batch(new_addresses)
            
            stats['wallets_added_to_webhook'] = webhook_stats['added']
            stats['wallets_failed_webhook'] = webhook_stats['failed']
            
            # Notify for each wallet
            if notify_callback:
                for wallet in stats['verified_wallets']:
                    notify_callback('discovery', wallet['address'], wallet)
        
        # Map to expected stats format
        return {
            'tokens_discovered': 0,
            'wallet_candidates_found': stats.get('candidates_found', 0),
            'wallets_prefiltered_out': 0,
            'wallets_profiled': stats.get('profiled', 0),
            'wallets_verified': stats.get('verified', 0),
            'wallets_rejected': stats.get('profiled', 0) - stats.get('verified', 0),
            'helius_api_calls': stats.get('api_credits_used', 0),
            'verified_wallets': stats.get('verified_wallets', []),
            'wallets_added_to_webhook': stats.get('wallets_added_to_webhook', 0),
            'wallets_failed_webhook': stats.get('wallets_failed_webhook', 0)
        }


# =============================================================================
# QUICK TEST
# =============================================================================

if __name__ == "__main__":
    from core.database_v2 import DatabaseV2
    
    print("\n" + "="*70)
    print("🧪 TESTING DISCOVERY INTEGRATION")
    print("="*70)
    
    db = DatabaseV2()
    
    # Test with small budget
    stats = run_improved_discovery(
        db=db,
        webhook_manager=None,
        notifier=None,
        api_budget=500,
        max_wallets=5
    )
    
    print(f"\n✅ Test complete!")
    print(f"   Verified: {stats.get('verified', 0)} wallets")
