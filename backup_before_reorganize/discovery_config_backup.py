"""
Discovery Configuration and Rate Limiting
Trading System V2 - Sustainable wallet discovery

This module defines strict limits for API usage during wallet discovery
to ensure we stay within Helius API budget for a full month.
"""

from dataclasses import dataclass


@dataclass
class DiscoveryConfig:
    """Configuration for wallet discovery with API rate limiting"""
    
    # =========================================================================
    # DISCOVERY SCHEDULE
    # =========================================================================
    discovery_interval_hours: int = 24  # Run once per day
    
    # =========================================================================
    # PER-CYCLE LIMITS (Keeps API usage low)
    # =========================================================================
    max_tokens_to_scan: int = 20  # Limit token launches to check
    max_candidates_to_profile: int = 10  # Max wallets to profile per cycle
    max_api_calls_per_discovery: int = 300  # Hard API call limit per cycle
    
    # =========================================================================
    # PRE-FILTERS (Check BEFORE profiling to save API calls)
    # =========================================================================
    min_trades_to_consider: int = 3  # Must have at least 3 trades visible
    
    # =========================================================================
    # VERIFICATION THRESHOLDS (Post-profiling)
    # =========================================================================
    min_win_rate: float = 0.50  # 50% win rate minimum
    min_pnl: float = 2.0  # At least 2 SOL profit in 7 days
    min_completed_swings: int = 3  # At least 3 complete buy→sell cycles
    min_roi_7d: float = 0.15  # 15% ROI minimum
    
    # =========================================================================
    # DAILY & TOTAL LIMITS
    # =========================================================================
    max_new_wallets_per_day: int = 5  # Don't add too many at once
    max_total_wallets: int = 50  # Total wallet cap (prevents webhook overload)
    
    # =========================================================================
    # API BUDGET ALLOCATION (Free tier = 1M credits/month)
    # =========================================================================
    # Expected daily usage:
    # - Webhooks (50 wallets × 50 trades/day): ~2,500/day
    # - Discovery (once/day): ~300/day
    # - Position checks: ~100/day (via DexScreener, free)
    # - Buffer: ~1,000/day
    # Total: ~3,900/day = ~117,000/month (well under 1M limit)
    
    def get_remaining_wallet_slots(self, current_count: int) -> int:
        """Calculate how many more wallets we can add"""
        return max(0, self.max_total_wallets - current_count)
    
    def can_run_discovery(self, current_count: int) -> bool:
        """Check if we should run discovery"""
        return current_count < self.max_total_wallets
    
    def get_max_wallets_this_cycle(self, current_count: int) -> int:
        """Get max wallets to verify in this cycle"""
        remaining = self.get_remaining_wallet_slots(current_count)
        return min(self.max_new_wallets_per_day, remaining)


# Global config instance
config = DiscoveryConfig()


# =============================================================================
# USAGE TRACKER
# =============================================================================

class DiscoveryUsageTracker:
    """Track API usage during discovery to enforce limits"""
    
    def __init__(self, budget: int):
        self.budget = budget
        self.used = 0
        self.calls_by_method = {}
    
    def record_call(self, method: str, count: int = 1):
        """Record API calls"""
        self.used += count
        self.calls_by_method[method] = self.calls_by_method.get(method, 0) + count
    
    def can_make_call(self, estimated_cost: int = 1) -> bool:
        """Check if we have budget for more calls"""
        return (self.used + estimated_cost) <= self.budget
    
    def get_remaining(self) -> int:
        """Get remaining budget"""
        return max(0, self.budget - self.used)
    
    def get_summary(self) -> dict:
        """Get usage summary"""
        return {
            'budget': self.budget,
            'used': self.used,
            'remaining': self.get_remaining(),
            'utilization_pct': (self.used / self.budget * 100) if self.budget > 0 else 0,
            'calls_by_method': self.calls_by_method
        }


if __name__ == "__main__":
    print("Discovery Configuration")
    print("="*60)
    print(f"Discovery interval: {config.discovery_interval_hours}h")
    print(f"Max API calls per cycle: {config.max_api_calls_per_discovery}")
    print(f"Max new wallets per day: {config.max_new_wallets_per_day}")
    print(f"Max total wallets: {config.max_total_wallets}")
    print(f"\nVerification thresholds:")
    print(f"  Win rate: {config.min_win_rate:.0%}")
    print(f"  PnL: {config.min_pnl} SOL")
    print(f"  Completed swings: {config.min_completed_swings}")
    
    # Test tracker
    print("\n" + "="*60)
    print("Testing Usage Tracker:")
    tracker = DiscoveryUsageTracker(300)
    tracker.record_call("get_signatures", 10)
    tracker.record_call("parse_trades", 20)
    
    summary = tracker.get_summary()
    print(f"Budget: {summary['budget']}")
    print(f"Used: {summary['used']}")
    print(f"Remaining: {summary['remaining']}")
    print(f"Utilization: {summary['utilization_pct']:.1f}%")
