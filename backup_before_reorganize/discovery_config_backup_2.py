"""
Discovery Configuration and Rate Limiting
Trading System V2 - OPTIMIZED for maximum wallet discovery

Updated v2: Uses more of your 1M Helius credits/month allocation

BUDGET MATH:
- Monthly credits: 1,000,000
- Daily budget: 33,333 credits/day
- Webhook monitoring: ~5,000/day (conservative estimate)
- Discovery: ~15,000/day (allows 2-3 runs)
- Buffer: ~13,000/day
- Monthly total: ~600,000 (60% utilization - safe margin)
"""

from dataclasses import dataclass


@dataclass
class DiscoveryConfig:
    """Configuration for wallet discovery with API rate limiting"""
    
    # =========================================================================
    # DISCOVERY SCHEDULE - RUN MORE FREQUENTLY
    # =========================================================================
    discovery_interval_hours: int = 12  # Run twice per day (was 24)
    
    # =========================================================================
    # PER-CYCLE LIMITS - INCREASED FOR BETTER COVERAGE
    # =========================================================================
    max_tokens_to_scan: int = 25        # Scan more tokens (was 20)
    max_candidates_to_profile: int = 30  # Profile more candidates (was 10)
    max_api_calls_per_discovery: int = 5000  # Much higher budget per cycle (was 300)
    
    # =========================================================================
    # PRE-FILTERS (Check BEFORE profiling to save API calls)
    # =========================================================================
    min_trades_to_consider: int = 3     # Must have at least 3 trades visible
    min_buys_required: int = 1          # Must have at least 1 buy
    min_sells_required: int = 1         # Must have at least 1 sell (CRITICAL!)
    min_volume_sol: float = 0.1         # Minimum trading volume
    
    # =========================================================================
    # VERIFICATION THRESHOLDS (Post-profiling)
    # =========================================================================
    min_win_rate: float = 0.50          # 50% win rate minimum
    min_pnl: float = 2.0                # At least 2 SOL profit in 7 days
    min_completed_swings: int = 3       # At least 3 complete buy→sell cycles
    min_roi_7d: float = 0.15            # 15% ROI minimum
    
    # =========================================================================
    # DAILY & TOTAL LIMITS
    # =========================================================================
    max_new_wallets_per_day: int = 15   # Increased from 5
    max_total_wallets: int = 100        # Increased from 50 (webhook scales)
    
    # =========================================================================
    # API BUDGET ALLOCATION (1M credits/month)
    # =========================================================================
    # Expected daily usage:
    # - Webhooks (100 wallets × 50 trades/day): ~5,000/day
    # - Discovery (2 runs × 5,000): ~10,000/day  
    # - Position checks: ~500/day
    # - Buffer: ~17,500/day
    # Total: ~15,500/day = ~465,000/month (46.5% utilization - very safe)
    #
    # This leaves plenty of room for:
    # - More discovery runs if needed
    # - More wallets being monitored
    # - Unexpected spikes in activity
    
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
    
    def get_daily_budget(self) -> int:
        """Get recommended daily API budget"""
        return 33_333  # 1M / 30 days
    
    def get_discovery_budget(self) -> int:
        """Get budget per discovery run"""
        return self.max_api_calls_per_discovery


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


# =============================================================================
# MONTHLY BUDGET CALCULATOR
# =============================================================================

def calculate_monthly_budget():
    """Calculate and display monthly budget allocation"""
    
    monthly_credits = 1_000_000
    days_per_month = 30
    daily_credits = monthly_credits // days_per_month
    
    # Webhook monitoring estimate
    max_wallets = config.max_total_wallets
    trades_per_wallet_day = 50  # Conservative estimate
    webhook_daily = max_wallets * trades_per_wallet_day
    
    # Discovery estimate
    discovery_runs = 24 // config.discovery_interval_hours
    discovery_per_run = config.max_api_calls_per_discovery
    discovery_daily = discovery_runs * discovery_per_run
    
    # Position checks (via DexScreener = free)
    position_checks = 500
    
    total_daily = webhook_daily + discovery_daily + position_checks
    total_monthly = total_daily * days_per_month
    utilization = (total_monthly / monthly_credits) * 100
    
    print("\n" + "="*60)
    print("📊 MONTHLY BUDGET CALCULATOR")
    print("="*60)
    print(f"\n💰 Total monthly credits: {monthly_credits:,}")
    print(f"📅 Daily budget: {daily_credits:,}")
    
    print(f"\n📈 Estimated daily usage:")
    print(f"   Webhook monitoring ({max_wallets} wallets): ~{webhook_daily:,}")
    print(f"   Discovery ({discovery_runs} runs × {discovery_per_run:,}): ~{discovery_daily:,}")
    print(f"   Position checks: ~{position_checks:,}")
    print(f"   ─────────────────────────────────")
    print(f"   TOTAL DAILY: ~{total_daily:,}")
    
    print(f"\n📊 Monthly projection:")
    print(f"   Estimated monthly: ~{total_monthly:,}")
    print(f"   Budget utilization: {utilization:.1f}%")
    print(f"   Remaining buffer: ~{monthly_credits - total_monthly:,}")
    
    if utilization < 50:
        print(f"\n✅ VERY SAFE - Plenty of room for more activity")
    elif utilization < 70:
        print(f"\n✅ SAFE - Good utilization with buffer")
    elif utilization < 90:
        print(f"\n⚠️ MODERATE - Monitor usage closely")
    else:
        print(f"\n🚨 HIGH - Risk of exceeding budget")
    
    print("="*60 + "\n")
    
    return {
        'monthly_credits': monthly_credits,
        'daily_budget': daily_credits,
        'estimated_daily': total_daily,
        'estimated_monthly': total_monthly,
        'utilization_pct': utilization
    }


if __name__ == "__main__":
    print("\n" + "="*60)
    print("DISCOVERY CONFIGURATION v2")
    print("="*60)
    print(f"\n🕐 Discovery interval: {config.discovery_interval_hours}h")
    print(f"💳 Max API calls per cycle: {config.max_api_calls_per_discovery:,}")
    print(f"🆕 Max new wallets per day: {config.max_new_wallets_per_day}")
    print(f"📊 Max total wallets: {config.max_total_wallets}")
    
    print(f"\n🔍 Pre-filters:")
    print(f"   Min trades to consider: {config.min_trades_to_consider}")
    print(f"   Min buys required: {config.min_buys_required}")
    print(f"   Min sells required: {config.min_sells_required}")
    
    print(f"\n✅ Verification thresholds:")
    print(f"   Win rate: ≥{config.min_win_rate:.0%}")
    print(f"   PnL: ≥{config.min_pnl} SOL")
    print(f"   Completed swings: ≥{config.min_completed_swings}")
    
    # Calculate budget
    calculate_monthly_budget()
