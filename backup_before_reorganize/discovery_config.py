"""
Discovery Configuration - CONSERVATIVE for API Budget
Trading System V2 - Optimized to stay under 500k credits/month

REVISED BUDGET MATH (with optimized swap-based discovery):
- Monthly credits: 1,000,000
- Daily budget: 33,333 credits/day

EXPECTED DAILY USAGE:
- Webhooks (100 wallets × 50 trades/day): ~5,000/day
- Discovery (2 runs × 500 credits): ~1,000/day
- Position checks (via DexScreener): FREE
- Buffer: ~27,000/day
- TOTAL: ~6,000/day = ~180,000/month (18% utilization) ✅✅✅

This gives you MASSIVE headroom for:
- Scaling to 200+ wallets
- Running discovery more frequently
- Unexpected activity spikes
"""

from dataclasses import dataclass


@dataclass
class DiscoveryConfig:
    """Configuration for wallet discovery with CONSERVATIVE API limits"""
    
    # =========================================================================
    # DISCOVERY SCHEDULE
    # =========================================================================
    discovery_interval_hours: int = 12  # 2 runs per day
    
    # =========================================================================
    # PER-CYCLE LIMITS - OPTIMIZED for swap-based discovery
    # =========================================================================
    max_tokens_to_scan: int = 25        # Scan 25 tokens (was 25)
    max_candidates_to_profile: int = 30  # Profile up to 30 (was 30)
    max_api_calls_per_discovery: int = 1000  # REDUCED from 5000 → 1000
    
    # With optimized swap-based discovery:
    # - Extract traders: 25 tokens × 4 credits = 100 credits
    # - Profile wallets: 30 wallets × 25 credits = 750 credits
    # - Total: ~850 credits per run (well under 1000 limit)
    
    # =========================================================================
    # PRE-FILTERS (Check BEFORE profiling to save API calls)
    # =========================================================================
    min_trades_to_consider: int = 2     # At least 2 trades visible
    min_buys_required: int = 1          # Must have at least 1 buy
    min_sells_required: int = 1         # Must have at least 1 sell (CRITICAL!)
    min_volume_sol: float = 0.1         # Minimum 0.1 SOL trading volume
    
    # =========================================================================
    # VERIFICATION THRESHOLDS (Post-profiling)
    # =========================================================================
    # SLIGHTLY RELAXED to find more wallets
    min_win_rate: float = 0.48          # 48% win rate (was 50%)
    min_pnl: float = 1.5                # 1.5 SOL profit (was 2.0)
    min_completed_swings: int = 2       # 2 swings (was 3)
    min_roi_7d: float = 0.12            # 12% ROI (was 15%)
    
    # =========================================================================
    # DAILY & TOTAL LIMITS
    # =========================================================================
    max_new_wallets_per_day: int = 15   # Max 15 new wallets per day
    max_total_wallets: int = 150        # Increased to 150 (webhooks scale!)
    
    # =========================================================================
    # MONTHLY BUDGET PROJECTION
    # =========================================================================
    # With optimized discovery:
    # - Discovery: 2 runs/day × 1,000 = 2,000/day = 60,000/month
    # - Webhooks: 150 wallets × 50 trades × 1 credit = 7,500/day = 225,000/month
    # - Position checks: FREE (DexScreener)
    # - Total: ~9,500/day = 285,000/month (28.5% utilization) ✅
    #
    # You have 715,000 credits/month buffer! (~71% unused capacity)
    
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
# USAGE TRACKER (unchanged)
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
# MONTHLY BUDGET CALCULATOR - UPDATED
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
    
    # Discovery estimate (OPTIMIZED)
    discovery_runs = 24 // config.discovery_interval_hours
    discovery_per_run = config.max_api_calls_per_discovery
    discovery_daily = discovery_runs * discovery_per_run
    
    # Position checks via DexScreener (FREE)
    position_checks = 0  # FREE!
    
    total_daily = webhook_daily + discovery_daily + position_checks
    total_monthly = total_daily * days_per_month
    utilization = (total_monthly / monthly_credits) * 100
    
    print("\n" + "="*60)
    print("📊 MONTHLY BUDGET CALCULATOR - OPTIMIZED")
    print("="*60)
    print(f"\n💰 Total monthly credits: {monthly_credits:,}")
    print(f"📅 Daily budget: {daily_credits:,}")
    
    print(f"\n📈 Estimated daily usage:")
    print(f"   Webhooks ({max_wallets} wallets × 50 trades): ~{webhook_daily:,}")
    print(f"   Discovery ({discovery_runs} runs × {discovery_per_run:,}): ~{discovery_daily:,}")
    print(f"   Position checks (DexScreener): FREE ✅")
    print(f"   ─────────────────────────────────────────")
    print(f"   TOTAL DAILY: ~{total_daily:,}")
    
    print(f"\n📊 Monthly projection:")
    print(f"   Estimated monthly: ~{total_monthly:,}")
    print(f"   Budget utilization: {utilization:.1f}%")
    print(f"   Remaining buffer: ~{monthly_credits - total_monthly:,}")
    
    if utilization < 30:
        print(f"\n✅ EXCELLENT - Massive headroom for scaling!")
    elif utilization < 50:
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
    print("DISCOVERY CONFIGURATION - OPTIMIZED")
    print("="*60)
    print(f"\n🕐 Discovery interval: {config.discovery_interval_hours}h ({24//config.discovery_interval_hours} runs/day)")
    print(f"💳 Max API calls per cycle: {config.max_api_calls_per_discovery:,}")
    print(f"🆕 Max new wallets per day: {config.max_new_wallets_per_day}")
    print(f"📊 Max total wallets: {config.max_total_wallets}")
    
    print(f"\n🔍 Pre-filters:")
    print(f"   Min trades: {config.min_trades_to_consider}")
    print(f"   Min buys: {config.min_buys_required}")
    print(f"   Min sells: {config.min_sells_required} (MUST have sells!)")
    print(f"   Min volume: {config.min_volume_sol} SOL")
    
    print(f"\n✅ Verification thresholds (SLIGHTLY RELAXED):")
    print(f"   Win rate: ≥{config.min_win_rate:.0%}")
    print(f"   PnL: ≥{config.min_pnl} SOL")
    print(f"   Completed swings: ≥{config.min_completed_swings}")
    print(f"   ROI (7d): ≥{config.min_roi_7d:.0%}")
    
    # Calculate budget
    calculate_monthly_budget()
