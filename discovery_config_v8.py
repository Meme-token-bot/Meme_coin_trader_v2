"""
Discovery Configuration v8
==========================

LOWERED THRESHOLDS for better wallet discovery.

Edit these values to tune discovery behavior.
"""

from dataclasses import dataclass


@dataclass
class DiscoveryConfigV8:
    """
    Discovery configuration with LOWERED thresholds.
    
    Key changes from v7:
    - Win rate: 60% → 50%
    - PnL minimum: 2 SOL → 0.5 SOL
    - Completed swings: 3 → 1
    """
    
    # ==========================================================================
    # VERIFICATION THRESHOLDS (LOWERED)
    # ==========================================================================
    
    # Minimum win rate to verify a wallet (50% = half their trades profitable)
    min_win_rate: float = 0.50  # Was 0.60
    
    # Minimum profit in SOL over the analysis period
    min_pnl_sol: float = 0.5    # Was 2.0
    
    # Minimum completed swing trades (buy + sell of same token)
    min_completed_swings: int = 1  # Was 3
    
    # ==========================================================================
    # PRE-FILTER THRESHOLDS
    # ==========================================================================
    
    # Minimum buys to consider profiling
    min_buys: int = 1
    
    # Minimum sells to consider profiling
    min_sells: int = 1
    
    # Minimum total trades (buys + sells)
    min_total_trades: int = 2
    
    # Minimum trading volume in SOL
    min_volume_sol: float = 0.1
    
    # ==========================================================================
    # API BUDGET ALLOCATION
    # ==========================================================================
    
    # Total budget per discovery cycle
    total_budget: int = 5000
    
    # Percentage allocation for each source
    extraction_budget_pct: float = 0.20      # 20% for token extraction
    validation_budget_pct: float = 0.15      # 15% for wallet validation
    profiling_budget_pct: float = 0.35       # 35% for profiling
    birdeye_budget_pct: float = 0.15         # 15% for Birdeye leaderboard
    reverse_discovery_pct: float = 0.15      # 15% for reverse discovery
    
    # ==========================================================================
    # DISCOVERY SOURCES
    # ==========================================================================
    
    # Maximum tokens to scan per cycle
    max_tokens_to_scan: int = 15
    
    # Maximum wallets to verify per cycle
    max_wallets_per_cycle: int = 20
    
    # Focus on tokens created within this many hours
    new_token_max_age_hours: int = 24
    
    # Minimum price gain to consider a "pumping" token
    min_pump_gain_pct: float = 30
    
    # ==========================================================================
    # SCORING WEIGHTS
    # ==========================================================================
    
    # Weight for trading volume in scoring
    score_weight_volume: float = 1.0
    
    # Weight for number of trades
    score_weight_trades: float = 1.5
    
    # Weight for balanced buy/sell ratio
    score_weight_balance: float = 2.0
    
    # ==========================================================================
    # DISCOVERY TIMING
    # ==========================================================================
    
    # Hours between discovery cycles
    discovery_interval_hours: int = 12
    
    # Maximum wallets to track total
    max_total_wallets: int = 200


# Default config instance
config = DiscoveryConfigV8()


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_budget_allocation(total_budget: int = None) -> dict:
    """Get budget allocation breakdown"""
    budget = total_budget or config.total_budget
    
    return {
        'extraction': int(budget * config.extraction_budget_pct),
        'validation': int(budget * config.validation_budget_pct),
        'profiling': int(budget * config.profiling_budget_pct),
        'birdeye': int(budget * config.birdeye_budget_pct),
        'reverse_discovery': int(budget * config.reverse_discovery_pct),
        'total': budget
    }


def get_thresholds() -> dict:
    """Get current verification thresholds"""
    return {
        'win_rate': config.min_win_rate,
        'pnl_sol': config.min_pnl_sol,
        'completed_swings': config.min_completed_swings
    }


def print_config():
    """Print current configuration"""
    print("\n" + "="*60)
    print("DISCOVERY CONFIGURATION v8")
    print("="*60)
    
    print("\nVerification Thresholds (LOWERED):")
    print(f"  Win Rate: ≥{config.min_win_rate:.0%}")
    print(f"  PnL: ≥{config.min_pnl_sol} SOL")
    print(f"  Completed Swings: ≥{config.min_completed_swings}")
    
    print("\nPre-filter Thresholds:")
    print(f"  Min Buys: {config.min_buys}")
    print(f"  Min Sells: {config.min_sells}")
    print(f"  Min Volume: {config.min_volume_sol} SOL")
    
    print("\nBudget Allocation:")
    allocation = get_budget_allocation()
    for key, value in allocation.items():
        if key != 'total':
            print(f"  {key}: {value} credits")
    print(f"  TOTAL: {allocation['total']} credits")
    
    print("\nDiscovery Sources:")
    print(f"  New Token Age: <{config.new_token_max_age_hours}h")
    print(f"  Pump Threshold: >{config.min_pump_gain_pct}%")
    print(f"  Max Tokens/Cycle: {config.max_tokens_to_scan}")
    print(f"  Max Wallets/Cycle: {config.max_wallets_per_cycle}")
    
    print("="*60)


# =============================================================================
# COMPARE WITH v7
# =============================================================================

def compare_with_v7():
    """Show what changed from v7"""
    print("\n" + "="*60)
    print("CHANGES FROM v7 → v8")
    print("="*60)
    
    print("\nThreshold Changes:")
    print("  Win Rate:        60% → 50% (more lenient)")
    print("  PnL Minimum:     2.0 SOL → 0.5 SOL (4x lower)")
    print("  Completed Swings: 3 → 1 (much easier)")
    
    print("\nNew Discovery Sources:")
    print("  ✅ Birdeye Leaderboard - Proven profitable traders")
    print("  ✅ New Tokens (<24h) - Active swing trading")
    print("  ✅ Reverse Discovery - Find profitable trades first")
    
    print("\nExpected Impact:")
    print("  • More wallets should pass verification")
    print("  • Better quality candidates from proven sources")
    print("  • More active swing traders from new tokens")
    
    print("="*60)


if __name__ == "__main__":
    print_config()
    compare_with_v7()
