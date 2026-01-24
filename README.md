# Trading System V2 - Optimized Memecoin Trading Bot

## Quick Start

```bash
# Install dependencies
pip install anthropic requests python-dotenv numpy

# Set environment variables
export HELIUS_KEY="your_helius_api_key"
export ANTHROPIC_API_KEY="your_anthropic_api_key"  # Optional
export TELEGRAM_BOT_TOKEN="your_telegram_token"    # Optional
export TELEGRAM_CHAT_ID="your_chat_id"             # Optional

# Run
python master_v2.py
```

## Key Improvements Over V1

### 1. SMART LLM GATING (Saves $$$!)

**Problem Solved:** V1 called LLM for every signal → $2.50/day wasted

**Solution:** Only call LLM when:
- Pre-filters pass (liquidity, volume, age)
- Base conviction score >= 50
- Signal has a chance of being approved

```
V1: Signal → LLM Call → Filter → Decision
V2: Signal → Filter → Base Score → IF score >= 50 THEN LLM Call → Decision
```

### 2. Database Schema Migration (Fixes your error!)

The `market_cap_entry` error is fixed. V2 includes automatic schema migration that adds all missing columns.

### 3. Exit Signal Tracking

New feature: Track when successful wallets EXIT tokens you're holding.
- 2+ SNIPERs exit → Full exit signal
- 3+ any wallets exit → Full exit signal

### 4. Regime-Adaptive Exits

Exit parameters auto-adjust to market conditions:
- Bull market: Wider stops, higher targets
- Bear market: Tighter stops, lower targets

### 5. Unified Architecture

```
master_v2.py          # Main orchestrator (replaces master.py)
├── database_v2.py    # Unified DB with migrations
├── historian.py      # Wallet discovery & monitoring (replaces scanner.py + profiler.py)
├── strategist_v2.py  # Signal analysis (replaces strategist.py + all sub-modules)
└── utils.py          # Helpers
```

## Configuration

### Entry Thresholds (strategist_v2.py)
```python
min_liquidity: float = 30000          # $30k minimum
min_volume_24h: float = 15000         # $15k minimum  
single_wallet_min_conviction: int = 75 # High bar for 1 wallet
aggregated_min_conviction: int = 60    # Lower bar for multi-wallet
llm_conviction_threshold: int = 50     # Must score 50+ for LLM
```

### Timing (master_v2.py)
```python
monitoring_interval: int = 120         # 2 minutes
discovery_interval: int = 86400        # 24 hours
```

## LLM Cost Tracking

```python
# View costs
strategist.get_llm_cost_today()
# {'calls': 5, 'cost_usd': 0.0123}

# Disable LLM entirely
CONFIG.use_llm = False
```

## Files

| File | Lines | Purpose |
|------|-------|---------|
| database_v2.py | ~600 | Unified DB with schema migrations |
| historian.py | ~400 | Wallet discovery & monitoring |
| strategist_v2.py | ~550 | Signal analysis with smart LLM gating |
| master_v2.py | ~500 | Main orchestrator |
| utils.py | ~100 | Utilities |

**Total: ~2,150 lines** (vs ~3,500+ lines in V1)
