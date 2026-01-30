# The Executioner - Live Trading Module for Solana Memecoin Bot

## Overview

The Executioner is the live trading component that executes real trades based on signals from the Strategist. It includes comprehensive transaction logging designed for **New Zealand tax compliance**.

## Components

### 1. `executioner_v1.py` - Core Execution Engine

The main module that handles:
- **Jupiter DEX Integration** - Routes trades through Jupiter for best prices
- **Wallet Management** - Secure private key handling
- **Tax Record Database** - SQLite database with all NZ-required fields
- **FIFO Cost Basis Tracking** - Automatic cost basis calculation for capital gains
- **Position Management** - Track open positions with exit conditions

### 2. `executioner_integration.py` - Master Integration Bridge

Connects the Executioner to your existing `master_v2.py`:
- **Gradual Rollout** - Configure % of trades that go live
- **Paper Trading Validation** - Requires profitable paper trading before live
- **Safety Controls** - Daily limits, cool down periods, emergency stop
- **Background Monitoring** - Auto-executes exit conditions

## Installation

```bash
# Install Solana dependencies
pip install solana solders requests --break-system-packages

# Copy files to your project
cp executioner_v1.py /path/to/your/project/core/
cp executioner_integration.py /path/to/your/project/core/
```

## Environment Variables

Add to your `.env`:

```bash
# REQUIRED for live trading
SOLANA_PRIVATE_KEY=your_base58_private_key_here

# Existing keys (you should already have these)
HELIUS_KEY=your_helius_api_key

# OPTIONAL - Live trading controls
ENABLE_LIVE_TRADING=false          # Set to 'true' when ready
MAX_POSITION_SOL=0.25              # Max SOL per trade
MIN_CONVICTION=60                  # Minimum conviction score
LIVE_TRADE_PCT=0                   # 0-100, % of signals to trade live
```

## NZ Tax Compliance

### What Gets Recorded

Every transaction includes:

| Field | Description |
|-------|-------------|
| `timestamp` | UTC timestamp of trade |
| `transaction_type` | BUY, SELL, FEE |
| `token_symbol` | Token ticker (e.g., BONK) |
| `token_amount` | Quantity of tokens |
| `sol_amount` | SOL spent/received |
| `price_per_token_nzd` | NZD value at trade time |
| `total_value_nzd` | Total transaction value in NZD |
| `fee_nzd` | Transaction fees in NZD |
| `cost_basis_nzd` | (For sells) Original cost in NZD |
| `gain_loss_nzd` | (For sells) Capital gain/loss |
| `signature` | Solana transaction signature |

### Tax Year Handling

New Zealand's tax year runs **April 1 - March 31**. The system automatically:
- Groups transactions by NZ tax year
- Calculates annual summaries
- Exports in IRD-friendly format

### Export Tax Records

```bash
# Export current tax year
python executioner_v1.py export

# Export specific year
python executioner_v1.py export --year 2024-2025

# View summary
python executioner_v1.py tax --year 2024-2025
```

This creates a CSV with:
- All transactions for the tax year
- Summary section at bottom
- Ready for your accountant or IRD submission

### FIFO Cost Basis

The system uses **First In, First Out (FIFO)** for cost basis:
- Each buy creates a "lot" with acquisition date and cost
- Sells consume lots in order of acquisition
- Automatically tracks partial lot consumption

## Usage

### Basic Integration

```python
from core.executioner_v1 import Executioner, ExecutionConfig
from core.executioner_integration import ExecutionerBridge, IntegrationConfig

# Initialize Executioner
exec_config = ExecutionConfig(
    enable_live_trading=False,  # Start with paper
    max_position_size_sol=0.25,
    min_conviction_score=60
)

executioner = Executioner(
    private_key=os.getenv('SOLANA_PRIVATE_KEY'),
    config=exec_config
)

# Create bridge to existing system
bridge = ExecutionerBridge(
    executioner=executioner,
    paper_trader=your_paper_trader,
    db=your_database
)
```

### Gradual Rollout

The safe path to live trading:

```python
# Step 1: Paper trade only (default)
bridge.set_live_percentage(0)

# Step 2: After 14+ days profitable paper trading
# Start with 10% of signals going live
bridge.set_live_percentage(10)

# Step 3: Gradually increase as confidence grows
bridge.set_live_percentage(25)
bridge.set_live_percentage(50)
bridge.set_live_percentage(100)
```

### Process Signals

In your `master_v2.py` webhook handler:

```python
# Instead of just paper trading:
# result = paper_trader.process_signal(signal, wallet_data)

# Use the bridge:
result = bridge.process_signal(signal, wallet_data)
# Returns: {
#   'paper_result': {...},  # Paper trade result
#   'live_result': {...},   # Live trade result (if executed)
#   'execution_path': 'paper' or 'both',
#   'reason': 'why live was/wasn't executed'
# }
```

### Manual Operations

```bash
# Check status
python executioner_v1.py status

# View open positions
python executioner_v1.py positions

# Exit a position
python executioner_v1.py exit --token TOKEN_ADDRESS

# Tax summary
python executioner_v1.py tax
```

## Safety Features

### Automatic Protections

| Feature | Default | Purpose |
|---------|---------|---------|
| Max Position Size | 0.5 SOL | Limit per-trade risk |
| Max Open Positions | 10 | Portfolio diversification |
| Max Portfolio % | 20% | No single token dominance |
| Min Liquidity | $10,000 | Avoid illiquid tokens |
| Slippage Protection | 1% | Limit price impact |
| Daily Loss Limit | 1 SOL | Stop trading after losses |
| Consecutive Loss Limit | 3 | Cool down after losing streak |

### Paper Trading Validation

Before live trading is allowed:
- Minimum 14 days of paper trading
- Minimum 50 paper trades
- Minimum 55% win rate
- Positive overall PnL

### Emergency Stop

```python
# Immediately halt all live trading and close positions
bridge.emergency_stop()
```

## Database Schema

### `tax_transactions` - All trades

```sql
id, timestamp, transaction_type, token_address, token_symbol,
token_amount, sol_amount, price_per_token_nzd, total_value_nzd,
fee_nzd, cost_basis_nzd, gain_loss_nzd, signature, notes
```

### `cost_basis_lots` - FIFO tracking

```sql
id, token_address, acquisition_date, tokens_acquired, tokens_remaining,
cost_per_token_nzd, total_cost_nzd, acquisition_signature
```

### `live_positions` - Current holdings

```sql
token_address, token_symbol, tokens_held, entry_price_usd,
total_cost_nzd, stop_loss_pct, take_profit_pct, trailing_stop_pct
```

### `tax_year_summary` - Annual totals

```sql
tax_year, total_acquisitions_nzd, total_disposals_nzd,
total_gains_nzd, total_losses_nzd, net_gain_loss_nzd, total_fees_nzd
```

## Recommended Workflow

1. **Week 1-2**: Run paper trading, monitor performance
2. **Week 3**: If profitable, enable 10% live trading
3. **Week 4+**: Gradually increase based on results
4. **Monthly**: Export and review tax records
5. **End of Tax Year**: Generate final export for accountant

## Important Notes

⚠️ **DO NOT**:
- Enable live trading without paper validation
- Share your private key
- Trade with money you can't afford to lose
- Ignore tax obligations

✅ **DO**:
- Start with small positions
- Monitor daily performance
- Keep regular tax record exports
- Test thoroughly in paper mode first

## Support

For issues specific to the Executioner:
1. Check logs in `execution_log` table
2. Verify wallet balance with `python executioner_v1.py status`
3. Confirm private key is loaded correctly
