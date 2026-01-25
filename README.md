# Solana Memecoin Trading Bot

An automated trading system that discovers profitable Solana wallets, learns from their trading patterns, and develops data-driven trading strategies through paper trading before deploying to live markets.

## 🎯 Project Overview

This bot operates in four phases:

1. **Discovery** - Finds profitable traders using multiple strategies (Birdeye leaderboard, new tokens, reverse discovery)
2. **Tracking** - Monitors discovered wallets via Helius webhooks for real-time trade signals
3. **Learning** - Analyzes trade outcomes to optimize strategies and parameters
4. **Execution** - Paper trades to validate strategies before live deployment

## 📋 Prerequisites

- Python 3.9+
- Solana wallet knowledge
- API Keys:
  - **Helius API** (1M credits/month free tier) - Required
  - **Anthropic Claude API** (for LLM analysis) - Optional but recommended
  - **Birdeye API** (for discovery) - Optional

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/BryanNsoh/solana-copy-trading-bot.git
cd solana-copy-trading-bot

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

Copy `.env.example` to `.env` and fill in your API keys:

```bash
cp .env.example .env
```

Edit `.env`:

```bash
# Required - Get from https://www.helius.dev/
HELIUS_KEY=your_helius_api_key_here

# Optional - Separate key for discovery quota management
HELIUS_DISCOVERY_KEY=separate_key_for_discovery

# Optional - Get from https://public-api.birdeye.so/
BIRDEYE_API_KEY=your_birdeye_key_here

# Optional - Get from https://console.anthropic.com/
ANTHROPIC_API_KEY=your_claude_api_key_here

# Optional - For Telegram notifications
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
TELEGRAM_CHAT_ID=your_telegram_chat_id

# Required - Your webhook endpoint (update with your server IP/domain)
WEBHOOK_URL=http://your-server-ip:5000/webhook/helius

# Optional - Will be auto-created on first run
HELIUS_WEBHOOK_ID=your_webhook_id
```

### 3. Initial Setup

```bash
# Run discovery to find your first wallets
python run_discovery.py 5000  # Use 5000 API credits

# Check what was found
python run_discovery.py candidates

# Validate the system
python system_analysis.py
```

### 4. Start the Bot

```bash
# Start the main trading system
python master_v2.py
```

The system will:
- Start a webhook server on port 5000
- Run discovery every 12 hours
- Monitor tracked wallets for trading signals
- Paper trade promising opportunities
- Run learning loop every 6 hours to improve strategies

## 📊 System Architecture

### Core Components

**Master (`master_v2.py`)**
- Webhook server for receiving trade signals
- Background discovery scheduler
- Position monitoring
- Learning loop coordinator

**Discovery (`improved_discovery_v8.py`)**
- Birdeye Leaderboard - Proven profitable traders
- New Tokens (<24h) - Active swing trading opportunities  
- Reverse Discovery - Find profitable trades first, then track wallets
- Multi-source token scanning with smart pre-filtering

**Strategist (`strategist_v2.py`)**
- Signal analysis with LLM gating (only calls Claude when worthwhile)
- Multi-wallet signal aggregation
- Market regime detection
- Automatic strategy evolution and promotion
- Parameter optimization based on results

**Paper Trader (`effective_paper_trader.py`)**
- Robust position tracking with database persistence
- Automatic exit monitoring (stop loss, take profit, trailing stop, time)
- Comprehensive analytics for strategy feedback
- Trade journal for post-mortem analysis

**Database (`database_v2.py`)**
- SQLite with automatic schema migrations
- Stores wallets, positions, market data, strategies
- Thread-safe with connection pooling

### Discovery Budget

Default: 5000 Helius credits per cycle, 2 cycles/day = 10,000/day

**Monthly projection (1M credits):**
- Discovery: ~300,000 credits (30%)
- Webhook monitoring: ~225,000 credits (23%)
- **Buffer: ~475,000 credits (47%)**

Plenty of headroom for scaling!

## 🎮 Usage

### Discovery Management

```bash
# Run discovery with custom budget
python run_discovery.py 10000

# View discovery history
python run_discovery.py history

# Show last run candidates (including rejected)
python run_discovery.py candidates

# Check budget allocation
python run_discovery.py budget
```

### System Monitoring

```bash
# Get comprehensive status
curl http://localhost:5000/diagnostics

# View open positions
curl http://localhost:5000/positions

# Check recently discovered wallets
curl http://localhost:5000/new_wallets

# View learning insights
curl http://localhost:5000/learning/insights
```

### Manual Operations

```bash
# Trigger discovery immediately
curl -X POST http://localhost:5000/discovery/run

# Run learning loop
curl -X POST http://localhost:5000/learning/run

# Check discovery performance
python discovery_dashboard.py summary 30  # Last 30 days
```

### System Analysis

```bash
# Validate wallet data and model profitability
python system_analysis.py

# Check paper trading performance
python effective_paper_trader.py dashboard

# Get strategy feedback
python effective_paper_trader.py feedback
```

## 🔧 Configuration

### Discovery Settings (`discovery_config.py`)

```python
# Verification thresholds (lowered for v8)
min_win_rate: float = 0.50        # 50% (was 60%)
min_pnl_sol: float = 0.5          # 0.5 SOL (was 2.0)
min_completed_swings: int = 1     # 1 swing (was 3)

# Discovery schedule
discovery_interval_hours: int = 12  # Run every 12 hours
max_new_wallets_per_day: int = 15   # Max 15 new wallets/day
max_total_wallets: int = 150        # Total wallet limit
```

### Paper Trading (`effective_paper_trader.py`)

```python
starting_balance_sol: float = 10.0
max_open_positions: int = 5
max_position_size_sol: float = 0.5

# Exit parameters (adjusted by strategist learning loop)
default_stop_loss_pct: float = -12.0    # -12%
default_take_profit_pct: float = 30.0   # +30%
default_trailing_stop_pct: float = 8.0  # 8% from peak
max_hold_hours: int = 12
```

## 📈 Path to Profitability

Based on `system_analysis.py` modeling:

### Conservative Path (Lower Risk)
- **Capital**: 50-100 SOL ($5,000-$10,000 USD)
- **Trades/day**: 4-6
- **Win rate needed**: ≥55%
- **Avg position**: 10-20 SOL
- **Strategy**: Copy only ELITE wallets (WR≥70%)
- **Target**: $200 NZD/day (~$116 USD/day)

### Aggressive Path (Higher Risk)
- **Capital**: 20-30 SOL ($2,000-$3,000 USD)
- **Trades/day**: 8-12
- **Win rate needed**: ≥60%
- **Avg position**: 5-10 SOL
- **Strategy**: Copy all STRONG+ wallets (WR≥60%)
- **Target**: $200 NZD/day (~$116 USD/day)

### Recommended Approach

**Phase 1: Validation (Current)**
- Run continuously for 7+ days
- Collect 50+ paper trades
- Validate wallet data accuracy
- Remove underperforming wallets

**Phase 2: Optimization**
- Analyze which wallets generate profitable signals
- Tune entry/exit timing
- Implement position sizing based on wallet quality
- **Target**: 55%+ win rate on paper trades

**Phase 3: Small Live Testing**
- Start with 5 SOL capital
- Max position: 1 SOL
- Validate paper vs live performance
- **Target**: Consistent daily profit for 14+ days

**Phase 4: Scale to Target**
- Increase capital to 50-100 SOL
- Position size: 10-20 SOL
- 4-8 trades/day
- **Target**: $200 NZD/day ≈ 1.2 SOL/day (at $100/SOL)

## 🔍 Key Features

### Learning System
- **Automatic Strategy Promotion** - Challenger strategies that outperform get promoted
- **Parameter Optimization** - Learns optimal stop/take profit from results
- **Trade Outcome Analysis** - Identifies why trades fail to improve filters
- **Adaptive Thresholds** - Adjusts based on actual performance

### Discovery Features
- **Wallet-Level Validation** - Verifies trading activity before profiling (saves API credits)
- **Multi-Source Discovery** - Birdeye + New Tokens + Reverse Discovery
- **Smart Pre-Filtering** - Only profiles wallets with both buys AND sells
- **Budget Management** - Tracks API usage to stay within limits

### Paper Trading
- **Database Persistence** - Survives restarts and updates
- **Automatic Exit Monitoring** - Background thread checks positions every 30s
- **Rich Analytics** - Detailed breakdown by conviction, exit reason, token characteristics
- **Strategy A/B Testing** - Compare different approaches

## 🛠️ Troubleshooting

### No wallets being discovered?
1. Check `run_discovery.py candidates` to see rejection reasons
2. Verify API keys in `.env`
3. Consider lowering thresholds in `discovery_config.py`
4. Run `python system_analysis.py` to validate data quality

### Webhooks not receiving signals?
1. Verify `WEBHOOK_URL` is publicly accessible
2. Check webhook exists: `python helius_webhook_manager.py list`
3. Sync wallets: `python multi_webhook_manager.py sync`
4. Test endpoint: `curl http://your-server:5000/test`

### Paper trading positions not closing?
The `effective_paper_trader.py` automatically monitors positions every 30 seconds. Check:
1. Are positions showing in `curl http://localhost:5000/positions`?
2. Is the master system running?
3. Check token prices are updating (DexScreener API)

### High API usage?
1. Check `run_discovery.py budget` for projections
2. Reduce `max_api_calls_per_discovery` in `discovery_config.py`
3. Increase `discovery_interval_hours` to run less frequently
4. Monitor with `curl http://localhost:5000/diagnostics`

## 📚 File Structure

```
├── master_v2.py                    # Main system orchestrator
├── database_v2.py                  # Database layer with migrations
├── strategist_v2.py                # Strategy engine with learning loop
├── improved_discovery_v8.py        # Multi-strategy discovery system
├── discovery_config.py             # Discovery configuration & budgets
├── discovery_integration.py        # Discovery system integration
├── profiler.py                     # Wallet performance profiling
├── effective_paper_trader.py       # Robust paper trading engine
├── paper_engine_replacement.py     # Paper trading wrapper for master
├── multi_webhook_manager.py        # Scales beyond 25 wallets/webhook
├── helius_webhook_manager.py       # Individual webhook management
├── run_discovery.py                # Discovery CLI tool
├── system_analysis.py              # System validation & profitability
├── discovery_dashboard.py          # Discovery performance monitoring
├── requirements.txt                # Python dependencies
├── .env.example                    # Environment template
├── .gitignore                      # Git ignore rules
└── README.md                       # This file
```

## 🧰 Diagnostic & Utility Scripts

### Discovery Diagnostics
```bash
# Debug discovery system
python diagnose_discovery.py        # Full system diagnostic
python diagnose_prefilter.py        # Why are candidates rejected?
python debug_diagnostic.py          # Transaction parsing analysis
python debug_profiler.py <wallet>   # Deep wallet analysis

# Test specific components
python test_discovery.py            # Test discovery with small budget
python test_api.py                  # Verify DexScreener API access
```

### System Analysis
```bash
# Validate system health
python analyze_positions.py         # Analyze open positions
python analyze_positions.py close   # Close stale positions

# System-wide analysis
python system_analysis.py           # Complete profitability analysis
```

### Webhook Management
```bash
# List and manage webhooks
python helius_webhook_manager.py list
python helius_webhook_manager.py sync

# Multi-webhook operations
python multi_webhook_manager.py sync
python multi_webhook_manager.py status

# Register webhooks
python register_webhook.py list
python register_webhook.py <webhook_id>
python fix_webhooks.py              # Fix empty/broken webhooks
```

### Database & Migration
```bash
# Paper trader migration
python migrate_paper_trader.py analyze        # Analyze old positions
python migrate_paper_trader.py close-stale    # Close stale positions
python migrate_paper_trader.py all            # Full cleanup

# Manual wallet seeding
python seed_wallets_script.py                 # Add wallets manually
python seed_wallets_script.py list            # List tracked wallets
```

## 🔬 Development Scripts

These scripts are for testing and development only:

- `diagnose_issues.py` - Diagnose Birdeye & Helius API issues
- `discover_endpoints.py` - Test DexScreener API endpoints
- `debug_diagnostic.py` - Debug transaction parsing
- `debug_profiler.py` - Debug wallet profiling
- `diagnose_prefilter.py` - Analyze pre-filter rejection rates

## ⚠️ Important Notes

### DO NOT
- ❌ Run with real money until 14+ days of profitable paper trading
- ❌ Exceed Helius API limits (monitor with diagnostics endpoint)
- ❌ Delete the database files (`swing_traders.db`, `paper_trades_v3.db`)
- ❌ Commit your `.env` file to version control (it's in `.gitignore`)
- ❌ Share your API keys publicly

### DO
- ✅ Keep the system running continuously for accurate data
- ✅ Monitor paper trading results before going live
- ✅ Review learning insights regularly (`/learning/insights`)
- ✅ Back up your database files regularly
- ✅ Start small when transitioning to live trading
- ✅ Use the diagnostic scripts when troubleshooting
- ✅ Check `discovery_debug.json` after each discovery run

## 🐛 Known Issues & Solutions

### Discovery finds no wallets
**Problem**: Discovery runs but verifies 0 wallets

**Solutions**:
1. Check thresholds in `discovery_config.py` (they may be too strict)
2. Run `python diagnose_discovery.py` for full diagnostic
3. View rejected candidates: `python run_discovery.py candidates`
4. Consider lowering verification thresholds temporarily

### Webhooks not working
**Problem**: System running but no trades detected

**Solutions**:
1. Verify webhook URL is publicly accessible
2. Run `python fix_webhooks.py` to repair empty webhooks
3. Check webhook status: `python multi_webhook_manager.py status`
4. Ensure wallets are registered: `python multi_webhook_manager.py sync`

### High API usage
**Problem**: Approaching Helius API limits

**Solutions**:
1. Check budget: `python run_discovery.py budget`
2. Reduce `max_api_calls_per_discovery` in `discovery_config.py`
3. Increase `discovery_interval_hours` (run less frequently)
4. Monitor usage: `curl http://localhost:5000/diagnostics`

### Paper positions not closing
**Problem**: Positions held beyond max time

**Solutions**:
1. Verify master system is running: `curl http://localhost:5000/status`
2. Check position monitoring is enabled in `paper_engine_replacement.py`
3. Manually close stale positions: `python analyze_positions.py close`
4. Review exit conditions in `effective_paper_trader.py`

### Database errors
**Problem**: Schema or migration errors

**Solutions**:
1. Database automatically migrates on startup
2. If corrupted, backup and delete database files
3. System will recreate with correct schema
4. Restore data from backup if needed

## 📊 Monitoring & Metrics

### Real-time Monitoring
```bash
# System health check
curl http://localhost:5000/status

# Detailed diagnostics
curl http://localhost:5000/diagnostics | jq

# View open positions
curl http://localhost:5000/positions | jq

# Learning insights
curl http://localhost:5000/learning/insights | jq
```

### Discovery Metrics
```bash
# Last 7 days summary
python discovery_dashboard.py summary 7

# Last 30 days
python discovery_dashboard.py summary 30

# Export detailed report
python discovery_dashboard.py export
```

### Paper Trading Dashboard
```bash
# Current status
python effective_paper_trader.py status

# 14-day analysis
python effective_paper_trader.py analyze

# Export for strategist
python effective_paper_trader.py export
```

## 🔐 Security Best Practices

1. **API Keys**: Never commit `.env` to version control
2. **Webhook Endpoint**: Use HTTPS in production, consider authentication
3. **Database Backups**: Regularly backup database files
4. **Access Control**: Restrict access to the server running the bot
5. **Monitoring**: Set up alerts for unusual API usage or trading activity

## 🚀 Production Deployment

### Recommended Setup
1. Deploy on a VPS (DigitalOcean, AWS, etc.)
2. Use a process manager (systemd, supervisord, or PM2)
3. Set up HTTPS with reverse proxy (nginx)
4. Configure automatic restarts on failure
5. Set up monitoring and alerting
6. Enable Telegram notifications

### Example systemd Service
```ini
[Unit]
Description=Solana Copy Trading Bot
After=network.target

[Service]
Type=simple
User=your-user
WorkingDirectory=/path/to/solana-copy-trading-bot
Environment="PATH=/path/to/venv/bin"
ExecStart=/path/to/venv/bin/python master_v2.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

### Health Check Endpoint
```bash
# Add to monitoring system (e.g., UptimeRobot, Pingdom)
curl http://your-server:5000/status
```

## 📄 License

MIT License - See LICENSE file for details

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

### Development Setup
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Testing
- Use diagnostic scripts to verify changes
- Run discovery with small budgets during testing
- Ensure paper trading works before submitting

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/BryanNsoh/solana-copy-trading-bot/issues)
- **Discussions**: [GitHub Discussions](https://github.com/BryanNsoh/solana-copy-trading-bot/discussions)
- **Documentation**: See this README and inline code comments

## 🙏 Acknowledgments

- Built with [Helius](https://helius.dev) for Solana data
- LLM analysis powered by [Anthropic Claude](https://anthropic.com)
- Market data from [DexScreener](https://dexscreener.com) and [Birdeye](https://birdeye.so)

## ⚡ Performance Notes

### Current Capabilities
- Tracks up to 150 wallets simultaneously (multi-webhook scaling)
- Processes ~50-100 trade signals per day
- Discovery budget: 5,000 credits/cycle (2 cycles/day)
- Monthly API usage: ~285,000 credits (28.5% of 1M quota)
- Paper trading: Unlimited positions (database-backed)

### Optimization Tips
1. **Reduce API Calls**: Increase discovery interval, lower candidate limits
2. **Improve Win Rate**: Tighten verification thresholds, focus on elite wallets
3. **Scale Trading**: Increase `max_open_positions` and capital after validation
4. **Better Signals**: Enable LLM analysis for high-conviction trades

## 📈 Roadmap

- [ ] Live trading engine (manual approval initially)
- [ ] Web dashboard for monitoring
- [ ] Advanced strategy backtesting
- [ ] Multi-exchange support
- [ ] Portfolio optimization
- [ ] Risk management improvements
- [ ] Machine learning signal enhancement

---

**Current Status**: System validated and ready for continuous paper trading. Collect 50+ trades before considering live deployment.

**Version**: 2.0 (Learning Loop + Discovery v8)