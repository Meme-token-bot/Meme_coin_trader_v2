"""
MASTER V2 LIVE TRADING INTEGRATION
===================================

Add these changes to your master_v2.py to enable live trading.

This file shows the modifications needed - copy the relevant sections
into your existing master_v2.py.
"""

# =============================================================================
# 1. ADD THESE IMPORTS AT THE TOP
# =============================================================================

IMPORTS_TO_ADD = """
# Live trading integration
try:
    from core.hybrid_trading_engine import HybridTradingEngine, HybridTradingConfig, create_hybrid_engine
    HYBRID_ENGINE_AVAILABLE = True
except ImportError:
    HYBRID_ENGINE_AVAILABLE = False

try:
    from core.live_trading_engine import LiveTradingEngine, LiveTradingConfig
    LIVE_ENGINE_AVAILABLE = True
except ImportError:
    LIVE_ENGINE_AVAILABLE = False
"""


# =============================================================================
# 2. ADD THESE CONFIG OPTIONS
# =============================================================================

CONFIG_ADDITIONS = """
# In your CONFIG dataclass, add:

    # Live trading
    enable_live_trading: bool = False      # Enable real trades
    position_size_sol: float = 0.08        # SOL per trade
    max_open_live_positions: int = 10      # Max live positions
    max_daily_loss_sol: float = 0.25       # Daily stop
    blocked_hours_utc: List[int] = field(default_factory=lambda: [1, 3, 5, 19, 23])
"""


# =============================================================================
# 3. UPDATE TradingSystem.__init__() - Add hybrid engine
# =============================================================================

INIT_UPDATE = """
# In TradingSystem.__init__(), after paper_engine initialization:

        # Initialize hybrid trading engine (paper + live)
        self.hybrid_engine = None
        
        if HYBRID_ENGINE_AVAILABLE:
            try:
                self.hybrid_engine = create_hybrid_engine(
                    paper_engine=self.paper_engine,
                    notifier=self.notifier
                )
                
                # Start live position monitoring if enabled
                if self.hybrid_engine.config.enable_live_trading:
                    if self.hybrid_engine.live_engine:
                        self.hybrid_engine.live_engine.start_monitoring()
                        print("🔴 LIVE TRADING ENABLED - Real money at risk!")
                        
            except Exception as e:
                print(f"⚠️ Hybrid engine init failed: {e}")
                self.hybrid_engine = None
"""


# =============================================================================
# 4. UPDATE _process_buy() - Route through hybrid engine
# =============================================================================

PROCESS_BUY_UPDATE = """
# Replace the paper trading call in _process_buy() with:

        # Route through hybrid engine if available
        if self.hybrid_engine:
            # Build signal
            signal_data = {
                'token_address': token_out,
                'token_symbol': token_symbol,
                'price_usd': price_usd,
                'conviction_score': conviction,
                'liquidity_usd': liquidity_usd,
                'volume_24h': volume_24h or 0
            }
            
            wallet_data = {
                'address': wallet_address,
                'win_rate': wallet_info.get('win_rate', 0.5) if wallet_info else 0.5
            }
            
            # Process through hybrid (paper + live)
            hybrid_result = self.hybrid_engine.process_signal(signal_data, wallet_data)
            
            if hybrid_result.get('filter_passed'):
                if hybrid_result.get('live_result', {}).get('success'):
                    print(f"🔴 LIVE: {token_symbol} | {self.hybrid_engine.config.position_size_sol} SOL")
                
                if hybrid_result.get('paper_result', {}).get('success'):
                    print(f"📝 Paper: {token_symbol}")
            else:
                print(f"⏭️ Filtered: {token_symbol} | {hybrid_result.get('filter_reason')}")
                
            return hybrid_result
        
        # Fallback to paper-only if hybrid not available
        # ... existing paper trading code ...
"""


# =============================================================================
# 5. ADD NEW ENDPOINTS
# =============================================================================

NEW_ENDPOINTS = '''
@app.route('/live/status', methods=['GET'])
def live_status():
    """Get live trading status"""
    global trading_system
    
    if not trading_system or not trading_system.hybrid_engine:
        return jsonify({'error': 'Hybrid engine not available'}), 503
    
    return jsonify(trading_system.hybrid_engine.get_status())


@app.route('/live/enable', methods=['POST'])
def enable_live():
    """Enable live trading (requires confirmation)"""
    global trading_system
    
    confirm = request.args.get('confirm', '')
    if confirm != 'LIVE':
        return jsonify({
            'error': 'Confirmation required',
            'message': 'Add ?confirm=LIVE to enable live trading'
        }), 400
    
    if trading_system and trading_system.hybrid_engine:
        trading_system.hybrid_engine.config.enable_live_trading = True
        trading_system.hybrid_engine.mode = TradingMode.HYBRID
        return jsonify({
            'success': True,
            'message': '🔴 LIVE TRADING ENABLED',
            'status': trading_system.hybrid_engine.get_status()
        })
    
    return jsonify({'error': 'Hybrid engine not available'}), 503


@app.route('/live/disable', methods=['POST'])
def disable_live():
    """Disable live trading"""
    global trading_system
    
    if trading_system and trading_system.hybrid_engine:
        trading_system.hybrid_engine.config.enable_live_trading = False
        trading_system.hybrid_engine.mode = TradingMode.PAPER_ONLY
        return jsonify({
            'success': True,
            'message': 'Live trading disabled',
            'status': trading_system.hybrid_engine.get_status()
        })
    
    return jsonify({'error': 'Hybrid engine not available'}), 503


@app.route('/live/kill', methods=['POST'])
def kill_switch():
    """Emergency kill switch - close all positions"""
    global trading_system
    
    confirm = request.args.get('confirm', '')
    if confirm != 'KILL':
        return jsonify({
            'error': 'Confirmation required',
            'message': 'Add ?confirm=KILL to activate kill switch'
        }), 400
    
    if trading_system and trading_system.hybrid_engine:
        trading_system.hybrid_engine._activate_kill_switch('Manual activation')
        return jsonify({
            'success': True,
            'message': '🚨 KILL SWITCH ACTIVATED - All positions closed'
        })
    
    return jsonify({'error': 'Hybrid engine not available'}), 503


@app.route('/live/positions', methods=['GET'])
def live_positions():
    """Get live positions"""
    global trading_system
    
    if not trading_system or not trading_system.hybrid_engine:
        return jsonify({'error': 'Hybrid engine not available'}), 503
    
    if not trading_system.hybrid_engine.live_engine:
        return jsonify({'error': 'Live engine not available'}), 503
    
    positions = trading_system.hybrid_engine.live_engine.get_open_positions()
    return jsonify({
        'count': len(positions),
        'positions': positions
    })
'''


# =============================================================================
# 6. UPDATE BACKGROUND TASKS - Add live monitoring
# =============================================================================

BACKGROUND_UPDATE = """
# In background_tasks(), add live position monitoring:

            # Check live position exits (every 30 seconds via live engine)
            # This is handled by live_engine.start_monitoring() but we can also force check
            if trading_system.hybrid_engine and trading_system.hybrid_engine.live_engine:
                if trading_system.hybrid_engine.config.enable_live_trading:
                    try:
                        exits = trading_system.hybrid_engine.live_engine.check_exit_conditions()
                        for exit in exits:
                            if exit.get('success'):
                                # Record exit for daily stats
                                pnl = exit.get('pnl_sol', 0)
                                is_win = pnl > 0
                                trading_system.hybrid_engine.record_exit(True, pnl, is_win)
                                
                                # Notify
                                if trading_system.notifier:
                                    emoji = "✅" if is_win else "❌"
                                    trading_system.notifier.send(
                                        f"{emoji} LIVE EXIT: {exit.get('token_symbol')}\\n"
                                        f"Reason: {exit.get('exit_reason')}\\n"
                                        f"PnL: {pnl:+.4f} SOL ({exit.get('pnl_pct', 0):+.1f}%)"
                                    )
                    except Exception as e:
                        print(f"Live exit check error: {e}")
"""


# =============================================================================
# 7. STARTUP CHANGES
# =============================================================================

STARTUP_UPDATE = """
# Update the startup prints:

        if trading_system.hybrid_engine:
            mode = trading_system.hybrid_engine.mode.value
            live_status = "🔴 LIVE" if trading_system.hybrid_engine.config.enable_live_trading else "📝 Paper Only"
            print(f"   Mode: {mode} ({live_status})")
            print(f"   Position Size: {trading_system.hybrid_engine.config.position_size_sol} SOL")
            print(f"   Max Positions: {trading_system.hybrid_engine.config.max_open_positions}")
            
            if trading_system.hybrid_engine.config.enable_live_trading:
                print("\\n   ⚠️  WARNING: LIVE TRADING ENABLED - REAL MONEY AT RISK!")
                print("   ⚠️  Use /live/kill?confirm=KILL for emergency stop")
"""


# =============================================================================
# FULL EXAMPLE: Updated _process_buy method
# =============================================================================

def example_process_buy(self, tx_data: Dict, token_data: Dict, wallet_info: Dict):
    """
    Example of updated _process_buy that routes through hybrid engine.
    
    This shows the complete logic flow.
    """
    # Extract signal data
    token_out = tx_data.get('token_out', '')
    token_symbol = token_data.get('symbol', 'UNKNOWN')
    price_usd = token_data.get('price_usd', 0)
    liquidity_usd = token_data.get('liquidity_usd', 0)
    volume_24h = token_data.get('volume_24h', 0)
    wallet_address = tx_data.get('wallet_address', '')
    
    # Calculate conviction (your existing logic)
    conviction = self._calculate_conviction(wallet_info, token_data)
    
    # Build signal for hybrid engine
    signal_data = {
        'token_address': token_out,
        'token_symbol': token_symbol,
        'price_usd': price_usd,
        'conviction_score': conviction,
        'liquidity_usd': liquidity_usd,
        'volume_24h': volume_24h
    }
    
    wallet_data = {
        'address': wallet_address,
        'win_rate': wallet_info.get('win_rate', 0.5) if wallet_info else 0.5
    }
    
    # Route through hybrid engine
    if self.hybrid_engine:
        result = self.hybrid_engine.process_signal(signal_data, wallet_data)
        
        # Log results
        if result.get('filter_passed'):
            live_result = result.get('live_result', {})
            paper_result = result.get('paper_result', {})
            
            if live_result.get('success'):
                self.diagnostics.live_trades_opened += 1
                print(f"🔴 LIVE BUY: {token_symbol} | {self.hybrid_engine.config.position_size_sol} SOL")
            
            if paper_result.get('success'):
                self.diagnostics.positions_opened += 1
        else:
            # Signal was filtered
            self.diagnostics.signals_filtered += 1
        
        return result
    
    # Fallback to paper-only
    return self._process_buy_paper_only(signal_data, wallet_data)


# =============================================================================
# PRINT SUMMARY
# =============================================================================

if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                     MASTER V2 LIVE TRADING INTEGRATION                       ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  To enable live trading in your master_v2.py:                                ║
║                                                                              ║
║  1. Copy hybrid_trading_engine.py to core/                                   ║
║  2. Copy live_trading_engine.py to core/                                     ║
║  3. Add the imports shown in IMPORTS_TO_ADD                                  ║
║  4. Add config options shown in CONFIG_ADDITIONS                             ║
║  5. Update __init__() as shown in INIT_UPDATE                                ║
║  6. Update _process_buy() as shown in PROCESS_BUY_UPDATE                     ║
║  7. Add new endpoints shown in NEW_ENDPOINTS                                 ║
║  8. Update background_tasks() as shown in BACKGROUND_UPDATE                  ║
║  9. Add environment variables to .env                                        ║
║                                                                              ║
║  Environment Variables Required:                                             ║
║  ─────────────────────────────────────────────────────────────────────────   ║
║  SOLANA_PRIVATE_KEY=your_base58_private_key                                  ║
║  ENABLE_LIVE_TRADING=true                                                    ║
║  POSITION_SIZE_SOL=0.08                                                      ║
║  MAX_OPEN_POSITIONS=10                                                       ║
║  MAX_DAILY_LOSS_SOL=0.25                                                     ║
║                                                                              ║
║  API Endpoints:                                                              ║
║  ─────────────────────────────────────────────────────────────────────────   ║
║  GET  /live/status           - Check live trading status                     ║
║  POST /live/enable?confirm=LIVE  - Enable live trading                       ║
║  POST /live/disable          - Disable live trading                          ║
║  POST /live/kill?confirm=KILL    - Emergency kill switch                     ║
║  GET  /live/positions        - View live positions                           ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")
