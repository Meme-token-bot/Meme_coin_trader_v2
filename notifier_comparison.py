"""
PAPER vs LIVE COMPARISON NOTIFIER
==================================

Clear side-by-side comparison of Paper and Live trading results.

Replace the Notifier class in master_v2.py with this one.
"""

class Notifier:
    """Telegram notifications comparing Paper vs Live trading"""
    
    def __init__(self, token: str = None, chat_id: str = None):
        self.token = token or get_secret('TELEGRAM_BOT_TOKEN')
        self.chat_id = chat_id or get_secret('TELEGRAM_CHAT_ID')
        self.enabled = bool(self.token and self.chat_id)
        self._last_30min_update = datetime.now() - timedelta(minutes=30)
        self._last_stats = {'paper': {}, 'live': {}}
    
    def send(self, message: str):
        if not self.enabled:
            return
        try:
            import requests
            url = f"https://api.telegram.org/bot{self.token}/sendMessage"
            requests.post(url, json={'chat_id': self.chat_id, 'text': message, 'parse_mode': 'HTML'}, timeout=10)
        except Exception as e:
            print(f"  ⚠️ Telegram error: {e}")
    
    def send_critical_alert(self, message: str):
        self.send(f"🚨 <b>CRITICAL</b>\n\n{message}")
    
    # =========================================================================
    # ENTRY ALERTS (Both Paper and Live)
    # =========================================================================
    
    def send_entry_alert(self, signal: Dict, decision: Dict, quality = None, is_live: bool = False):
        """Notify when a position is opened"""
        mode = "🔴 LIVE" if is_live else "📝 PAPER"
        token = signal.get('token_symbol', 'UNKNOWN')
        conviction = decision.get('conviction', signal.get('conviction_score', 0))
        
        msg = f"""{mode} <b>BUY</b>

Token: <b>{token}</b>
Conviction: {conviction}%"""
        
        self.send(msg)
    
    def send_live_entry(self, token_symbol: str, token_address: str, 
                        amount_sol: float, conviction: int, wallet_id: int = 1):
        """Notify when a LIVE position is opened"""
        msg = f"""🔴 <b>LIVE BUY</b>

Token: <b>{token_symbol}</b>
Size: {amount_sol:.2f} SOL
Conviction: {conviction}%
Wallet: #{wallet_id}"""
        self.send(msg)
    
    # =========================================================================
    # EXIT ALERTS (Both Paper and Live)  
    # =========================================================================
    
    def send_exit_alert(self, position: Dict, reason: str, pnl_pct: float, 
                        result: Dict = None, is_live: bool = False):
        """Notify when a position is closed"""
        mode = "🔴 LIVE" if is_live else "📝 PAPER"
        emoji = "✅" if pnl_pct >= 0 else "❌"
        token = position.get('token_symbol', 'UNKNOWN')
        pnl_sol = position.get('pnl_sol', 0)
        
        msg = f"""{emoji} {mode} <b>SELL</b>

Token: <b>{token}</b>
PnL: <b>{pnl_sol:+.4f} SOL</b> ({pnl_pct:+.1f}%)
Reason: {reason}"""
        
        self.send(msg)
    
    def send_live_exit(self, token_symbol: str, pnl_sol: float, pnl_pct: float,
                       exit_reason: str, hold_time_mins: float = 0):
        """Notify when a LIVE position is closed"""
        emoji = "✅" if pnl_sol >= 0 else "❌"
        
        msg = f"""{emoji} 🔴 <b>LIVE SELL</b>

Token: <b>{token_symbol}</b>
PnL: <b>{pnl_sol:+.4f} SOL</b> ({pnl_pct:+.1f}%)
Reason: {exit_reason}
Hold: {hold_time_mins:.0f} min"""
        self.send(msg)
    
    # =========================================================================
    # 30-MINUTE COMPARISON UPDATE
    # =========================================================================
    
    def send_30min_update(self, paper_stats: Dict, diag: Dict, live_stats: Dict = None):
        """30-minute update comparing Paper vs Live"""
        now = datetime.now()
        if (now - self._last_30min_update).total_seconds() < 1800:
            return
        
        self._last_30min_update = now
        
        # Paper stats
        p_balance = paper_stats.get('balance', 0)
        p_open = paper_stats.get('open_positions', 0)
        p_max = paper_stats.get('max_positions', 3)
        p_trades = paper_stats.get('total_trades', 0)
        p_wr = paper_stats.get('win_rate', 0)
        p_pnl = paper_stats.get('total_pnl', 0)
        
        # Live stats from diagnostics
        live_diag = diag.get('live_trading', {})
        l_opened = live_diag.get('opened', 0)
        l_closed = live_diag.get('closed', 0)
        l_pnl = live_diag.get('pnl_sol', 0)
        
        # Live engine stats (if provided)
        if live_stats:
            l_balance = live_stats.get('balance_sol', 0)
            l_open = live_stats.get('open_positions', 0)
            l_max = live_stats.get('max_positions', 3)
            l_trades = live_stats.get('daily_trades', 0)
            l_wins = live_stats.get('daily_wins', 0)
            l_wr = (l_wins / l_trades * 100) if l_trades > 0 else 0
        else:
            l_balance = 0
            l_open = 0
            l_max = 3
            l_trades = l_opened
            l_wr = 0
        
        # Calculate deltas from last update
        prev_p = self._last_stats.get('paper', {})
        prev_l = self._last_stats.get('live', {})
        
        p_pnl_delta = p_pnl - prev_p.get('pnl', 0)
        l_pnl_delta = l_pnl - prev_l.get('pnl', 0)
        
        # Save for next comparison
        self._last_stats = {
            'paper': {'pnl': p_pnl, 'trades': p_trades},
            'live': {'pnl': l_pnl, 'trades': l_trades}
        }
        
        # Determine overall emoji
        if l_pnl > 0 and p_pnl > 0:
            emoji = "📈"
        elif l_pnl < 0 and p_pnl < 0:
            emoji = "📉"
        else:
            emoji = "📊"
        
        msg = f"""{emoji} <b>30-Min Update</b>

┌─────────────────────┐
│ 📝 <b>PAPER</b>              │
├─────────────────────┤
│ Balance: {p_balance:.2f} SOL     
│ Open: {p_open}/{p_max} positions  
│ Trades: {p_trades}              
│ Win Rate: {p_wr:.0%}            
│ PnL: <b>{p_pnl:+.4f}</b> SOL       
│ Δ30m: {p_pnl_delta:+.4f} SOL    
└─────────────────────┘

┌─────────────────────┐
│ 🔴 <b>LIVE</b>               │
├─────────────────────┤
│ Balance: {l_balance:.4f} SOL   
│ Open: {l_open}/{l_max} positions  
│ Trades: {l_trades}              
│ Win Rate: {l_wr:.0f}%            
│ PnL: <b>{l_pnl:+.4f}</b> SOL       
│ Δ30m: {l_pnl_delta:+.4f} SOL    
└─────────────────────┘

Webhooks: {diag['webhooks']['received']}"""
        
        self.send(msg)
    
    # =========================================================================
    # DAILY SUMMARY
    # =========================================================================
    
    def send_daily_summary(self, paper_stats: Dict, live_stats: Dict):
        """End of day comparison"""
        
        p_pnl = paper_stats.get('total_pnl', 0)
        p_trades = paper_stats.get('total_trades', 0)
        p_wr = paper_stats.get('win_rate', 0)
        
        l_pnl = live_stats.get('daily_pnl_sol', 0)
        l_trades = live_stats.get('daily_trades', 0)
        l_wins = live_stats.get('daily_wins', 0)
        l_wr = (l_wins / l_trades * 100) if l_trades > 0 else 0
        l_balance = live_stats.get('balance_sol', 0)
        
        # Who won?
        if l_pnl > p_pnl:
            winner = "🔴 LIVE WINS!"
        elif p_pnl > l_pnl:
            winner = "📝 PAPER WINS!"
        else:
            winner = "🤝 TIE!"
        
        msg = f"""📊 <b>DAILY SUMMARY</b>

{winner}

<b>📝 PAPER:</b>
Trades: {p_trades} | WR: {p_wr:.0%}
PnL: <b>{p_pnl:+.4f} SOL</b>

<b>🔴 LIVE:</b>
Balance: {l_balance:.4f} SOL
Trades: {l_trades} | WR: {l_wr:.0f}%
PnL: <b>{l_pnl:+.4f} SOL</b>

Difference: {l_pnl - p_pnl:+.4f} SOL"""
        
        self.send(msg)
    
    # =========================================================================
    # HARVEST ALERTS
    # =========================================================================
    
    def send_harvest_alert(self, wallet_id: int, amount_sol: float, burner_address: str):
        """Alert when profits are harvested to burner"""
        msg = f"""💰 <b>HARVEST</b>

Wallet #{wallet_id} → Burner
Amount: {amount_sol:.2f} SOL
To: {burner_address[:12]}..."""
        self.send(msg)
    
    def send_burner_ready(self, burner_address: str, balance_sol: float):
        """Alert when burner is ready for CEX withdrawal"""
        msg = f"""🏦 <b>BURNER READY</b>

Address: {burner_address[:16]}...
Balance: {balance_sol:.2f} SOL

⚡ Send to CEX at random time!"""
        self.send(msg)
    
    # Disabled methods (kept for compatibility)
    def send_cluster_alert(self, token_symbol: str, wallet_count: int, wallets: List[str]):
        pass
    
    def send_baseline_report(self, report: Dict):
        pass
