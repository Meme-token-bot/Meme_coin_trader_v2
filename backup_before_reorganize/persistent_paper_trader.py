"""
PERSISTENT PAPER TRADING TRACKER
================================

This tracker stores paper trades in the DATABASE, not in memory.
It survives system restarts and updates.

Usage:
    Replace the in-memory paper trading engine with this.
"""

import os
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass
from database_v2 import DatabaseV2


@dataclass
class PaperPosition:
    """A paper trading position"""
    id: int
    wallet_source: str
    token_address: str
    token_symbol: str
    entry_price: float
    entry_time: datetime
    size_sol: float
    current_price: float = 0
    exit_price: float = 0
    exit_time: Optional[datetime] = None
    pnl_sol: float = 0
    pnl_pct: float = 0
    status: str = 'open'  # open, closed, stopped


class PersistentPaperTrader:
    """
    Paper trading engine that persists to database.
    
    Unlike in-memory engines, this survives restarts.
    """
    
    def __init__(self, db: DatabaseV2, starting_balance: float = 10.0):
        self.db = db
        self.starting_balance = starting_balance
        
        # Initialize tables
        self._init_tables()
        
        # Load or create account
        self._load_or_create_account()
        
        print(f"📊 Persistent Paper Trader initialized")
        print(f"   Balance: {self.balance:.4f} SOL")
        print(f"   Total trades: {self.total_trades}")
        print(f"   Open positions: {len(self.get_open_positions())}")
    
    def _init_tables(self):
        """Create paper trading tables if they don't exist"""
        with self.db.connection() as conn:
            # Paper account
            conn.execute("""
                CREATE TABLE IF NOT EXISTS paper_account (
                    id INTEGER PRIMARY KEY,
                    starting_balance REAL,
                    current_balance REAL,
                    total_trades INTEGER DEFAULT 0,
                    winning_trades INTEGER DEFAULT 0,
                    total_pnl REAL DEFAULT 0,
                    created_at TIMESTAMP,
                    last_updated TIMESTAMP
                )
            """)
            
            # Paper positions (both open and closed)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS paper_positions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    wallet_source TEXT,
                    token_address TEXT,
                    token_symbol TEXT,
                    entry_price REAL,
                    entry_time TIMESTAMP,
                    size_sol REAL,
                    exit_price REAL,
                    exit_time TIMESTAMP,
                    pnl_sol REAL DEFAULT 0,
                    pnl_pct REAL DEFAULT 0,
                    status TEXT DEFAULT 'open',
                    notes TEXT
                )
            """)
            
            # Paper trade history (detailed log)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS paper_trade_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    position_id INTEGER,
                    action TEXT,
                    price REAL,
                    size_sol REAL,
                    timestamp TIMESTAMP,
                    wallet_source TEXT,
                    signal_data TEXT,
                    FOREIGN KEY (position_id) REFERENCES paper_positions(id)
                )
            """)
            
            # Daily summary
            conn.execute("""
                CREATE TABLE IF NOT EXISTS paper_daily_summary (
                    date DATE PRIMARY KEY,
                    starting_balance REAL,
                    ending_balance REAL,
                    trades INTEGER,
                    wins INTEGER,
                    pnl_sol REAL,
                    notes TEXT
                )
            """)
    
    def _load_or_create_account(self):
        """Load existing account or create new one"""
        with self.db.connection() as conn:
            account = conn.execute("""
                SELECT * FROM paper_account ORDER BY id DESC LIMIT 1
            """).fetchone()
            
            if account:
                self.balance = account['current_balance']
                self.total_trades = account['total_trades']
                self.winning_trades = account['winning_trades']
                self.total_pnl = account['total_pnl']
                self.account_id = account['id']
            else:
                # Create new account
                conn.execute("""
                    INSERT INTO paper_account 
                    (starting_balance, current_balance, total_trades, winning_trades, 
                     total_pnl, created_at, last_updated)
                    VALUES (?, ?, 0, 0, 0, ?, ?)
                """, (self.starting_balance, self.starting_balance, 
                      datetime.now(), datetime.now()))
                
                self.balance = self.starting_balance
                self.total_trades = 0
                self.winning_trades = 0
                self.total_pnl = 0
                self.account_id = 1
    
    def open_position(self, wallet_source: str, token_address: str, 
                      token_symbol: str, entry_price: float, 
                      size_sol: float, notes: str = "") -> Optional[int]:
        """
        Open a new paper position.
        
        Returns position ID if successful.
        """
        # Check balance
        if size_sol > self.balance:
            print(f"   ⚠️ Insufficient balance: {self.balance:.4f} < {size_sol:.4f}")
            return None
        
        # Check max positions
        open_positions = self.get_open_positions()
        if len(open_positions) >= 5:
            print(f"   ⚠️ Max positions reached (5)")
            return None
        
        # Check if already have position in this token
        for pos in open_positions:
            if pos['token_address'] == token_address:
                print(f"   ⚠️ Already have position in {token_symbol}")
                return None
        
        with self.db.connection() as conn:
            # Create position
            cursor = conn.execute("""
                INSERT INTO paper_positions 
                (wallet_source, token_address, token_symbol, entry_price, 
                 entry_time, size_sol, status, notes)
                VALUES (?, ?, ?, ?, ?, ?, 'open', ?)
            """, (wallet_source, token_address, token_symbol, entry_price,
                  datetime.now(), size_sol, notes))
            
            position_id = cursor.lastrowid
            
            # Log the trade
            conn.execute("""
                INSERT INTO paper_trade_log 
                (position_id, action, price, size_sol, timestamp, wallet_source)
                VALUES (?, 'BUY', ?, ?, ?, ?)
            """, (position_id, entry_price, size_sol, datetime.now(), wallet_source))
            
            # Update balance
            self.balance -= size_sol
            conn.execute("""
                UPDATE paper_account SET current_balance = ?, last_updated = ?
                WHERE id = ?
            """, (self.balance, datetime.now(), self.account_id))
        
        print(f"   📥 PAPER BUY: {size_sol:.4f} SOL of {token_symbol} @ ${entry_price:.8f}")
        print(f"      Source: {wallet_source[:12]}...")
        print(f"      Balance: {self.balance:.4f} SOL")
        
        return position_id
    
    def close_position(self, position_id: int, exit_price: float, 
                       notes: str = "") -> Optional[Dict]:
        """
        Close a paper position.
        
        Returns trade result if successful.
        """
        with self.db.connection() as conn:
            # Get position
            pos = conn.execute("""
                SELECT * FROM paper_positions WHERE id = ? AND status = 'open'
            """, (position_id,)).fetchone()
            
            if not pos:
                print(f"   ⚠️ Position {position_id} not found or already closed")
                return None
            
            # Calculate PnL
            entry_price = pos['entry_price']
            size_sol = pos['size_sol']
            
            if entry_price > 0:
                # Calculate token amount bought
                tokens_bought = size_sol / entry_price
                # Calculate SOL value at exit
                exit_value = tokens_bought * exit_price
                pnl_sol = exit_value - size_sol
                pnl_pct = (exit_price - entry_price) / entry_price * 100
            else:
                pnl_sol = 0
                pnl_pct = 0
                exit_value = size_sol
            
            # Update position
            conn.execute("""
                UPDATE paper_positions 
                SET exit_price = ?, exit_time = ?, pnl_sol = ?, pnl_pct = ?, 
                    status = 'closed', notes = COALESCE(notes || ' | ', '') || ?
                WHERE id = ?
            """, (exit_price, datetime.now(), pnl_sol, pnl_pct, notes, position_id))
            
            # Log the trade
            conn.execute("""
                INSERT INTO paper_trade_log 
                (position_id, action, price, size_sol, timestamp, wallet_source)
                VALUES (?, 'SELL', ?, ?, ?, ?)
            """, (position_id, exit_price, exit_value, datetime.now(), pos['wallet_source']))
            
            # Update account
            self.balance += exit_value
            self.total_trades += 1
            self.total_pnl += pnl_sol
            
            if pnl_sol > 0:
                self.winning_trades += 1
            
            conn.execute("""
                UPDATE paper_account 
                SET current_balance = ?, total_trades = ?, winning_trades = ?,
                    total_pnl = ?, last_updated = ?
                WHERE id = ?
            """, (self.balance, self.total_trades, self.winning_trades,
                  self.total_pnl, datetime.now(), self.account_id))
            
            # Update daily summary
            self._update_daily_summary(pnl_sol)
        
        result_emoji = "✅" if pnl_sol > 0 else "❌"
        print(f"   📤 PAPER SELL: {pos['token_symbol']} @ ${exit_price:.8f}")
        print(f"      {result_emoji} PnL: {pnl_sol:+.4f} SOL ({pnl_pct:+.1f}%)")
        print(f"      Balance: {self.balance:.4f} SOL")
        
        return {
            'position_id': position_id,
            'token_symbol': pos['token_symbol'],
            'entry_price': entry_price,
            'exit_price': exit_price,
            'size_sol': size_sol,
            'pnl_sol': pnl_sol,
            'pnl_pct': pnl_pct,
            'wallet_source': pos['wallet_source']
        }
    
    def _update_daily_summary(self, pnl_sol: float):
        """Update daily summary table"""
        today = datetime.now().date()
        
        with self.db.connection() as conn:
            existing = conn.execute("""
                SELECT * FROM paper_daily_summary WHERE date = ?
            """, (today,)).fetchone()
            
            if existing:
                conn.execute("""
                    UPDATE paper_daily_summary 
                    SET ending_balance = ?, trades = trades + 1,
                        wins = wins + ?, pnl_sol = pnl_sol + ?
                    WHERE date = ?
                """, (self.balance, 1 if pnl_sol > 0 else 0, pnl_sol, today))
            else:
                conn.execute("""
                    INSERT INTO paper_daily_summary 
                    (date, starting_balance, ending_balance, trades, wins, pnl_sol)
                    VALUES (?, ?, ?, 1, ?, ?)
                """, (today, self.balance - pnl_sol, self.balance, 
                      1 if pnl_sol > 0 else 0, pnl_sol))
    
    def get_open_positions(self) -> List[Dict]:
        """Get all open positions"""
        with self.db.connection() as conn:
            rows = conn.execute("""
                SELECT * FROM paper_positions WHERE status = 'open'
                ORDER BY entry_time DESC
            """).fetchall()
            return [dict(row) for row in rows]
    
    def get_trade_history(self, limit: int = 50) -> List[Dict]:
        """Get recent trade history"""
        with self.db.connection() as conn:
            rows = conn.execute("""
                SELECT * FROM paper_positions WHERE status = 'closed'
                ORDER BY exit_time DESC LIMIT ?
            """, (limit,)).fetchall()
            return [dict(row) for row in rows]
    
    def get_daily_summary(self, days: int = 7) -> List[Dict]:
        """Get daily summary for last N days"""
        with self.db.connection() as conn:
            rows = conn.execute("""
                SELECT * FROM paper_daily_summary
                ORDER BY date DESC LIMIT ?
            """, (days,)).fetchall()
            return [dict(row) for row in rows]
    
    def get_stats(self) -> Dict:
        """Get comprehensive statistics"""
        open_positions = self.get_open_positions()
        trade_history = self.get_trade_history(100)
        daily_summary = self.get_daily_summary(30)
        
        # Calculate stats
        win_rate = self.winning_trades / max(1, self.total_trades)
        return_pct = (self.balance - self.starting_balance) / self.starting_balance * 100
        
        # Calculate average win/loss
        wins = [t for t in trade_history if t['pnl_sol'] > 0]
        losses = [t for t in trade_history if t['pnl_sol'] <= 0]
        
        avg_win = sum(t['pnl_sol'] for t in wins) / max(1, len(wins))
        avg_loss = sum(t['pnl_sol'] for t in losses) / max(1, len(losses))
        
        # Best/worst days
        if daily_summary:
            best_day = max(daily_summary, key=lambda x: x['pnl_sol'])
            worst_day = min(daily_summary, key=lambda x: x['pnl_sol'])
        else:
            best_day = worst_day = None
        
        return {
            'balance': self.balance,
            'starting_balance': self.starting_balance,
            'return_pct': return_pct,
            'total_pnl': self.total_pnl,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'open_positions': len(open_positions),
            'days_trading': len(daily_summary),
            'best_day': best_day,
            'worst_day': worst_day
        }
    
    def print_status(self):
        """Print detailed status"""
        stats = self.get_stats()
        
        print(f"\n📊 PAPER TRADING STATUS")
        print(f"{'='*50}")
        print(f"Balance: {stats['balance']:.4f} SOL ({stats['return_pct']:+.1f}%)")
        print(f"Total PnL: {stats['total_pnl']:+.4f} SOL")
        print(f"Trades: {stats['total_trades']} ({stats['win_rate']:.1%} win rate)")
        print(f"Avg Win: {stats['avg_win']:.4f} SOL | Avg Loss: {stats['avg_loss']:.4f} SOL")
        print(f"Open positions: {stats['open_positions']}")
        print(f"Days trading: {stats['days_trading']}")
        
        if stats['best_day']:
            print(f"\nBest day: {stats['best_day']['date']} ({stats['best_day']['pnl_sol']:+.4f} SOL)")
        if stats['worst_day']:
            print(f"Worst day: {stats['worst_day']['date']} ({stats['worst_day']['pnl_sol']:+.4f} SOL)")
        
        # Show open positions
        open_pos = self.get_open_positions()
        if open_pos:
            print(f"\n📈 Open Positions:")
            for pos in open_pos:
                print(f"   {pos['token_symbol']}: {pos['size_sol']:.4f} SOL @ ${pos['entry_price']:.8f}")
        
        # Show recent trades
        recent = self.get_trade_history(5)
        if recent:
            print(f"\n📜 Recent Trades:")
            for trade in recent:
                emoji = "✅" if trade['pnl_sol'] > 0 else "❌"
                print(f"   {emoji} {trade['token_symbol']}: {trade['pnl_sol']:+.4f} SOL ({trade['pnl_pct']:+.1f}%)")


# CLI for testing
if __name__ == "__main__":
    import sys
    
    db = DatabaseV2()
    trader = PersistentPaperTrader(db, starting_balance=10.0)
    
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == 'status':
            trader.print_status()
        
        elif command == 'history':
            history = trader.get_trade_history(20)
            print(f"\n📜 Trade History ({len(history)} trades):")
            for t in history:
                emoji = "✅" if t['pnl_sol'] > 0 else "❌"
                print(f"   {emoji} {t['exit_time']} | {t['token_symbol']}: {t['pnl_sol']:+.4f} SOL")
        
        elif command == 'daily':
            daily = trader.get_daily_summary(14)
            print(f"\n📅 Daily Summary:")
            for d in daily:
                emoji = "✅" if d['pnl_sol'] > 0 else "❌"
                print(f"   {emoji} {d['date']}: {d['pnl_sol']:+.4f} SOL ({d['trades']} trades, {d['wins']} wins)")
        
        else:
            print("Usage: python persistent_paper_trader.py [status|history|daily]")
    else:
        trader.print_status()
