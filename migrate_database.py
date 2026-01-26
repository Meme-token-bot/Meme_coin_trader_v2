"""
DATABASE MIGRATION & ANALYSIS SCRIPT
====================================

This script:
1. Analyzes corrupted paper trading data
2. Recalculates correct balance from trade history
3. Can reset or migrate to the fixed V6 schema

Run this BEFORE using the fixed paper trading platform!
"""

import sqlite3
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from contextlib import contextmanager
import os


class DatabaseAnalyzer:
    """Analyze and fix corrupted paper trading database"""
    
    def __init__(self, 
                 old_db_path: str = "robust_paper_trades.db",
                 new_db_path: str = "robust_paper_trades_v6.db"):
        self.old_db_path = old_db_path
        self.new_db_path = new_db_path
    
    @contextmanager
    def _get_connection(self, db_path: str):
        conn = sqlite3.connect(db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()
    
    def analyze_old_database(self) -> Dict:
        """
        Analyze the old corrupted database and identify issues.
        """
        if not os.path.exists(self.old_db_path):
            print(f"❌ Old database not found: {self.old_db_path}")
            return {}
        
        print("\n" + "=" * 70)
        print("🔍 ANALYZING OLD DATABASE")
        print("=" * 70)
        
        results = {}
        
        with self._get_connection(self.old_db_path) as conn:
            # Check which tables exist
            tables = conn.execute("""
                SELECT name FROM sqlite_master WHERE type='table'
            """).fetchall()
            table_names = [t['name'] for t in tables]
            print(f"\nTables found: {table_names}")
            
            # Find the account table (v3, v5, etc)
            account_table = None
            positions_table = None
            
            for version in ['v5', 'v3', '']:
                test_account = f"paper_account_{version}" if version else "paper_account"
                test_positions = f"paper_positions_{version}" if version else "paper_positions"
                
                if test_account in table_names:
                    account_table = test_account
                if test_positions in table_names:
                    positions_table = test_positions
            
            if not account_table or not positions_table:
                print("❌ Could not find account/positions tables")
                return {}
            
            print(f"\nUsing tables: {account_table}, {positions_table}")
            
            # Get account state
            account = conn.execute(f"SELECT * FROM {account_table} WHERE id = 1").fetchone()
            if account:
                results['account'] = dict(account)
                print(f"\n📊 ACCOUNT STATE:")
                print(f"   Starting balance: {account['starting_balance']:.4f} SOL")
                print(f"   Current balance:  {account['current_balance']:.4f} SOL")
                print(f"   Reserved balance: {account.get('reserved_balance', 0):.4f} SOL")
                print(f"   Total trades:     {account.get('total_trades', 0)}")
                print(f"   Winning trades:   {account.get('winning_trades', 0)}")
                print(f"   Total PnL:        {account.get('total_pnl_sol', 0):+.4f} SOL")
                
                # Calculate return percentage
                if account['starting_balance'] > 0:
                    return_pct = ((account['current_balance'] / account['starting_balance']) - 1) * 100
                    print(f"   Return:           {return_pct:+.1f}%")
            
            # Count positions by status
            open_count = conn.execute(f"""
                SELECT COUNT(*) FROM {positions_table} WHERE status = 'open'
            """).fetchone()[0]
            
            closed_count = conn.execute(f"""
                SELECT COUNT(*) FROM {positions_table} WHERE status = 'closed'
            """).fetchone()[0]
            
            print(f"\n📈 POSITIONS:")
            print(f"   Open:   {open_count}")
            print(f"   Closed: {closed_count}")
            
            results['open_positions'] = open_count
            results['closed_positions'] = closed_count
            
            # Recalculate what the balance SHOULD be
            if closed_count > 0:
                # Get all closed trades and recalculate
                closed_trades = conn.execute(f"""
                    SELECT id, token_symbol, size_sol, entry_price, exit_price, 
                           pnl_sol, pnl_pct, tokens_bought
                    FROM {positions_table} 
                    WHERE status = 'closed'
                    ORDER BY id
                """).fetchall()
                
                recalculated_pnl = 0
                recalculated_wins = 0
                bug_examples = []
                
                for trade in closed_trades:
                    # Recalculate correct PnL
                    size_sol = trade['size_sol']
                    tokens = trade['tokens_bought'] or (size_sol / trade['entry_price'] if trade['entry_price'] > 0 else 0)
                    exit_value = tokens * (trade['exit_price'] or 0)
                    correct_pnl = exit_value - size_sol
                    stored_pnl = trade['pnl_sol'] or 0
                    
                    # Check for discrepancy (this would indicate the bug)
                    if abs(correct_pnl - stored_pnl) > 0.001:
                        if len(bug_examples) < 5:
                            bug_examples.append({
                                'id': trade['id'],
                                'symbol': trade['token_symbol'],
                                'stored_pnl': stored_pnl,
                                'correct_pnl': correct_pnl,
                                'difference': stored_pnl - correct_pnl
                            })
                    
                    recalculated_pnl += correct_pnl
                    if correct_pnl > 0:
                        recalculated_wins += 1
                
                expected_balance = account['starting_balance'] + recalculated_pnl
                actual_balance = account['current_balance']
                discrepancy = actual_balance - expected_balance
                
                print(f"\n🔬 RECALCULATION:")
                print(f"   Recalculated PnL:      {recalculated_pnl:+.4f} SOL")
                print(f"   Expected balance:      {expected_balance:.4f} SOL")
                print(f"   Actual balance:        {actual_balance:.4f} SOL")
                print(f"   DISCREPANCY:           {discrepancy:+.4f} SOL")
                
                if closed_count > 0:
                    recalc_wr = recalculated_wins / closed_count * 100
                    stored_wr = account.get('winning_trades', 0) / closed_count * 100 if closed_count > 0 else 0
                    print(f"\n   Recalculated WR:       {recalc_wr:.1f}%")
                    print(f"   Stored WR:             {stored_wr:.1f}%")
                
                results['recalculated_pnl'] = recalculated_pnl
                results['expected_balance'] = expected_balance
                results['discrepancy'] = discrepancy
                results['recalculated_wins'] = recalculated_wins
                
                if bug_examples:
                    print(f"\n⚠️  BUG EVIDENCE (sample trades with wrong PnL):")
                    for ex in bug_examples:
                        print(f"   Trade #{ex['id']} ({ex['symbol']}): "
                              f"stored={ex['stored_pnl']:+.4f}, "
                              f"correct={ex['correct_pnl']:+.4f}, "
                              f"diff={ex['difference']:+.4f}")
                
                # Identify the bug type
                if discrepancy > 1.0:
                    print(f"\n🐛 BUG IDENTIFIED: Balance inflated by {discrepancy:.2f} SOL")
                    print("   This matches the 'exit_value instead of pnl_sol' bug")
                    
                    # Calculate theoretical inflation
                    avg_position_size = 0.3  # Typical position size
                    theoretical_inflation = avg_position_size * closed_count * 0.85  # Assuming ~85% survival to exit
                    print(f"   Theoretical inflation: ~{theoretical_inflation:.2f} SOL")
            
            # Check for position limit violations
            if open_count > 5:  # Assuming default limit was 5
                print(f"\n⚠️  POSITION LIMIT BUG: {open_count} open positions (should be ≤5)")
                print("   This indicates the race condition bug")
        
        print("\n" + "=" * 70)
        return results
    
    def migrate_to_v6(self, starting_balance: float = 10.0, 
                      close_open_positions: bool = True) -> bool:
        """
        Migrate to the new V6 database with correct data.
        
        Options:
        - close_open_positions: If True, close all open positions at entry price (0 PnL)
        """
        print("\n" + "=" * 70)
        print("🔄 MIGRATING TO V6 DATABASE")
        print("=" * 70)
        
        # Create new V6 database
        with self._get_connection(self.new_db_path) as conn:
            # Create tables
            conn.execute("""
                CREATE TABLE IF NOT EXISTS paper_account_v6 (
                    id INTEGER PRIMARY KEY,
                    starting_balance REAL,
                    current_balance REAL,
                    reserved_balance REAL DEFAULT 0,
                    total_trades INTEGER DEFAULT 0,
                    winning_trades INTEGER DEFAULT 0,
                    total_pnl_sol REAL DEFAULT 0,
                    peak_balance REAL,
                    max_open_positions INTEGER DEFAULT 5,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS paper_positions_v6 (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    token_address TEXT NOT NULL,
                    token_symbol TEXT,
                    entry_price REAL NOT NULL,
                    entry_time TIMESTAMP NOT NULL,
                    size_sol REAL NOT NULL,
                    tokens_bought REAL NOT NULL,
                    stop_loss_pct REAL,
                    take_profit_pct REAL,
                    trailing_stop_pct REAL,
                    trailing_activation_pct REAL,
                    max_hold_hours INTEGER,
                    scale_out_enabled BOOLEAN DEFAULT FALSE,
                    current_price REAL,
                    peak_price REAL,
                    lowest_price REAL,
                    last_price_update TIMESTAMP,
                    status TEXT DEFAULT 'open',
                    exit_price REAL,
                    exit_time TIMESTAMP,
                    exit_reason TEXT,
                    pnl_sol REAL,
                    pnl_pct REAL,
                    hold_duration_minutes REAL,
                    conviction_score REAL,
                    quality_score REAL,
                    is_cluster_signal BOOLEAN,
                    entry_context_json TEXT,
                    signal_id TEXT,
                    ab_test_id TEXT,
                    ab_variant TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Initialize with fresh account
            conn.execute("""
                INSERT OR REPLACE INTO paper_account_v6
                (id, starting_balance, current_balance, peak_balance, max_open_positions)
                VALUES (1, ?, ?, ?, 5)
            """, (starting_balance, starting_balance, starting_balance))
        
        print(f"✅ Created new V6 database: {self.new_db_path}")
        print(f"   Starting balance: {starting_balance} SOL")
        print(f"   Max positions: 5")
        print(f"   All positions: closed/reset")
        
        return True
    
    def recalculate_and_fix_v5(self) -> bool:
        """
        Recalculate correct balances in the existing V5 database.
        This fixes the data in place rather than migrating.
        """
        if not os.path.exists(self.old_db_path):
            print(f"❌ Database not found: {self.old_db_path}")
            return False
        
        print("\n" + "=" * 70)
        print("🔧 RECALCULATING V5 DATABASE")
        print("=" * 70)
        
        with self._get_connection(self.old_db_path) as conn:
            # Find tables
            tables = conn.execute("""
                SELECT name FROM sqlite_master WHERE type='table'
            """).fetchall()
            table_names = [t['name'] for t in tables]
            
            # Determine version
            if 'paper_account_v5' in table_names:
                account_table = 'paper_account_v5'
                positions_table = 'paper_positions_v5'
            elif 'paper_account_v3' in table_names:
                account_table = 'paper_account_v3'
                positions_table = 'paper_positions_v3'
            else:
                print("❌ Could not find paper_account table")
                return False
            
            # Get starting balance
            account = conn.execute(f"SELECT * FROM {account_table} WHERE id = 1").fetchone()
            starting_balance = account['starting_balance']
            
            print(f"Using tables: {account_table}, {positions_table}")
            print(f"Starting balance: {starting_balance}")
            
            # Recalculate all closed trades
            closed_trades = conn.execute(f"""
                SELECT id, size_sol, entry_price, exit_price, tokens_bought
                FROM {positions_table}
                WHERE status = 'closed'
            """).fetchall()
            
            total_pnl = 0
            wins = 0
            
            for trade in closed_trades:
                size_sol = trade['size_sol']
                tokens = trade['tokens_bought']
                exit_price = trade['exit_price'] or 0
                
                exit_value = tokens * exit_price
                correct_pnl = exit_value - size_sol
                pnl_pct = ((exit_price / trade['entry_price']) - 1) * 100 if trade['entry_price'] > 0 else 0
                
                # Update the trade with correct PnL
                conn.execute(f"""
                    UPDATE {positions_table}
                    SET pnl_sol = ?, pnl_pct = ?
                    WHERE id = ?
                """, (correct_pnl, pnl_pct, trade['id']))
                
                total_pnl += correct_pnl
                if correct_pnl > 0:
                    wins += 1
            
            # Close all open positions at entry price (0 PnL)
            open_positions = conn.execute(f"""
                SELECT id FROM {positions_table} WHERE status = 'open'
            """).fetchall()
            
            for pos in open_positions:
                conn.execute(f"""
                    UPDATE {positions_table}
                    SET status = 'closed', 
                        exit_reason = 'RESET',
                        pnl_sol = 0,
                        pnl_pct = 0,
                        exit_time = ?
                    WHERE id = ?
                """, (datetime.utcnow(), pos['id']))
            
            # Calculate correct balance
            correct_balance = starting_balance + total_pnl
            
            # Update account
            conn.execute(f"""
                UPDATE {account_table}
                SET current_balance = ?,
                    reserved_balance = 0,
                    total_trades = ?,
                    winning_trades = ?,
                    total_pnl_sol = ?,
                    updated_at = ?
                WHERE id = 1
            """, (correct_balance, len(closed_trades), wins, total_pnl, datetime.utcnow()))
            
            print(f"\n✅ DATABASE FIXED:")
            print(f"   Trades recalculated: {len(closed_trades)}")
            print(f"   Positions closed: {len(open_positions)}")
            print(f"   Total PnL: {total_pnl:+.4f} SOL")
            print(f"   Wins: {wins} ({wins/len(closed_trades)*100:.1f}% WR)" if closed_trades else "   Wins: 0")
            print(f"   Correct balance: {correct_balance:.4f} SOL")
        
        return True
    
    def create_fresh_v6(self, starting_balance: float = 10.0) -> bool:
        """Create a completely fresh V6 database (no migration)"""
        print("\n" + "=" * 70)
        print("🆕 CREATING FRESH V6 DATABASE")
        print("=" * 70)
        
        # Remove old if exists
        if os.path.exists(self.new_db_path):
            os.remove(self.new_db_path)
            print(f"   Removed existing: {self.new_db_path}")
        
        return self.migrate_to_v6(starting_balance)


def main():
    import sys
    
    analyzer = DatabaseAnalyzer()
    
    print("\n" + "=" * 70)
    print("📊 PAPER TRADING DATABASE MIGRATION TOOL")
    print("=" * 70)
    
    if len(sys.argv) < 2:
        print("\nUsage: python migrate_database.py <command>")
        print("\nCommands:")
        print("  analyze     - Analyze old database and identify bugs")
        print("  fix         - Recalculate and fix the old database in place")
        print("  migrate     - Create new V6 database (fresh start)")
        print("  fresh       - Create completely fresh V6 database")
        print("\nExample:")
        print("  python migrate_database.py analyze")
        print("  python migrate_database.py fresh")
        return
    
    command = sys.argv[1].lower()
    
    if command == 'analyze':
        analyzer.analyze_old_database()
    
    elif command == 'fix':
        print("\n⚠️  This will modify the existing database!")
        print("   - Recalculate all trade PnLs")
        print("   - Close all open positions at 0 PnL")
        print("   - Reset balance to correct value")
        confirm = input("\nType 'FIX' to confirm: ")
        if confirm == 'FIX':
            analyzer.recalculate_and_fix_v5()
        else:
            print("Cancelled.")
    
    elif command == 'migrate':
        print("\n⚠️  This will create a new V6 database!")
        print("   - Fresh start with 10 SOL")
        print("   - No trade history")
        confirm = input("\nType 'MIGRATE' to confirm: ")
        if confirm == 'MIGRATE':
            analyzer.migrate_to_v6()
        else:
            print("Cancelled.")
    
    elif command == 'fresh':
        balance = 10.0
        if len(sys.argv) > 2:
            try:
                balance = float(sys.argv[2])
            except:
                pass
        
        print(f"\n🆕 Creating fresh V6 database with {balance} SOL")
        analyzer.create_fresh_v6(balance)
    
    else:
        print(f"Unknown command: {command}")


if __name__ == "__main__":
    main()
