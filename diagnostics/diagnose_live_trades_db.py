#!/usr/bin/env python3
"""
LIVE TRADES TAX DB DIAGNOSTIC TOOL
===================================

PURPOSE:
    Diagnose issues with live_trades_tax.db to understand why
    transactions might not be saving properly.

CHECKS:
    1. Database file exists and is accessible
    2. Tables are properly created
    3. Recent transactions
    4. Open positions
    5. Cost basis lots
    6. Daily statistics
    7. Execution log for errors

USAGE:
    python diagnose_live_trades_db.py
    python diagnose_live_trades_db.py --db /path/to/live_trades_tax.db

AUTHOR: Trading Bot System
VERSION: 1.0.0
"""

import argparse
import os
import sqlite3
from datetime import datetime, timedelta
from contextlib import contextmanager

# =============================================================================
# DATABASE CONNECTION
# =============================================================================

@contextmanager
def get_connection(db_path: str):
    """Get database connection with row factory"""
    conn = sqlite3.connect(db_path, timeout=30)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def check_database_exists(db_path: str) -> bool:
    """Check if database file exists"""
    return os.path.exists(db_path)


def check_tables(db_path: str) -> dict:
    """Check which tables exist"""
    expected_tables = [
        'tax_transactions',
        'cost_basis_lots',
        'open_positions',
        'daily_stats',
        'execution_log'
    ]
    
    with get_connection(db_path) as conn:
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
        existing = [row['name'] for row in cursor.fetchall()]
    
    return {
        'expected': expected_tables,
        'found': existing,
        'missing': [t for t in expected_tables if t not in existing]
    }


def get_transaction_counts(db_path: str) -> dict:
    """Get transaction counts"""
    with get_connection(db_path) as conn:
        # Total transactions
        total = conn.execute(
            "SELECT COUNT(*) as count FROM tax_transactions"
        ).fetchone()['count']
        
        # By type
        by_type = conn.execute("""
            SELECT transaction_type, COUNT(*) as count 
            FROM tax_transactions 
            GROUP BY transaction_type
        """).fetchall()
        
        # Recent (last 24h)
        yesterday = (datetime.utcnow() - timedelta(days=1)).isoformat()
        recent = conn.execute(
            "SELECT COUNT(*) as count FROM tax_transactions WHERE timestamp > ?",
            (yesterday,)
        ).fetchone()['count']
        
        # Live vs paper
        live_count = conn.execute(
            "SELECT COUNT(*) as count FROM tax_transactions WHERE is_live = 1"
        ).fetchone()['count']
    
    return {
        'total': total,
        'by_type': {row['transaction_type']: row['count'] for row in by_type},
        'last_24h': recent,
        'live_trades': live_count
    }


def get_recent_transactions(db_path: str, limit: int = 10) -> list:
    """Get recent transactions"""
    with get_connection(db_path) as conn:
        rows = conn.execute("""
            SELECT timestamp, transaction_type, token_symbol, sol_amount, 
                   signature, is_live, notes
            FROM tax_transactions 
            ORDER BY timestamp DESC 
            LIMIT ?
        """, (limit,)).fetchall()
    
    return [dict(row) for row in rows]


def get_open_positions(db_path: str) -> list:
    """Get open positions"""
    with get_connection(db_path) as conn:
        rows = conn.execute("""
            SELECT token_address, token_symbol, tokens_held, entry_price_usd,
                   entry_time, total_cost_sol, stop_loss_pct, take_profit_pct,
                   conviction_score
            FROM open_positions
        """).fetchall()
    
    return [dict(row) for row in rows]


def get_cost_basis_lots(db_path: str, limit: int = 10) -> list:
    """Get recent cost basis lots"""
    with get_connection(db_path) as conn:
        rows = conn.execute("""
            SELECT token_address, acquisition_date, tokens_acquired, 
                   tokens_remaining, cost_per_token_nzd, total_cost_nzd
            FROM cost_basis_lots 
            ORDER BY acquisition_date DESC 
            LIMIT ?
        """, (limit,)).fetchall()
    
    return [dict(row) for row in rows]


def get_daily_stats(db_path: str, days: int = 7) -> list:
    """Get recent daily stats"""
    with get_connection(db_path) as conn:
        rows = conn.execute("""
            SELECT date, trades, wins, losses, pnl_sol, fees_sol
            FROM daily_stats 
            ORDER BY date DESC 
            LIMIT ?
        """, (days,)).fetchall()
    
    return [dict(row) for row in rows]


def get_execution_errors(db_path: str, limit: int = 20) -> list:
    """Get recent execution errors"""
    with get_connection(db_path) as conn:
        rows = conn.execute("""
            SELECT timestamp, action, token_symbol, status, error
            FROM execution_log 
            WHERE status IN ('ERROR', 'FAILED', 'UNCONFIRMED', 'TIMEOUT')
            ORDER BY timestamp DESC 
            LIMIT ?
        """, (limit,)).fetchall()
    
    return [dict(row) for row in rows]


def get_execution_log_summary(db_path: str) -> dict:
    """Get execution log summary"""
    with get_connection(db_path) as conn:
        rows = conn.execute("""
            SELECT status, COUNT(*) as count 
            FROM execution_log 
            GROUP BY status
        """).fetchall()
    
    return {row['status']: row['count'] for row in rows}


# =============================================================================
# DIAGNOSTIC REPORT
# =============================================================================

def run_diagnostics(db_path: str):
    """Run full diagnostics"""
    
    print("\n" + "=" * 70)
    print("🔍 LIVE TRADES TAX DB DIAGNOSTIC REPORT")
    print("=" * 70)
    print(f"   Database: {db_path}")
    print(f"   Time: {datetime.utcnow().isoformat()}Z")
    print("=" * 70)
    
    # Check 1: Database exists
    print("\n📁 CHECK 1: Database File")
    print("-" * 40)
    
    if not check_database_exists(db_path):
        print("   ❌ Database file NOT FOUND!")
        print(f"   Expected at: {os.path.abspath(db_path)}")
        print("\n   DIAGNOSIS: Database hasn't been created yet.")
        print("   FIX: Run the trading engine to initialize the database.")
        return
    
    file_size = os.path.getsize(db_path)
    print(f"   ✅ Database exists")
    print(f"   📊 Size: {file_size:,} bytes ({file_size/1024:.1f} KB)")
    
    # Check 2: Tables
    print("\n📋 CHECK 2: Database Tables")
    print("-" * 40)
    
    tables = check_tables(db_path)
    
    if tables['missing']:
        print(f"   ⚠️ Missing tables: {tables['missing']}")
        print("   DIAGNOSIS: Schema may be incomplete.")
    else:
        print(f"   ✅ All expected tables exist")
    
    print(f"   Tables found: {tables['found']}")
    
    # Check 3: Transaction counts
    print("\n📈 CHECK 3: Transaction Counts")
    print("-" * 40)
    
    try:
        counts = get_transaction_counts(db_path)
        print(f"   Total transactions: {counts['total']}")
        print(f"   Last 24 hours: {counts['last_24h']}")
        print(f"   Live trades: {counts['live_trades']}")
        print(f"   By type: {counts['by_type']}")
        
        if counts['total'] == 0:
            print("\n   ⚠️ NO TRANSACTIONS RECORDED!")
            print("   DIAGNOSIS: Either no trades executed, or records not saving.")
    except Exception as e:
        print(f"   ❌ Error reading transactions: {e}")
    
    # Check 4: Open positions
    print("\n📊 CHECK 4: Open Positions")
    print("-" * 40)
    
    try:
        positions = get_open_positions(db_path)
        print(f"   Open positions: {len(positions)}")
        
        for pos in positions:
            print(f"\n   🪙 {pos['token_symbol']}")
            print(f"      Address: {pos['token_address'][:16]}...")
            print(f"      Tokens: {pos['tokens_held']:.4f}")
            print(f"      Entry: ${pos['entry_price_usd']:.6f}")
            print(f"      Cost: {pos['total_cost_sol']:.4f} SOL")
            print(f"      SL/TP: {pos['stop_loss_pct']}% / {pos['take_profit_pct']}%")
            print(f"      Entry time: {pos['entry_time']}")
    except Exception as e:
        print(f"   ❌ Error reading positions: {e}")
    
    # Check 5: Cost basis lots
    print("\n💰 CHECK 5: Cost Basis Lots (FIFO)")
    print("-" * 40)
    
    try:
        lots = get_cost_basis_lots(db_path)
        print(f"   Recent lots: {len(lots)}")
        
        if lots:
            for lot in lots[:5]:
                remaining_pct = (lot['tokens_remaining'] / lot['tokens_acquired'] * 100) if lot['tokens_acquired'] > 0 else 0
                print(f"\n   📦 {lot['acquisition_date'][:10]}")
                print(f"      Token: {lot['token_address'][:16]}...")
                print(f"      Acquired: {lot['tokens_acquired']:.4f}")
                print(f"      Remaining: {lot['tokens_remaining']:.4f} ({remaining_pct:.0f}%)")
                print(f"      Cost: ${lot['total_cost_nzd']:.2f} NZD")
        else:
            print("   ⚠️ No cost basis lots found")
    except Exception as e:
        print(f"   ❌ Error reading cost basis lots: {e}")
    
    # Check 6: Daily stats
    print("\n📅 CHECK 6: Daily Statistics")
    print("-" * 40)
    
    try:
        stats = get_daily_stats(db_path)
        
        if stats:
            print(f"   Days with data: {len(stats)}")
            print()
            print("   Date       | Trades | Wins | Losses | P&L (SOL)")
            print("   " + "-" * 50)
            
            for day in stats:
                print(f"   {day['date']} |   {day['trades']:3}  |  {day['wins']:2}  |   {day['losses']:2}   | {day['pnl_sol']:+.4f}")
        else:
            print("   ⚠️ No daily stats recorded")
    except Exception as e:
        print(f"   ❌ Error reading daily stats: {e}")
    
    # Check 7: Execution errors
    print("\n🚨 CHECK 7: Execution Errors")
    print("-" * 40)
    
    try:
        summary = get_execution_log_summary(db_path)
        print(f"   Execution status summary: {summary}")
        
        errors = get_execution_errors(db_path)
        
        if errors:
            print(f"\n   Recent errors ({len(errors)}):")
            for err in errors[:10]:
                print(f"\n   ⚠️ {err['timestamp'][:19]}")
                print(f"      Action: {err['action']}")
                print(f"      Token: {err['token_symbol']}")
                print(f"      Status: {err['status']}")
                print(f"      Error: {err['error']}")
        else:
            print("   ✅ No recent errors")
    except Exception as e:
        print(f"   ❌ Error reading execution log: {e}")
    
    # Check 8: Recent transactions
    print("\n📝 CHECK 8: Recent Transactions")
    print("-" * 40)
    
    try:
        recent = get_recent_transactions(db_path)
        
        if recent:
            print(f"   Most recent {len(recent)} transactions:")
            print()
            
            for tx in recent:
                live_marker = "🟢" if tx['is_live'] else "⚪"
                tx_type = "BUY " if tx['transaction_type'] == 'BUY' else "SELL"
                sig = tx['signature'][:12] + "..." if tx['signature'] else "N/A"
                print(f"   {live_marker} {tx['timestamp'][:16]} | {tx_type} | {tx['token_symbol'][:8]:8} | {tx['sol_amount']:.4f} SOL | {sig}")
        else:
            print("   ⚠️ No transactions found")
    except Exception as e:
        print(f"   ❌ Error reading transactions: {e}")
    
    # Summary
    print("\n" + "=" * 70)
    print("📋 DIAGNOSIS SUMMARY")
    print("=" * 70)
    
    issues = []
    
    if not check_database_exists(db_path):
        issues.append("Database file does not exist")
    
    if tables['missing']:
        issues.append(f"Missing tables: {tables['missing']}")
    
    try:
        counts = get_transaction_counts(db_path)
        if counts['total'] == 0:
            issues.append("No transactions recorded")
        if counts['live_trades'] == 0:
            issues.append("No LIVE trades recorded (only paper?)")
    except:
        issues.append("Could not read transaction counts")
    
    try:
        errors = get_execution_errors(db_path)
        if len(errors) > 5:
            issues.append(f"High error rate: {len(errors)} recent errors")
    except:
        pass
    
    if issues:
        print("\n⚠️ ISSUES FOUND:")
        for i, issue in enumerate(issues, 1):
            print(f"   {i}. {issue}")
        
        print("\n🔧 POSSIBLE FIXES:")
        print("   1. Ensure live_trading_engine_v3.py is being used")
        print("   2. Check that TaxDatabase is properly initialized")
        print("   3. Verify record_trade() is called after successful swaps")
        print("   4. Check for exceptions in the execution flow")
        print("   5. Verify database file permissions")
    else:
        print("\n✅ No major issues detected")
        print("   Database appears to be properly tracking trades")
    
    print("\n" + "=" * 70 + "\n")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Diagnose live_trades_tax.db issues"
    )
    parser.add_argument(
        "--db",
        default="live_trades_tax.db",
        help="Path to database file (default: live_trades_tax.db)"
    )
    
    args = parser.parse_args()
    run_diagnostics(args.db)


if __name__ == "__main__":
    main()
