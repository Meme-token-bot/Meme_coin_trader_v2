#!/usr/bin/env python3
"""
RESET AND SYNC PAPER TRADER WITH LIVE
======================================

This script:
1. Resets paper trading database (fresh start)
2. Syncs paper trader settings to match live trader
3. Both start from the same point for fair comparison

Run on EC2:
    python reset_paper_to_match_live.py

"""

import os
import sys
import shutil
import sqlite3
from datetime import datetime

# Paper trading database files to reset
PAPER_DB_FILES = [
    'paper_trading.db',
    'robust_paper_trades_v6.db',
    'paper_trades.db',
]

# Settings to sync (Paper should match Live)
SYNC_SETTINGS = {
    'position_size_sol': 0.3,      # Same as live
    'max_positions': 3,             # Bootstrap mode (2.91 SOL)
    'starting_balance': 2.91,       # Same as live wallet
    'stop_loss_pct': -0.15,         # -15%
    'take_profit_pct': 0.30,        # +30%
    'max_hold_hours': 12,
    'blocked_hours': [1, 3, 5, 19, 23],
    'min_conviction': 60,
}


def backup_databases():
    """Backup existing paper trading databases"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_dir = f'backups/paper_trading_{timestamp}'
    
    os.makedirs(backup_dir, exist_ok=True)
    
    backed_up = []
    for db_file in PAPER_DB_FILES:
        if os.path.exists(db_file):
            backup_path = os.path.join(backup_dir, db_file)
            shutil.copy2(db_file, backup_path)
            backed_up.append(db_file)
            print(f"  ✅ Backed up: {db_file} → {backup_path}")
    
    if backed_up:
        print(f"\n📁 Backups saved to: {backup_dir}")
    else:
        print("  ℹ️  No existing databases to backup")
    
    return backup_dir


def reset_databases():
    """Delete paper trading databases for fresh start"""
    deleted = []
    for db_file in PAPER_DB_FILES:
        if os.path.exists(db_file):
            os.remove(db_file)
            deleted.append(db_file)
            print(f"  🗑️  Deleted: {db_file}")
    
    if not deleted:
        print("  ℹ️  No databases to delete (already clean)")
    
    return deleted


def create_fresh_paper_db(settings: dict):
    """Create fresh paper trading database with synced settings"""
    db_path = 'robust_paper_trades_v6.db'
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create tables
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS positions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            token_address TEXT NOT NULL,
            token_symbol TEXT,
            entry_price REAL NOT NULL,
            entry_time TEXT NOT NULL,
            position_size_sol REAL NOT NULL,
            tokens_held REAL NOT NULL,
            stop_loss_price REAL,
            take_profit_price REAL,
            trailing_stop_pct REAL,
            highest_price REAL,
            status TEXT DEFAULT 'OPEN',
            exit_price REAL,
            exit_time TEXT,
            exit_reason TEXT,
            pnl_sol REAL,
            pnl_pct REAL,
            conviction_score INTEGER,
            wallet_address TEXT,
            entry_mcap REAL,
            entry_liquidity REAL,
            entry_holder_count INTEGER
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS balance_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            balance REAL NOT NULL,
            event_type TEXT,
            position_id INTEGER
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS config (
            key TEXT PRIMARY KEY,
            value TEXT
        )
    ''')
    
    # Insert synced settings
    for key, value in settings.items():
        cursor.execute(
            'INSERT OR REPLACE INTO config (key, value) VALUES (?, ?)',
            (key, str(value))
        )
    
    # Record starting balance
    cursor.execute('''
        INSERT INTO balance_history (timestamp, balance, event_type)
        VALUES (?, ?, 'INITIAL')
    ''', (datetime.utcnow().isoformat(), settings['starting_balance']))
    
    conn.commit()
    conn.close()
    
    print(f"  ✅ Created fresh database: {db_path}")
    print(f"     Starting balance: {settings['starting_balance']} SOL")
    print(f"     Max positions: {settings['max_positions']}")
    print(f"     Position size: {settings['position_size_sol']} SOL")


def print_sync_settings(settings: dict):
    """Display the synced settings"""
    print("\n" + "=" * 50)
    print("📊 PAPER ↔ LIVE SYNC SETTINGS")
    print("=" * 50)
    print(f"""
    Starting Balance: {settings['starting_balance']} SOL
    Position Size:    {settings['position_size_sol']} SOL
    Max Positions:    {settings['max_positions']}
    Stop Loss:        {settings['stop_loss_pct']*100:.0f}%
    Take Profit:      {settings['take_profit_pct']*100:.0f}%
    Max Hold:         {settings['max_hold_hours']} hours
    Min Conviction:   {settings['min_conviction']}%
    Blocked Hours:    {settings['blocked_hours']} UTC
    """)


def main():
    print("\n" + "=" * 60)
    print("🔄 RESET PAPER TRADER TO MATCH LIVE")
    print("=" * 60)
    
    print("""
This will:
1. Backup existing paper trading data
2. Delete paper trading databases
3. Create fresh database with settings matching LIVE trader

Paper and Live will start from the SAME point for fair comparison.
    """)
    
    # Show what settings will be used
    print_sync_settings(SYNC_SETTINGS)
    
    response = input("\nProceed? (yes/no): ").strip().lower()
    if response != 'yes':
        print("Cancelled.")
        return
    
    print("\n" + "-" * 40)
    print("Step 1: Backing up existing data...")
    print("-" * 40)
    backup_databases()
    
    print("\n" + "-" * 40)
    print("Step 2: Deleting old databases...")
    print("-" * 40)
    reset_databases()
    
    print("\n" + "-" * 40)
    print("Step 3: Creating fresh synced database...")
    print("-" * 40)
    create_fresh_paper_db(SYNC_SETTINGS)
    
    print("\n" + "=" * 60)
    print("✅ RESET COMPLETE!")
    print("=" * 60)
    print("""
Both Paper and Live now start from:
  • Balance: 2.91 SOL
  • Max Positions: 3
  • Position Size: 0.3 SOL

Restart the bot to apply:
  sudo systemctl restart solana-trader
    """)


if __name__ == "__main__":
    main()
