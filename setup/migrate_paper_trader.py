"""
PAPER TRADER MIGRATION UTILITY
==============================

This script helps you:
1. Clean up the existing 56+ positions mess
2. Migrate data to the new paper trader
3. Analyze what went wrong

Run with: python migrate_paper_trader.py [command]
"""

import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # Go up ONE level
import json
import sqlite3
import requests
from datetime import datetime, timedelta
from collections import defaultdict


def get_token_price(token_address: str) -> float:
    """Get current token price"""
    try:
        url = f"https://api.dexscreener.com/latest/dex/tokens/{token_address}"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            pairs = response.json().get('pairs', [])
            if pairs:
                pair = max(pairs, key=lambda p: float(p.get('liquidity', {}).get('usd', 0) or 0))
                return float(pair.get('priceUsd', 0) or 0)
        return 0
    except:
        return 0


def analyze_old_positions(db_path: str = 'swing_traders.db'):
    """
    Analyze the existing position problem.
    """
    print("\n" + "="*70)
    print("📊 ANALYZING EXISTING POSITIONS")
    print("="*70)
    
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    
    # Check tracked_positions table
    positions = conn.execute("""
        SELECT * FROM tracked_positions WHERE status = 'open'
    """).fetchall()
    
    print(f"\n📦 Found {len(positions)} open positions in tracked_positions")
    
    if not positions:
        # Try paper_positions table
        try:
            positions = conn.execute("""
                SELECT * FROM paper_positions WHERE status = 'open'
            """).fetchall()
            print(f"📦 Found {len(positions)} open positions in paper_positions")
        except:
            pass
    
    if not positions:
        print("No open positions found in database.")
        conn.close()
        return
    
    # Analyze
    analysis = {
        'total': len(positions),
        'by_token': defaultdict(lambda: {'count': 0, 'total_sol': 0}),
        'by_wallet': defaultdict(lambda: {'count': 0}),
        'old_positions': [],  # > 12h
        'very_old_positions': [],  # > 24h
    }
    
    for pos in positions:
        pos = dict(pos)
        symbol = pos.get('token_symbol', 'UNKNOWN')
        size = pos.get('position_size_sol', 0) or pos.get('size_sol', 0) or 0
        wallet = pos.get('wallet_address', 'unknown')[:12]
        entry_time = pos.get('entry_time')
        
        analysis['by_token'][symbol]['count'] += 1
        analysis['by_token'][symbol]['total_sol'] += size
        analysis['by_wallet'][wallet]['count'] += 1
        
        if entry_time:
            if isinstance(entry_time, str):
                try:
                    entry_time = datetime.fromisoformat(entry_time.replace('Z', ''))
                except:
                    entry_time = None
            
            if entry_time:
                hold_hours = (datetime.now() - entry_time).total_seconds() / 3600
                if hold_hours > 24:
                    analysis['very_old_positions'].append({
                        'id': pos.get('id'),
                        'symbol': symbol,
                        'hold_hours': hold_hours
                    })
                elif hold_hours > 12:
                    analysis['old_positions'].append({
                        'id': pos.get('id'),
                        'symbol': symbol,
                        'hold_hours': hold_hours
                    })
    
    # Print analysis
    print(f"\n📊 TOKEN CONCENTRATION:")
    sorted_tokens = sorted(analysis['by_token'].items(), key=lambda x: -x[1]['count'])
    for token, data in sorted_tokens[:10]:
        print(f"   {token}: {data['count']} positions, {data['total_sol']:.4f} SOL")
    
    print(f"\n⏰ STALE POSITIONS:")
    print(f"   > 12 hours: {len(analysis['old_positions'])}")
    print(f"   > 24 hours: {len(analysis['very_old_positions'])}")
    
    for pos in analysis['very_old_positions'][:5]:
        print(f"      {pos['symbol']}: {pos['hold_hours']:.1f}h")
    
    conn.close()
    return analysis


def close_all_stale_positions(db_path: str = 'swing_traders.db', max_hours: float = 12):
    """
    Close all positions older than max_hours.
    """
    print(f"\n🧹 CLOSING POSITIONS OLDER THAN {max_hours} HOURS")
    print("="*70)
    
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    
    # Get stale positions
    positions = conn.execute("""
        SELECT * FROM tracked_positions WHERE status = 'open'
    """).fetchall()
    
    if not positions:
        try:
            positions = conn.execute("""
                SELECT * FROM paper_positions WHERE status = 'open'
            """).fetchall()
            table = 'paper_positions'
        except:
            print("No positions found")
            conn.close()
            return
    else:
        table = 'tracked_positions'
    
    closed_count = 0
    total_pnl = 0
    
    for pos in positions:
        pos = dict(pos)
        entry_time = pos.get('entry_time')
        
        if not entry_time:
            continue
        
        if isinstance(entry_time, str):
            try:
                entry_time = datetime.fromisoformat(entry_time.replace('Z', ''))
            except:
                continue
        
        hold_hours = (datetime.now() - entry_time).total_seconds() / 3600
        
        if hold_hours > max_hours:
            # Get current price
            token_address = pos.get('token_address', '')
            current_price = get_token_price(token_address)
            entry_price = pos.get('entry_price', 0)
            size_sol = pos.get('position_size_sol', 0) or pos.get('size_sol', 0)
            
            if entry_price > 0 and current_price > 0:
                pnl_pct = ((current_price / entry_price) - 1) * 100
                pnl_sol = size_sol * (pnl_pct / 100)
            else:
                pnl_pct = 0
                pnl_sol = 0
            
            # Close position
            conn.execute(f"""
                UPDATE {table}
                SET status = 'closed', exit_reason = 'TIME_STOP', 
                    exit_price = ?, exit_time = ?, profit_pct = ?, profit_sol = ?
                WHERE id = ?
            """, (current_price, datetime.now(), pnl_pct, pnl_sol, pos['id']))
            
            emoji = "✅" if pnl_sol > 0 else "❌"
            print(f"   {emoji} Closed {pos.get('token_symbol', 'UNKNOWN')}: {pnl_pct:+.1f}% ({pnl_sol:+.4f} SOL) - held {hold_hours:.1f}h")
            
            closed_count += 1
            total_pnl += pnl_sol
    
    conn.commit()
    conn.close()
    
    print(f"\n📊 SUMMARY:")
    print(f"   Closed: {closed_count} positions")
    print(f"   Total PnL: {total_pnl:+.4f} SOL")


def close_duplicate_positions(db_path: str = 'swing_traders.db'):
    """
    Close duplicate positions in the same token (keep oldest).
    """
    print("\n🧹 CLOSING DUPLICATE POSITIONS")
    print("="*70)
    
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    
    # Get all open positions
    positions = conn.execute("""
        SELECT * FROM tracked_positions WHERE status = 'open'
        ORDER BY entry_time ASC
    """).fetchall()
    
    if not positions:
        try:
            positions = conn.execute("""
                SELECT * FROM paper_positions WHERE status = 'open'
                ORDER BY entry_time ASC
            """).fetchall()
            table = 'paper_positions'
        except:
            print("No positions found")
            conn.close()
            return
    else:
        table = 'tracked_positions'
    
    # Group by token
    by_token = defaultdict(list)
    for pos in positions:
        pos = dict(pos)
        token = pos.get('token_address', '')
        by_token[token].append(pos)
    
    closed_count = 0
    
    for token, token_positions in by_token.items():
        if len(token_positions) > 1:
            # Keep the first (oldest), close the rest
            symbol = token_positions[0].get('token_symbol', 'UNKNOWN')
            print(f"   {symbol}: {len(token_positions)} positions, keeping oldest")
            
            for pos in token_positions[1:]:  # Skip first
                conn.execute(f"""
                    UPDATE {table}
                    SET status = 'closed', exit_reason = 'DUPLICATE_CLEANUP',
                        exit_time = ?
                    WHERE id = ?
                """, (datetime.now(), pos['id']))
                closed_count += 1
    
    conn.commit()
    conn.close()
    
    print(f"\n📊 Closed {closed_count} duplicate positions")


def reset_paper_account(db_path: str = 'swing_traders.db', starting_balance: float = 10.0):
    """
    Reset the paper trading account to start fresh.
    This closes all positions and resets the balance.
    """
    print("\n🔄 RESETTING PAPER ACCOUNT")
    print("="*70)
    print(f"⚠️ This will close all positions and reset to {starting_balance} SOL")
    
    confirm = input("Type 'RESET' to confirm: ")
    if confirm != 'RESET':
        print("Aborted.")
        return
    
    conn = sqlite3.connect(db_path)
    
    # Close all open positions
    tables = ['tracked_positions', 'paper_positions']
    for table in tables:
        try:
            conn.execute(f"""
                UPDATE {table}
                SET status = 'closed', exit_reason = 'ACCOUNT_RESET', exit_time = ?
                WHERE status = 'open'
            """, (datetime.now(),))
            print(f"   Closed all positions in {table}")
        except Exception as e:
            pass
    
    # Reset account if it exists
    try:
        conn.execute("""
            UPDATE paper_account
            SET current_balance = ?, total_trades = 0, winning_trades = 0, 
                total_pnl = 0, last_updated = ?
        """, (starting_balance, datetime.now()))
        print("   Reset paper_account")
    except:
        pass
    
    conn.commit()
    conn.close()
    
    print(f"\n✅ Account reset to {starting_balance} SOL")


def migrate_to_new_system(old_db: str = 'swing_traders.db', new_db: str = 'paper_trades_v3.db'):
    """
    Migrate closed trade history to the new paper trader for analysis.
    """
    print("\n📦 MIGRATING TRADE HISTORY TO NEW SYSTEM")
    print("="*70)
    
    old_conn = sqlite3.connect(old_db)
    old_conn.row_factory = sqlite3.Row
    
    # Get closed positions
    try:
        closed = old_conn.execute("""
            SELECT * FROM tracked_positions 
            WHERE status = 'closed' 
            ORDER BY exit_time DESC LIMIT 500
        """).fetchall()
    except:
        closed = []
    
    if not closed:
        try:
            closed = old_conn.execute("""
                SELECT * FROM paper_positions 
                WHERE status = 'closed' 
                ORDER BY exit_time DESC LIMIT 500
            """).fetchall()
        except:
            closed = []
    
    print(f"Found {len(closed)} closed trades to migrate")
    
    if not closed:
        old_conn.close()
        return
    
    # Import to new system
    from core.effective_paper_trader import EffectivePaperTrader, PaperTraderConfig
    
    config = PaperTraderConfig(enable_auto_exits=False)
    trader = EffectivePaperTrader(db_path=new_db, config=config)
    
    new_conn = sqlite3.connect(new_db)
    
    migrated = 0
    for pos in closed:
        pos = dict(pos)
        
        try:
            new_conn.execute("""
                INSERT OR IGNORE INTO paper_positions_v3
                (token_address, token_symbol, entry_price, entry_time, size_sol, tokens_bought,
                 stop_loss_pct, take_profit_pct, trailing_stop_pct, max_hold_hours,
                 status, exit_price, exit_time, exit_reason, pnl_sol, pnl_pct, hold_duration_minutes,
                 peak_unrealized_pct, entry_context_json, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                pos.get('token_address', ''),
                pos.get('token_symbol', 'UNKNOWN'),
                pos.get('entry_price', 0),
                pos.get('entry_time'),
                pos.get('position_size_sol', 0) or pos.get('size_sol', 0),
                0,  # tokens_bought - calculate if needed
                pos.get('stop_loss_pct', -12),
                pos.get('take_profit_pct', 30),
                pos.get('trailing_stop_pct', 8),
                pos.get('max_hold_hours', 12),
                'closed',
                pos.get('exit_price', 0),
                pos.get('exit_time'),
                pos.get('exit_reason', 'UNKNOWN'),
                pos.get('profit_sol', 0) or pos.get('pnl_sol', 0),
                pos.get('profit_pct', 0) or pos.get('pnl_pct', 0),
                pos.get('hold_duration_minutes', 0),
                pos.get('peak_unrealized_pct', 0),
                json.dumps({
                    'wallet_source': pos.get('wallet_address', ''),
                    'conviction_score': pos.get('conviction_score', 50),
                    'liquidity_usd': pos.get('entry_liquidity', 0),
                    'strategy_name': pos.get('strategy_name', 'migrated')
                }),
                'Migrated from old system'
            ))
            migrated += 1
        except Exception as e:
            pass
    
    new_conn.commit()
    new_conn.close()
    old_conn.close()
    
    print(f"✅ Migrated {migrated} trades to new system")


def generate_postmortem(db_path: str = 'swing_traders.db'):
    """
    Generate a postmortem analysis of what went wrong.
    """
    print("\n" + "="*70)
    print("📋 POSITION EXPLOSION POSTMORTEM")
    print("="*70)
    
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    
    # Get all positions (open and closed)
    positions = conn.execute("""
        SELECT * FROM tracked_positions
        ORDER BY entry_time ASC
    """).fetchall()
    
    if not positions:
        try:
            positions = conn.execute("""
                SELECT * FROM paper_positions
                ORDER BY entry_time ASC
            """).fetchall()
        except:
            pass
    
    if not positions:
        print("No positions found.")
        conn.close()
        return
    
    # Analyze the pattern
    positions = [dict(p) for p in positions]
    
    print(f"\n📊 OVERVIEW:")
    print(f"   Total positions: {len(positions)}")
    print(f"   Open: {sum(1 for p in positions if p.get('status') == 'open')}")
    print(f"   Closed: {sum(1 for p in positions if p.get('status') != 'open')}")
    
    # Time distribution
    print(f"\n⏰ POSITION OPENING TIMELINE:")
    by_hour = defaultdict(int)
    for p in positions:
        entry_time = p.get('entry_time')
        if entry_time:
            if isinstance(entry_time, str):
                try:
                    entry_time = datetime.fromisoformat(entry_time.replace('Z', ''))
                    hour = entry_time.strftime('%Y-%m-%d %H:00')
                    by_hour[hour] += 1
                except:
                    pass
    
    for hour in sorted(by_hour.keys())[-10:]:
        print(f"   {hour}: {by_hour[hour]} positions opened")
    
    # Root cause analysis
    print(f"\n🔍 ROOT CAUSE ANALYSIS:")
    
    # Check for duplicate tokens
    by_token = defaultdict(list)
    for p in positions:
        if p.get('status') == 'open':
            token = p.get('token_address', '')
            by_token[token].append(p)
    
    duplicates = {k: v for k, v in by_token.items() if len(v) > 1}
    print(f"   Duplicate positions (same token): {len(duplicates)} tokens")
    
    # Check if position limit was ignored
    open_at_any_time = []
    for p in positions:
        entry_time = p.get('entry_time')
        exit_time = p.get('exit_time')
        if entry_time:
            open_at_any_time.append({
                'entry': entry_time,
                'exit': exit_time or 'still_open',
                'symbol': p.get('token_symbol')
            })
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    print("   1. The position limit check wasn't atomic - another thread/process")
    print("      could open positions between the check and insert.")
    print("   2. Consider using database-level constraints or transactions.")
    print("   3. The exit monitoring loop may not have been running.")
    print("   4. Implement a rate limiter on position opens (max 1 per minute).")
    print("   5. Add alerts when position count exceeds threshold.")
    
    conn.close()


def main():
    """Main entry point"""
    if len(sys.argv) < 2:
        print("""
Paper Trader Migration Utility
==============================

Commands:
  analyze         - Analyze existing positions
  close-stale     - Close positions older than 12 hours
  close-duplicates - Close duplicate positions (keep oldest)
  reset           - Reset paper account (CAUTION: closes all)
  migrate         - Migrate trade history to new system
  postmortem      - Generate analysis of what went wrong
  all             - Run close-stale + close-duplicates + migrate

Examples:
  python migrate_paper_trader.py analyze
  python migrate_paper_trader.py close-stale
  python migrate_paper_trader.py all
        """)
        return
    
    command = sys.argv[1].lower()
    
    if command == 'analyze':
        analyze_old_positions()
    
    elif command == 'close-stale':
        max_hours = float(sys.argv[2]) if len(sys.argv) > 2 else 12
        close_all_stale_positions(max_hours=max_hours)
    
    elif command == 'close-duplicates':
        close_duplicate_positions()
    
    elif command == 'reset':
        balance = float(sys.argv[2]) if len(sys.argv) > 2 else 10.0
        reset_paper_account(starting_balance=balance)
    
    elif command == 'migrate':
        migrate_to_new_system()
    
    elif command == 'postmortem':
        generate_postmortem()
    
    elif command == 'all':
        print("Running full cleanup and migration...")
        close_all_stale_positions()
        close_duplicate_positions()
        migrate_to_new_system()
        print("\n✅ Cleanup complete!")
    
    else:
        print(f"Unknown command: {command}")


if __name__ == "__main__":
    main()
