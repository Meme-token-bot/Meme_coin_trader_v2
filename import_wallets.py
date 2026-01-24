# create import_wallets.py
from database_v2 import DatabaseV2
import sqlite3

def import_from_old_db():
    """Import top wallets from old database"""
    
    # Connect to OLD database
    old_db = sqlite3.connect('/home/barefootbushman/Documents/Swing_trading/swing_traders.db')  # Adjust path
    old_db.row_factory = sqlite3.Row
    
    # Connect to NEW database
    new_db = DatabaseV2()
    
    # Get top wallets from old DB (with quality filters)
    cursor = old_db.execute("""
        SELECT * FROM verified_wallets
        WHERE win_rate >= 0.55
        AND pnl_7d > 3.0
        AND completed_swings >= 5
        AND is_active = 1
        ORDER BY win_rate DESC, pnl_7d DESC
        LIMIT 60
    """)
    
    wallets = [dict(row) for row in cursor.fetchall()]
    
    print(f"\n{'='*60}")
    print(f"IMPORTING {len(wallets)} TOP WALLETS FROM OLD DATABASE")
    print(f"{'='*60}\n")
    
    for wallet in wallets:
        addr = wallet['address']
        
        # Prepare data for new DB
        wallet_data = {
            'win_rate': wallet.get('win_rate', 0),
            'pnl': wallet.get('pnl_7d', 0),
            'roi_7d': wallet.get('roi_7d', 0),
            'completed_swings': wallet.get('completed_swings', 0),
            'avg_hold_hours': wallet.get('avg_hold_hours', 0),
            'risk_reward_ratio': wallet.get('risk_reward_ratio', 0),
            'best_trade_pct': wallet.get('best_trade_pct', 0),
            'worst_trade_pct': wallet.get('worst_trade_pct', 0),
            'total_volume_sol': wallet.get('total_volume_sol', 0),
            'avg_position_size_sol': wallet.get('avg_position_size_sol', 0)
        }
        
        new_db.add_verified_wallet(addr, wallet_data)
        
        print(f"✅ {addr[:8]}... | WR: {wallet_data['win_rate']:.1%} | PnL: {wallet_data['pnl']:.2f} SOL")
    
    old_db.close()
    
    print(f"\n{'='*60}")
    print(f"✅ Imported {len(wallets)} wallets")
    print(f"{'='*60}")
    print("\n⚠️ NEXT STEPS:")
    print("1. Add these wallet addresses to your Helius webhook")
    print("2. You can export the list with: python import_wallets.py list")
    
    return wallets

def export_addresses_for_helius():
    """Export just the addresses for easy copy-paste into Helius"""
    db = DatabaseV2()
    wallets = db.get_all_verified_wallets()
    
    print("\n" + "="*60)
    print(f"WALLET ADDRESSES FOR HELIUS ({len(wallets)} total)")
    print("="*60)
    print("\nCopy these addresses to your Helius webhook:\n")
    
    for wallet in wallets:
        print(wallet['address'])
    
    print("\n" + "="*60 + "\n")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'list':
        export_addresses_for_helius()
    else:
        import_from_old_db()