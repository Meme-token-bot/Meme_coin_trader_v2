"""
Seed Wallets Script
Use this to manually add verified wallets to your database
"""

from core.database_v2 import DatabaseV2
from datetime import datetime

def seed_wallets():
    """
    Manually add known profitable wallets to the database.
    
    Add wallet addresses you want to track. You'll need to:
    1. Find these wallets on Solscan, DEXScreener, or Bubblemaps
    2. Verify they're active traders
    3. Add them to this list
    4. Run this script
    5. Add the same addresses to your Helius webhook
    """
    
    db = DatabaseV2()
    
    # Example wallets - REPLACE THESE with actual wallet addresses
    # These are placeholder values
    wallets_to_add = [
        {
            'address': 'HDBd1j2GCbzJ8iaVygWKTqR4bw6FYdauMta3mUX4KwZ9',
            'win_rate': 0.888888888888889,
            'pnl': 0.334524394,
            'roi_7d': 75.8548977541245,
            'completed_swings': 9,
            'avg_hold_hours': 4.5,
            'risk_reward_ratio': 2.0,
            'best_trade_pct': 125.648835030496,
            'worst_trade_pct': -5.04631208654079,
            'total_volume_sol': 50.0,
            'avg_position_size_sol': 3.0
        },
        # Add more wallets here...
        # {
        #     'address': 'ANOTHER_WALLET_ADDRESS_HERE',
        #     'win_rate': 0.60,
        #     'pnl': 8.0,
        #     ...
        # },
    ]
    
    print("\n" + "="*60)
    print("WALLET SEEDING SCRIPT")
    print("="*60)
    print(f"\nAdding {len(wallets_to_add)} wallet(s)...\n")
    
    for wallet_data in wallets_to_add:
        addr = wallet_data.pop('address')
        
        # Check if already exists
        if db.is_wallet_tracked(addr):
            print(f"⏭️  {addr[:8]}... - Already tracked")
            continue
        
        db.add_verified_wallet(addr, wallet_data)
        print(f"✅ {addr[:8]}... - Added")
        print(f"   WR: {wallet_data.get('win_rate', 0):.1%} | PnL: {wallet_data.get('pnl', 0):.2f} SOL")
    
    print(f"\n" + "="*60)
    print(f"✅ Seeding complete! Total wallets: {db.get_wallet_count()}")
    print("="*60)
    
    print("\n⚠️  IMPORTANT NEXT STEPS:")
    print("1. Add these wallet addresses to your Helius webhook")
    print("2. Go to Helius dashboard → Webhooks → Your webhook")
    print("3. Click 'Manage Addresses' and paste each address")
    print("4. Save and wait for trades!\n")


def list_tracked_wallets():
    """Show all currently tracked wallets"""
    db = DatabaseV2()
    
    wallets = db.get_all_verified_wallets()
    
    print("\n" + "="*60)
    print(f"TRACKED WALLETS ({len(wallets)} total)")
    print("="*60)
    
    if not wallets:
        print("\n  No wallets tracked yet.")
        print("  Run this script with wallet addresses to seed the database.\n")
        return
    
    for wallet in wallets:
        addr = wallet['address']
        wr = wallet.get('win_rate', 0)
        pnl = wallet.get('pnl_7d', 0)
        swings = wallet.get('completed_swings', 0)
        cluster = wallet.get('cluster', 'BALANCED')
        
        print(f"\n  {addr}")
        print(f"    Cluster: {cluster} | WR: {wr:.1%} | PnL: {pnl:.2f} SOL | Swings: {swings}")
    
    print("\n" + "="*60 + "\n")


def remove_wallet(address: str):
    """Remove a wallet from tracking"""
    db = DatabaseV2()
    
    with db.connection() as conn:
        result = conn.execute(
            "UPDATE verified_wallets SET is_active = 0 WHERE address = ?",
            (address,)
        )
        
        if result.rowcount > 0:
            print(f"✅ Deactivated wallet: {address[:8]}...")
        else:
            print(f"❌ Wallet not found: {address[:8]}...")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == 'list':
            list_tracked_wallets()
        elif command == 'remove' and len(sys.argv) > 2:
            remove_wallet(sys.argv[2])
        elif command == 'help':
            print("""
Usage:
  python seed_wallets.py          - Add wallets from the script
  python seed_wallets.py list     - List all tracked wallets
  python seed_wallets.py remove <address>  - Remove a wallet
  python seed_wallets.py help     - Show this help
            """)
        else:
            print("Unknown command. Use 'help' for usage.")
    else:
        # Default: seed wallets
        seed_wallets()
