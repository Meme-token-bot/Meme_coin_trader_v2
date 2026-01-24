#!/usr/bin/env python3
"""
Profiler Diagnostic - See exactly what the profiler detects

Run: python profiler_diagnostic.py <wallet_address>
"""

import os
import sys
import requests
import time
from datetime import datetime, timedelta
from collections import defaultdict
from dotenv import load_dotenv

load_dotenv()

HELIUS_KEY = os.getenv('HELIUS_KEY')
HELIUS_API = f"https://api.helius.xyz/v0"

STABLES = {
    "So11111111111111111111111111111111111111112",   # WSOL
    "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v", # USDC
    "Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB", # USDT
}


def rate_limit():
    time.sleep(0.2)


def parse_swap(tx: dict, wallet: str) -> dict:
    """Parse a swap transaction"""
    token_transfers = tx.get('tokenTransfers', [])
    native_transfers = tx.get('nativeTransfers', [])
    timestamp = tx.get('timestamp', 0)
    signature = tx.get('signature', '')[:12]
    
    sol_in = sol_out = 0
    tokens_in = {}
    tokens_out = {}
    
    for transfer in token_transfers:
        mint = transfer.get('mint', '')
        amount = float(transfer.get('tokenAmount', 0) or 0)
        from_addr = transfer.get('fromUserAccount', '')
        to_addr = transfer.get('toUserAccount', '')
        
        if from_addr == wallet:
            if mint in STABLES:
                sol_out += amount
            else:
                tokens_out[mint] = tokens_out.get(mint, 0) + amount
        elif to_addr == wallet:
            if mint in STABLES:
                sol_in += amount
            else:
                tokens_in[mint] = tokens_in.get(mint, 0) + amount
    
    for transfer in native_transfers:
        amount = float(transfer.get('amount', 0) or 0) / 1e9
        from_addr = transfer.get('fromUserAccount', '')
        to_addr = transfer.get('toUserAccount', '')
        
        if from_addr == wallet:
            sol_out += amount
        elif to_addr == wallet:
            sol_in += amount
    
    # Determine swap type
    if len(tokens_in) >= 1 and sol_out > 0:
        token = max(tokens_in.keys(), key=lambda t: tokens_in[t])
        return {
            'type': 'BUY',
            'token': token[:12],
            'token_full': token,
            'token_amount': tokens_in[token],
            'sol_amount': sol_out,
            'timestamp': timestamp,
            'signature': signature,
        }
    elif len(tokens_out) >= 1 and sol_in > 0:
        token = max(tokens_out.keys(), key=lambda t: tokens_out[t])
        return {
            'type': 'SELL',
            'token': token[:12],
            'token_full': token,
            'token_amount': tokens_out[token],
            'sol_amount': sol_in,
            'timestamp': timestamp,
            'signature': signature,
        }
    
    # Couldn't determine type
    return {
        'type': 'UNKNOWN',
        'sol_in': sol_in,
        'sol_out': sol_out,
        'tokens_in': list(tokens_in.keys()),
        'tokens_out': list(tokens_out.keys()),
        'signature': signature,
    }


def get_wallet_swaps(wallet: str, days: int = 7) -> list:
    """Get swap transactions for wallet"""
    print(f"\n📡 Fetching swaps for {wallet[:16]}...")
    
    swaps = []
    before_sig = None
    
    for i in range(5):  # Max 5 pages
        rate_limit()
        
        url = f"{HELIUS_API}/addresses/{wallet}/transactions"
        params = {
            "api-key": HELIUS_KEY,
            "type": "SWAP",
            "limit": 100
        }
        
        if before_sig:
            params["before"] = before_sig
        
        try:
            res = requests.get(url, params=params, timeout=15)
            
            if res.status_code != 200:
                print(f"   Error: {res.status_code}")
                break
            
            data = res.json()
            
            if not data:
                break
            
            cutoff = datetime.now() - timedelta(days=days)
            
            for tx in data:
                timestamp = tx.get('timestamp', 0)
                
                if timestamp == 0:
                    continue
                
                tx_time = datetime.fromtimestamp(timestamp)
                
                if tx_time < cutoff:
                    print(f"   Reached {days}-day cutoff")
                    return swaps
                
                swaps.append(tx)
            
            print(f"   Page {i+1}: {len(data)} transactions")
            
            if len(data) < 100:
                break
            
            before_sig = data[-1].get('signature')
            
        except Exception as e:
            print(f"   Error: {e}")
            break
    
    return swaps


def analyze_wallet(wallet: str, days: int = 7):
    """Full analysis of wallet's trading"""
    print(f"\n{'='*70}")
    print(f"PROFILER DIAGNOSTIC: {wallet}")
    print(f"{'='*70}")
    
    raw_txs = get_wallet_swaps(wallet, days)
    print(f"   Total SWAP transactions: {len(raw_txs)}")
    
    if not raw_txs:
        print("\n❌ No swap transactions found!")
        return
    
    # Parse all swaps
    parsed = []
    for tx in raw_txs:
        swap = parse_swap(tx, wallet)
        parsed.append(swap)
    
    # Count by type
    buys = [s for s in parsed if s['type'] == 'BUY']
    sells = [s for s in parsed if s['type'] == 'SELL']
    unknown = [s for s in parsed if s['type'] == 'UNKNOWN']
    
    print(f"\n📊 Swap Detection Results:")
    print(f"   BUYs: {len(buys)}")
    print(f"   SELLs: {len(sells)}")
    print(f"   UNKNOWN: {len(unknown)}")
    
    # Group by token
    tokens_bought = defaultdict(list)
    tokens_sold = defaultdict(list)
    
    for s in buys:
        tokens_bought[s['token_full']].append(s)
    for s in sells:
        tokens_sold[s['token_full']].append(s)
    
    print(f"\n📈 Tokens Bought ({len(tokens_bought)} unique):")
    for token, txs in sorted(tokens_bought.items(), key=lambda x: -len(x[1]))[:10]:
        total_sol = sum(t['sol_amount'] for t in txs)
        print(f"   {token[:12]}... | {len(txs)} buys | {total_sol:.4f} SOL")
    
    print(f"\n📉 Tokens Sold ({len(tokens_sold)} unique):")
    for token, txs in sorted(tokens_sold.items(), key=lambda x: -len(x[1]))[:10]:
        total_sol = sum(t['sol_amount'] for t in txs)
        print(f"   {token[:12]}... | {len(txs)} sells | {total_sol:.4f} SOL")
    
    # Find completed swings (bought AND sold same token)
    completed_tokens = set(tokens_bought.keys()) & set(tokens_sold.keys())
    
    print(f"\n🔄 Completed Swings ({len(completed_tokens)} tokens):")
    
    if not completed_tokens:
        print("   ❌ NO COMPLETED SWINGS!")
        print("\n   This happens when:")
        print("   - Wallet buys Token A but only sells Token B, C, D...")
        print("   - Wallet hasn't sold any of their purchased tokens yet")
        print("\n   Recent buys (no matching sells):")
        for token in list(tokens_bought.keys())[:5]:
            if token not in tokens_sold:
                print(f"      {token[:20]}...")
    else:
        total_profit = 0
        winning = 0
        
        for token in completed_tokens:
            buy_txs = tokens_bought[token]
            sell_txs = tokens_sold[token]
            
            total_bought = sum(t['sol_amount'] for t in buy_txs)
            total_sold = sum(t['sol_amount'] for t in sell_txs)
            profit = total_sold - total_bought
            
            status = "✅" if profit > 0 else "❌"
            if profit > 0:
                winning += 1
            
            print(f"   {status} {token[:12]}... | Bought: {total_bought:.4f} SOL | Sold: {total_sold:.4f} SOL | P&L: {profit:+.4f} SOL")
            total_profit += profit
        
        print(f"\n📊 Summary:")
        print(f"   Completed swings: {len(completed_tokens)}")
        print(f"   Winning trades: {winning}")
        print(f"   Win rate: {100*winning/len(completed_tokens):.1f}%")
        print(f"   Total P&L: {total_profit:+.4f} SOL")
    
    # Show unknown transactions
    if unknown:
        print(f"\n⚠️ UNKNOWN Transactions ({len(unknown)}):")
        for u in unknown[:5]:
            print(f"   {u['signature']}... | SOL in: {u.get('sol_in', 0):.4f} | SOL out: {u.get('sol_out', 0):.4f}")
            print(f"      Tokens in: {u.get('tokens_in', [])} | Tokens out: {u.get('tokens_out', [])}")


def find_wallet_from_partial(partial: str) -> str:
    """Try to find full wallet address from partial match in database"""
    import sqlite3
    
    try:
        conn = sqlite3.connect('swing_traders.db')
        conn.row_factory = sqlite3.Row
        
        # Try verified_wallets table
        cursor = conn.execute(
            "SELECT address FROM verified_wallets WHERE address LIKE ?",
            (f"{partial}%",)
        )
        row = cursor.fetchone()
        if row:
            conn.close()
            return row['address']
        
        # Try wallet_webhook_assignments table
        cursor = conn.execute(
            "SELECT wallet_address FROM wallet_webhook_assignments WHERE wallet_address LIKE ?",
            (f"{partial}%",)
        )
        row = cursor.fetchone()
        if row:
            conn.close()
            return row['wallet_address']
        
        conn.close()
    except Exception as e:
        print(f"Database lookup failed: {e}")
    
    return None


def list_recent_wallets():
    """List recent wallets from database and debug file"""
    print("\n📋 Recent wallets you can analyze:\n")
    
    # From database
    import sqlite3
    try:
        conn = sqlite3.connect('swing_traders.db')
        conn.row_factory = sqlite3.Row
        
        cursor = conn.execute("""
            SELECT address, win_rate, pnl_7d, completed_swings 
            FROM verified_wallets 
            ORDER BY discovered_at DESC 
            LIMIT 10
        """)
        rows = cursor.fetchall()
        
        if rows:
            print("From database (verified):")
            for row in rows:
                print(f"   {row['address']}")
                print(f"      WR: {row['win_rate']:.0f}% | PnL: {row['pnl_7d']:.2f} SOL | Swings: {row['completed_swings']}")
        
        conn.close()
    except Exception as e:
        print(f"Database error: {e}")
    
    # From debug file
    try:
        import json
        with open('discovery_debug.json', 'r') as f:
            data = json.load(f)
        
        profiled = data.get('profiled_wallets', [])
        if profiled:
            print("\nFrom last discovery (profiled but may not be verified):")
            for w in profiled[:10]:
                if isinstance(w, dict):
                    addr = w.get('address', 'unknown')
                    wr = w.get('win_rate', 0)
                    pnl = w.get('pnl', 0)
                    swings = w.get('completed_swings', 0)
                    print(f"   {addr}")
                    print(f"      WR: {wr:.0f}% | PnL: {pnl:.2f} SOL | Swings: {swings}")
                else:
                    print(f"   {w}")
    except FileNotFoundError:
        print("\n(discovery_debug.json not found)")
    except Exception as e:
        print(f"\nDebug file error: {e}")


if __name__ == "__main__":
    if not HELIUS_KEY:
        print("❌ HELIUS_KEY not found in .env")
        sys.exit(1)
    
    if len(sys.argv) < 2:
        print("Usage: python profiler_diagnostic.py <wallet_address>")
        print("       python profiler_diagnostic.py list")
        print("       (can use partial address like 'EzB3Zv...')")
        list_recent_wallets()
        sys.exit(0)
    
    if sys.argv[1].lower() == 'list':
        list_recent_wallets()
        sys.exit(0)
    
    wallet = sys.argv[1]
    
    # If partial address, try to find full one
    if len(wallet) < 30:
        print(f"🔍 Looking up partial address: {wallet}")
        full_wallet = find_wallet_from_partial(wallet)
        if full_wallet:
            print(f"   Found: {full_wallet}")
            wallet = full_wallet
        else:
            print(f"❌ Could not find wallet starting with '{wallet}' in database")
            list_recent_wallets()
            sys.exit(1)
    
    analyze_wallet(wallet)
