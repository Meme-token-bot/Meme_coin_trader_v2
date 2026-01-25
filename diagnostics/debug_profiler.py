"""
Debug Wallet Profiling
See exactly what's happening when we try to profile a wallet
"""

import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # Go up ONE level
import requests
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

def debug_wallet_profile(wallet_address: str):
    """Debug why a wallet shows 0 trades"""
    
    helius_key = os.getenv('HELIUS_KEY')
    if not helius_key:
        print("❌ HELIUS_KEY not set")
        return
    
    helius_rpc = f"https://mainnet.helius-rpc.com/?api-key={helius_key}"
    helius_api = f"https://api.helius.xyz/v0"
    
    print(f"\n{'='*70}")
    print(f"🔍 DEBUG WALLET PROFILING")
    print(f"   Wallet: {wallet_address}")
    print(f"{'='*70}")
    
    # Step 1: Get recent signatures
    print(f"\n1. FETCHING RECENT SIGNATURES...")
    
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "getSignaturesForAddress",
        "params": [wallet_address, {"limit": 50}]
    }
    
    res = requests.post(helius_rpc, json=payload, timeout=15)
    signatures = res.json().get('result', [])
    
    print(f"   Found {len(signatures)} signatures")
    
    if not signatures:
        print("   ❌ No signatures found - wallet might be inactive")
        return
    
    # Show some signature info
    print(f"\n   Recent transactions:")
    for sig in signatures[:5]:
        sig_str = sig.get('signature', '')[:20]
        block_time = sig.get('blockTime', 0)
        time_str = datetime.fromtimestamp(block_time).strftime('%Y-%m-%d %H:%M') if block_time else 'Unknown'
        print(f"      {sig_str}... | {time_str}")
    
    # Step 2: Parse transactions with Helius
    print(f"\n2. PARSING TRANSACTIONS WITH HELIUS...")
    
    sig_list = [s.get('signature') for s in signatures[:20]]
    
    parse_url = f"{helius_api}/transactions?api-key={helius_key}"
    parse_res = requests.post(parse_url, json={"transactions": sig_list}, timeout=15)
    parsed_txs = parse_res.json()
    
    if not isinstance(parsed_txs, list):
        print(f"   ❌ Unexpected response: {type(parsed_txs)}")
        print(f"   Response: {str(parsed_txs)[:200]}")
        return
    
    print(f"   Parsed {len(parsed_txs)} transactions")
    
    # Count transaction types
    tx_types = {}
    swaps = []
    
    for tx in parsed_txs:
        if not isinstance(tx, dict):
            continue
        
        tx_type = tx.get('type', 'UNKNOWN')
        tx_types[tx_type] = tx_types.get(tx_type, 0) + 1
        
        if tx_type == 'SWAP':
            swaps.append(tx)
    
    print(f"\n   Transaction types:")
    for tx_type, count in sorted(tx_types.items(), key=lambda x: -x[1]):
        print(f"      {tx_type}: {count}")
    
    # Step 3: Analyze swaps
    print(f"\n3. ANALYZING SWAP TRANSACTIONS...")
    print(f"   Found {len(swaps)} SWAP transactions")
    
    if not swaps:
        print("\n   ❌ NO SWAPS FOUND!")
        print("   This wallet has transactions but none are SWAP type.")
        print("   Possible reasons:")
        print("   - Wallet does transfers, not swaps")
        print("   - Wallet interacts with DeFi in non-swap ways")
        print("   - Recent activity is all non-trading")
        return
    
    # Parse each swap
    print(f"\n   Swap details:")
    
    buys = []
    sells = []
    
    WSOL = "So11111111111111111111111111111111111111112"
    
    for i, swap in enumerate(swaps[:10]):
        fee_payer = swap.get('feePayer', '')
        token_transfers = swap.get('tokenTransfers', [])
        native_transfers = swap.get('nativeTransfers', [])
        
        print(f"\n   Swap #{i+1}:")
        print(f"      Fee payer: {fee_payer[:12]}...")
        print(f"      Token transfers: {len(token_transfers)}")
        print(f"      Native transfers: {len(native_transfers)}")
        
        # Analyze transfers
        sol_in, sol_out = 0, 0
        tokens_in, tokens_out = {}, {}
        
        for transfer in token_transfers:
            mint = transfer.get('mint', '')
            amount = float(transfer.get('tokenAmount', 0) or 0)
            from_addr = transfer.get('fromUserAccount', '')
            to_addr = transfer.get('toUserAccount', '')
            
            print(f"      Token: {mint[:8]}... | Amount: {amount:.4f}")
            print(f"         From: {from_addr[:12] if from_addr else 'N/A'}...")
            print(f"         To: {to_addr[:12] if to_addr else 'N/A'}...")
            
            if from_addr == fee_payer:
                if mint == WSOL:
                    sol_out += amount
                else:
                    tokens_out[mint] = tokens_out.get(mint, 0) + amount
            elif to_addr == fee_payer:
                if mint == WSOL:
                    sol_in += amount
                else:
                    tokens_in[mint] = tokens_in.get(mint, 0) + amount
        
        for transfer in native_transfers:
            amount = float(transfer.get('amount', 0) or 0) / 1e9
            from_addr = transfer.get('fromUserAccount', '')
            to_addr = transfer.get('toUserAccount', '')
            
            if from_addr == fee_payer:
                sol_out += amount
            elif to_addr == fee_payer:
                sol_in += amount
        
        # Determine trade type
        print(f"\n      Analysis:")
        print(f"         SOL in: {sol_in:.4f}")
        print(f"         SOL out: {sol_out:.4f}")
        print(f"         Tokens in: {list(tokens_in.keys())}")
        print(f"         Tokens out: {list(tokens_out.keys())}")
        
        if len(tokens_in) == 1 and sol_out > 0:
            token = list(tokens_in.keys())[0]
            print(f"         → BUY {tokens_in[token]:.4f} of {token[:8]}... for {sol_out:.4f} SOL")
            buys.append({'token': token, 'amount': tokens_in[token], 'sol': sol_out})
        elif len(tokens_out) == 1 and sol_in > 0:
            token = list(tokens_out.keys())[0]
            print(f"         → SELL {tokens_out[token]:.4f} of {token[:8]}... for {sol_in:.4f} SOL")
            sells.append({'token': token, 'amount': tokens_out[token], 'sol': sol_in})
        else:
            print(f"         → UNRECOGNIZED SWAP PATTERN")
    
    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"   Total transactions: {len(signatures)}")
    print(f"   SWAP transactions: {len(swaps)}")
    print(f"   Detected BUYs: {len(buys)}")
    print(f"   Detected SELLs: {len(sells)}")
    
    if buys or sells:
        print(f"\n   ✅ This wallet HAS trading activity!")
        print(f"   The profiler should be able to find trades.")
        print(f"\n   If it's still showing 0, the issue might be:")
        print(f"   - Trades are for different tokens (no buy+sell pairs)")
        print(f"   - Time window mismatch")
    else:
        print(f"\n   ❌ No clear BUY/SELL patterns detected")
        print(f"   The swap transactions might be complex (multi-hop, etc.)")
    
    print(f"{'='*70}\n")


def load_wallets_from_discovery():
    """Load wallet addresses from discovery_debug.json"""
    try:
        import json
        with open('discovery_debug.json', 'r') as f:
            data = json.load(f)
        
        wallets = []
        for w in data.get('profiled_wallets', []):
            addr = w.get('address')
            if addr:
                wallets.append({
                    'address': addr,
                    'win_rate': w.get('win_rate', 0),
                    'pnl': w.get('pnl', 0),
                    'swings': w.get('completed_swings', 0)
                })
        return wallets
    except FileNotFoundError:
        print("❌ discovery_debug.json not found - run discovery first")
        return []
    except Exception as e:
        print(f"❌ Error loading discovery data: {e}")
        return []


def list_discovery_wallets():
    """List all wallets from last discovery"""
    wallets = load_wallets_from_discovery()
    
    if not wallets:
        return
    
    print(f"\n{'='*70}")
    print(f"📋 WALLETS FROM LAST DISCOVERY ({len(wallets)} total)")
    print(f"{'='*70}")
    
    for i, w in enumerate(wallets, 1):
        status = '✅' if (w['win_rate'] >= 0.5 and w['pnl'] >= 2 and w['swings'] >= 3) else '❌'
        print(f"\n{i}. {status} {w['address']}")
        print(f"      WR: {w['win_rate']:.1%} | PnL: {w['pnl']:.2f} | Swings: {w['swings']}")
    
    print(f"\n{'='*70}")
    print(f"To debug a wallet: python debug_profiler.py <number>")
    print(f"Example: python debug_profiler.py 1")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        
        if arg == 'list':
            list_discovery_wallets()
        elif arg.isdigit():
            # Load wallet by number from discovery
            wallets = load_wallets_from_discovery()
            idx = int(arg) - 1
            if 0 <= idx < len(wallets):
                debug_wallet_profile(wallets[idx]['address'])
            else:
                print(f"❌ Invalid wallet number. Use 1-{len(wallets)}")
                print("   Run: python debug_profiler.py list")
        else:
            # Assume it's a wallet address
            debug_wallet_profile(arg)
    else:
        # No args - show list and prompt
        wallets = load_wallets_from_discovery()
        
        if wallets:
            list_discovery_wallets()
            
            choice = input("Enter wallet number to debug (or 'q' to quit): ").strip()
            if choice.isdigit():
                idx = int(choice) - 1
                if 0 <= idx < len(wallets):
                    debug_wallet_profile(wallets[idx]['address'])
                else:
                    print(f"Invalid number. Choose 1-{len(wallets)}")
        else:
            wallet = input("Enter wallet address to debug: ").strip()
            if wallet:
                debug_wallet_profile(wallet)
