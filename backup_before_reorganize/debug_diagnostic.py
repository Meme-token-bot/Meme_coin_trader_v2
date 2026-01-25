#!/usr/bin/env python3
"""
Debug Diagnostic - See exactly what's in token transactions
"""

import os
import requests
import time
from collections import defaultdict
from dotenv import load_dotenv

load_dotenv()

HELIUS_KEY = os.getenv('HELIUS_KEY')
HELIUS_API = f"https://api.helius.xyz/v0"
HELIUS_RPC = f"https://mainnet.helius-rpc.com/?api-key={HELIUS_KEY}"

SKIP_ADDRESSES = {
    "5Q544fKrFoe6tsEbD7S8EmxGTJYAKtTVhAW5Q5pge4j1",
    "675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8",
    "JUP6LkbZbjS1jKKwapdHNy74zcZ3tLUZoi5QNyVTaV4",
}
STABLES = {
    "So11111111111111111111111111111111111111112",
    "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
    "Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB",
}

def rate_limit():
    time.sleep(0.2)


def get_active_token():
    """Find a token with actual trading activity"""
    print("🔍 Finding active trading token...")
    
    # Try multiple search terms to find active tokens
    searches = [
        "solana pump",
        "solana meme", 
        "solana ai",
    ]
    
    for search in searches:
        try:
            res = requests.get(
                f"https://api.dexscreener.com/latest/dex/search?q={search}",
                timeout=10
            )
            data = res.json()
            pairs = data.get('pairs', [])
            
            for pair in pairs:
                if pair.get('chainId') != 'solana':
                    continue
                    
                vol = float(pair.get('volume', {}).get('h24', 0) or 0)
                txns = pair.get('txns', {}).get('h24', {})
                buys = txns.get('buys', 0)
                sells = txns.get('sells', 0)
                
                # Need volume AND actual trades
                if vol > 50000 and buys > 100 and sells > 100:
                    token = pair.get('baseToken', {}).get('address')
                    symbol = pair.get('baseToken', {}).get('symbol', '?')
                    print(f"   Found ${symbol}: ${vol:,.0f} vol, {buys}B/{sells}S")
                    return token, symbol
                    
        except Exception as e:
            print(f"   Search '{search}' failed: {e}")
            continue
    
    return None, None


def analyze_token(token_address: str, token_symbol: str):
    """Analyze transactions for a specific token"""
    print(f"\n📊 Analyzing ${token_symbol} ({token_address[:12]}...)")
    
    # Get signatures
    rate_limit()
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "getSignaturesForAddress",
        "params": [token_address, {"limit": 50}]
    }
    
    res = requests.post(HELIUS_RPC, json=payload, timeout=15)
    result = res.json()
    
    if 'error' in result:
        print(f"   Error getting signatures: {result['error']}")
        return
    
    signatures = result.get('result', [])
    print(f"   Found {len(signatures)} recent signatures")
    
    if not signatures:
        return
    
    # Parse transactions
    rate_limit()
    sig_list = [s.get('signature') for s in signatures[:30] if s.get('signature')]
    parse_url = f"{HELIUS_API}/transactions?api-key={HELIUS_KEY}"
    
    parse_res = requests.post(parse_url, json={"transactions": sig_list}, timeout=20)
    txs = parse_res.json()
    
    if not isinstance(txs, list):
        print(f"   Error parsing: {txs}")
        return
    
    print(f"   Parsed {len(txs)} transactions")
    
    # Analyze transaction types
    tx_types = defaultdict(int)
    for tx in txs:
        tx_types[tx.get('type', 'UNKNOWN')] += 1
    
    print(f"\n   Transaction types:")
    for t, count in sorted(tx_types.items(), key=lambda x: -x[1]):
        print(f"      {t}: {count}")
    
    # Extract candidates from SWAPs
    swaps = [tx for tx in txs if tx.get('type') == 'SWAP']
    print(f"\n   SWAP transactions: {len(swaps)}")
    
    if not swaps:
        print("   ⚠️ No SWAP transactions found!")
        print("   This token may not have recent trades, or trades are not being parsed as SWAP")
        
        # Show what we do have
        if txs:
            print(f"\n   Sample non-SWAP transaction:")
            tx = txs[0]
            print(f"      Type: {tx.get('type')}")
            print(f"      Source: {tx.get('source')}")
            print(f"      Description: {tx.get('description', 'N/A')[:80]}")
        return
    
    # Show SWAP details
    print(f"\n   Sample SWAPs:")
    candidates = {}
    
    for tx in swaps[:10]:
        fee_payer = tx.get('feePayer', '')
        sig = tx.get('signature', '')[:12]
        
        # Show details
        token_transfers = tx.get('tokenTransfers', [])
        native_transfers = tx.get('nativeTransfers', [])
        
        print(f"\n   {sig}...")
        print(f"      feePayer: {fee_payer[:16]}..." if fee_payer else "      feePayer: NONE")
        print(f"      tokenTransfers: {len(token_transfers)}")
        print(f"      nativeTransfers: {len(native_transfers)}")
        
        # Try to determine buy/sell
        if not fee_payer or len(fee_payer) < 32 or fee_payer in SKIP_ADDRESSES:
            print(f"      Status: SKIPPED (invalid/protocol)")
            continue
        
        sol_in = sol_out = 0
        tokens_in = []
        tokens_out = []
        
        for transfer in token_transfers:
            mint = transfer.get('mint', '')
            from_addr = transfer.get('fromUserAccount', '')
            to_addr = transfer.get('toUserAccount', '')
            amount = transfer.get('tokenAmount', 0)
            
            if from_addr == fee_payer:
                if mint in STABLES:
                    sol_out += float(amount or 0)
                else:
                    tokens_out.append(mint[:8])
            elif to_addr == fee_payer:
                if mint in STABLES:
                    sol_in += float(amount or 0)
                else:
                    tokens_in.append(mint[:8])
        
        for transfer in native_transfers:
            from_addr = transfer.get('fromUserAccount', '')
            to_addr = transfer.get('toUserAccount', '')
            amount = float(transfer.get('amount', 0) or 0) / 1e9
            
            if from_addr == fee_payer:
                sol_out += amount
            elif to_addr == fee_payer:
                sol_in += amount
        
        # Determine direction
        if tokens_in and sol_out > 0:
            direction = "BUY"
        elif tokens_out and sol_in > 0:
            direction = "SELL"
        else:
            direction = "UNKNOWN"
        
        print(f"      SOL in: {sol_in:.4f} | SOL out: {sol_out:.4f}")
        print(f"      Tokens in: {tokens_in} | Tokens out: {tokens_out}")
        print(f"      Direction: {direction}")
        
        if fee_payer not in candidates:
            candidates[fee_payer] = {'buys': 0, 'sells': 0}
        
        if direction == 'BUY':
            candidates[fee_payer]['buys'] += 1
        elif direction == 'SELL':
            candidates[fee_payer]['sells'] += 1
    
    # Summary
    print(f"\n" + "="*60)
    print(f"CANDIDATE EXTRACTION SUMMARY")
    print(f"="*60)
    print(f"Unique wallets found: {len(candidates)}")
    
    if candidates:
        no_sells = sum(1 for c in candidates.values() if c['sells'] == 0 and c['buys'] > 0)
        no_buys = sum(1 for c in candidates.values() if c['buys'] == 0 and c['sells'] > 0)
        has_both = sum(1 for c in candidates.values() if c['buys'] > 0 and c['sells'] > 0)
        
        print(f"Has both buys+sells: {has_both}")
        print(f"Has buys only (no sells): {no_sells}")
        print(f"Has sells only (no buys): {no_buys}")
        
        print(f"\nCandidate details:")
        for addr, stats in list(candidates.items())[:10]:
            print(f"   {addr[:16]}... | buys: {stats['buys']} | sells: {stats['sells']}")


def main():
    print("="*60)
    print("TRANSACTION DEBUG DIAGNOSTIC")
    print("="*60)
    
    if not HELIUS_KEY:
        print("❌ HELIUS_KEY not found in .env")
        return
    
    token, symbol = get_active_token()
    
    if not token:
        print("❌ Could not find an active trading token")
        return
    
    analyze_token(token, symbol)


if __name__ == "__main__":
    main()
