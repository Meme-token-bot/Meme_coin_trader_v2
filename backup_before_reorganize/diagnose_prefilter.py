#!/usr/bin/env python3
"""
Discovery Pre-Filter Diagnostic Tool v2

Investigates WHY 97% of candidate wallets are rejected with "no_sells" or "no_buys"

This tool:
1. Runs a mini-discovery to get real candidate wallets
2. For wallets marked "no_sells" - checks their ACTUAL transaction history
3. Determines if they really have no sells, or if our detection is broken

USAGE:
  python diagnose_prefilter.py              # Run full diagnostic
  python diagnose_prefilter.py <wallet>     # Deep-dive on specific wallet
"""

import os
import sys
import json
import requests
import time
from datetime import datetime
from collections import defaultdict
from dotenv import load_dotenv

load_dotenv()

HELIUS_KEY = os.getenv('HELIUS_KEY')
HELIUS_API = f"https://api.helius.xyz/v0"
HELIUS_RPC = f"https://mainnet.helius-rpc.com/?api-key={HELIUS_KEY}"

# Stablecoins treated as "SOL equivalent"
STABLES = {
    "So11111111111111111111111111111111111111112",   # WSOL
    "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v", # USDC
    "Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB", # USDT
}

# Known protocol/bot addresses to skip
SKIP_ADDRESSES = {
    "5Q544fKrFoe6tsEbD7S8EmxGTJYAKtTVhAW5Q5pge4j1",  # Raydium
    "675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8",  # Raydium AMM
    "JUP6LkbZbjS1jKKwapdHNy74zcZ3tLUZoi5QNyVTaV4",   # Jupiter
}

def rate_limit():
    time.sleep(0.2)


def get_trending_token():
    """Get a trending Solana memecoin from DexScreener"""
    print("\n🔍 Finding a trending token to analyze...")
    
    try:
        # Try boosted tokens first (more likely to have active trading)
        res = requests.get(
            "https://api.dexscreener.com/token-boosts/top/v1",
            timeout=10
        )
        data = res.json()
        
        for token in data[:20]:
            if token.get('chainId') == 'solana':
                address = token.get('tokenAddress')
                if address and len(address) > 30:
                    print(f"   Found boosted token: {address[:12]}...")
                    return address
        
        # Fallback: search for trending
        res = requests.get(
            "https://api.dexscreener.com/latest/dex/search?q=solana",
            timeout=10
        )
        data = res.json()
        pairs = data.get('pairs', [])
        
        # Find high volume Solana tokens
        for pair in sorted(pairs, key=lambda p: float(p.get('volume', {}).get('h24', 0) or 0), reverse=True):
            if pair.get('chainId') == 'solana':
                address = pair.get('baseToken', {}).get('address')
                if address and len(address) > 30:
                    symbol = pair.get('baseToken', {}).get('symbol', '?')
                    volume = float(pair.get('volume', {}).get('h24', 0) or 0)
                    print(f"   Found ${symbol} with ${volume:,.0f} 24h volume")
                    return address
        
        return None
        
    except Exception as e:
        print(f"   Error: {e}")
        return None


def get_token_transactions(token_address: str, limit: int = 100) -> list:
    """Get recent transactions for a token"""
    rate_limit()
    
    # Get signatures for the token
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "getSignaturesForAddress",
        "params": [token_address, {"limit": limit}]
    }
    
    try:
        res = requests.post(HELIUS_RPC, json=payload, timeout=15)
        signatures = res.json().get('result', [])
        
        if not signatures:
            print(f"   No transactions found for token")
            return []
        
        print(f"   Found {len(signatures)} recent transactions")
        
        # Parse transactions in batches
        rate_limit()
        sig_list = [s.get('signature') for s in signatures if s.get('signature')]
        
        parse_url = f"{HELIUS_API}/transactions?api-key={HELIUS_KEY}"
        parse_res = requests.post(parse_url, json={"transactions": sig_list[:50]}, timeout=20)
        
        result = parse_res.json()
        if isinstance(result, list):
            return result
        return []
        
    except Exception as e:
        print(f"   Error: {e}")
        return []


def get_wallet_transactions(wallet: str, limit: int = 100) -> list:
    """Get parsed transactions for a wallet"""
    rate_limit()
    
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "getSignaturesForAddress",
        "params": [wallet, {"limit": limit}]
    }
    
    try:
        res = requests.post(HELIUS_RPC, json=payload, timeout=15)
        signatures = res.json().get('result', [])
        
        if not signatures:
            return []
        
        rate_limit()
        sig_list = [s.get('signature') for s in signatures if s.get('signature')]
        
        parse_url = f"{HELIUS_API}/transactions?api-key={HELIUS_KEY}"
        parse_res = requests.post(parse_url, json={"transactions": sig_list}, timeout=20)
        
        result = parse_res.json()
        return result if isinstance(result, list) else []
    except Exception as e:
        print(f"   Error: {e}")
        return []


def analyze_swap(tx: dict, wallet: str) -> dict:
    """Analyze a swap transaction from wallet's perspective"""
    result = {
        'signature': tx.get('signature', '')[:16],
        'type': tx.get('type', 'UNKNOWN'),
        'timestamp': tx.get('timestamp', 0),
        'detected_as': None,
        'sol_in': 0,
        'sol_out': 0,
        'tokens_in': {},
        'tokens_out': {},
    }
    
    if tx.get('type') != 'SWAP':
        return result
    
    token_transfers = tx.get('tokenTransfers', [])
    native_transfers = tx.get('nativeTransfers', [])
    
    sol_in = sol_out = 0
    tokens_in = {}
    tokens_out = {}
    
    # Process token transfers
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
    
    # Process native SOL transfers
    for transfer in native_transfers:
        amount = float(transfer.get('amount', 0) or 0) / 1e9
        from_addr = transfer.get('fromUserAccount', '')
        to_addr = transfer.get('toUserAccount', '')
        
        if from_addr == wallet:
            sol_out += amount
        elif to_addr == wallet:
            sol_in += amount
    
    result['sol_in'] = sol_in
    result['sol_out'] = sol_out
    result['tokens_in'] = tokens_in
    result['tokens_out'] = tokens_out
    
    # Determine trade type
    if len(tokens_in) >= 1 and sol_out > 0:
        result['detected_as'] = 'BUY'
    elif len(tokens_out) >= 1 and sol_in > 0:
        result['detected_as'] = 'SELL'
    
    return result


def extract_candidates_from_token(token_address: str) -> list:
    """Extract candidate wallets from a token's transactions"""
    print(f"\n📊 Extracting candidates from token transactions...")
    
    txs = get_token_transactions(token_address, limit=100)
    
    if not txs:
        return []
    
    candidates = {}
    
    for tx in txs:
        if tx.get('type') != 'SWAP':
            continue
        
        fee_payer = tx.get('feePayer', '')
        
        # Skip invalid or known addresses
        if not fee_payer or len(fee_payer) < 32:
            continue
        if fee_payer in SKIP_ADDRESSES:
            continue
        
        if fee_payer not in candidates:
            candidates[fee_payer] = {
                'address': fee_payer,
                'buy_count': 0,
                'sell_count': 0,
                'tokens_seen': set(),
            }
        
        # Analyze the swap from this wallet's perspective
        analysis = analyze_swap(tx, fee_payer)
        
        if analysis['detected_as'] == 'BUY':
            candidates[fee_payer]['buy_count'] += 1
            candidates[fee_payer]['tokens_seen'].update(analysis['tokens_in'].keys())
        elif analysis['detected_as'] == 'SELL':
            candidates[fee_payer]['sell_count'] += 1
            candidates[fee_payer]['tokens_seen'].update(analysis['tokens_out'].keys())
    
    # Convert to list and clean up
    result = []
    for addr, data in candidates.items():
        data['tokens_seen'] = len(data['tokens_seen'])
        result.append(data)
    
    print(f"   Found {len(result)} unique wallets")
    return result


def deep_analyze_wallet(wallet: str):
    """Deep analysis of a single wallet's trading history"""
    print(f"\n{'='*70}")
    print(f"DEEP ANALYSIS: {wallet}")
    print(f"{'='*70}")
    
    print(f"\n📡 Fetching last 100 transactions...")
    txs = get_wallet_transactions(wallet, limit=100)
    
    if not txs:
        print("  ❌ No transactions found")
        return
    
    print(f"   Found {len(txs)} transactions")
    
    # Categorize transactions
    swaps = []
    non_swaps = defaultdict(int)
    
    for tx in txs:
        if tx.get('type') == 'SWAP':
            analysis = analyze_swap(tx, wallet)
            swaps.append(analysis)
        else:
            non_swaps[tx.get('type', 'UNKNOWN')] += 1
    
    # Summary
    print(f"\n📊 Transaction Types:")
    print(f"   SWAP: {len(swaps)}")
    for tx_type, count in sorted(non_swaps.items(), key=lambda x: -x[1])[:5]:
        print(f"   {tx_type}: {count}")
    
    if not swaps:
        print("\n   ⚠️ No SWAP transactions found!")
        print("   This wallet is probably not an active trader.")
        return
    
    # Analyze swaps
    buys = [s for s in swaps if s['detected_as'] == 'BUY']
    sells = [s for s in swaps if s['detected_as'] == 'SELL']
    undetected = [s for s in swaps if s['detected_as'] is None]
    
    print(f"\n📈 Swap Detection Results:")
    print(f"   BUYs detected: {len(buys)}")
    print(f"   SELLs detected: {len(sells)}")
    print(f"   Undetected: {len(undetected)}")
    
    # Unique tokens
    all_tokens = set()
    for s in swaps:
        all_tokens.update(s['tokens_in'].keys())
        all_tokens.update(s['tokens_out'].keys())
    print(f"   Unique tokens: {len(all_tokens)}")
    
    # Sample trades
    if buys:
        print(f"\n   Sample BUYs:")
        for b in buys[:3]:
            sol = b['sol_out']
            tokens = list(b['tokens_in'].keys())
            token_short = tokens[0][:8] + "..." if tokens else "?"
            print(f"      {b['signature']}... | {sol:.3f} SOL → {token_short}")
    
    if sells:
        print(f"\n   Sample SELLs:")
        for s in sells[:3]:
            sol = s['sol_in']
            tokens = list(s['tokens_out'].keys())
            token_short = tokens[0][:8] + "..." if tokens else "?"
            print(f"      {s['signature']}... | {token_short} → {sol:.3f} SOL")
    
    # Diagnosis
    print(f"\n🔍 DIAGNOSIS:")
    if len(buys) > 0 and len(sells) > 0:
        print(f"   ✅ This wallet HAS both buys ({len(buys)}) and sells ({len(sells)})!")
        print(f"   If pre-filter rejected them, it's because we only see trades")
        print(f"   for ONE token at a time during discovery, not all their trades.")
    elif len(buys) > 0 and len(sells) == 0:
        print(f"   ⚠️ Wallet has {len(buys)} buys but 0 sells")
        print(f"   They may be a holder, or sells are in different tokens.")
    elif len(sells) > 0 and len(buys) == 0:
        print(f"   ⚠️ Wallet has {len(sells)} sells but 0 buys")
        print(f"   Buys may have been before our time window.")
    else:
        print(f"   ❌ No buys or sells detected - detection logic issue?")
    
    return {
        'buys': len(buys),
        'sells': len(sells),
        'total_swaps': len(swaps),
        'unique_tokens': len(all_tokens),
    }


def run_diagnostic():
    """Run the full diagnostic"""
    print(f"\n{'='*70}")
    print("DISCOVERY PRE-FILTER DIAGNOSTIC v2")
    print(f"{'='*70}")
    
    # Load debug data if available
    try:
        with open('discovery_debug.json', 'r') as f:
            data = json.load(f)
        
        stats = data.get('stats', {})
        prefilter = data.get('prefilter_reasons', {})
        
        print(f"\n📊 Last Discovery Summary:")
        print(f"   Total candidates: {stats.get('wallet_candidates_found', 0)}")
        passed = stats.get('wallet_candidates_found', 0) - stats.get('wallets_prefiltered_out', 0)
        print(f"   Passed pre-filter: {passed}")
        print(f"   Filtered out: {stats.get('wallets_prefiltered_out', 0)}")
        
        print(f"\n📋 Pre-filter Breakdown:")
        total = stats.get('wallet_candidates_found', 1)
        for reason, count in sorted(prefilter.items(), key=lambda x: -x[1]):
            pct = 100 * count / total
            print(f"   {reason}: {count} ({pct:.0f}%)")
    except FileNotFoundError:
        print("\n⚠️ discovery_debug.json not found - will run fresh analysis")
    
    # Get a trending token to analyze
    token = get_trending_token()
    
    if not token:
        print("\n❌ Could not find a token to analyze")
        return
    
    # Extract candidates from this token
    candidates = extract_candidates_from_token(token)
    
    if not candidates:
        print("\n❌ No candidates found from token")
        return
    
    # Categorize candidates (simulating pre-filter)
    no_sells = [c for c in candidates if c['sell_count'] == 0 and c['buy_count'] > 0]
    no_buys = [c for c in candidates if c['buy_count'] == 0 and c['sell_count'] > 0]
    has_both = [c for c in candidates if c['buy_count'] > 0 and c['sell_count'] > 0]
    
    print(f"\n📊 Candidate Breakdown (from this token):")
    print(f"   Has both buys+sells: {len(has_both)} (would pass pre-filter)")
    print(f"   Has buys, no sells: {len(no_sells)} (would be rejected)")
    print(f"   Has sells, no buys: {len(no_buys)} (would be rejected)")
    
    # Deep analyze some "no_sells" wallets
    if no_sells:
        print(f"\n{'='*70}")
        print("CHECKING 'no_sells' WALLETS - Do they REALLY have no sells?")
        print(f"{'='*70}")
        
        real_traders = 0
        sample = no_sells[:3]
        
        for c in sample:
            wallet = c['address']
            print(f"\n🔍 Checking: {wallet[:16]}...")
            print(f"   From token analysis: {c['buy_count']} buys, {c['sell_count']} sells")
            
            result = deep_analyze_wallet(wallet)
            
            if result and result['sells'] > 0:
                print(f"\n   🔴 THIS WALLET HAS {result['sells']} SELLS! Pre-filter would have missed them!")
                real_traders += 1
            elif result:
                print(f"\n   ✅ Confirmed: genuinely no sells in recent history")
        
        if real_traders > 0:
            print(f"\n{'='*70}")
            print("🚨 PROBLEM IDENTIFIED")
            print(f"{'='*70}")
            print(f"   {real_traders}/{len(sample)} 'no_sells' wallets actually HAVE sells!")
            print(f"   The pre-filter is rejecting valid traders because it only")
            print(f"   sees trades for ONE token at a time.")
    
    # Summary and recommendations
    print(f"\n{'='*70}")
    print("RECOMMENDATIONS")
    print(f"{'='*70}")
    print("""
The high rejection rate happens because:

1. TOKEN-CENTRIC VIEW: When scanning $MEME transactions, we only
   see that wallet's trades in $MEME. If they bought $MEME but
   sold $PEPE, we see "no sells".

2. LIMITED WINDOW: Only recent transactions are scanned. A wallet's
   buy and sell for the SAME token may not both be in the window.

SOLUTION: Add a wallet-level verification step before profiling.
After finding a candidate, fetch THEIR recent transactions (not
the token's) to count total buys/sells across ALL tokens.

This costs ~2 extra API calls per candidate but dramatically 
improves accuracy.
""")


if __name__ == "__main__":
    if not HELIUS_KEY:
        print("❌ HELIUS_KEY not set in environment")
        sys.exit(1)
    
    if len(sys.argv) > 1:
        wallet = sys.argv[1]
        if len(wallet) > 30:
            deep_analyze_wallet(wallet)
        else:
            print(f"Invalid wallet address: {wallet}")
    else:
        run_diagnostic()
