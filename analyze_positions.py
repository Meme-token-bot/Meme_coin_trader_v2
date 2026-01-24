"""
POSITION ANALYSIS & REALITY CHECK
=================================

Analyzes your 55 open positions to understand:
1. What's the actual unrealized PnL?
2. How many are profitable vs losing?
3. What's the realistic performance?
"""

import os
import requests
from datetime import datetime
from collections import defaultdict
from dotenv import load_dotenv
load_dotenv()

from database_v2 import DatabaseV2

HELIUS_KEY = os.getenv('HELIUS_KEY')


def get_token_price(token_address: str) -> float:
    """Get current token price from DexScreener"""
    try:
        url = f"https://api.dexscreener.com/latest/dex/tokens/{token_address}"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            pairs = data.get('pairs', [])
            
            if pairs:
                # Get highest liquidity pair
                pair = max(pairs, key=lambda p: float(p.get('liquidity', {}).get('usd', 0) or 0))
                return float(pair.get('priceUsd', 0) or 0)
        
        return 0
    except:
        return 0


def analyze_positions():
    """Analyze all open positions"""
    print("\n" + "="*70)
    print("📊 OPEN POSITION ANALYSIS")
    print("="*70)
    
    db = DatabaseV2()
    positions = db.get_open_positions()
    
    print(f"\nTotal open positions: {len(positions)}")
    
    if not positions:
        print("No open positions found.")
        return
    
    # Analyze each position
    results = {
        'total': len(positions),
        'profitable': 0,
        'losing': 0,
        'unknown': 0,
        'total_invested': 0,
        'total_current_value': 0,
        'unrealized_pnl': 0,
        'by_token': defaultdict(lambda: {'count': 0, 'invested': 0, 'current': 0}),
        'positions': []
    }
    
    print(f"\nFetching current prices (this may take a minute)...\n")
    
    for i, pos in enumerate(positions):
        token_address = pos.get('token_address', '')
        token_symbol = pos.get('token_symbol', 'UNKNOWN')
        entry_price = pos.get('entry_price', 0)
        position_size = pos.get('position_size_sol', 0) or 0
        entry_time = pos.get('entry_time')
        
        # Get current price
        current_price = get_token_price(token_address)
        
        # Calculate PnL
        if entry_price > 0 and current_price > 0:
            pnl_pct = (current_price - entry_price) / entry_price * 100
            current_value = position_size * (1 + pnl_pct/100)
            pnl_sol = current_value - position_size
        else:
            pnl_pct = 0
            current_value = position_size
            pnl_sol = 0
        
        # Categorize
        if current_price == 0:
            results['unknown'] += 1
            status = "❓"
        elif pnl_pct > 0:
            results['profitable'] += 1
            status = "✅"
        else:
            results['losing'] += 1
            status = "❌"
        
        results['total_invested'] += position_size
        results['total_current_value'] += current_value
        results['unrealized_pnl'] += pnl_sol
        
        results['by_token'][token_symbol]['count'] += 1
        results['by_token'][token_symbol]['invested'] += position_size
        results['by_token'][token_symbol]['current'] += current_value
        
        # Calculate hold time
        if entry_time:
            if isinstance(entry_time, str):
                entry_time = datetime.fromisoformat(entry_time.replace('Z', '+00:00').replace('+00:00', ''))
            hold_hours = (datetime.now() - entry_time).total_seconds() / 3600
        else:
            hold_hours = 0
        
        results['positions'].append({
            'symbol': token_symbol,
            'size': position_size,
            'entry': entry_price,
            'current': current_price,
            'pnl_pct': pnl_pct,
            'pnl_sol': pnl_sol,
            'hold_hours': hold_hours,
            'status': status
        })
        
        # Print progress
        if (i + 1) % 10 == 0:
            print(f"   Processed {i + 1}/{len(positions)} positions...")
    
    # Summary
    print(f"\n" + "="*70)
    print(f"📊 RESULTS SUMMARY")
    print(f"="*70)
    
    print(f"\n📈 Position Status:")
    print(f"   ✅ Profitable: {results['profitable']} ({100*results['profitable']/results['total']:.1f}%)")
    print(f"   ❌ Losing: {results['losing']} ({100*results['losing']/results['total']:.1f}%)")
    print(f"   ❓ Unknown (no price): {results['unknown']}")
    
    print(f"\n💰 Financial Summary:")
    print(f"   Total invested: {results['total_invested']:.4f} SOL")
    print(f"   Current value: {results['total_current_value']:.4f} SOL")
    print(f"   Unrealized PnL: {results['unrealized_pnl']:+.4f} SOL ({100*results['unrealized_pnl']/max(0.01, results['total_invested']):+.1f}%)")
    
    # Win rate if we closed now
    if results['profitable'] + results['losing'] > 0:
        actual_wr = results['profitable'] / (results['profitable'] + results['losing'])
        print(f"\n📊 If closed now:")
        print(f"   Win rate: {actual_wr:.1%}")
    
    # Top performers
    sorted_by_pnl = sorted(results['positions'], key=lambda x: x['pnl_sol'], reverse=True)
    
    print(f"\n🏆 Top 5 Performers:")
    for pos in sorted_by_pnl[:5]:
        print(f"   {pos['status']} {pos['symbol']}: {pos['pnl_pct']:+.1f}% ({pos['pnl_sol']:+.4f} SOL) | {pos['hold_hours']:.1f}h")
    
    print(f"\n💀 Bottom 5 Performers:")
    for pos in sorted_by_pnl[-5:]:
        print(f"   {pos['status']} {pos['symbol']}: {pos['pnl_pct']:+.1f}% ({pos['pnl_sol']:+.4f} SOL) | {pos['hold_hours']:.1f}h")
    
    # By token concentration
    print(f"\n📦 Token Concentration:")
    sorted_tokens = sorted(results['by_token'].items(), key=lambda x: x[1]['count'], reverse=True)
    for symbol, data in sorted_tokens[:10]:
        token_pnl = data['current'] - data['invested']
        print(f"   {symbol}: {data['count']} positions, {data['invested']:.3f} SOL invested, {token_pnl:+.4f} SOL PnL")
    
    # Positions held too long
    old_positions = [p for p in results['positions'] if p['hold_hours'] > 12]
    if old_positions:
        print(f"\n⏰ Positions held > 12 hours (should have time-stopped): {len(old_positions)}")
        for pos in old_positions[:5]:
            print(f"   {pos['symbol']}: {pos['hold_hours']:.1f}h | {pos['pnl_pct']:+.1f}%")
    
    # Critical assessment
    print(f"\n" + "="*70)
    print(f"🔍 CRITICAL ASSESSMENT")
    print(f"="*70)
    
    if results['losing'] > results['profitable']:
        print(f"""
⚠️ WARNING: More losing positions than winning!
   
   The +39% paper gain shown in the bot is UNREALIZED.
   If you closed all 55 positions now, you would likely LOSE money.
   
   Actual unrealized PnL: {results['unrealized_pnl']:+.4f} SOL
""")
    
    if len(old_positions) > 10:
        print(f"""
⚠️ WARNING: {len(old_positions)} positions exceeded max hold time!
   
   The TIME_STOP exit condition (12h) is not working properly.
   Positions should be closed automatically after 12 hours.
""")
    
    print(f"""
📋 RECOMMENDATIONS:

1. FIX MAX POSITIONS: The system opened 55 positions when limit is 5.
   Add this check to open_position():
   
   positions = self.db.get_open_positions()
   if len(positions) >= CONFIG.max_open_positions:
       return None

2. CLOSE STALE POSITIONS: {len(old_positions)} positions are over 12h old.
   Run: python close_stale_positions.py

3. REVIEW EXIT CONDITIONS: Positions aren't being closed properly.
   Check that the position checking loop is running.

4. DON'T TRUST THE +39%: That's unrealized PnL on 55 positions.
   Real performance = unrealized PnL = {results['unrealized_pnl']:+.4f} SOL
""")
    
    return results


def close_stale_positions():
    """Close all positions over 12 hours old"""
    print("\n" + "="*70)
    print("🧹 CLOSING STALE POSITIONS")
    print("="*70)
    
    db = DatabaseV2()
    positions = db.get_open_positions()
    
    closed = 0
    for pos in positions:
        entry_time = pos.get('entry_time')
        
        if entry_time:
            if isinstance(entry_time, str):
                entry_time = datetime.fromisoformat(entry_time.replace('Z', '+00:00').replace('+00:00', ''))
            
            hold_hours = (datetime.now() - entry_time).total_seconds() / 3600
            
            if hold_hours > 12:
                # Get current price
                current_price = get_token_price(pos.get('token_address', ''))
                
                if current_price <= 0:
                    current_price = pos.get('entry_price', 0)  # Use entry price as fallback
                
                # Close the position
                entry_price = pos.get('entry_price', 0)
                position_size = pos.get('position_size_sol', 0) or 0
                
                if entry_price > 0:
                    pnl_pct = (current_price - entry_price) / entry_price * 100
                else:
                    pnl_pct = 0
                
                db.close_position(pos['id'], {
                    'exit_time': datetime.now(),
                    'exit_price': current_price,
                    'exit_reason': 'MANUAL_TIME_STOP',
                    'pnl_pct': pnl_pct,
                    'pnl_sol': position_size * (pnl_pct / 100)
                })
                
                print(f"   Closed {pos.get('token_symbol', 'UNKNOWN')}: {hold_hours:.1f}h old, {pnl_pct:+.1f}%")
                closed += 1
    
    print(f"\n✅ Closed {closed} stale positions")
    return closed


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'close':
        close_stale_positions()
    else:
        analyze_positions()
        
        print("\n" + "="*70)
        print("To close stale positions, run: python analyze_positions.py close")
        print("="*70)
