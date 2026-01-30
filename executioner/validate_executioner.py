#!/usr/bin/env python3
"""
EXECUTIONER SETUP VALIDATOR
===========================

Validates that the Executioner is properly configured before live trading.
Run this after installation to verify everything is working.

Usage:
    python validate_executioner.py
"""

import os
import sys
from datetime import datetime


def print_header(text):
    print(f"\n{'='*60}")
    print(f"  {text}")
    print(f"{'='*60}")


def print_check(name, passed, details=""):
    status = "✅" if passed else "❌"
    print(f"  {status} {name}")
    if details:
        print(f"      {details}")


def main():
    print_header("EXECUTIONER SETUP VALIDATOR")
    print(f"  Running at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    all_passed = True
    
    # Check 1: Environment file
    print_header("1. ENVIRONMENT CONFIGURATION")
    
    from dotenv import load_dotenv
    load_dotenv()
    
    helius_key = os.getenv('HELIUS_KEY')
    private_key = os.getenv('SOLANA_PRIVATE_KEY')
    live_enabled = os.getenv('ENABLE_LIVE_TRADING', '').lower() == 'true'
    
    print_check("HELIUS_KEY", bool(helius_key), 
                "Found" if helius_key else "Missing - required for RPC")
    print_check("SOLANA_PRIVATE_KEY", bool(private_key),
                "Found (hidden)" if private_key else "Missing - required for live trading")
    print_check("ENABLE_LIVE_TRADING", True, 
                f"{'ENABLED ⚠️' if live_enabled else 'Disabled (safe)'}")
    
    if not helius_key:
        all_passed = False
    
    # Check 2: Python dependencies
    print_header("2. PYTHON DEPENDENCIES")
    
    try:
        from solders.keypair import Keypair
        print_check("solders", True, "Installed")
    except ImportError:
        print_check("solders", False, "pip install solders --break-system-packages")
        all_passed = False
    
    try:
        from solana.rpc.api import Client
        print_check("solana-py", True, "Installed")
    except ImportError:
        print_check("solana-py", False, "pip install solana --break-system-packages")
        all_passed = False
    
    try:
        import requests
        print_check("requests", True, "Installed")
    except ImportError:
        print_check("requests", False, "pip install requests")
        all_passed = False
    
    # Check 3: Executioner module
    print_header("3. EXECUTIONER MODULE")
    
    try:
        from executioner_v1 import Executioner, ExecutionConfig, TaxDatabase
        print_check("executioner_v1.py", True, "Module loaded")
    except ImportError as e:
        print_check("executioner_v1.py", False, f"Import error: {e}")
        all_passed = False
    except Exception as e:
        print_check("executioner_v1.py", False, f"Error: {e}")
        all_passed = False
    
    # Check 4: Wallet validation (if key provided)
    print_header("4. WALLET VALIDATION")
    
    if private_key:
        try:
            from solders.keypair import Keypair
            kp = Keypair.from_base58_string(private_key)
            pubkey = str(kp.pubkey())
            print_check("Private key valid", True, f"Pubkey: {pubkey[:8]}...{pubkey[-4:]}")
        except Exception as e:
            print_check("Private key valid", False, f"Error: {e}")
            all_passed = False
    else:
        print_check("Private key valid", False, "No key provided")
    
    # Check 5: API connectivity
    print_header("5. API CONNECTIVITY")
    
    try:
        import requests
        
        # Test CoinGecko (for NZD prices)
        resp = requests.get(
            "https://api.coingecko.com/api/v3/simple/price?ids=solana&vs_currencies=nzd,usd",
            timeout=10
        )
        if resp.status_code == 200:
            data = resp.json()
            sol_nzd = data['solana']['nzd']
            print_check("CoinGecko API", True, f"SOL price: ${sol_nzd:.2f} NZD")
        else:
            print_check("CoinGecko API", False, f"Status: {resp.status_code}")
    except Exception as e:
        print_check("CoinGecko API", False, f"Error: {e}")
    
    try:
        # Test Jupiter
        resp = requests.get(
            "https://price.jup.ag/v6/price?ids=So11111111111111111111111111111111111111112",
            timeout=10
        )
        if resp.status_code == 200:
            print_check("Jupiter API", True, "Connected")
        else:
            print_check("Jupiter API", False, f"Status: {resp.status_code}")
    except Exception as e:
        print_check("Jupiter API", False, f"Error: {e}")
    
    if helius_key:
        try:
            resp = requests.post(
                f"https://mainnet.helius-rpc.com/?api-key={helius_key}",
                json={"jsonrpc": "2.0", "id": 1, "method": "getHealth"},
                timeout=10
            )
            if resp.status_code == 200:
                print_check("Helius RPC", True, "Connected")
            else:
                print_check("Helius RPC", False, f"Status: {resp.status_code}")
        except Exception as e:
            print_check("Helius RPC", False, f"Error: {e}")
    
    # Check 6: Tax database
    print_header("6. TAX DATABASE")
    
    try:
        from executioner_v1 import TaxDatabase
        tax_db = TaxDatabase("test_tax_records.db")
        print_check("TaxDatabase", True, "Initialized successfully")
        
        # Clean up test file
        import os
        if os.path.exists("test_tax_records.db"):
            os.remove("test_tax_records.db")
            print_check("Test cleanup", True, "Removed test database")
    except Exception as e:
        print_check("TaxDatabase", False, f"Error: {e}")
        all_passed = False
    
    # Check 7: Balance check (if wallet configured)
    print_header("7. WALLET BALANCE")
    
    if private_key and helius_key:
        try:
            from solana.rpc.api import Client
            from solders.pubkey import Pubkey
            
            client = Client(f"https://mainnet.helius-rpc.com/?api-key={helius_key}")
            kp = Keypair.from_base58_string(private_key)
            pubkey = kp.pubkey()
            
            resp = client.get_balance(pubkey)
            balance_sol = resp.value / 1_000_000_000
            
            print_check("Balance check", True, f"{balance_sol:.4f} SOL")
            
            if balance_sol < 0.01:
                print("      ⚠️  Warning: Low balance. Deposit SOL before live trading.")
        except Exception as e:
            print_check("Balance check", False, f"Error: {e}")
    else:
        print_check("Balance check", False, "Missing wallet or Helius key")
    
    # Summary
    print_header("SUMMARY")
    
    if all_passed:
        print("  🎉 All critical checks passed!")
        print("\n  Next steps:")
        print("  1. Run paper trading for 14+ days")
        print("  2. Monitor performance in existing system")
        print("  3. When ready, set LIVE_TRADE_PCT=10 to start gradual rollout")
        print("  4. Export tax records monthly: python executioner_v1.py export")
    else:
        print("  ⚠️  Some checks failed. Please fix before proceeding.")
        print("\n  Common fixes:")
        print("  - pip install solana solders --break-system-packages")
        print("  - Add missing environment variables to .env")
        print("  - Ensure private key is valid base58 format")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
