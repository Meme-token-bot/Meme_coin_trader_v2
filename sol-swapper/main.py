#!/usr/bin/env python3
"""
CLI entry-point for the SOL <-> Token swapper.

Examples
--------
  # Buy a token with 0.5 SOL
  python main.py buy  TOKEN_MINT  0.5

  # Sell 1 000 000 raw units of a token for SOL
  python main.py sell TOKEN_MINT  1000000

  # Override defaults
  python main.py buy TOKEN_MINT 1.0 --slippage 200 --priority-fee 200000
"""

import argparse
import asyncio
import sys

from swapper import create_swapper, LAMPORTS_PER_SOL, MAX_TPS, DEFAULT_TIP_LAMPORTS


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="SOL <-> Token swapper  (Helius Sender SWQOS-Only · Jupiter v1)",
    )
    sub = p.add_subparsers(dest="command", required=True)

    # ── buy ──────────────────────────────────────────────────────────────
    buy = sub.add_parser("buy", help="Buy a token by spending SOL")
    buy.add_argument("token_mint", help="SPL token mint address")
    buy.add_argument("sol_amount", type=float, help="Amount of SOL to spend")

    # ── sell ─────────────────────────────────────────────────────────────
    sell = sub.add_parser("sell", help="Sell a token for SOL")
    sell.add_argument("token_mint", help="SPL token mint address")
    sell.add_argument("token_amount", type=str, help="Token amount (human-readable or raw units)")

    # ── shared flags ─────────────────────────────────────────────────────
    for sp in (buy, sell):
        sp.add_argument(
            "--slippage", type=int, default=300,
            help="Slippage tolerance in bps (default: 100)",
        )
        sp.add_argument(
            "--priority-fee", type=int, default=100_000,
            help="Priority fee in lamports (default: 100000)",
        )
        sp.add_argument(
            "--jito-tip", type=int, default=DEFAULT_TIP_LAMPORTS,
            help=f"Jito tip in lamports for Sender (default: {DEFAULT_TIP_LAMPORTS})",
        )
        sp.add_argument(
            "--region", default="eu-west-1",
            help="AWS region for Secrets Manager (default: eu-west-1)",
        )
        sp.add_argument(
            "--wallet-secret", default="HOT_WALLET_1",
            help="JSON key for wallet private key in Secrets Manager",
        )
        sp.add_argument(
            "--helius-secret", default="HELIUS_KEY",
            help="JSON key for Helius API key in Secrets Manager",
        )
        sp.add_argument(
            "--jupiter-secret", default="JUPITER_API_KEY",
            help="JSON key for Jupiter API key in Secrets Manager",
        )
        sp.add_argument(
            "--max-tps", type=int, default=MAX_TPS,
            help=f"Rate-limit ceiling (default: {MAX_TPS})",
        )

    return p


async def _run(args: argparse.Namespace) -> int:
    swapper = create_swapper(
        aws_region=args.region,
        wallet_secret_id=args.wallet_secret,
        helius_secret_id=args.helius_secret,
        jupiter_secret_id=args.jupiter_secret,
        max_tps=args.max_tps,
    )

    try:
        if args.command == "buy":
            result = await swapper.buy_token(
                token_mint=args.token_mint,
                sol_amount=args.sol_amount,
                slippage_bps=args.slippage,
                priority_fee=args.priority_fee,
                jito_tip=args.jito_tip,
            )
        else:
            # If user passed a decimal, convert to raw units using on-chain decimals
            raw_amount = args.token_amount
            if '.' in raw_amount:
                from decimal import Decimal
                decimals = await swapper.get_token_decimals(args.token_mint)  # you'll need this method
                raw_amount = int(Decimal(raw_amount) * (10 ** decimals))
            else:
                raw_amount = int(raw_amount)

            result = await swapper.sell_token(
                token_mint=args.token_mint,
                token_amount_raw=raw_amount,
                slippage_bps=args.slippage,
                priority_fee=args.priority_fee,
                jito_tip=args.jito_tip,
            )

        if result.success:
            print(f"\n{'═' * 60}")
            print(f"  ✓  SWAP CONFIRMED")
            print(f"{'═' * 60}")
            print(f"  Signature : {result.tx_signature}")
            print(f"  In        : {result.input_amount}")
            print(f"  Out       : {result.output_amount}")
            print(f"  Impact    : {result.price_impact_pct:.4f}%")
            print(f"  Elapsed   : {result.elapsed_ms:.0f} ms")
            print(f"  Explorer  : https://solscan.io/tx/{result.tx_signature}")
            print(f"{'═' * 60}\n")
            return 0
        else:
            print(f"\n✗  Swap failed: {result.error}  ({result.elapsed_ms:.0f} ms)")
            return 1

    finally:
        await swapper.close()


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    sys.exit(asyncio.run(_run(args)))


if __name__ == "__main__":
    main()
