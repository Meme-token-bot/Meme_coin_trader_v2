from __future__ import annotations

import argparse
import base64
import logging
import time
from decimal import Decimal, InvalidOperation
from typing import Optional

import requests
from solders.transaction import Transaction

from core.live_trading_engine import LiveTradingEngine
from core.secrets_manager import init_secrets, get_secret

SOL_MINT = "So11111111111111111111111111111111111111112"

logger = logging.getLogger("TokenSwapToSolV2")


def _get_token_decimals(rpc_url: str, mint: str) -> int:
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "getTokenSupply",
        "params": [mint],
    }
    response = requests.post(rpc_url, json=payload, timeout=20)
    response.raise_for_status()
    result = response.json().get("result", {})
    value = result.get("value", {})
    decimals = value.get("decimals")
    if decimals is None:
        raise RuntimeError("Unable to fetch token decimals from RPC.")
    return int(decimals)


def _to_base_units(amount: str, decimals: int) -> int:
    try:
        parsed = Decimal(amount)
    except InvalidOperation as exc:
        raise ValueError(f"Invalid amount: {amount}") from exc
    if parsed <= 0:
        raise ValueError("Amount must be greater than 0.")
    return int(parsed * (10 ** decimals))


def _get_jupiter_quote(input_mint: str, output_mint: str, amount: int, slippage_bps: int) -> dict:
    params = {
        "inputMint": input_mint,
        "outputMint": output_mint,
        "amount": str(amount),
        "slippageBps": slippage_bps,
    }
    metis_url = get_secret("QUICKNODE_METIS_URL")
    quote_url = f"{metis_url.rstrip('/')}/quote" if metis_url else "https://quote-api.jup.ag/v6/quote"
    resp = requests.get(quote_url, params=params, timeout=20)
    if resp.status_code != 200:
        raise RuntimeError(f"Jupiter quote failed: {resp.status_code} - {resp.text[:200]}")
    return resp.json()


def _get_jupiter_swap(
    quote: dict,
    wallet_pubkey: str,
    compute_unit_price: Optional[int],
    prioritization_fee: Optional[str],
    as_legacy: bool,
) -> str:
    if compute_unit_price and prioritization_fee:
        raise ValueError("Do not set computeUnitPriceMicroLamports and prioritizationFeeLamports together.")

    payload = {
        "quoteResponse": quote,
        "userPublicKey": wallet_pubkey,
        "wrapAndUnwrapSol": True,
        "dynamicComputeUnitLimit": True,
    }
    if compute_unit_price:
        payload["computeUnitPriceMicroLamports"] = compute_unit_price
    if prioritization_fee:
        payload["prioritizationFeeLamports"] = prioritization_fee
    if as_legacy:
        payload["asLegacyTransaction"] = True
        payload["useSharedAccounts"] = False

    metis_url = get_secret("QUICKNODE_METIS_URL")
    swap_url = f"{metis_url.rstrip('/')}/swap" if metis_url else "https://quote-api.jup.ag/v6/swap"
    resp = requests.post(swap_url, json=payload, timeout=30)
    if resp.status_code != 200:
        raise RuntimeError(f"Jupiter swap failed: {resp.status_code} - {resp.text[:200]}")
    data = resp.json()
    swap_tx = data.get("swapTransaction")
    if not swap_tx:
        raise RuntimeError("Jupiter swap response missing swapTransaction.")
    return swap_tx


def _send_via_helius_sender(
    swap_tx_base64: str,
    sender_url: str,
    skip_preflight: bool,
    max_retries: int,
) -> str:
    payload = {
        "jsonrpc": "2.0",
        "id": str(int(time.time() * 1000)),
        "method": "sendTransaction",
        "params": [
            swap_tx_base64,
            {
                "encoding": "base64",
                "skipPreflight": skip_preflight,
                "maxRetries": max_retries,
            },
        ],
    }
    headers = {"Content-Type": "application/json"}
    resp = requests.post(sender_url, json=payload, headers=headers, timeout=30)
    if resp.status_code != 200:
        raise RuntimeError(f"Helius sender HTTP error: {resp.status_code} - {resp.text[:200]}")
    data = resp.json()
    if "result" not in data:
        raise RuntimeError(f"Helius sender error: {data.get('error')}")
    return str(data["result"])


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    parser = argparse.ArgumentParser(description="Swap a token to SOL via Jupiter + Helius Sender.")
    parser.add_argument("--mint", required=True, help="Token mint address to swap to SOL.")
    parser.add_argument("--amount", required=True, help="Token amount (UI units).")
    parser.add_argument("--slippage-bps", type=int, default=100, help="Slippage in basis points.")
    parser.add_argument("--cu-price", type=int, default=3000, help="Compute unit price (micro-lamports).")
    parser.add_argument(
        "--prioritization-fee",
        default=None,
        help="Jupiter prioritization fee (lamports) or 'auto'. Mutually exclusive with --cu-price.",
    )
    parser.add_argument(
        "--no-skip-preflight",
        action="store_false",
        dest="skip_preflight",
        help="Disable skip-preflight checks (not supported by Helius Sender).",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Max retries for the sender RPC (ignored if skip-preflight is required).",
    )
    parser.add_argument(
        "--confirm-timeout",
        type=int,
        default=90,
        help="Seconds to wait for confirmation before failing.",
    )
    parser.set_defaults(skip_preflight=True)
    args = parser.parse_args()
    if not args.skip_preflight:
        logger.warning("Helius Sender does not support preflight checks; forcing skip-preflight.")
        args.skip_preflight = True

    init_secrets()
    engine = LiveTradingEngine()

    if not engine.wallet_pubkey:
        raise RuntimeError("Wallet not loaded. Check secrets configuration.")
    if not engine.helius_key:
        raise RuntimeError("HELIUS_KEY not configured. RPC unavailable.")

    sender_url = get_secret("HELIUS_SENDER_URL")
    if not sender_url:
        sender_url = f"https://sender.helius-rpc.com/fast?api-key={engine.helius_key}"

    logger.info("🔄 SWAP: %s TOKEN → SOL", args.amount)
    decimals = _get_token_decimals(engine.rpc_url, args.mint)
    amount_base_units = _to_base_units(args.amount, decimals)

    logger.info("📊 Getting Jupiter quote...")
    quote = _get_jupiter_quote(args.mint, SOL_MINT, amount_base_units, args.slippage_bps)
    out_amount = quote.get("outAmount")
    if out_amount:
        logger.info("📈 Quote: %s TOKEN → %s SOL (lamports)", args.amount, out_amount)

    prioritization_fee = args.prioritization_fee
    compute_unit_price = args.cu_price if args.cu_price else None
    if prioritization_fee and compute_unit_price:
        raise ValueError("Refusing to set both compute unit price and prioritization fee.")

    logger.info("🔧 Building Jupiter swap transaction...")
    swap_tx_base64 = _get_jupiter_swap(
        quote,
        engine.wallet_pubkey,
        compute_unit_price=compute_unit_price,
        prioritization_fee=prioritization_fee,
        as_legacy=True,
    )

    tx_bytes = base64.b64decode(swap_tx_base64)
    try:
        legacy_tx = Transaction.from_bytes(tx_bytes)
    except Exception as exc:
        raise RuntimeError("Expected legacy transaction from Jupiter; enable asLegacyTransaction.") from exc
    legacy_tx = engine._inject_helius_tip(legacy_tx)
    legacy_tx.sign([engine.keypair], legacy_tx.message.recent_blockhash)
    signed_swap = base64.b64encode(bytes(legacy_tx)).decode("utf-8")

    logger.info("📤 Sending via Helius Sender...")
    signature = _send_via_helius_sender(
        signed_swap,
        sender_url,
        skip_preflight=args.skip_preflight,
        max_retries=args.max_retries,
    )
    logger.info("✅ Sent: %s", signature)

    logger.info("⏳ Confirming...")
    confirmed = engine._wait_for_confirmation(signature, timeout=args.confirm_timeout)
    if not confirmed:
        raise RuntimeError("Transaction not confirmed within timeout.")
    logger.info("✅ Confirmed!")


if __name__ == "__main__":
    main()

