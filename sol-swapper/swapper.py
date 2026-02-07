"""
SOL <-> Token Swapper  v2
=========================
Helius Sender SWQOS-Only routing · Jupiter v1 swap API · 15 TPS cap
Keys loaded from AWS Secrets Manager (EC2 instance, eu-west-1).

Fixes over v1
-------------
- Jupiter API key sent via ``x-api-key`` header on quote + swap calls
- Helius Sender endpoint (``lon-sender.helius-rpc.com``) with ``?swqos_only=true``
- Jito tip transfer instruction injected into every swap transaction
"""

import json
import time
import random
import struct
import base64
import asyncio
import logging
from enum import Enum
from dataclasses import dataclass
from typing import Optional, List

import boto3
import aiohttp
from botocore.exceptions import ClientError

# Solana / solders
from solders.keypair import Keypair                             # type: ignore
from solders.pubkey import Pubkey                               # type: ignore
from solders.hash import Hash                                   # type: ignore
from solders.transaction import VersionedTransaction            # type: ignore
from solders.message import MessageV0, MessageHeader            # type: ignore
from solders.instruction import CompiledInstruction             # type: ignore
from solders.system_program import ID as SYS_PROGRAM_ID         # type: ignore

# ─── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-7s │ %(message)s",
)
log = logging.getLogger(__name__)

# ─── Constants ──────────────────────────────────────────────────────────────
SOL_MINT = "So11111111111111111111111111111111111111112"
LAMPORTS_PER_SOL = 1_000_000_000
MAX_TPS = 15

# Jupiter v1 endpoints (paid / api-key gated)
JUPITER_QUOTE_URL = "https://api.jup.ag/swap/v1/quote"
JUPITER_SWAP_URL  = "https://api.jup.ag/swap/v1/swap"

# AWS — all secrets live under one SecretId
AWS_SECRET_CONTAINER = "prod/solana-bot/keys"

# ─── Helius Sender ──────────────────────────────────────────────────────────
# London regional endpoint — closest to eu-west-1 (Ireland)
# SWQOS-only routing via query param (per Helius docs)
# No API key needed — Sender is free, zero credits consumed
SENDER_ENDPOINT = "http://lon-sender.helius-rpc.com/fast?swqos_only=true"

# ─── Jito Tip Accounts (mainnet-beta) ──────────────────────────────────────
# Required by Helius Sender: every tx must include a SOL transfer to one
# of these accounts.  Min tip: 5 000 lamports for SWQOS-only.
JITO_TIP_ACCOUNTS: List[str] = [
    "4ACfpUFoaSD9bfPdeu6DBt89gB6ENTeHBXCAi87NhDEE",
    "D2L6yPZ2FmmmTKPgzaMKdhu6EWZcTpLy1Vhx8uvZe7NZ",
    "9bnz4RShgq1hAnLnZbP8kbgBg1kEmcJBYQq3gQbmnSta",
    "5VY91ws6B2hMmBFRsXkoAAdsPHBJwRfBht4DXox3xkwn",
    "2nyhqdwKcJZR2vcqCyrYsaPVdAnFoJjiksCXJ7hfEYgD",
    "2q5pghRs6arqVjRvT5gfgWfWcHWmw1ZuCzphgd5KfWGJ",
    "wyvPkWjVZz1M8fHQnMMCDTQDbkManefNNhweYk5WkcF",
    "3KCKozbAaF75qEU33jtzozcJ29yJuaLJTy2jFdzUY8bT",
    "4vieeGHPYPG2MmyPRcYjdiDmmhN3ww7hsFNap8pVN3Ey",
    "4TQLFNWK8AovT1gFvda5jfw2oJeRMKEmw7aH6MGBJ3or",
]

SWQOS_MIN_TIP_LAMPORTS = 5_000      # 0.000005 SOL
DEFAULT_TIP_LAMPORTS   = 10_000     # 2× minimum — small safety margin


# ─── Enums / Config ────────────────────────────────────────────────────────
class SwapDirection(Enum):
    BUY  = "buy"    # SOL  → Token
    SELL = "sell"   # Token → SOL


@dataclass
class SwapConfig:
    token_mint: str
    direction: SwapDirection
    amount_lamports: int
    slippage_bps: int = 100
    priority_fee_lamports: int = 100_000
    jito_tip_lamports: int = DEFAULT_TIP_LAMPORTS
    max_retries: int = 3
    retry_delay_s: float = 1.0


@dataclass
class SwapResult:
    success: bool
    tx_signature: Optional[str] = None
    input_amount: int = 0
    output_amount: int = 0
    price_impact_pct: float = 0.0
    error: Optional[str] = None
    elapsed_ms: float = 0.0


# ─── Rate Limiter ───────────────────────────────────────────────────────────
class RateLimiter:
    """Async token-bucket limiter capped at *max_tps* per second."""

    def __init__(self, max_tps: int = MAX_TPS):
        self.max_tps = max_tps
        self.tokens = float(max_tps)
        self.last_refill = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self.last_refill
            self.tokens = min(self.max_tps, self.tokens + elapsed * self.max_tps)
            self.last_refill = now

            if self.tokens < 1.0:
                wait = (1.0 - self.tokens) / self.max_tps
                log.debug("Rate-limit: sleeping %.3fs", wait)
                await asyncio.sleep(wait)
                self.tokens = 0.0
            else:
                self.tokens -= 1.0


# ─── Secrets Manager ────────────────────────────────────────────────────────
class SecretsLoader:
    """Pull secrets from a single AWS Secrets Manager JSON blob."""

    def __init__(self, region: str = "eu-west-1"):
        self._client = boto3.client("secretsmanager", region_name=region)

    def get_secret(self, secret_id: str, json_key: Optional[str] = None) -> str:
        try:
            resp = self._client.get_secret_value(SecretId=secret_id)
            raw = resp.get("SecretString") or base64.b64decode(
                resp["SecretBinary"]
            ).decode()

            if json_key:
                try:
                    data = json.loads(raw)
                    if isinstance(data, dict) and json_key in data:
                        return str(data[json_key])
                except json.JSONDecodeError:
                    pass

            # Fallback
            try:
                data = json.loads(raw)
                if isinstance(data, dict) and not json_key:
                    return next(iter(data.values()))
                return str(data)
            except (json.JSONDecodeError, StopIteration):
                return raw

        except ClientError as exc:
            log.error("SecretsManager error for %s (key=%s): %s", secret_id, json_key, exc)
            raise

    def load_keypair(self, secret_id: str, json_key: Optional[str] = None) -> Keypair:
        pk_str = self.get_secret(secret_id, json_key).strip()
        return Keypair.from_base58_string(pk_str)


# ─── Jito Tip Injection ────────────────────────────────────────────────────
def _inject_jito_tip(
    raw_tx: bytes,
    payer_pubkey: Pubkey,
    tip_lamports: int = DEFAULT_TIP_LAMPORTS,
) -> bytes:
    """
    Take a Jupiter-built VersionedTransaction (V0), append a SOL transfer
    to a random Jito tip account, and return the modified *unsigned* tx bytes.

    Required because Helius Sender mandates a tip instruction inside the
    transaction itself — Jupiter does not add one.

    The function:
      1. Copies the existing V0 message fields
      2. Adds the tip account pubkey to static account keys (writable)
      3. Locates (or adds) the System Program in the account list
      4. Appends a compiled Transfer instruction
      5. Rebuilds a new MessageV0 preserving all address-table lookups
    """
    tx = VersionedTransaction.from_bytes(raw_tx)
    msg = tx.message

    # ── Copy existing message components ─────────────────────────────────
    static_keys: list       = list(msg.account_keys)
    instructions: list      = list(msg.instructions)
    header: MessageHeader   = msg.header
    lookups: list           = list(msg.address_table_lookups)
    blockhash: Hash         = msg.recent_blockhash

    num_signers    = header.num_required_signatures
    num_ro_signed  = header.num_readonly_signed_accounts
    num_ro_unsigned = header.num_readonly_unsigned_accounts

    # ── Payer is always index 0 ──────────────────────────────────────────
    payer_idx = 0
    assert static_keys[payer_idx] == payer_pubkey, (
        f"Expected payer {payer_pubkey} at index 0, got {static_keys[payer_idx]}"
    )

    # ── Account layout in a V0 message (positions matter for mutability) ─
    #
    #   [0 .. num_signers - num_ro_signed - 1]   = writable signers
    #   [num_signers - num_ro_signed .. num_signers - 1] = readonly signers
    #   [num_signers .. total - num_ro_unsigned - 1]     = writable non-signers
    #   [total - num_ro_unsigned .. total - 1]           = readonly non-signers
    #
    # We need to insert:
    #   - tip account  → writable non-signer
    #   - system prog  → readonly non-signer  (if not already present)

    # ── Locate the System Program ────────────────────────────────────────
    sys_idx: Optional[int] = None
    for i, key in enumerate(static_keys):
        if key == SYS_PROGRAM_ID:
            sys_idx = i
            break

    if sys_idx is None:
        # Append as readonly non-signer
        static_keys.append(SYS_PROGRAM_ID)
        sys_idx = len(static_keys) - 1
        num_ro_unsigned += 1

    # ── Insert tip account as writable non-signer ────────────────────────
    # Insert it just before the readonly-unsigned block so it lands in the
    # writable-non-signer region.
    insert_pos = len(static_keys) - num_ro_unsigned
    tip_pubkey = Pubkey.from_string(random.choice(JITO_TIP_ACCOUNTS))
    static_keys.insert(insert_pos, tip_pubkey)
    tip_idx = insert_pos

    # Because we inserted before the readonly-unsigned block, any existing
    # instruction account indices >= insert_pos need to be bumped by 1.
    # The sys_idx may also have shifted.
    if sys_idx >= insert_pos:
        sys_idx += 1

    shifted_instructions = []
    for ix in instructions:
        new_prog_idx = ix.program_id_index + (1 if ix.program_id_index >= insert_pos else 0)
        new_accounts = bytes(
            (a + 1 if a >= insert_pos else a) for a in ix.accounts
        )
        shifted_instructions.append(
            CompiledInstruction(
                program_id_index=new_prog_idx,
                accounts=new_accounts,
                data=ix.data,
            )
        )

    # ── Build the transfer instruction ───────────────────────────────────
    # SystemProgram::Transfer = instruction index 2
    # Layout: u32 instruction_type (LE) + u64 lamports (LE)
    transfer_data = struct.pack("<I", 2) + struct.pack("<Q", tip_lamports)

    tip_ix = CompiledInstruction(
        program_id_index=sys_idx,
        accounts=bytes([payer_idx, tip_idx]),
        data=transfer_data,
    )
    shifted_instructions.append(tip_ix)

    # ── Rebuild the V0 message ───────────────────────────────────────────
    new_header = MessageHeader(
        num_signers,
        num_ro_signed,
        num_ro_unsigned,
    )

    new_msg = MessageV0(
        new_header,
        static_keys,
        blockhash,
        shifted_instructions,
        lookups,
    )

    # Return as raw bytes — caller will sign
    unsigned_tx = VersionedTransaction(new_msg, [])
    return bytes(unsigned_tx)


# ─── Core Swapper ───────────────────────────────────────────────────────────
class SolTokenSwapper:
    """
    Production swapper:
      1. Fetches a Jupiter v1 quote  (with x-api-key header)
      2. Builds the swap transaction via Jupiter  (with x-api-key header)
      3. Injects Jito tip transfer into the transaction
      4. Signs locally
      5. Sends via Helius Sender  (lon-sender, ?swqos_only=true)
      6. Confirms via Helius RPC
    """

    def __init__(
        self,
        keypair: Keypair,
        helius_api_key: str,
        jupiter_api_key: str,
        max_tps: int = MAX_TPS,
    ):
        self.keypair = keypair
        self.pubkey = keypair.pubkey()
        self.pubkey_str = str(self.pubkey)

        # Helius *RPC* — used only for confirmation polling
        self.helius_rpc_url = (
            f"https://mainnet.helius-rpc.com/?api-key={helius_api_key}"
        )

        # Helius *Sender* — used for submitting transactions
        # London endpoint, SWQOS-only routing
        self.sender_url = SENDER_ENDPOINT

        # Jupiter API key — sent as x-api-key header
        self.jupiter_api_key = jupiter_api_key

        self.limiter = RateLimiter(max_tps)
        self._session: Optional[aiohttp.ClientSession] = None

    # ── lifecycle ────────────────────────────────────────────────────────
    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=30)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()

    # ── Jupiter common headers ───────────────────────────────────────────
    def _jupiter_headers(self) -> dict:
        return {
            "Content-Type": "application/json",
            "x-api-key": self.jupiter_api_key,
        }

    # ── Jupiter: quote ───────────────────────────────────────────────────
    async def _get_quote(self, cfg: SwapConfig) -> dict:
        session = await self._get_session()

        if cfg.direction == SwapDirection.BUY:
            input_mint, output_mint = SOL_MINT, cfg.token_mint
        else:
            input_mint, output_mint = cfg.token_mint, SOL_MINT

        params = {
            "inputMint": input_mint,
            "outputMint": output_mint,
            "amount": str(cfg.amount_lamports),
            "slippageBps": cfg.slippage_bps,
        }

        await self.limiter.acquire()
        async with session.get(
            JUPITER_QUOTE_URL, params=params, headers=self._jupiter_headers()
        ) as resp:
            if resp.status != 200:
                body = await resp.text()
                raise RuntimeError(f"Jupiter quote failed ({resp.status}): {body}")
            return await resp.json()

    # ── Jupiter: swap transaction ────────────────────────────────────────
    async def _get_swap_tx(self, quote: dict, cfg: SwapConfig) -> bytes:
        session = await self._get_session()

        payload = {
            "quoteResponse": quote,
            "userPublicKey": self.pubkey_str,
            "wrapAndUnwrapSol": True,
            "dynamicComputeUnitLimit": True,
            "prioritizationFeeLamports": cfg.priority_fee_lamports,
        }

        await self.limiter.acquire()
        async with session.post(
            JUPITER_SWAP_URL, json=payload, headers=self._jupiter_headers()
        ) as resp:
            if resp.status != 200:
                body = await resp.text()
                raise RuntimeError(f"Jupiter swap build failed ({resp.status}): {body}")
            data = await resp.json()
            return base64.b64decode(data["swapTransaction"])

    # ── Sign ─────────────────────────────────────────────────────────────
    def _sign_transaction(self, raw_tx: bytes) -> str:
        """Deserialise, sign, return base-64 wire format."""
        tx = VersionedTransaction.from_bytes(raw_tx)
        signed = VersionedTransaction(tx.message, [self.keypair])
        return base64.b64encode(bytes(signed)).decode()

    # ── Helius Sender (SWQOS-Only) ───────────────────────────────────────
    async def _send_via_sender(self, signed_b64: str) -> str:
        """
        Submit a signed transaction through Helius Sender.

        Endpoint : http://lon-sender.helius-rpc.com/fast?swqos_only=true
        Protocol : standard JSON-RPC sendTransaction
        Routing  : SWQOS-only (query param, NOT custom headers)

        Per Helius docs:
          - skipPreflight: true   (mandatory)
          - maxRetries: 0         (we handle retries ourselves)
          - No API key needed     (Sender is free, zero credits)
          - No custom headers     (routing is via ?swqos_only=true)
        """
        session = await self._get_session()

        rpc_payload = {
            "jsonrpc": "2.0",
            "id": str(int(time.time() * 1000)),
            "method": "sendTransaction",
            "params": [
                signed_b64,
                {
                    "encoding": "base64",
                    "skipPreflight": True,
                    "maxRetries": 0,
                },
            ],
        }

        headers = {"Content-Type": "application/json"}

        await self.limiter.acquire()
        async with session.post(
            self.sender_url, json=rpc_payload, headers=headers
        ) as resp:
            data = await resp.json()

        if "error" in data:
            raise RuntimeError(f"Helius Sender error: {data['error']}")

        return data["result"]

    # ── Confirm (via regular Helius RPC) ─────────────────────────────────
    async def _confirm_transaction(
        self, signature: str, timeout_s: float = 60.0, poll_s: float = 2.0
    ) -> bool:
        session = await self._get_session()
        deadline = time.monotonic() + timeout_s

        while time.monotonic() < deadline:
            rpc = {
                "jsonrpc": "2.0",
                "id": "1",
                "method": "getSignatureStatuses",
                "params": [[signature], {"searchTransactionHistory": True}],
            }
            await self.limiter.acquire()
            async with session.post(self.helius_rpc_url, json=rpc) as resp:
                data = await resp.json()

            statuses = data.get("result", {}).get("value", [None])
            status = statuses[0] if statuses else None

            if status is not None:
                if status.get("err"):
                    log.warning("Tx %s failed on-chain: %s", signature, status["err"])
                    return False
                conf = status.get("confirmationStatus", "")
                if conf in ("confirmed", "finalized"):
                    return True

            await asyncio.sleep(poll_s)

        log.warning("Tx %s confirmation timed out after %.0fs", signature, timeout_s)
        return False

    # ── Public API ───────────────────────────────────────────────────────
    async def swap(self, cfg: SwapConfig) -> SwapResult:
        """Execute: quote → build → tip-inject → sign → send → confirm."""
        t0 = time.monotonic()

        for attempt in range(1, cfg.max_retries + 1):
            try:
                log.info(
                    "Swap attempt %d/%d  %s  %s  amount=%s  slippage=%dbps  tip=%d",
                    attempt, cfg.max_retries,
                    cfg.direction.value, cfg.token_mint,
                    cfg.amount_lamports, cfg.slippage_bps,
                    cfg.jito_tip_lamports,
                )

                # 1. Quote
                quote = await self._get_quote(cfg)
                in_amount  = int(quote.get("inAmount", cfg.amount_lamports))
                out_amount = int(quote.get("outAmount", 0))
                impact     = float(quote.get("priceImpactPct", 0))
                log.info("Quote: in=%s  out=%s  impact=%.4f%%", in_amount, out_amount, impact)

                # 2. Build swap tx (from Jupiter)
                raw_tx = await self._get_swap_tx(quote, cfg)

                # 3. Inject Jito tip
                log.info("Injecting Jito tip (%d lamports) for SWQOS-only…", cfg.jito_tip_lamports)
                tipped_tx = _inject_jito_tip(raw_tx, self.pubkey, cfg.jito_tip_lamports)

                # 4. Sign
                signed_b64 = self._sign_transaction(tipped_tx)

                # 5. Send via Helius Sender
                sig = await self._send_via_sender(signed_b64)
                log.info("Sent via Sender: %s", sig)

                # 6. Confirm via Helius RPC
                confirmed = await self._confirm_transaction(sig)
                elapsed = (time.monotonic() - t0) * 1000

                if confirmed:
                    log.info("✓ Confirmed in %.0fms: %s", elapsed, sig)
                    return SwapResult(
                        success=True,
                        tx_signature=sig,
                        input_amount=in_amount,
                        output_amount=out_amount,
                        price_impact_pct=impact,
                        elapsed_ms=elapsed,
                    )
                else:
                    log.warning("Tx sent but not confirmed – retrying…")

            except Exception as exc:
                log.error("Attempt %d failed: %s", attempt, exc)
                if attempt < cfg.max_retries:
                    await asyncio.sleep(cfg.retry_delay_s * attempt)

        elapsed = (time.monotonic() - t0) * 1000
        return SwapResult(success=False, error="All retry attempts exhausted", elapsed_ms=elapsed)

    # ── Convenience helpers ──────────────────────────────────────────────
    async def buy_token(
        self,
        token_mint: str,
        sol_amount: float,
        slippage_bps: int = 100,
        priority_fee: int = 100_000,
        jito_tip: int = DEFAULT_TIP_LAMPORTS,
    ) -> SwapResult:
        cfg = SwapConfig(
            token_mint=token_mint,
            direction=SwapDirection.BUY,
            amount_lamports=int(sol_amount * LAMPORTS_PER_SOL),
            slippage_bps=slippage_bps,
            priority_fee_lamports=priority_fee,
            jito_tip_lamports=jito_tip,
        )
        return await self.swap(cfg)

    async def sell_token(
        self,
        token_mint: str,
        token_amount_raw: int,
        slippage_bps: int = 100,
        priority_fee: int = 100_000,
        jito_tip: int = DEFAULT_TIP_LAMPORTS,
    ) -> SwapResult:
        cfg = SwapConfig(
            token_mint=token_mint,
            direction=SwapDirection.SELL,
            amount_lamports=token_amount_raw,
            slippage_bps=slippage_bps,
            priority_fee_lamports=priority_fee,
            jito_tip_lamports=jito_tip,
        )
        return await self.swap(cfg)


# ─── Factory ────────────────────────────────────────────────────────────────
def create_swapper(
    aws_region: str = "eu-west-1",
    wallet_secret_id: str = "HOT_WALLET_1",
    helius_secret_id: str = "HELIUS_KEY",
    jupiter_secret_id: str = "JUPITER_API_KEY",
    max_tps: int = MAX_TPS,
) -> SolTokenSwapper:
    """Build a ready-to-use swapper with all three secrets from AWS."""

    secrets = SecretsLoader(region=aws_region)

    # 1. Wallet keypair
    log.info("Loading wallet key '%s' from secret '%s'…", wallet_secret_id, AWS_SECRET_CONTAINER)
    keypair = secrets.load_keypair(AWS_SECRET_CONTAINER, json_key=wallet_secret_id)
    log.info("Wallet public key: %s", keypair.pubkey())

    # 2. Helius API key (for RPC confirmation only)
    log.info("Loading Helius API key '%s' from secret '%s'…", helius_secret_id, AWS_SECRET_CONTAINER)
    helius_key = secrets.get_secret(AWS_SECRET_CONTAINER, json_key=helius_secret_id)

    # 3. Jupiter API key (for quote + swap endpoints)
    log.info("Loading Jupiter API key '%s' from secret '%s'…", jupiter_secret_id, AWS_SECRET_CONTAINER)
    jupiter_key = secrets.get_secret(AWS_SECRET_CONTAINER, json_key=jupiter_secret_id)

    return SolTokenSwapper(
        keypair=keypair,
        helius_api_key=helius_key,
        jupiter_api_key=jupiter_key,
        max_tps=max_tps,
    )
