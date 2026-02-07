"""Bridge to execute swaps using the proven sol-swapper implementation."""

from __future__ import annotations

import asyncio
import importlib.util
import logging
import os
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class SolSwapperBridge:
    """Thin sync wrapper around `sol-swapper/swapper.py` async API."""

    def __init__(self):
        repo_root = Path(__file__).resolve().parents[1]
        module_path = repo_root / "sol-swapper" / "swapper.py"
        spec = importlib.util.spec_from_file_location("sol_swapper_module", module_path)
        if not spec or not spec.loader:
            raise RuntimeError(f"Unable to load sol-swapper module from {module_path}")

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self._module = module

        self.aws_region = os.getenv("SOL_SWAPPER_AWS_REGION", "eu-west-1")
        self.wallet_secret = os.getenv("SOL_SWAPPER_WALLET_SECRET", "HOT_WALLET_1")
        self.helius_secret = os.getenv("SOL_SWAPPER_HELIUS_SECRET", "HELIUS_KEY")
        self.jupiter_secret = os.getenv("SOL_SWAPPER_JUPITER_SECRET", "JUPITER_API_KEY")
        self.max_tps = int(os.getenv("SOL_SWAPPER_MAX_TPS", str(module.MAX_TPS)))

    async def _execute(self, *, is_buy: bool, token_mint: str, amount: float | int,
                       slippage_bps: int, priority_fee: int, jito_tip: int) -> Dict:
        swapper = self._module.create_swapper(
            aws_region=self.aws_region,
            wallet_secret_id=self.wallet_secret,
            helius_secret_id=self.helius_secret,
            jupiter_secret_id=self.jupiter_secret,
            max_tps=self.max_tps,
        )
        try:
            if is_buy:
                result = await swapper.buy_token(
                    token_mint=token_mint,
                    sol_amount=float(amount),
                    slippage_bps=slippage_bps,
                    priority_fee=priority_fee,
                    jito_tip=jito_tip,
                )
            else:
                result = await swapper.sell_token(
                    token_mint=token_mint,
                    token_amount_raw=int(amount),
                    slippage_bps=slippage_bps,
                    priority_fee=priority_fee,
                    jito_tip=jito_tip,
                )
            return {
                "success": bool(result.success),
                "signature": result.tx_signature,
                "input_amount": result.input_amount,
                "output_amount": result.output_amount,
                "price_impact_pct": result.price_impact_pct,
                "elapsed_ms": result.elapsed_ms,
                "error": result.error,
            }
        finally:
            await swapper.close()

    def buy(self, token_mint: str, sol_amount: float, slippage_bps: int,
            priority_fee: int, jito_tip: int) -> Dict:
        return asyncio.run(self._execute(
            is_buy=True,
            token_mint=token_mint,
            amount=sol_amount,
            slippage_bps=slippage_bps,
            priority_fee=priority_fee,
            jito_tip=jito_tip,
        ))

    def sell(self, token_mint: str, token_amount_raw: int, slippage_bps: int,
             priority_fee: int, jito_tip: int) -> Dict:
        return asyncio.run(self._execute(
            is_buy=False,
            token_mint=token_mint,
            amount=token_amount_raw,
            slippage_bps=slippage_bps,
            priority_fee=priority_fee,
            jito_tip=jito_tip,
        ))