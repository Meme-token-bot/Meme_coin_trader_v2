# SOL ↔ Token Swapper

Async Python swapper using **Helius SWQOS-Only Alternative routing** and the **Jupiter v6 aggregator**. Rate-limited to 15 TPS. Secrets loaded from AWS Secrets Manager.

---

## Architecture

```
┌──────────────┐      quote / swap-tx      ┌─────────────────┐
│   Swapper    │ ◄──────────────────────►   │   Jupiter v6    │
│  (this app)  │                            └─────────────────┘
│              │   sendTransaction           ┌─────────────────┐
│  ┌────────┐  │ ──── SWQOS-Only Alt ─────► │   Helius RPC    │
│  │Keypair │  │   X-Helius-SWQOS: true     │  (staked conn)  │
│  └────────┘  │   X-Helius-Routing:        └─────────────────┘
│      ▲       │     swqos-only-alt
│      │       │
│  AWS Secrets │
│  Manager     │
└──────────────┘
```

## Prerequisites

| Requirement | Detail |
|---|---|
| Python | 3.10+ |
| AWS Secrets | `HOT_WALLET_1` → base-58 private key |
| | `HELIUS_KEY` → Helius API key (string) |
| IAM | EC2 instance role with `secretsmanager:GetSecretValue` on both secrets |
| Helius Plan | Must support SWQOS / staked connections |

## EC2 Deployment

```bash
# 1. Clone / copy files onto the instance
scp -r sol-swapper/ ec2-user@<IP>:~/

# 2. Install
cd ~/sol-swapper
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 3. Run a buy (0.1 SOL → token)
python main.py buy <TOKEN_MINT> 0.1

# 4. Run a sell (1000000 raw units → SOL)
python main.py sell <TOKEN_MINT> 1000000
```

## CLI Flags

| Flag | Default | Description |
|---|---|---|
| `--slippage` | `100` | Slippage tolerance in basis points |
| `--priority-fee` | `100000` | Compute-unit tip in lamports |
| `--region` | `us-east-1` | AWS region for Secrets Manager |
| `--wallet-secret` | `HOT_WALLET_1` | Secret ID for wallet private key |
| `--helius-secret` | `HELIUS_KEY` | Secret ID for Helius API key |
| `--max-tps` | `15` | Max transactions per second |

## Programmatic Usage

```python
import asyncio
from swapper import create_swapper

async def main():
    s = create_swapper()  # pulls keys from Secrets Manager

    # Buy token with 0.5 SOL
    result = await s.buy_token("TOKEN_MINT_HERE", sol_amount=0.5)
    print(result)

    # Sell 5 000 000 raw units back to SOL
    result = await s.sell_token("TOKEN_MINT_HERE", token_amount_raw=5_000_000)
    print(result)

    await s.close()

asyncio.run(main())
```

## IAM Policy (minimum)

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": "secretsmanager:GetSecretValue",
      "Resource": [
        "arn:aws:secretsmanager:us-east-1:<ACCOUNT_ID>:secret:HOT_WALLET_1-*",
        "arn:aws:secretsmanager:us-east-1:<ACCOUNT_ID>:secret:HELIUS_KEY-*"
      ]
    }
  ]
}
```

## Notes

- **SWQOS-Only Alt routing** sends your transaction exclusively through Helius's staked validator connections via the `X-Helius-SWQOS` and `X-Helius-Routing` headers, bypassing the public mempool for faster landing.
- **Rate limiter** uses an async token-bucket algorithm capped at 15 TPS across all outbound calls (Jupiter + Helius combined).
- **Retries** default to 3 attempts with exponential back-off per swap.
- The swapper uses `skipPreflight: true` and `maxRetries: 0` on the RPC call — preflight is skipped to reduce latency, and retries are handled application-side.
