"""
THE STRATEGIST V2 - Smart Signal Analysis with LEARNING LOOP
Trading System V2

LEARNING FEATURES:
1. Automatic strategy promotion - Challenger beats champion? Promote it!
2. Parameter optimization - Learns optimal stop/TP from actual results
3. Entry threshold tuning - Adjusts conviction thresholds based on hit rate
4. Trade outcome analysis - Tracks WHY trades fail to improve filters
5. Adaptive cluster weights - Already existed, now enhanced

Key Features:
1. SMART LLM GATING - Only call LLM if pre-filters pass
2. Multi-wallet signal aggregation
3. Exit signal tracking
4. Regime-adaptive exits
5. Adaptive cluster weights
"""

import json
import os
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from dataclasses import dataclass, field

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False


@dataclass
class StrategistConfig:
    """Configuration - now with adaptive thresholds"""
    min_liquidity: float = 30000
    min_volume_24h: float = 15000
    min_token_age_hours: float = 0.5
    max_token_age_hours: float = 168
    min_wallets_for_entry: int = 2
    single_wallet_min_conviction: int = 75
    aggregated_min_conviction: int = 60
    llm_conviction_threshold: int = 50
    llm_enabled: bool = True
    exit_sniper_count: int = 2
    exit_total_count: int = 3
    default_stop_loss: float = -0.12
    default_take_profit: float = 0.25
    default_trailing_stop: float = 0.08
    max_hold_hours: int = 12
    
    # Learning settings
    learning_enabled: bool = True
    min_trades_for_learning: int = 10  # Minimum trades before adjusting
    learning_interval_hours: int = 6   # How often to run learning loop
    promotion_min_trades: int = 15     # Min trades before considering promotion
    promotion_outperform_pct: float = 0.10  # Challenger must beat champion by 10%

config = StrategistConfig()


class RegimeDetector:
    """Market regime detection"""
    
    EXPANSION = "EXPANSION"
    CONSOLIDATION = "CONSOLIDATION"
    CONTRACTION = "CONTRACTION"
    
    REGIME_CONFIG = {
        'EXPANSION': {'position_mult': 1.3, 'stop_mult': 1.2, 'tp_mult': 1.3, 'trail_mult': 1.1},
        'CONSOLIDATION': {'position_mult': 0.8, 'stop_mult': 1.0, 'tp_mult': 1.0, 'trail_mult': 1.0},
        'CONTRACTION': {'position_mult': 0.5, 'stop_mult': 0.7, 'tp_mult': 0.6, 'trail_mult': 0.7}
    }
    
    def __init__(self, db):
        self.db = db
        self.current_regime = self.CONSOLIDATION
        self.confidence = 0.5
        self.sol_price = 0
        self.sol_change_24h = 0
    
    def update(self, market_conditions: Dict = None) -> Tuple[str, float]:
        if market_conditions:
            self.sol_price = market_conditions.get('sol_price_usd', 0)
            self.sol_change_24h = market_conditions.get('sol_24h_change_pct', 0)
        
        conditions = self.db.get_recent_market_conditions(hours=168)
        
        if len(conditions) < 5:
            return self.current_regime, self.confidence
        
        changes = [c.get('sol_24h_change_pct', 0) for c in conditions]
        recent = changes[:10] if len(changes) >= 10 else changes
        avg = np.mean(recent)
        vol = np.std(recent)
        
        if avg > 3:
            self.current_regime = self.EXPANSION
            self.confidence = min(0.95, 0.5 + abs(avg) / 15)
        elif avg < -3:
            self.current_regime = self.CONTRACTION
            self.confidence = min(0.95, 0.5 + abs(avg) / 15)
        else:
            self.current_regime = self.CONSOLIDATION
            self.confidence = 0.6 if vol > 5 else 0.55
        
        return self.current_regime, self.confidence
    
    def get_regime_multipliers(self) -> Dict:
        return self.REGIME_CONFIG.get(self.current_regime, self.REGIME_CONFIG['CONSOLIDATION'])
    
    def get_adjusted_exits(self, base_stop: float, base_tp: float, base_trail: float) -> Dict:
        mults = self.get_regime_multipliers()
        return {
            'stop_loss': base_stop * mults['stop_mult'],
            'take_profit': base_tp * mults['tp_mult'],
            'trailing_stop': base_trail * mults['trail_mult'],
            'regime': self.current_regime
        }
    
    def should_trade(self) -> Tuple[bool, str]:
        if self.current_regime == self.CONTRACTION and self.confidence > 0.7:
            return False, "Strong bear - minimal trading"
        return True, f"{self.current_regime} - trading OK"


class WalletClusterer:
    """Wallet clustering with adaptive weights - ENHANCED"""
    
    SNIPER = "SNIPER"
    RIDER = "RIDER"
    BALANCED = "BALANCED"
    GAMBLER = "GAMBLER"
    
    BASE_WEIGHTS = {SNIPER: 1.3, RIDER: 1.2, BALANCED: 1.0, GAMBLER: 0.6}
    
    def __init__(self, db):
        self.db = db
        self._weights = self.BASE_WEIGHTS.copy()
        self._last_update = datetime.now() - timedelta(hours=2)
        self._cluster_stats = {}  # Track per-cluster performance
    
    def get_wallet_cluster(self, wallet: str) -> str:
        data = self.db.get_wallet(wallet)
        if not data:
            return self.BALANCED
        
        cluster = data.get('cluster')
        if cluster:
            return cluster
        
        cluster = self._classify(data)
        self.db.update_wallet_cluster(wallet, cluster)
        return cluster
    
    def _classify(self, data: Dict) -> str:
        hold = data.get('avg_hold_hours', 3)
        wr = data.get('win_rate', 0.5)
        rr = data.get('risk_reward_ratio', 1)
        
        if hold < 3 and wr >= 0.55:
            return self.SNIPER
        if hold >= 6 and wr >= 0.50 and rr >= 1.5:
            return self.RIDER
        if wr < 0.40:
            return self.GAMBLER
        return self.BALANCED
    
    def get_adaptive_weight(self, cluster: str) -> float:
        self._maybe_update()
        return self._weights.get(cluster, 1.0)
    
    def _maybe_update(self):
        if (datetime.now() - self._last_update).seconds < 3600:
            return
        
        self._last_update = datetime.now()
        closed = self.db.get_closed_positions(days=14)
        
        stats = defaultdict(lambda: {'wins': 0, 'total': 0, 'pnl': 0})
        for pos in closed:
            w = self.db.get_wallet(pos['wallet_address'])
            if not w:
                continue
            c = w.get('cluster', self.BALANCED)
            stats[c]['total'] += 1
            profit = pos.get('profit_pct') or 0
            if profit > 0:
                stats[c]['wins'] += 1
            stats[c]['pnl'] += pos.get('profit_sol', 0) or 0
        
        self._cluster_stats = dict(stats)
        
        for c in self.BASE_WEIGHTS:
            if stats[c]['total'] >= 10:
                wr = stats[c]['wins'] / stats[c]['total']
                # Weight adjustment based on win rate AND profitability
                wr_bonus = (wr - 0.5) * 0.5
                pnl_bonus = min(0.2, max(-0.2, stats[c]['pnl'] / 10))  # Cap at +/-20%
                self._weights[c] = self.BASE_WEIGHTS[c] * (1 + wr_bonus + pnl_bonus)
        
        print(f"  📊 Cluster weights updated: {self._format_weights()}")
    
    def _format_weights(self) -> str:
        return ", ".join([f"{k}:{v:.2f}" for k, v in self._weights.items()])
    
    def get_cluster_stats(self) -> Dict:
        """Get cluster performance stats for analysis"""
        return self._cluster_stats


@dataclass
class TokenSignal:
    """Aggregated signals for a token"""
    token_address: str
    token_symbol: str
    first_seen: datetime
    last_updated: datetime
    liquidity: float
    volume_24h: float
    token_age_hours: float
    wallets: List[Dict] = field(default_factory=list)
    
    def add_wallet(self, wallet: str, cluster: str, win_rate: float,
                  roi: float, price: float, conviction: float):
        for w in self.wallets:
            if w['address'] == wallet:
                return
        self.wallets.append({
            'address': wallet, 'cluster': cluster, 'win_rate': win_rate,
            'roi': roi, 'entry_price': price, 'base_conviction': conviction,
            'timestamp': datetime.now()
        })
        self.last_updated = datetime.now()


class SignalAggregator:
    """Multi-wallet signal aggregation"""
    
    CLUSTER_BONUS = {2: 1.15, 3: 1.30, 4: 1.45}
    DIVERSITY_BONUS = {2: 1.10, 3: 1.25, 4: 1.40}
    
    def __init__(self):
        self.signals: Dict[str, TokenSignal] = {}
        self.triggered: set = set()
    
    def add_signal(self, token_address: str, token_symbol: str, wallet: str,
                  cluster: str, win_rate: float, roi: float, entry_price: float,
                  liquidity: float, volume_24h: float, token_age_hours: float,
                  base_conviction: float) -> Optional[Dict]:
        
        self._cleanup()
        
        if token_address not in self.signals:
            self.signals[token_address] = TokenSignal(
                token_address, token_symbol, datetime.now(), datetime.now(),
                liquidity, volume_24h, token_age_hours
            )
        
        sig = self.signals[token_address]
        sig.add_wallet(wallet, cluster, win_rate, roi, entry_price, base_conviction)
        
        if token_address in self.triggered:
            return None
        
        agg = self._calc_conviction(sig)
        count = len(sig.wallets)
        conv = agg['conviction']
        
        trigger = False
        if count >= config.min_wallets_for_entry and conv >= config.aggregated_min_conviction:
            trigger = True
        elif count == 1 and conv >= config.single_wallet_min_conviction:
            trigger = True
        
        if trigger:
            self.triggered.add(token_address)
            return self._build_decision(sig, agg)
        
        return None
    
    def _calc_conviction(self, sig: TokenSignal) -> Dict:
        if not sig.wallets:
            return {'conviction': 0, 'factors': []}
        
        base = np.mean([w['base_conviction'] for w in sig.wallets])
        clusters = defaultdict(int)
        for w in sig.wallets:
            clusters[w['cluster']] += 1
        
        factors = []
        conv = base
        
        max_same = max(clusters.values()) if clusters else 0
        if max_same >= 2:
            bonus = self.CLUSTER_BONUS.get(min(4, max_same), 1.0)
            conv *= bonus
            dom = max(clusters.keys(), key=lambda k: clusters[k])
            factors.append(f"{max_same}x {dom}")
        
        unique = len([c for c in clusters.values() if c > 0])
        if unique >= 2:
            bonus = self.DIVERSITY_BONUS.get(min(4, unique), 1.0)
            conv *= bonus
            factors.append(f"{unique} clusters")
        
        avg_wr = np.mean([w['win_rate'] for w in sig.wallets])
        if avg_wr >= 0.65:
            conv *= 1.1
            factors.append(f"High quality")
        
        return {
            'conviction': min(100, conv),
            'wallet_count': len(sig.wallets),
            'clusters': dict(clusters),
            'factors': factors,
            'avg_win_rate': avg_wr
        }
    
    def _build_decision(self, sig: TokenSignal, agg: Dict) -> Dict:
        prices = [w['entry_price'] for w in sig.wallets]
        return {
            'action': 'AGGREGATED_BUY',
            'should_enter': True,
            'conviction_score': agg['conviction'],
            'token_address': sig.token_address,
            'token_symbol': sig.token_symbol,
            'wallet_count': agg['wallet_count'],
            'clusters': agg['clusters'],
            'factors': agg['factors'],
            'wallets': [w['address'] for w in sig.wallets],
            'best_entry_price': min(prices),
            'reason': f"Multi-wallet: {', '.join(agg['factors'])}"
        }
    
    def _cleanup(self):
        cutoff = datetime.now() - timedelta(minutes=120)
        expired = [k for k, v in self.signals.items() if v.last_updated < cutoff]
        for k in expired:
            del self.signals[k]
            self.triggered.discard(k)
    
    def get_pending(self) -> List[Dict]:
        return [
            {'token': k, 'symbol': v.token_symbol, 'wallets': len(v.wallets),
             'conviction': self._calc_conviction(v)['conviction']}
            for k, v in self.signals.items() if k not in self.triggered
        ]


class ExitSignalTracker:
    """Tracks wallet exits"""
    
    def __init__(self, db):
        self.db = db
        self._exits: Dict[str, List[Dict]] = defaultdict(list)
    
    def process_exit(self, token: str, symbol: str, wallet: str,
                    cluster: str, price: float) -> Optional[Dict]:
        self._exits[token].append({
            'wallet': wallet, 'cluster': cluster, 'price': price,
            'timestamp': datetime.now()
        })
        
        cutoff = datetime.now() - timedelta(minutes=30)
        self._exits[token] = [e for e in self._exits[token] if e['timestamp'] > cutoff]
        
        exits = self._exits[token]
        sniper_count = sum(1 for e in exits if e['cluster'] == 'SNIPER')
        total = len(exits)
        
        if sniper_count >= config.exit_sniper_count:
            return {'action': 'FULL_EXIT', 'urgency': 'HIGH',
                   'reason': f"{sniper_count} SNIPERs exited", 'token': token}
        elif total >= config.exit_total_count:
            return {'action': 'FULL_EXIT', 'urgency': 'MEDIUM',
                   'reason': f"{total} wallets exited", 'token': token}
        elif sniper_count >= 1:
            return {'action': 'PARTIAL_EXIT', 'urgency': 'LOW',
                   'percentage': 50, 'reason': "SNIPER warning", 'token': token}
        return None


class ReasoningAgent:
    """LLM analysis with cost tracking"""
    
    def __init__(self, api_key: str = None, db=None):
        self.api_key = api_key or os.getenv('ANTHROPIC_API_KEY')
        self.db = db
        self.client = None
        self.enabled = False
        self.model = "claude-sonnet-4-20250514"
        
        if self.api_key and ANTHROPIC_AVAILABLE:
            try:
                self.client = anthropic.Anthropic(api_key=self.api_key)
                self.enabled = True
            except:
                pass
    
    def analyze_signal(self, signal: Dict, market: Dict) -> Dict:
        if not self.enabled:
            return {'adjustment': 0, 'summary': 'LLM disabled'}
        
        prompt = f"""Analyze this memecoin signal briefly:
TOKEN: ${signal.get('token_symbol')} | Liq: ${signal.get('liquidity', 0):,.0f} | Age: {signal.get('token_age_hours', 0):.1f}h
WALLETS: {signal.get('wallet_count', 1)} buying | Avg WR: {signal.get('avg_wallet_win_rate', 0.5):.0%}
MARKET: {market.get('regime', 'UNKNOWN')}

Respond with ONLY valid JSON:
{{"bull_score": 0-100, "bear_score": 0-100, "summary": "brief reason"}}"""

        try:
            resp = self.client.messages.create(
                model=self.model,
                max_tokens=150,
                messages=[{"role": "user", "content": prompt}]
            )
            
            text = resp.content[0].text.strip()
            
            if self.db:
                self.db.log_llm_call(
                    call_type='signal_analysis',
                    token_symbol=signal.get('token_symbol'),
                    input_tokens=resp.usage.input_tokens,
                    output_tokens=resp.usage.output_tokens,
                    result_summary=text[:200]
                )
            
            try:
                result = json.loads(text)
            except:
                import re
                match = re.search(r'\{[^}]+\}', text)
                if match:
                    result = json.loads(match.group())
                else:
                    return {'adjustment': 0, 'summary': 'Parse error'}
            
            bull = result.get('bull_score', 50)
            bear = result.get('bear_score', 50)
            adj = (bull - 50) - ((bear - 50) * 0.5)
            
            return {'adjustment': adj, 'bull': bull, 'bear': bear,
                   'summary': result.get('summary', '')}
        except Exception as e:
            return {'adjustment': 0, 'summary': f'Error: {e}'}


# =============================================================================
# LEARNING COMPONENTS - NEW!
# =============================================================================

class TradeOutcomeAnalyzer:
    """
    Analyzes trade outcomes to understand WHY trades succeed or fail.
    Used to improve entry filters and exit timing.
    """
    
    def __init__(self, db):
        self.db = db
        self._last_analysis = datetime.now() - timedelta(hours=24)
        self._insights = {}
    
    def analyze(self, days: int = 14) -> Dict:
        """Analyze recent trade outcomes"""
        closed = self.db.get_closed_positions(days=days)
        
        if len(closed) < 5:
            return {'status': 'insufficient_data', 'trades': len(closed)}
        
        insights = {
            'total_trades': len(closed),
            'by_exit_reason': self._analyze_by_exit_reason(closed),
            'by_conviction': self._analyze_by_conviction(closed),
            'by_liquidity': self._analyze_by_liquidity(closed),
            'by_token_age': self._analyze_by_token_age(closed),
            'by_hold_time': self._analyze_by_hold_time(closed),
            'optimal_ranges': self._find_optimal_ranges(closed),
            'problematic_patterns': self._find_problems(closed),
        }
        
        self._insights = insights
        self._last_analysis = datetime.now()
        
        return insights
    
    def _analyze_by_exit_reason(self, trades: List[Dict]) -> Dict:
        """Group performance by exit reason"""
        by_reason = defaultdict(lambda: {'count': 0, 'wins': 0, 'pnl': 0})
        
        for t in trades:
            reason = t.get('exit_reason', 'unknown')
            by_reason[reason]['count'] += 1
            if (t.get('profit_pct') or 0) > 0:
                by_reason[reason]['wins'] += 1
            by_reason[reason]['pnl'] += t.get('profit_sol', 0) or 0
        
        # Calculate win rates
        for reason in by_reason:
            total = by_reason[reason]['count']
            if total > 0:
                by_reason[reason]['win_rate'] = by_reason[reason]['wins'] / total
        
        return dict(by_reason)
    
    def _analyze_by_conviction(self, trades: List[Dict]) -> Dict:
        """Analyze performance by conviction score buckets"""
        buckets = {
            'low_50_60': {'range': (50, 60), 'trades': []},
            'med_60_75': {'range': (60, 75), 'trades': []},
            'high_75_90': {'range': (75, 90), 'trades': []},
            'very_high_90+': {'range': (90, 100), 'trades': []},
        }
        
        for t in trades:
            conv = t.get('conviction_score', 50) or 50
            for bucket, data in buckets.items():
                if data['range'][0] <= conv < data['range'][1]:
                    data['trades'].append(t)
                    break
        
        results = {}
        for bucket, data in buckets.items():
            trades_list = data['trades']
            if trades_list:
                wins = sum(1 for t in trades_list if (t.get('profit_pct') or 0) > 0)
                results[bucket] = {
                    'count': len(trades_list),
                    'win_rate': wins / len(trades_list),
                    'avg_pnl_pct': np.mean([t.get('profit_pct', 0) or 0 for t in trades_list]),
                    'total_pnl_sol': sum(t.get('profit_sol', 0) or 0 for t in trades_list),
                }
        
        return results
    
    def _analyze_by_liquidity(self, trades: List[Dict]) -> Dict:
        """Analyze performance by liquidity buckets"""
        buckets = {
            'low_30k_50k': (30000, 50000),
            'med_50k_100k': (50000, 100000),
            'high_100k+': (100000, float('inf')),
        }
        
        results = {}
        for bucket, (min_liq, max_liq) in buckets.items():
            matching = [t for t in trades 
                       if min_liq <= (t.get('entry_liquidity', 0) or 0) < max_liq]
            if matching:
                wins = sum(1 for t in matching if (t.get('profit_pct') or 0) > 0)
                results[bucket] = {
                    'count': len(matching),
                    'win_rate': wins / len(matching),
                    'avg_pnl_pct': np.mean([t.get('profit_pct', 0) or 0 for t in matching]),
                }
        
        return results
    
    def _analyze_by_token_age(self, trades: List[Dict]) -> Dict:
        """Analyze performance by token age"""
        buckets = {
            'very_new_0_2h': (0, 2),
            'new_2_12h': (2, 12),
            'established_12_48h': (12, 48),
            'mature_48h+': (48, float('inf')),
        }
        
        results = {}
        for bucket, (min_age, max_age) in buckets.items():
            matching = [t for t in trades 
                       if min_age <= (t.get('token_age_hours', 0) or 0) < max_age]
            if matching:
                wins = sum(1 for t in matching if (t.get('profit_pct') or 0) > 0)
                results[bucket] = {
                    'count': len(matching),
                    'win_rate': wins / len(matching),
                    'avg_pnl_pct': np.mean([t.get('profit_pct', 0) or 0 for t in matching]),
                }
        
        return results
    
    def _analyze_by_hold_time(self, trades: List[Dict]) -> Dict:
        """Analyze if we're holding too long or exiting too early"""
        results = {
            'stopped_out_early': [],  # Hit stop loss within 30 min
            'timed_out': [],          # Hit max hold time
            'good_exits': [],         # Take profit or trailing stop
        }
        
        for t in trades:
            reason = t.get('exit_reason', '')
            hold_mins = t.get('hold_duration_minutes', 0) or 0
            
            if reason == 'STOP_LOSS' and hold_mins < 30:
                results['stopped_out_early'].append(t)
            elif reason == 'TIME_STOP':
                results['timed_out'].append(t)
            elif reason in ['TAKE_PROFIT', 'TRAILING_STOP']:
                results['good_exits'].append(t)
        
        summary = {}
        for category, trades_list in results.items():
            if trades_list:
                summary[category] = {
                    'count': len(trades_list),
                    'avg_pnl_pct': np.mean([t.get('profit_pct', 0) or 0 for t in trades_list]),
                }
        
        return summary
    
    def _find_optimal_ranges(self, trades: List[Dict]) -> Dict:
        """Find optimal parameter ranges based on winning trades"""
        winners = [t for t in trades if (t.get('profit_pct') or 0) > 0]
        
        if len(winners) < 5:
            return {}
        
        return {
            'liquidity': {
                'min': np.percentile([t.get('entry_liquidity', 0) or 0 for t in winners], 10),
                'median': np.median([t.get('entry_liquidity', 0) or 0 for t in winners]),
                'max': np.percentile([t.get('entry_liquidity', 0) or 0 for t in winners], 90),
            },
            'token_age_hours': {
                'min': np.percentile([t.get('token_age_hours', 0) or 0 for t in winners], 10),
                'median': np.median([t.get('token_age_hours', 0) or 0 for t in winners]),
                'max': np.percentile([t.get('token_age_hours', 0) or 0 for t in winners], 90),
            },
            'conviction': {
                'min': np.percentile([t.get('conviction_score', 50) or 50 for t in winners], 10),
                'median': np.median([t.get('conviction_score', 50) or 50 for t in winners]),
            },
        }
    
    def _find_problems(self, trades: List[Dict]) -> List[str]:
        """Identify problematic patterns"""
        problems = []
        
        losers = [t for t in trades if (t.get('profit_pct') or 0) <= 0]
        
        if not losers:
            return problems
        
        # Check for early stop-outs
        early_stops = [t for t in losers 
                      if t.get('exit_reason') == 'STOP_LOSS' 
                      and (t.get('hold_duration_minutes', 0) or 0) < 30]
        if len(early_stops) > len(losers) * 0.3:
            problems.append(f"HIGH_EARLY_STOPOUT: {len(early_stops)}/{len(losers)} losers stopped out within 30min - consider wider stops or better entry timing")
        
        # Check for time-outs with profit potential
        timeouts = [t for t in losers if t.get('exit_reason') == 'TIME_STOP']
        profitable_timeouts = [t for t in timeouts if (t.get('peak_unrealized_pct') or 0) > 10]
        if len(profitable_timeouts) > 3:
            problems.append(f"MISSED_EXITS: {len(profitable_timeouts)} trades had 10%+ unrealized profit but hit time stop - consider trailing stops")
        
        # Check conviction correlation
        low_conv_losers = [t for t in losers if (t.get('conviction_score', 50) or 50) < 60]
        if len(low_conv_losers) > len(losers) * 0.5:
            problems.append(f"LOW_CONVICTION_LOSSES: {len(low_conv_losers)}/{len(losers)} losses were low conviction - raise entry threshold")
        
        return problems
    
    def get_insights(self) -> Dict:
        """Get cached insights"""
        return self._insights


class ParameterOptimizer:
    """
    Optimizes trading parameters based on actual results.
    Adjusts stop loss, take profit, and entry thresholds.
    """
    
    def __init__(self, db):
        self.db = db
        self._last_optimization = datetime.now() - timedelta(hours=24)
        self._optimized_params = {}
    
    def optimize(self, days: int = 14) -> Dict:
        """Run parameter optimization based on trade history"""
        closed = self.db.get_closed_positions(days=days)
        
        if len(closed) < config.min_trades_for_learning:
            return {
                'status': 'insufficient_data',
                'trades': len(closed),
                'required': config.min_trades_for_learning
            }
        
        recommendations = {
            'stop_loss': self._optimize_stop_loss(closed),
            'take_profit': self._optimize_take_profit(closed),
            'trailing_stop': self._optimize_trailing_stop(closed),
            'conviction_threshold': self._optimize_conviction(closed),
            'max_hold_hours': self._optimize_hold_time(closed),
        }
        
        self._optimized_params = recommendations
        self._last_optimization = datetime.now()
        
        return recommendations
    
    def _optimize_stop_loss(self, trades: List[Dict]) -> Dict:
        """Optimize stop loss based on drawdown analysis"""
        # Look at trades that hit stop loss
        stop_losses = [t for t in trades if t.get('exit_reason') == 'STOP_LOSS']
        
        if len(stop_losses) < 3:
            return {'current': config.default_stop_loss, 'recommended': config.default_stop_loss, 'reason': 'insufficient_data'}
        
        # Check if stops are too tight (many quick stop-outs)
        quick_stops = [t for t in stop_losses if (t.get('hold_duration_minutes', 0) or 0) < 15]
        
        # Check how many would have been profitable with wider stops
        # (by looking at peak unrealized profit)
        would_win = [t for t in stop_losses if (t.get('peak_unrealized_pct') or 0) > 5]
        
        current_stop = config.default_stop_loss
        
        if len(quick_stops) > len(stop_losses) * 0.4:
            # Too many quick stop-outs, widen stop
            recommended = current_stop * 1.25  # e.g., -12% -> -15%
            reason = f"40%+ quick stop-outs ({len(quick_stops)}/{len(stop_losses)})"
        elif len(would_win) > len(stop_losses) * 0.3:
            # Many would have won with patience
            recommended = current_stop * 1.15
            reason = f"30%+ stopped trades had profit potential ({len(would_win)}/{len(stop_losses)})"
        else:
            # Stops seem appropriate
            recommended = current_stop
            reason = "current stop loss performing well"
        
        return {
            'current': current_stop,
            'recommended': max(-0.25, min(-0.08, recommended)),  # Clamp between -8% and -25%
            'reason': reason,
            'quick_stops': len(quick_stops),
            'would_have_won': len(would_win),
        }
    
    def _optimize_take_profit(self, trades: List[Dict]) -> Dict:
        """Optimize take profit based on peak analysis"""
        winners = [t for t in trades if (t.get('profit_pct') or 0) > 0]
        
        if len(winners) < 3:
            return {'current': config.default_take_profit, 'recommended': config.default_take_profit, 'reason': 'insufficient_data'}
        
        # Look at actual profits achieved
        actual_profits = [t.get('profit_pct', 0) or 0 for t in winners]
        peak_profits = [t.get('peak_unrealized_pct', 0) or 0 for t in winners]
        
        avg_actual = np.mean(actual_profits)
        avg_peak = np.mean(peak_profits)
        
        current_tp = config.default_take_profit * 100  # Convert to percentage
        
        # If we're consistently hitting peaks higher than TP, raise it
        if avg_peak > current_tp * 1.3:
            recommended = min(50, avg_peak * 0.8)  # Target 80% of avg peak
            reason = f"avg peak ({avg_peak:.0f}%) > TP ({current_tp:.0f}%)"
        # If we rarely reach TP, lower it
        elif avg_actual < current_tp * 0.5:
            recommended = max(15, avg_actual * 1.2)
            reason = f"avg profit ({avg_actual:.0f}%) << TP ({current_tp:.0f}%)"
        else:
            recommended = current_tp
            reason = "take profit level appropriate"
        
        return {
            'current': config.default_take_profit,
            'recommended': recommended / 100,  # Convert back to decimal
            'reason': reason,
            'avg_profit_pct': avg_actual,
            'avg_peak_pct': avg_peak,
        }
    
    def _optimize_trailing_stop(self, trades: List[Dict]) -> Dict:
        """Optimize trailing stop based on profit giveback"""
        # Look at trades that had significant unrealized profit
        had_profit = [t for t in trades if (t.get('peak_unrealized_pct') or 0) > 15]
        
        if len(had_profit) < 3:
            return {'current': config.default_trailing_stop, 'recommended': config.default_trailing_stop, 'reason': 'insufficient_data'}
        
        # Calculate how much profit was given back
        givebacks = []
        for t in had_profit:
            peak = t.get('peak_unrealized_pct', 0) or 0
            actual = t.get('profit_pct', 0) or 0
            if peak > 0:
                giveback = (peak - actual) / peak
                givebacks.append(giveback)
        
        avg_giveback = np.mean(givebacks) if givebacks else 0
        
        current_trail = config.default_trailing_stop * 100
        
        if avg_giveback > 0.5:
            # Giving back more than 50% of peak - tighten trail
            recommended = max(5, current_trail * 0.8)
            reason = f"avg giveback {avg_giveback:.0%} - tighten trail"
        elif avg_giveback < 0.2:
            # Very tight exits, might be leaving money on table
            recommended = min(15, current_trail * 1.2)
            reason = f"avg giveback only {avg_giveback:.0%} - could loosen"
        else:
            recommended = current_trail
            reason = "trailing stop appropriate"
        
        return {
            'current': config.default_trailing_stop,
            'recommended': recommended / 100,
            'reason': reason,
            'avg_giveback': avg_giveback,
        }
    
    def _optimize_conviction(self, trades: List[Dict]) -> Dict:
        """Optimize conviction threshold based on win rates by conviction level"""
        # Group by conviction buckets
        buckets = defaultdict(lambda: {'wins': 0, 'total': 0})
        
        for t in trades:
            conv = t.get('conviction_score', 50) or 50
            bucket = int(conv // 10) * 10  # 50, 60, 70, etc.
            buckets[bucket]['total'] += 1
            if (t.get('profit_pct') or 0) > 0:
                buckets[bucket]['wins'] += 1
        
        # Find the conviction level where win rate > 50%
        current_threshold = config.single_wallet_min_conviction
        recommended = current_threshold
        
        for bucket in sorted(buckets.keys()):
            if buckets[bucket]['total'] >= 3:
                wr = buckets[bucket]['wins'] / buckets[bucket]['total']
                if wr >= 0.5:
                    recommended = bucket
                    break
        
        # Build reason
        bucket_summary = {b: f"{d['wins']}/{d['total']}" for b, d in sorted(buckets.items()) if d['total'] >= 2}
        
        return {
            'current': current_threshold,
            'recommended': recommended,
            'reason': f"First 50%+ WR at {recommended}",
            'by_bucket': bucket_summary,
        }
    
    def _optimize_hold_time(self, trades: List[Dict]) -> Dict:
        """Optimize max hold time based on performance over time"""
        # Group by hold duration
        by_duration = defaultdict(lambda: {'wins': 0, 'total': 0, 'pnl': 0})
        
        for t in trades:
            hours = (t.get('hold_duration_minutes', 0) or 0) / 60
            if hours < 2:
                bucket = '0-2h'
            elif hours < 6:
                bucket = '2-6h'
            elif hours < 12:
                bucket = '6-12h'
            else:
                bucket = '12h+'
            
            by_duration[bucket]['total'] += 1
            if (t.get('profit_pct') or 0) > 0:
                by_duration[bucket]['wins'] += 1
            by_duration[bucket]['pnl'] += t.get('profit_sol', 0) or 0
        
        # Check if holding longer hurts
        current_max = config.max_hold_hours
        
        if by_duration['12h+']['total'] >= 3:
            long_wr = by_duration['12h+']['wins'] / by_duration['12h+']['total']
            short_wr = sum(by_duration[b]['wins'] for b in ['0-2h', '2-6h']) / max(1, sum(by_duration[b]['total'] for b in ['0-2h', '2-6h']))
            
            if long_wr < 0.3 and short_wr > 0.5:
                recommended = 8
                reason = f"Long holds underperform ({long_wr:.0%} vs {short_wr:.0%} short)"
            else:
                recommended = current_max
                reason = "hold time appropriate"
        else:
            recommended = current_max
            reason = "insufficient long hold data"
        
        return {
            'current': current_max,
            'recommended': recommended,
            'reason': reason,
            'by_duration': {k: f"{v['wins']}/{v['total']}" for k, v in by_duration.items()},
        }
    
    def get_optimized_params(self) -> Dict:
        """Get current optimized parameters"""
        return self._optimized_params


class StrategyLab:
    """
    Strategy evolution with AUTOMATIC PROMOTION.
    
    Now actually promotes challengers that outperform!
    """
    
    def __init__(self, db):
        self.db = db
        self.strategies = self._load()
        self._last_evaluation = datetime.now() - timedelta(hours=24)
    
    def _load(self) -> Dict:
        strategies = self.db.get_all_strategies()
        if not strategies:
            strategies = {
                'champion': {
                    'name': 'Conservative Consensus', 'status': 'CHAMPION',
                    'config': {
                        'min_wallets': 2, 'min_wallet_quality': 0.55,
                        'min_liquidity': 30000, 'token_age_min': 1, 'token_age_max': 72,
                        'stop_loss': -0.12, 'take_profit': 0.25, 'trailing_stop': 0.08,
                        'max_hold_hours': 12
                    },
                    'performance': {},
                    'generation': 1
                },
                'challenger_1': {
                    'name': 'Aggressive Sniper', 'status': 'CHALLENGER',
                    'config': {
                        'min_wallets': 1, 'min_wallet_quality': 0.60,
                        'min_liquidity': 25000, 'token_age_min': 0.5, 'token_age_max': 24,
                        'stop_loss': -0.15, 'take_profit': 0.35, 'trailing_stop': 0.10,
                        'max_hold_hours': 6
                    },
                    'performance': {},
                    'generation': 1
                },
                'challenger_2': {
                    'name': 'Patient Rider', 'status': 'CHALLENGER',
                    'config': {
                        'min_wallets': 3, 'min_wallet_quality': 0.50,
                        'min_liquidity': 50000, 'token_age_min': 2, 'token_age_max': 96,
                        'stop_loss': -0.10, 'take_profit': 0.20, 'trailing_stop': 0.06,
                        'max_hold_hours': 18
                    },
                    'performance': {},
                    'generation': 1
                }
            }
            for sid, s in strategies.items():
                self.db.save_strategy(sid, s)
        return strategies
    
    def get_champion_config(self) -> Dict:
        return self.strategies.get('champion', {}).get('config', {})
    
    def evaluate(self, days: int = 7) -> Dict:
        """Evaluate all strategies and potentially promote a challenger"""
        closed = self.db.get_closed_positions(days=days)
        results = {}
        
        for sid, strat in self.strategies.items():
            cfg = strat.get('config', {})
            matching = [p for p in closed if self._matches(p, cfg)]
            
            if not matching:
                results[sid] = {'trades': 0, 'win_rate': 0, 'pnl': 0}
                continue
            
            wins = sum(1 for p in matching if (p.get('profit_pct') or 0) > 0)
            pnl = sum(p.get('profit_sol', 0) or 0 for p in matching)
            
            results[sid] = {
                'trades': len(matching), 
                'wins': wins,
                'win_rate': wins / len(matching) if matching else 0,
                'pnl': pnl,
                'avg_pnl_pct': np.mean([p.get('profit_pct', 0) or 0 for p in matching]),
            }
            
            strat['performance'] = results[sid]
            self.db.save_strategy(sid, strat)
        
        # Check for promotion
        promotion = self._check_promotion(results)
        if promotion:
            results['promotion'] = promotion
        
        self._last_evaluation = datetime.now()
        
        return results
    
    def _matches(self, pos: Dict, cfg: Dict) -> bool:
        liq = pos.get('entry_liquidity', 0) or 0
        age = pos.get('token_age_hours', 0) or 0
        return liq >= cfg.get('min_liquidity', 0) and \
               cfg.get('token_age_min', 0) <= age <= cfg.get('token_age_max', 999)
    
    def _check_promotion(self, results: Dict) -> Optional[Dict]:
        """Check if any challenger should be promoted"""
        champion_perf = results.get('champion', {})
        champion_pnl = champion_perf.get('pnl', 0)
        champion_wr = champion_perf.get('win_rate', 0)
        champion_trades = champion_perf.get('trades', 0)
        
        best_challenger = None
        best_score = 0
        
        for sid, perf in results.items():
            if sid == 'champion' or not sid.startswith('challenger'):
                continue
            
            # Need minimum trades
            if perf.get('trades', 0) < config.promotion_min_trades:
                continue
            
            # Calculate composite score (PnL + win rate bonus)
            score = perf.get('pnl', 0) + (perf.get('win_rate', 0) - 0.5) * 2
            
            if score > best_score:
                best_score = score
                best_challenger = sid
        
        if not best_challenger:
            return None
        
        challenger_perf = results[best_challenger]
        
        # Check if challenger significantly outperforms
        # Either: higher PnL by margin, OR similar PnL with better win rate
        pnl_improvement = (challenger_perf['pnl'] - champion_pnl) / max(0.1, abs(champion_pnl)) if champion_pnl != 0 else 1
        wr_improvement = challenger_perf['win_rate'] - champion_wr
        
        should_promote = False
        reason = ""
        
        if pnl_improvement > config.promotion_outperform_pct:
            should_promote = True
            reason = f"PnL +{pnl_improvement:.0%} vs champion"
        elif pnl_improvement > 0 and wr_improvement > 0.1:
            should_promote = True
            reason = f"Better PnL and +{wr_improvement:.0%} win rate"
        
        if should_promote and champion_trades >= config.promotion_min_trades:
            self._promote_strategy(best_challenger)
            return {
                'promoted': best_challenger,
                'reason': reason,
                'new_champion': self.strategies[best_challenger]['name'],
                'champion_pnl': champion_pnl,
                'challenger_pnl': challenger_perf['pnl'],
            }
        
        return None
    
    def _promote_strategy(self, challenger_id: str):
        """Promote a challenger to champion"""
        old_champion = self.strategies['champion'].copy()
        new_champion = self.strategies[challenger_id].copy()
        
        # New champion becomes champion
        new_champion['status'] = 'CHAMPION'
        new_champion['promoted_at'] = datetime.now().isoformat()
        new_champion['generation'] = old_champion.get('generation', 1) + 1
        
        # Old champion becomes a challenger
        old_champion['status'] = 'CHALLENGER'
        old_champion['evolved_from'] = 'demoted_champion'
        
        # Swap
        self.strategies['champion'] = new_champion
        self.strategies[challenger_id] = old_champion
        
        # Save
        self.db.save_strategy('champion', new_champion)
        self.db.save_strategy(challenger_id, old_champion)
        
        print(f"\n🏆 STRATEGY PROMOTION!")
        print(f"   New champion: {new_champion['name']} (gen {new_champion['generation']})")
        print(f"   Old champion demoted to: {challenger_id}")
    
    def evolve_strategies(self, optimizer: ParameterOptimizer, analyzer: TradeOutcomeAnalyzer):
        """Evolve strategies based on learned insights"""
        optimized = optimizer.get_optimized_params()
        insights = analyzer.get_insights()
        
        if not optimized or not insights:
            return
        
        # Update champion with optimized parameters (conservative)
        champion = self.strategies.get('champion', {})
        cfg = champion.get('config', {})
        
        # Only apply optimizations if they're significant improvements
        if 'stop_loss' in optimized:
            rec = optimized['stop_loss'].get('recommended')
            if rec and abs(rec - cfg.get('stop_loss', -0.12)) > 0.02:
                cfg['stop_loss'] = rec
                print(f"  📊 Adjusted champion stop_loss to {rec:.0%}")
        
        if 'take_profit' in optimized:
            rec = optimized['take_profit'].get('recommended')
            if rec and abs(rec - cfg.get('take_profit', 0.25)) > 0.03:
                cfg['take_profit'] = rec
                print(f"  📊 Adjusted champion take_profit to {rec:.0%}")
        
        champion['config'] = cfg
        self.db.save_strategy('champion', champion)
        self.strategies['champion'] = champion


class Strategist:
    """
    THE STRATEGIST V2 - Main orchestrator with LEARNING LOOP.
    
    LLM is ONLY called if:
    1. Pre-filters pass
    2. Base score >= threshold
    3. LLM is enabled
    
    LEARNING features:
    1. Trade outcome analysis
    2. Parameter optimization
    3. Automatic strategy promotion
    4. Adaptive thresholds
    """
    
    def __init__(self, db, api_key: str = None):
        self.db = db
        self.api_key = api_key or os.getenv('ANTHROPIC_API_KEY')
        
        print("\n🧠 THE STRATEGIST V2 (with Learning)")
        print("="*50)
        
        # Core components
        self.regime_detector = RegimeDetector(db)
        self.wallet_clusterer = WalletClusterer(db)
        self.signal_aggregator = SignalAggregator()
        self.exit_tracker = ExitSignalTracker(db)
        self.strategy_lab = StrategyLab(db)
        self.reasoning_agent = ReasoningAgent(self.api_key, db)
        
        # Learning components - NEW!
        self.outcome_analyzer = TradeOutcomeAnalyzer(db)
        self.param_optimizer = ParameterOptimizer(db)
        self._last_learning_run = datetime.now() - timedelta(hours=24)
        
        print(f"  ✅ Core components loaded")
        print(f"  ✅ Learning components loaded")
        print(f"  {'✅' if self.reasoning_agent.enabled else '⚠️'} LLM: {'enabled' if self.reasoning_agent.enabled else 'disabled'}")
    
    def analyze_signal(self, signal_data: Dict, wallet_data: Dict,
                      use_llm: bool = True) -> Dict:
        """
        Analyze signal with smart LLM gating.
        """
        # STEP 1: Pre-filters (fast, free)
        filter_result = self._check_filters(signal_data)
        if not filter_result['pass']:
            return {'action': 'SKIP', 'should_enter': False,
                   'conviction_score': 0, 'reason': filter_result['reason'],
                   'llm_called': False}
        
        # STEP 2: Base score (fast, free)
        base = self._calc_base_score(signal_data, wallet_data)
        
        # STEP 3: Signal aggregation (fast, free)
        wallet = wallet_data.get('address', '')
        cluster = self.wallet_clusterer.get_wallet_cluster(wallet)
        
        aggregated = self.signal_aggregator.add_signal(
            token_address=signal_data.get('token_address', ''),
            token_symbol=signal_data.get('token_symbol', ''),
            wallet=wallet, cluster=cluster,
            win_rate=wallet_data.get('win_rate', 0.5),
            roi=wallet_data.get('roi_7d', 0),
            entry_price=signal_data.get('price', 0),
            liquidity=signal_data.get('liquidity', 0),
            volume_24h=signal_data.get('volume_24h', 0),
            token_age_hours=signal_data.get('token_age_hours', 0),
            base_conviction=base['score']
        )
        
        if aggregated:
            return self._finalize(aggregated, signal_data, False)
        
        # STEP 4: LLM GATING - Only call if worth it
        llm_adj = 0
        llm_called = False
        
        if (config.llm_enabled and use_llm and
            base['score'] >= config.llm_conviction_threshold and
            self.reasoning_agent.enabled):
            
            market = {'regime': self.regime_detector.current_regime,
                     'sol_price': self.regime_detector.sol_price}
            
            llm_result = self.reasoning_agent.analyze_signal(signal_data, market)
            llm_adj = llm_result.get('adjustment', 0)
            llm_called = True
        
        # STEP 5: Final decision
        final = base['score'] + llm_adj
        
        decision = {
            'action': 'BUY' if final >= config.single_wallet_min_conviction else 'SKIP',
            'should_enter': final >= config.single_wallet_min_conviction,
            'conviction_score': final,
            'base_score': base['score'],
            'llm_adjustment': llm_adj,
            'llm_called': llm_called,
            'reason': f"Base={base['score']:.0f}, LLM={llm_adj:+.0f}"
        }
        
        if decision['should_enter']:
            return self._finalize(decision, signal_data, llm_called)
        
        return decision
    
    def _check_filters(self, signal: Dict) -> Dict:
        liq = signal.get('liquidity', 0)
        vol = signal.get('volume_24h', 0)
        age = signal.get('token_age_hours', 0)
        
        if liq < config.min_liquidity:
            return {'pass': False, 'reason': f'Low liq: ${liq:,.0f}'}
        if vol < config.min_volume_24h:
            return {'pass': False, 'reason': f'Low vol: ${vol:,.0f}'}
        if age < config.min_token_age_hours:
            return {'pass': False, 'reason': f'Too new: {age:.1f}h'}
        if age > config.max_token_age_hours:
            return {'pass': False, 'reason': f'Too old: {age:.0f}h'}
        
        ok, reason = self.regime_detector.should_trade()
        if not ok:
            return {'pass': False, 'reason': reason}
        
        return {'pass': True, 'reason': 'OK'}
    
    def _calc_base_score(self, signal: Dict, wallet: Dict) -> Dict:
        score = 50
        wr = wallet.get('win_rate', 0.5)
        
        if wr >= 0.65:
            score += 20
        elif wr >= 0.55:
            score += 10
        elif wr < 0.45:
            score -= 10
        
        cluster = self.wallet_clusterer.get_wallet_cluster(wallet.get('address', ''))
        weight = self.wallet_clusterer.get_adaptive_weight(cluster)
        if weight >= 1.2:
            score += 10
        elif weight <= 0.7:
            score -= 10
        
        if signal.get('liquidity', 0) >= 100000:
            score += 10
        elif signal.get('liquidity', 0) >= 50000:
            score += 5
        
        if self.regime_detector.current_regime == 'EXPANSION':
            score += 10
        elif self.regime_detector.current_regime == 'CONTRACTION':
            score -= 10
        
        return {'score': min(100, max(0, score))}
    
    def _finalize(self, decision: Dict, signal: Dict, llm_called: bool) -> Dict:
        cfg = self.strategy_lab.get_champion_config()
        
        base_stop = cfg.get('stop_loss', -0.12)
        base_tp = cfg.get('take_profit', 0.25)
        base_trail = cfg.get('trailing_stop', 0.08)
        
        adjusted = self.regime_detector.get_adjusted_exits(base_stop, base_tp, base_trail)
        
        regime_mult = self.regime_detector.get_regime_multipliers()['position_mult']
        conv_mult = decision.get('conviction_score', 50) / 100
        size = max(0.1, min(1.0, 0.5 * regime_mult * conv_mult))
        
        decision.update({
            'position_size_sol': size,
            'stop_loss': adjusted['stop_loss'],
            'take_profit': adjusted['take_profit'],
            'trailing_stop': adjusted['trailing_stop'],
            'max_hold_hours': cfg.get('max_hold_hours', 12),
            'strategy': cfg.get('name', 'unknown'),
            'regime': adjusted['regime'],
            'llm_called': llm_called
        })
        
        return decision
    
    def process_exit_signal(self, token: str, symbol: str,
                           wallet: str, price: float) -> Optional[Dict]:
        cluster = self.wallet_clusterer.get_wallet_cluster(wallet)
        return self.exit_tracker.process_exit(token, symbol, wallet, cluster, price)
    
    def update_regime(self, market: Dict = None):
        self.regime_detector.update(market)
    
    def run_learning_loop(self, force: bool = False) -> Dict:
        """
        Run the learning loop to improve strategy based on trade outcomes.
        Called periodically (every 6 hours by default).
        """
        if not config.learning_enabled:
            return {'status': 'disabled'}
        
        hours_since_last = (datetime.now() - self._last_learning_run).total_seconds() / 3600
        
        if not force and hours_since_last < config.learning_interval_hours:
            return {'status': 'skipped', 'hours_until_next': config.learning_interval_hours - hours_since_last}
        
        print("\n" + "="*60)
        print("🧠 RUNNING LEARNING LOOP")
        print("="*60)
        
        results = {'status': 'completed', 'timestamp': datetime.now().isoformat()}
        
        # 1. Analyze trade outcomes
        print("\n📊 Analyzing trade outcomes...")
        analysis = self.outcome_analyzer.analyze(days=14)
        results['outcome_analysis'] = {
            'total_trades': analysis.get('total_trades', 0),
            'problems_found': len(analysis.get('problematic_patterns', [])),
        }
        
        if analysis.get('problematic_patterns'):
            print("  ⚠️ Problems identified:")
            for problem in analysis['problematic_patterns']:
                print(f"     - {problem}")
        
        # 2. Optimize parameters
        print("\n🔧 Optimizing parameters...")
        optimization = self.param_optimizer.optimize(days=14)
        results['parameter_optimization'] = optimization.get('status', 'completed')
        
        if optimization.get('status') != 'insufficient_data':
            for param, data in optimization.items():
                if isinstance(data, dict) and 'recommended' in data:
                    current = data.get('current')
                    recommended = data.get('recommended')
                    if current != recommended:
                        print(f"  💡 {param}: {current} -> {recommended} ({data.get('reason', '')})")
        
        # 3. Evaluate and potentially promote strategies
        print("\n🏆 Evaluating strategies...")
        evaluation = self.strategy_lab.evaluate(days=7)
        results['strategy_evaluation'] = {
            sid: {'trades': p.get('trades', 0), 'win_rate': p.get('win_rate', 0), 'pnl': p.get('pnl', 0)}
            for sid, p in evaluation.items() if isinstance(p, dict) and 'trades' in p
        }
        
        for sid, perf in evaluation.items():
            if isinstance(perf, dict) and 'trades' in perf:
                print(f"  {sid}: {perf.get('trades', 0)} trades, {perf.get('win_rate', 0):.0%} WR, {perf.get('pnl', 0):.2f} SOL")
        
        if 'promotion' in evaluation:
            results['promotion'] = evaluation['promotion']
            print(f"\n  🎉 PROMOTION: {evaluation['promotion']}")
        
        # 4. Evolve strategies with learned insights
        print("\n🧬 Evolving strategies...")
        self.strategy_lab.evolve_strategies(self.param_optimizer, self.outcome_analyzer)
        
        self._last_learning_run = datetime.now()
        
        print("\n" + "="*60)
        print("✅ Learning loop complete")
        print("="*60 + "\n")
        
        return results
    
    def get_status(self) -> Dict:
        return {
            'regime': self.regime_detector.current_regime,
            'confidence': self.regime_detector.confidence,
            'llm_enabled': self.reasoning_agent.enabled,
            'champion': self.strategy_lab.strategies.get('champion', {}).get('name'),
            'champion_generation': self.strategy_lab.strategies.get('champion', {}).get('generation', 1),
            'pending_signals': len(self.signal_aggregator.signals),
            'learning_enabled': config.learning_enabled,
            'hours_since_learning': (datetime.now() - self._last_learning_run).total_seconds() / 3600,
        }
    
    def get_llm_cost_today(self) -> Dict:
        return self.db.get_llm_cost_summary(days=1)
    
    def get_learning_insights(self) -> Dict:
        """Get current learning insights for diagnostics"""
        return {
            'outcome_analysis': self.outcome_analyzer.get_insights(),
            'optimized_params': self.param_optimizer.get_optimized_params(),
            'cluster_stats': self.wallet_clusterer.get_cluster_stats(),
            'strategies': {
                sid: {
                    'name': s.get('name'),
                    'status': s.get('status'),
                    'performance': s.get('performance', {}),
                }
                for sid, s in self.strategy_lab.strategies.items()
            }
        }


if __name__ == "__main__":
    print("\n" + "="*60)
    print("STRATEGIST V2 - Test with Learning")
    print("="*60)
    
    from core.database_v2 import DatabaseV2
    
    db = DatabaseV2("test_strat.db")
    strategist = Strategist(db)
    
    # Test signal analysis
    signal = {'token_symbol': 'TEST', 'token_address': 'test123',
              'price': 0.001, 'liquidity': 50000, 'volume_24h': 25000,
              'token_age_hours': 12}
    wallet = {'address': 'w123', 'win_rate': 0.65}
    
    result = strategist.analyze_signal(signal, wallet, use_llm=False)
    print(f"\nSignal Result: {result['action']} ({result['conviction_score']:.0f})")
    print(f"LLM called: {result['llm_called']}")
    
    # Test learning loop
    print("\n" + "-"*40)
    learning_result = strategist.run_learning_loop(force=True)
    print(f"\nLearning Status: {learning_result['status']}")
    
    # Cleanup
    import os
    os.remove("test_strat.db")
