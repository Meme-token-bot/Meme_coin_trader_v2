"""
THE STRATEGIST V2 - Smart Signal Analysis with LLM Gating
Trading System V2

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
    """Configuration"""
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
    """Wallet clustering with adaptive weights"""
    
    SNIPER = "SNIPER"
    RIDER = "RIDER"
    BALANCED = "BALANCED"
    GAMBLER = "GAMBLER"
    
    BASE_WEIGHTS = {SNIPER: 1.3, RIDER: 1.2, BALANCED: 1.0, GAMBLER: 0.6}
    
    def __init__(self, db):
        self.db = db
        self._weights = self.BASE_WEIGHTS.copy()
        self._last_update = datetime.now() - timedelta(hours=2)
    
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
        
        stats = defaultdict(lambda: {'wins': 0, 'total': 0})
        for pos in closed:
            w = self.db.get_wallet(pos['wallet_address'])
            if not w:
                continue
            c = w.get('cluster', self.BALANCED)
            stats[c]['total'] += 1
            if (pos.get('profit_pct') or 0) > 0:
                stats[c]['wins'] += 1
        
        for c in self.BASE_WEIGHTS:
            if stats[c]['total'] >= 10:
                wr = stats[c]['wins'] / stats[c]['total']
                bonus = (wr - 0.5) * 0.5
                self._weights[c] = self.BASE_WEIGHTS[c] * (1 + bonus)


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

Reply JSON only: {{"bull_score": 0-100, "bear_score": 0-100, "summary": "one sentence"}}"""

        try:
            response = self.client.messages.create(
                model=self.model, max_tokens=150,
                messages=[{"role": "user", "content": prompt}]
            )
            
            if self.db:
                self.db.log_llm_call('signal', signal.get('token_symbol'),
                                    response.usage.input_tokens, response.usage.output_tokens)
            
            text = response.content[0].text.strip()
            if '```' in text:
                text = text.split('```')[1].replace('json', '').strip()
            
            result = json.loads(text)
            bull = result.get('bull_score', 50)
            bear = result.get('bear_score', 50)
            adj = (bull - 50) - ((bear - 50) * 0.5)
            
            return {'adjustment': adj, 'bull': bull, 'bear': bear,
                   'summary': result.get('summary', '')}
        except Exception as e:
            return {'adjustment': 0, 'summary': f'Error: {e}'}


class StrategyLab:
    """Strategy evolution"""
    
    def __init__(self, db):
        self.db = db
        self.strategies = self._load()
    
    def _load(self) -> Dict:
        strategies = self.db.get_all_strategies()
        if not strategies:
            strategies = {
                'champion': {
                    'name': 'Conservative Consensus', 'status': 'CHAMPION',
                    'config': {
                        'min_wallets': 3, 'min_wallet_quality': 0.60,
                        'min_liquidity': 40000, 'token_age_min': 2, 'token_age_max': 72,
                        'stop_loss': -0.10, 'take_profit': 0.25, 'trailing_stop': 0.08,
                        'max_hold_hours': 12
                    },
                    'performance': {}
                },
                'challenger_1': {
                    'name': 'Aggressive Sniper', 'status': 'CHALLENGER',
                    'config': {
                        'min_wallets': 2, 'min_wallet_quality': 0.55,
                        'min_liquidity': 25000, 'token_age_min': 0.5, 'token_age_max': 24,
                        'stop_loss': -0.15, 'take_profit': 0.35, 'trailing_stop': 0.10,
                        'max_hold_hours': 6
                    },
                    'performance': {}
                }
            }
            for sid, s in strategies.items():
                self.db.save_strategy(sid, s)
        return strategies
    
    def get_champion_config(self) -> Dict:
        return self.strategies.get('champion', {}).get('config', {})
    
    def evaluate(self, days: int = 7) -> Dict:
        closed = self.db.get_closed_positions(days=days)
        results = {}
        
        for sid, strat in self.strategies.items():
            cfg = strat.get('config', {})
            matching = [p for p in closed if self._matches(p, cfg)]
            
            if not matching:
                results[sid] = {'trades': 0}
                continue
            
            wins = sum(1 for p in matching if (p.get('profit_pct') or 0) > 0)
            results[sid] = {
                'trades': len(matching), 'wins': wins,
                'win_rate': wins / len(matching),
                'pnl': sum(p.get('profit_sol', 0) or 0 for p in matching)
            }
            
            strat['performance'] = results[sid]
            self.db.save_strategy(sid, strat)
        
        return results
    
    def _matches(self, pos: Dict, cfg: Dict) -> bool:
        liq = pos.get('entry_liquidity', 0)
        age = pos.get('token_age_hours', 0)
        return liq >= cfg.get('min_liquidity', 0) and \
               cfg.get('token_age_min', 0) <= age <= cfg.get('token_age_max', 999)


class Strategist:
    """
    THE STRATEGIST V2 - Main orchestrator with SMART LLM GATING.
    
    LLM is ONLY called if:
    1. Pre-filters pass
    2. Base score >= threshold
    3. LLM is enabled
    """
    
    def __init__(self, db, api_key: str = None):
        self.db = db
        self.api_key = api_key or os.getenv('ANTHROPIC_API_KEY')
        
        print("\n🧠 THE STRATEGIST V2")
        print("="*50)
        
        self.regime_detector = RegimeDetector(db)
        self.wallet_clusterer = WalletClusterer(db)
        self.signal_aggregator = SignalAggregator()
        self.exit_tracker = ExitSignalTracker(db)
        self.strategy_lab = StrategyLab(db)
        self.reasoning_agent = ReasoningAgent(self.api_key, db)
        
        print(f"  ✅ All components loaded")
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
    
    def get_status(self) -> Dict:
        return {
            'regime': self.regime_detector.current_regime,
            'confidence': self.regime_detector.confidence,
            'llm_enabled': self.reasoning_agent.enabled,
            'champion': self.strategy_lab.strategies.get('champion', {}).get('name'),
            'pending_signals': len(self.signal_aggregator.signals)
        }
    
    def get_llm_cost_today(self) -> Dict:
        return self.db.get_llm_cost_summary(days=1)


if __name__ == "__main__":
    print("\n" + "="*60)
    print("STRATEGIST V2 - Test")
    print("="*60)
    
    from database_v2 import DatabaseV2
    
    db = DatabaseV2("test_strat.db")
    strategist = Strategist(db)
    
    signal = {'token_symbol': 'TEST', 'token_address': 'test123',
              'price': 0.001, 'liquidity': 50000, 'volume_24h': 25000,
              'token_age_hours': 12}
    wallet = {'address': 'w123', 'win_rate': 0.65}
    
    result = strategist.analyze_signal(signal, wallet, use_llm=False)
    print(f"\nResult: {result['action']} ({result['conviction_score']:.0f})")
    print(f"LLM called: {result['llm_called']}")
    
    import os
    os.remove("test_strat.db")
