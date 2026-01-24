"""
Database Schema Addition - Token & Market Data Tracking

Add these tables to database_v2.py to track tokens discovered
and market conditions for strategist analysis.
"""

# Add this to database_v2.py _init_schema() method:

def _add_token_tracking_tables(conn):
    """
    Add tables for tracking discovered tokens and their performance.
    This data helps the strategist learn which token characteristics
    lead to profitable trades.
    """
    
    # Table: discovered_tokens
    # Tracks all tokens we've discovered during discovery cycles
    conn.execute("""
        CREATE TABLE IF NOT EXISTS discovered_tokens (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            token_address TEXT UNIQUE NOT NULL,
            symbol TEXT,
            name TEXT,
            
            -- Discovery info
            discovered_at TIMESTAMP,
            discovery_source TEXT,
            
            -- Token metrics at discovery
            price_usd REAL,
            liquidity_usd REAL,
            volume_24h REAL,
            market_cap REAL,
            fdv REAL,
            price_change_24h REAL,
            age_hours REAL,
            
            -- Holder info
            holder_count INTEGER,
            top_10_concentration REAL,
            
            -- Security (from Birdeye)
            is_mintable INTEGER,
            is_freezable INTEGER,
            has_lp_locked INTEGER,
            lp_lock_days INTEGER,
            
            -- Performance tracking
            was_traded INTEGER DEFAULT 0,
            total_positions_opened INTEGER DEFAULT 0,
            winning_positions INTEGER DEFAULT 0,
            total_pnl_sol REAL DEFAULT 0,
            
            -- Status
            is_active INTEGER DEFAULT 1,
            last_updated TIMESTAMP
        )
    """)
    
    # Table: token_snapshots
    # Time-series data for discovered tokens
    conn.execute("""
        CREATE TABLE IF NOT EXISTS token_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            token_address TEXT NOT NULL,
            snapshot_time TIMESTAMP,
            
            price_usd REAL,
            liquidity_usd REAL,
            volume_24h REAL,
            market_cap REAL,
            price_change_1h REAL,
            price_change_24h REAL,
            holder_count INTEGER,
            
            FOREIGN KEY (token_address) REFERENCES discovered_tokens(token_address)
        )
    """)
    
    # Table: market_regime_history
    # Enhanced market conditions tracking
    conn.execute("""
        CREATE TABLE IF NOT EXISTS market_regime_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TIMESTAMP,
            
            -- SOL metrics
            sol_price_usd REAL,
            sol_24h_change_pct REAL,
            sol_volume_24h REAL,
            
            -- Market regime
            regime TEXT,
            regime_confidence REAL,
            
            -- Broader market
            btc_price_usd REAL,
            btc_24h_change_pct REAL,
            total_market_cap REAL,
            
            -- Solana ecosystem
            active_wallets_24h INTEGER,
            total_transactions_24h INTEGER,
            dex_volume_24h REAL,
            
            -- Sentiment indicators
            fear_greed_index INTEGER,
            trending_token_count INTEGER
        )
    """)
    
    # Table: discovery_cycles
    # Track each discovery cycle for analysis
    conn.execute("""
        CREATE TABLE IF NOT EXISTS discovery_cycles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            cycle_start TIMESTAMP,
            cycle_end TIMESTAMP,
            
            -- Discovery stats
            tokens_discovered INTEGER,
            wallets_found INTEGER,
            wallets_profiled INTEGER,
            wallets_verified INTEGER,
            
            -- API usage
            helius_calls_used INTEGER,
            api_budget INTEGER,
            
            -- Sources
            pumping_tokens INTEGER,
            trending_tokens INTEGER,
            new_tokens INTEGER,
            
            -- Efficiency metrics
            verification_rate REAL,
            cost_per_wallet REAL,
            
            -- Market context
            market_regime TEXT,
            sol_price_usd REAL
        )
    """)
    
    # Indices for performance
    conn.execute("CREATE INDEX IF NOT EXISTS idx_discovered_tokens_address ON discovered_tokens(token_address)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_token_snapshots_token ON token_snapshots(token_address)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_token_snapshots_time ON token_snapshots(snapshot_time)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_market_regime_time ON market_regime_history(timestamp)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_discovery_cycles_start ON discovery_cycles(cycle_start)")


# Add these methods to DatabaseV2 class:

def add_discovered_token(self, token_data: Dict):
    """Record a token discovered during discovery cycle"""
    with self.connection() as conn:
        conn.execute("""
            INSERT OR REPLACE INTO discovered_tokens 
            (token_address, symbol, name, discovered_at, discovery_source,
             price_usd, liquidity_usd, volume_24h, market_cap, fdv,
             price_change_24h, age_hours, last_updated)
            VALUES (?, ?, ?, COALESCE(
                (SELECT discovered_at FROM discovered_tokens WHERE token_address = ?), ?
            ), ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            token_data['address'],
            token_data.get('symbol'),
            token_data.get('name'),
            token_data['address'],
            datetime.now(),
            token_data.get('source'),
            token_data.get('price_usd', 0),
            token_data.get('liquidity', 0),
            token_data.get('volume_24h', 0),
            token_data.get('market_cap', 0),
            token_data.get('fdv', 0),
            token_data.get('price_change_24h', 0),
            token_data.get('age_hours', 0),
            datetime.now()
        ))

def update_token_performance(self, token_address: str, position_result: Dict):
    """Update token performance metrics when a position closes"""
    with self.connection() as conn:
        # Mark as traded
        conn.execute("""
            UPDATE discovered_tokens 
            SET was_traded = 1,
                total_positions_opened = total_positions_opened + 1,
                winning_positions = winning_positions + CASE WHEN ? > 0 THEN 1 ELSE 0 END,
                total_pnl_sol = total_pnl_sol + ?,
                last_updated = ?
            WHERE token_address = ?
        """, (
            position_result.get('profit_pct', 0),
            position_result.get('profit_sol', 0),
            datetime.now(),
            token_address
        ))

def add_token_snapshot(self, token_address: str, snapshot_data: Dict):
    """Add a time-series snapshot of token metrics"""
    with self.connection() as conn:
        conn.execute("""
            INSERT INTO token_snapshots
            (token_address, snapshot_time, price_usd, liquidity_usd,
             volume_24h, market_cap, price_change_1h, price_change_24h, holder_count)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            token_address,
            datetime.now(),
            snapshot_data.get('price_usd', 0),
            snapshot_data.get('liquidity', 0),
            snapshot_data.get('volume_24h', 0),
            snapshot_data.get('market_cap', 0),
            snapshot_data.get('price_change_1h', 0),
            snapshot_data.get('price_change_24h', 0),
            snapshot_data.get('holder_count', 0)
        ))

def log_discovery_cycle(self, cycle_stats: Dict):
    """Log a completed discovery cycle for analysis"""
    with self.connection() as conn:
        conn.execute("""
            INSERT INTO discovery_cycles
            (cycle_start, cycle_end, tokens_discovered, wallets_found,
             wallets_profiled, wallets_verified, helius_calls_used,
             api_budget, pumping_tokens, trending_tokens, new_tokens,
             verification_rate, cost_per_wallet, market_regime, sol_price_usd)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            cycle_stats.get('cycle_start'),
            datetime.now(),
            cycle_stats.get('tokens_discovered', 0),
            cycle_stats.get('wallet_candidates_found', 0),
            cycle_stats.get('wallets_profiled', 0),
            cycle_stats.get('wallets_verified', 0),
            cycle_stats.get('helius_api_calls', 0),
            cycle_stats.get('api_budget', 0),
            cycle_stats.get('token_sources', {}).get('pumping', 0),
            cycle_stats.get('token_sources', {}).get('trending', 0),
            cycle_stats.get('token_sources', {}).get('new', 0),
            cycle_stats.get('verification_rate', 0),
            cycle_stats.get('cost_per_wallet', 0),
            cycle_stats.get('market_regime'),
            cycle_stats.get('sol_price', 0)
        ))

def get_token_performance_analysis(self, days: int = 30) -> Dict:
    """
    Analyze which token characteristics lead to profitable trades.
    This helps the strategist learn and improve.
    """
    with self.connection() as conn:
        # Get tokens that were actually traded
        cursor = conn.execute("""
            SELECT 
                discovery_source,
                AVG(CASE WHEN total_pnl_sol > 0 THEN 1.0 ELSE 0.0 END) as win_rate,
                AVG(total_pnl_sol) as avg_pnl,
                COUNT(*) as count,
                AVG(liquidity_usd) as avg_liquidity,
                AVG(volume_24h) as avg_volume,
                AVG(price_change_24h) as avg_pump
            FROM discovered_tokens
            WHERE was_traded = 1
            AND discovered_at >= datetime('now', ? || ' days')
            GROUP BY discovery_source
        """, (f'-{days}',))
        
        sources = [dict(row) for row in cursor.fetchall()]
        
        # Get overall stats
        overall = conn.execute("""
            SELECT 
                COUNT(*) as total_discovered,
                SUM(was_traded) as total_traded,
                AVG(CASE WHEN was_traded = 1 THEN total_pnl_sol END) as avg_pnl_per_trade,
                SUM(CASE WHEN total_pnl_sol > 0 THEN 1 ELSE 0 END) as winning_tokens
            FROM discovered_tokens
            WHERE discovered_at >= datetime('now', ? || ' days')
        """, (f'-{days}',)).fetchone()
        
        return {
            'by_source': sources,
            'overall': dict(overall) if overall else {},
            'period_days': days
        }

def get_discovery_efficiency_stats(self) -> Dict:
    """Get discovery system efficiency metrics"""
    with self.connection() as conn:
        stats = conn.execute("""
            SELECT 
                COUNT(*) as total_cycles,
                SUM(wallets_verified) as total_wallets_found,
                SUM(helius_calls_used) as total_api_calls,
                AVG(verification_rate) as avg_verification_rate,
                AVG(cost_per_wallet) as avg_cost_per_wallet,
                SUM(helius_calls_used) / NULLIF(SUM(wallets_verified), 0) as actual_cost_per_wallet
            FROM discovery_cycles
            WHERE cycle_start >= datetime('now', '-30 days')
        """).fetchone()
        
        return dict(stats) if stats else {}


# Usage in hybrid_discovery.py:

def _store_token_data(self, token: Dict):
    """Store discovered token in database"""
    self.db.add_discovered_token(token)

# At end of run_discovery():
def run_discovery(self, api_budget: int = 300, max_wallets: int = 15) -> Dict:
    # ... existing code ...
    
    # Log the discovery cycle
    stats['cycle_start'] = cycle_start_time  # Set at beginning
    stats['market_regime'] = self.db.get_current_market_regime()
    stats['sol_price'] = self.db.get_current_sol_price()
    
    if stats['wallets_profiled'] > 0:
        stats['verification_rate'] = stats['wallets_verified'] / stats['wallets_profiled']
        stats['cost_per_wallet'] = stats['helius_api_calls'] / stats['wallets_verified'] if stats['wallets_verified'] > 0 else 0
    
    self.db.log_discovery_cycle(stats)
    
    return stats
