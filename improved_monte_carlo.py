"""
Improved Monte Carlo Profit Simulation

This is a standalone version of the improved Monte Carlo simulation code.
Key improvements:
1. Vectorized operations for better performance
2. Configurable parameters
3. Better error handling and validation
4. More detailed output metrics
5. Reproducibility through random seeds
"""

import pandas as pd
import numpy as np
import dataiku


def validate_parameters(fee_erosion_values, fee_erosion_probs):
    """Validate that parameters are consistent and reasonable."""
    if len(fee_erosion_values) != len(fee_erosion_probs):
        raise ValueError(
            f"Number of fee erosion values ({len(fee_erosion_values)}) must match "
            f"number of probabilities ({len(fee_erosion_probs)})"
        )
    
    prob_sum = sum(fee_erosion_probs)
    if abs(prob_sum - 1.0) > 1e-6:
        raise ValueError(
            f"Fee erosion probabilities must sum to 1.0, got {prob_sum}"
        )
    
    return True


def run_monte_carlo_simulation(
    df_input,
    n_simulations=1000,
    volume_shock_mean=1.0,
    volume_shock_std=0.15,
    cost_shock_alpha=2.0,
    cost_shock_beta=5.0,
    cost_shock_scale=0.04,
    fee_erosion_values=None,
    fee_erosion_probs=None,
    base_cost_to_serve=0.01,
    backend_rebate_pct=0.04,
    profit_threshold=0.0,
    random_seed=None
):
    """
    Run Monte Carlo simulation for profit and rebates.
    
    Parameters:
    -----------
    df_input : pd.DataFrame
        Input dataframe with deal information
    n_simulations : int
        Number of simulations per deal
    volume_shock_mean : float
        Mean of volume shock distribution
    volume_shock_std : float
        Standard deviation of volume shock distribution
    cost_shock_alpha : float
        Alpha parameter for cost shock beta distribution
    cost_shock_beta : float
        Beta parameter for cost shock beta distribution
    cost_shock_scale : float
        Scale factor for cost shock
    fee_erosion_values : list
        List of possible fee erosion values (as decimals)
    fee_erosion_probs : list
        List of probabilities for fee erosion values
    base_cost_to_serve : float
        Base cost to serve as decimal (e.g., 0.01 = 1%)
    backend_rebate_pct : float
        Backend rebate as decimal (e.g., 0.04 = 4%)
    profit_threshold : float
        Threshold for profitable vs loss scenarios
    random_seed : int or None
        Random seed for reproducibility
        
    Returns:
    --------
    pd.DataFrame
        Results dataframe with simulation outcomes
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # Default fee erosion if not provided
    if fee_erosion_values is None:
        fee_erosion_values = [0.0, 0.001, 0.002]
    if fee_erosion_probs is None:
        fee_erosion_probs = [0.7, 0.2, 0.1]
    
    # Validate parameters
    validate_parameters(fee_erosion_values, fee_erosion_probs)
    
    # Normalize probabilities
    fee_erosion_probs = np.array(fee_erosion_probs)
    fee_erosion_probs = fee_erosion_probs / fee_erosion_probs.sum()
    
    # Required columns
    required_cols = ['deal_id', 'total_volume', 'avg_rebate_pct', 'distribution_fee_pct']
    missing_cols = [col for col in required_cols if col not in df_input.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Prepare results list
    results = []
    
    # Iterate through each deal
    for index, row in df_input.iterrows():
        deal_id = row['deal_id']
        customer_name = row.get('customer_name', f'Deal_{deal_id}')
        base_volume = float(row['total_volume'])
        base_rebate = float(row['avg_rebate_pct']) / 100.0  # Convert percentage to decimal
        base_fee = float(row['distribution_fee_pct']) / 100.0  # Convert percentage to decimal
        
        # Generate risk factors for all simulations at once (vectorized)
        # Volume shock: normal distribution
        vol_shocks = np.random.normal(volume_shock_mean, volume_shock_std, n_simulations)
        # Ensure non-negative volumes
        vol_shocks = np.maximum(vol_shocks, 0.01)  # At least 1% of original volume
        
        # Cost shock: beta distribution scaled
        cost_shocks = np.random.beta(cost_shock_alpha, cost_shock_beta, n_simulations) * cost_shock_scale
        
        # Fee erosion: discrete choice
        fee_erosions = np.random.choice(fee_erosion_values, size=n_simulations, p=fee_erosion_probs)
        
        # Vectorized calculations
        sim_volumes = base_volume * vol_shocks
        sim_fees = np.maximum(base_fee - fee_erosions, 0.0)  # Ensure non-negative fees
        
        # Financial calculations (vectorized)
        gross_margin_pcts = sim_fees - base_rebate
        sim_costs_to_serve = base_cost_to_serve * (1 + cost_shocks)
        final_net_margin_pcts = gross_margin_pcts + backend_rebate_pct - sim_costs_to_serve
        final_profits = sim_volumes * final_net_margin_pcts
        
        # Determine outcomes
        scenarios = np.where(final_profits > profit_threshold, 'Profitable', 'Loss')
        
        # Store results for this deal
        for i in range(n_simulations):
            results.append({
                'deal_id': deal_id,
                'customer_name': customer_name,
                'simulation_run': i,
                'simulated_volume': sim_volumes[i],
                'simulated_fee_pct': sim_fees[i] * 100,  # Convert back to percentage
                'simulated_cost_to_serve_pct': sim_costs_to_serve[i] * 100,
                'simulated_net_margin_pct': final_net_margin_pcts[i] * 100,
                'simulated_profit': final_profits[i],
                'volume_shock': vol_shocks[i],
                'cost_shock': cost_shocks[i],
                'fee_erosion': fee_erosions[i] * 100,  # Convert to percentage
                'scenario_outcome': scenarios[i]
            })
    
    return pd.DataFrame(results)


# Example usage (when running standalone)
if __name__ == "__main__":
    # --- 1. Load Data ---
    # Ensure "current_negotiation" is in your Flow
    df_current = dataiku.Dataset("current_negotiation").get_dataframe()
    
    # --- 2. Run Simulation ---
    df_results = run_monte_carlo_simulation(
        df_input=df_current,
        n_simulations=1000,
        volume_shock_mean=1.0,
        volume_shock_std=0.15,
        cost_shock_alpha=2.0,
        cost_shock_beta=5.0,
        cost_shock_scale=0.04,
        fee_erosion_values=[0.0, 0.001, 0.002],
        fee_erosion_probs=[0.7, 0.2, 0.1],
        base_cost_to_serve=0.01,
        backend_rebate_pct=0.04,
        profit_threshold=0.0,
        random_seed=42  # For reproducibility
    )
    
    # --- 3. Output ---
    output_dataset = dataiku.Dataset("monte_carlo_results")
    output_dataset.write_with_schema(df_results)
    
    print(f"Done! Generated {len(df_results)} scenarios across {len(df_current)} deals.")
    print(f"Average scenarios per deal: {len(df_results) / len(df_current):.1f}")

