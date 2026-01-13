# Monte Carlo Profit Simulator - Dataiku Plugin

A Dataiku plugin that runs Monte Carlo simulations to simulate profit and rebates based on volume shocks, cost shocks, and fee erosion.

## Features

### Improvements Over Original Code

1. **Vectorized Operations**: Uses NumPy vectorization instead of explicit loops for better performance
2. **Configurable Parameters**: All simulation parameters are configurable through the plugin UI
3. **Better Error Handling**: Validates inputs and provides clear error messages
4. **More Detailed Output**: Includes additional metrics like volume shocks, cost shocks, and fee erosion values
5. **Reproducibility**: Optional random seed for reproducible results
6. **Data Validation**: Checks for required columns and validates parameter consistency

## Installation

1. Copy the plugin folder to your Dataiku instance's plugins directory
2. In Dataiku, go to **Administration** > **Plugins** > **Add Plugin** > **From a folder**
3. Select the plugin folder
4. The plugin will appear in your recipe list

## Usage

### As a Dataiku Plugin Recipe

1. In your Dataiku Flow, add a new recipe
2. Select **Monte Carlo Profit Simulation** from the recipe list
3. Configure the input dataset (must contain columns: `deal_id`, `total_volume`, `avg_rebate_pct`, `distribution_fee_pct`)
4. Configure simulation parameters:
   - **Number of Simulations**: How many Monte Carlo runs per deal (default: 1000)
   - **Volume Shock Parameters**: Mean and standard deviation for volume variations
   - **Cost Shock Parameters**: Beta distribution parameters for cost variations
   - **Fee Erosion**: Possible values and their probabilities
   - **Financial Parameters**: Base cost to serve, backend rebate, profit threshold
   - **Random Seed**: Optional seed for reproducibility
5. Select an output dataset
6. Run the recipe

### Required Input Columns

- `deal_id`: Unique identifier for each deal
- `total_volume`: Total volume for the deal
- `avg_rebate_pct`: Average rebate percentage
- `distribution_fee_pct`: Distribution fee percentage
- `customer_name`: (Optional) Customer name for filtering/grouping

### Output Columns

- `deal_id`: Deal identifier
- `customer_name`: Customer name
- `simulation_run`: Simulation run number (0 to n_simulations-1)
- `simulated_volume`: Simulated volume after shock
- `simulated_fee_pct`: Simulated fee percentage after erosion
- `simulated_cost_to_serve_pct`: Simulated cost to serve percentage
- `simulated_net_margin_pct`: Final net margin percentage
- `simulated_profit`: Final profit amount
- `volume_shock`: Volume shock multiplier applied
- `cost_shock`: Cost shock multiplier applied
- `fee_erosion`: Fee erosion percentage applied
- `scenario_outcome`: 'Profitable' or 'Loss' based on profit threshold

## Standalone Usage

You can also use the improved code directly in a Dataiku Python recipe:

```python
import pandas as pd
import numpy as np
import dataiku
from improved_monte_carlo import run_monte_carlo_simulation

# Load your data
df_input = dataiku.Dataset("your_input_dataset").get_dataframe()

# Run simulation
df_results = run_monte_carlo_simulation(
    df_input=df_input,
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
    random_seed=42
)

# Write results
output_dataset = dataiku.Dataset("your_output_dataset")
output_dataset.write_with_schema(df_results)
```

## Simulation Logic

The simulation models three types of shocks:

1. **Volume Shock**: Normal distribution around a mean (default: 1.0 = no change) with configurable standard deviation
2. **Cost Shock**: Beta distribution scaled by a factor, representing cost inflation
3. **Fee Erosion**: Discrete choice from a set of possible values with specified probabilities

The profit calculation:
```
Gross Margin % = Fee % - Rebate %
Cost to Serve % = Base Cost % × (1 + Cost Shock)
Net Margin % = Gross Margin % + Backend Rebate % - Cost to Serve %
Profit = Volume × Net Margin %
```

## Performance

The improved version uses vectorized NumPy operations, which provides significant performance improvements over the original loop-based approach, especially for large numbers of simulations.

## License

This plugin is provided as-is for use in Dataiku projects.

