# Installation Guide

## Quick Start

### Option 1: Install as Dataiku Plugin (Recommended)

1. **Copy the plugin folder** to your Dataiku instance:
   - The entire `monte_carlo` folder should be copied to your Dataiku plugins directory
   - Typically located at: `$DATA_DIR/plugins/installed/`

2. **Install in Dataiku UI**:
   - Go to **Administration** > **Plugins**
   - Click **Add Plugin** > **From a folder**
   - Select the `monte_carlo` folder
   - Click **Install**

3. **Use the recipe**:
   - In your Flow, click **+ Recipe**
   - Select **Monte Carlo Profit Simulation** from the recipe list
   - Configure your input dataset and parameters
   - Run!

### Option 2: Use as Standalone Python Code

1. **Copy the code**:
   - Copy `improved_monte_carlo.py` to your Dataiku project
   - Or use it in a Python recipe

2. **Import and use**:
   ```python
   from improved_monte_carlo import run_monte_carlo_simulation
   
   # Load your data
   df_input = dataiku.Dataset("your_dataset").get_dataframe()
   
   # Run simulation
   df_results = run_monte_carlo_simulation(
       df_input=df_input,
       n_simulations=1000,
       # ... other parameters
   )
   ```

## Plugin Structure

```
monte_carlo/
├── plugin.json                    # Plugin metadata
├── recipe-monte-carlo-recipe/
│   ├── recipe.json                # Recipe component definition
│   └── recipe.py                  # Recipe execution code
├── improved_monte_carlo.py        # Standalone improved code
├── requirements.txt               # Python dependencies
├── README.md                      # Main documentation
├── IMPROVEMENTS.md                # Code improvements summary
└── INSTALLATION.md                # This file
```

## Dependencies

The plugin requires:
- `pandas >= 1.3.0`
- `numpy >= 1.20.0`

These are typically already installed in Dataiku Python environments. If not, they will be installed automatically when you install the plugin.

## Troubleshooting

### Plugin doesn't appear in recipe list
- Make sure the plugin is installed (check Administration > Plugins)
- Refresh your browser
- Check that `plugin.json` is valid JSON

### Recipe fails with import errors
- Ensure pandas and numpy are installed in your code environment
- Check that the recipe.py file is in the correct location

### Parameter validation errors
- Make sure fee erosion values and probabilities have the same length
- Ensure probabilities sum to 1.0
- Check that all required input columns are present

### Performance issues
- Reduce `n_simulations` for faster execution
- Consider sampling your input data if you have many deals
- The vectorized code should be fast, but very large datasets may take time

## Next Steps

1. Read the [README.md](README.md) for detailed usage instructions
2. Review [IMPROVEMENTS.md](IMPROVEMENTS.md) to understand the enhancements
3. Test with a small dataset first
4. Adjust parameters based on your business requirements

