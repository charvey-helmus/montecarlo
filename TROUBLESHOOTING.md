# Troubleshooting: Plugin Shows "No Components"

If Dataiku shows "the plugin has no components", try these steps:

## 1. Verify Plugin Structure

Ensure your plugin folder has this structure:
```
monte_carlo/
├── plugin.json
└── recipe-monte-carlo-recipe/
    ├── recipe.json
    └── recipe.py
```

## 2. Check Installation Method

**If installed via UI:**
- Go to **Administration** > **Plugins**
- Find your plugin
- Click **Reload** or **Uninstall** then **Reinstall**

**If installed manually:**
- Ensure the entire `monte_carlo` folder is in: `$DATA_DIR/plugins/installed/monte-carlo-profit-simulator/`
- Restart Dataiku or reload plugins

## 3. Verify plugin.json is Valid JSON

The plugin.json should have:
- Valid JSON syntax (no trailing commas, proper quotes)
- A `components` array with at least one component
- Component ID matching folder name pattern

## 4. Check Component ID vs Folder Name

- Folder: `recipe-monte-carlo-recipe`
- Component ID in plugin.json: `monte-carlo-recipe`
- These should match (folder name without `recipe-` prefix)

## 5. Verify recipe.json Structure

The `recipe.json` file should have:
- `"kind": "PYTHON"` field
- `inputRoles` array
- `outputRoles` array
- `params` array

## 6. Common Issues and Fixes

### Issue: Components array is empty
**Fix:** Ensure `components` array in plugin.json contains the component definition

### Issue: Folder name doesn't match component ID
**Fix:** Folder should be `recipe-<component-id>`, so `recipe-monte-carlo-recipe` matches component ID `monte-carlo-recipe`

### Issue: Plugin not refreshed after changes
**Fix:** 
1. Go to Administration > Plugins
2. Find your plugin
3. Click "Reload" or restart Dataiku

### Issue: Missing recipe.json or recipe.py
**Fix:** Ensure both files exist in the `recipe-monte-carlo-recipe/` folder

## 7. Development Mode

If developing the plugin:
- Place it in `$DATA_DIR/plugins/dev/monte-carlo-profit-simulator/`
- Changes are auto-reloaded in development mode
- Check the plugin editor for errors

## 8. Check Dataiku Logs

If still not working:
1. Check Dataiku logs for plugin loading errors
2. Look for JSON parsing errors
3. Verify file permissions (plugin files should be readable)

## 9. Minimal Test

Try creating a minimal plugin.json to test:
```json
{
    "id": "test-plugin",
    "version": "1.0.0",
    "meta": {
        "label": "Test Plugin"
    },
    "components": [
        {
            "id": "test-recipe",
            "meta": {
                "label": "Test Recipe"
            }
        }
    ]
}
```

With a folder: `recipe-test-recipe/` containing `recipe.json` and `recipe.py`

## 10. Contact Support

If none of these work, the issue might be:
- Dataiku version compatibility
- Plugin installation directory permissions
- Corrupted plugin cache

Try clearing the plugin cache or reinstalling Dataiku plugins.

