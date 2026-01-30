#!/bin/bash
set -e

PLUGIN_DIR="$HOME/.claude/plugins/atif-exporter-plugin"
SETTINGS_FILE="$HOME/.claude/settings.json"

echo "Installing ATIF Exporter plugin for Claude Code..."

# Create plugins directory
mkdir -p "$HOME/.claude/plugins"

# Clone or update the plugin
if [ -d "$PLUGIN_DIR" ]; then
    echo "Updating existing installation..."
    cd "$PLUGIN_DIR"
    git pull origin main
else
    echo "Cloning plugin..."
    git clone https://github.com/refreshdotdev/atif-exporter-plugin.git "$PLUGIN_DIR"
fi

# Check if settings.json exists and update it
if [ -f "$SETTINGS_FILE" ]; then
    # Check if plugin is already in settings
    if grep -q "atif-exporter-plugin" "$SETTINGS_FILE"; then
        echo "Plugin already in settings.json"
    else
        echo ""
        echo "Add this to your ~/.claude/settings.json plugins array:"
        echo "  \"$PLUGIN_DIR\""
    fi
else
    # Create settings.json with the plugin
    mkdir -p "$HOME/.claude"
    echo '{
  "plugins": ["'"$PLUGIN_DIR"'"]
}' > "$SETTINGS_FILE"
    echo "Created ~/.claude/settings.json with plugin enabled"
fi

echo ""
echo "Installation complete!"
echo "Trajectories will be saved to: ~/.claude/atif-trajectories/"
echo ""
echo "Run 'claude' from any directory to start capturing trajectories."
