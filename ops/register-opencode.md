# Registering with OpenCode (operator action)

1. Back up first (mandatory):

   ```bash
   cp ~/.config/opencode/opencode.json ~/.config/opencode/opencode.json_$(date +%Y%m%d%H%M%S)_backup
   ```

2. Merge this entry into `opencode.json` (keep all existing keys):

   ```json
   "mcp": {
     "embedding": {
       "type": "remote",
       "url": "http://<EMBED_HOST>:<MCP_PORT>/mcp",
       "enabled": true
     }
   }
   ```

3. Validate + verify:

   ```bash
   python3 -c "import json; json.load(open('$HOME/.config/opencode/opencode.json')); print('valid')"
   opencode mcp list    # expect: embedding connected
   ```

Rollback: restore the backup file.
