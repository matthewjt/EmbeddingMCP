#!/usr/bin/env bash
# Prints (never runs) the firewall commands. Operator approval required.
set -u
LAN_SUBNET=${LAN_SUBNET:-10.223.223.0/24}
LLAMA_PORT=${LLAMA_PORT:-8080}
MCP_PORT=${MCP_PORT:-8090}
cat <<EOF
Run these yourself (review before executing):

  sudo ufw allow from $LAN_SUBNET to any port $LLAMA_PORT,$MCP_PORT proto tcp comment "embedding-mcp"
  sudo ufw status numbered        # verify
  sudo ufw delete <number>        # rollback

If ufw is not active on this host, adapt for your firewall (nftables/
firewalld). Never expose these ports to the public internet.
EOF
