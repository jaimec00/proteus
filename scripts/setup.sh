#!/bin/bash

# stuff for setup dev on new instances

# claude
curl -fsSL https://claude.ai/install.sh | bash

# pixi
curl -fsSL https://pixi.sh/install.sh | sh

# pymol
sudo apt install pymol -y

# github
sudo apt install gh -y

# setup tmux
echo "set -g mouse on" > $HOME/.tmux.conf
