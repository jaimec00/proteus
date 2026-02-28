#!/bin/bash

# stuff for setup dev on new instances

# claude
curl -fsSL https://claude.ai/install.sh | bash

# pixi
curl -fsSL https://pixi.sh/install.sh | sh

# pymol
# sudo apt install pymol -y

# github
sudo apt install gh -y

# setup tmux
git clone --single-branch https://github.com/gpakosz/.tmux.git $HOME/.tmux
ln -s -f $HOME/.tmux/.tmux.conf $HOME/.tmux.conf
cp $HOME/.tmux/.tmux.conf.local $HOME/.tmux.conf.local
echo "set -g mouse on" >> $HOME/.tmux.conf.local

