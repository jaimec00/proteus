#!/bin/bash

# stuff for setup dev on new instances

sudo apt update

# claude
curl -fsSL https://claude.ai/install.sh | bash

# pixi
curl -fsSL https://pixi.sh/install.sh | sh

# setup tmux
git clone --single-branch https://github.com/gpakosz/.tmux.git $HOME/.tmux
ln -s -f $HOME/.tmux/.tmux.conf $HOME/.tmux.conf
cp $HOME/.tmux/.tmux.conf.local $HOME/.tmux.conf.local
echo "set -g mouse on" >> $HOME/.tmux.conf.local

# codium extensions
codium --install-extension ms-python.python
codium --install-extension tamasfe.even-better-toml

# add git config stuff, will remove later
git config --global user.name "jaimec00"
git config --global user.email "hejaca00@gmail.com"
git config --global fetch.prune true
