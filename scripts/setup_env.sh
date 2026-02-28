#!/bin/bash

# stuff for setup dev on new instances

sudo apt upgrade

# claude
curl -fsSL https://claude.ai/install.sh | bash

# pixi
curl -fsSL https://pixi.sh/install.sh | sh

# aws
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install
rm -rf aws awscliv2.zip

# github
sudo apt install gh -y

# setup tmux
git clone --single-branch https://github.com/gpakosz/.tmux.git $HOME/.tmux
ln -s -f $HOME/.tmux/.tmux.conf $HOME/.tmux.conf
cp $HOME/.tmux/.tmux.conf.local $HOME/.tmux.conf.local
echo "set -g mouse on" >> $HOME/.tmux.conf.local

