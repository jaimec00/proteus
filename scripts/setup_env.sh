#!/bin/bash

# stuff for setup dev on new instances

sudo apt update

# claude
curl -fsSL https://claude.ai/install.sh | bash

# pixi
curl -fsSL https://pixi.sh/install.sh | sh

# aws
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
sudo apt install unzip -y
unzip awscliv2.zip
sudo ./aws/install
rm -r aws awscliv2.zip

# github
sudo apt install gh -y

# setup tmux
git clone --single-branch https://github.com/gpakosz/.tmux.git $HOME/.tmux
ln -s -f $HOME/.tmux/.tmux.conf $HOME/.tmux.conf
cp $HOME/.tmux/.tmux.conf.local $HOME/.tmux.conf.local
echo "set -g mouse on" >> $HOME/.tmux.conf.local

# add codium (what i use) extensions
codium --install-extension ms-python.python
codium --install-extension tamasfe.even-better-toml

# logins at the end
echo "logging in to aws..."
aws login

echo "logging in to claude..."
claude auth login

echo "logging in to github"
gh auth login

# add git config stuff, will remove later
git config --global user.name "jaimec00"
git config --global user.email "hejaca00@gmail.com"
git config --global fetch.prune true
