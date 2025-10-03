#!/usr/bin/env bash
set -euo pipefail

# 1) Docker
sudo apt-get update
sudo apt-get install -y ca-certificates curl gnupg
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
  https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo $VERSION_CODENAME) stable" \
  | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin
sudo usermod -aG docker $USER

# 2) Каталоги и файлы
sudo mkdir -p /opt/passivbot/{data,configs}
sudo mkdir -p /opt/passivbot/deploy
sudo chown -R $USER:$USER /opt/passivbot

# Скопируйте в /opt/passivbot: docker-compose.yml, *.service, *.timer, *.sh и ваш .env / api-keys.json
echo "Place your files into /opt/passivbot and run the rest steps."
echo "Then run: sudo systemctl daemon-reload && sudo systemctl enable --now updater.timer"
