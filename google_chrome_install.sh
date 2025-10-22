apt update -y
wget https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb
dpkg -i google-chrome-stable_current_amd64.deb
apt -f install -y

# Remove .deb file
rm google-chrome-stable_current_amd64.deb