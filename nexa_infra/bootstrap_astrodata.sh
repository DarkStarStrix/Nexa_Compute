#!/usr/bin/env bash
set -e

##############################################
#  NEXA — ASTRODATA BOOTSTRAP SCRIPT
#  Prepares the 48-core VM for data processing
#  - Installs rclone
#  - Creates directories
#  - Sets up Wasabi config
#  - Installs Python & deps
#  - Clones repo
##############################################

echo "=== Loading environment variables ==="
if [ -f ~/.nexa_env ]; then
    source ~/.nexa_env
else
    echo "ERROR: ~/.nexa_env not found. Exiting."
    exit 1
fi

echo ""
echo "=== Updating system ==="
sudo apt-get update -y

echo ""
echo "=== Installing dependencies ==="
sudo apt-get install -y python3 python3-pip python3-venv git curl

echo ""
echo "=== Installing rclone ==="
curl https://rclone.org/install.sh | sudo bash

echo ""
echo "=== Creating rclone config ==="
mkdir -p ~/.config/rclone
cat > ~/.config/rclone/rclone.conf <<EOF
[wasabi]
type = s3
provider = Wasabi
access_key_id = ${WASABI_ACCESS_KEY}
secret_access_key = ${WASABI_SECRET_KEY}
region = ${WASABI_REGION}
endpoint = ${WASABI_ENDPOINT}
EOF

echo ""
echo "=== Verifying rclone access ==="
rclone lsd wasabi: || { echo "rclone test failed"; exit 1; }

echo ""
echo "=== Setting up NVMe scratch directories ==="
sudo mkdir -p /nvme/raw
sudo mkdir -p /nvme/shards/train
sudo mkdir -p /nvme/shards/val
sudo mkdir -p /nvme/shards/test
sudo mkdir -p /nvme/manifests
sudo chmod -R 777 /nvme

echo ""
echo "=== Cloning NEXA repo ==="
if [ ! -d ~/nexa-ms ]; then
    git clone https://github.com/YOURUSERNAME/nexa-ms.git ~/nexa-ms
else
    echo "Repo already exists, pulling latest..."
    cd ~/nexa-ms && git pull
fi

echo ""
echo "=== Creating Python venv ==="
python3 -m venv ~/nexa-env
source ~/nexa-env/bin/activate
pip install --upgrade pip

echo ""
echo "=== Installing Python dependencies ==="
pip install -r ~/nexa-ms/requirements.txt

echo ""
echo "=== Bootstrapping complete ==="
echo "VM is now ready to:"
echo " - download raw HDF5 from HF"
echo " - preprocess into Arrow shards"
echo " - upload shards/manifests to Wasabi"
echo ""
echo "Run: source ~/nexa-env/bin/activate"
