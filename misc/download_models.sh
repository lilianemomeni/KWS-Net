#create output location
output_dir="misc/pretrained_models"
mkdir -p "${output_dir}"

# G2P baseline
wget https://www.robots.ox.ac.uk/~vgg/research/kws-net/pretrained_models/G2P_baseline.pth -P "${output_dir}"

# P2G baseline
wget https://www.robots.ox.ac.uk/~vgg/research/kws-net/pretrained_models/P2G_baseline.pth -P "${output_dir}"

# KWS-Net 
wget https://www.robots.ox.ac.uk/~vgg/research/kws-net/pretrained_models/KWS_Net.pth -P "${output_dir}"
