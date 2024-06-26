#!/bin/bash
urle () { [[ "${1}" ]] || return 1; local LANG=C i x; for (( i = 0; i < ${#1}; i++ )); do x="${1:i:1}"; [[ "${x}" == [a-zA-Z0-9.~-] ]] && echo -n "${x}" || printf '%%%02X' "'${x}"; done; echo; }

# SMPL-X model login data
echo -e "\nYou need to register at https://smpl-x.is.tue.mpg.de"
read -p "Username (SMPL-X):" username
read -p "Password (SMPL-X):" password
username=$(urle $username)
password=$(urle $password)

mkdir -p ../../transfer_data
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=smplx&sfile=model_transfer.zip' -O '../../transfer_data/model_transfer.zip' --no-check-certificate --continue
unzip ../../transfer_data/model_transfer.zip -d ../../transfer_data/
rm ../../transfer_data/model_transfer.zip

wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=smplx&sfile=sample_transfer_data.zip' -O '../../transfer_data/sample_transfer_data.zip' --no-check-certificate --continue
unzip ../../transfer_data/sample_transfer_data.zip -d ../../transfer_data/
rm ../../transfer_data/sample_transfer_data.zip
