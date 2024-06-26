#!/bin/bash
urle () { [[ "${1}" ]] || return 1; local LANG=C i x; for (( i = 0; i < ${#1}; i++ )); do x="${1:i:1}"; [[ "${x}" == [a-zA-Z0-9.~-] ]] && echo -n "${x}" || printf '%%%02X' "'${x}"; done; echo; }

# SMPL-X model login data
echo -e "\nYou need to register at https://smpl-x.is.tue.mpg.de"
read -p "Username (SMPL-X):" username
read -p "Password (SMPL-X):" password
username=$(urle $username)
password=$(urle $password)

mkdir -p ../models
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=smplx&sfile=models_smplx_v1_1.zip' -O '../models/models_smplx_v1_1.zip' --no-check-certificate --continue
unzip ../models/models_smplx_v1_1.zip -d ../models/
mv ../models/models/smplx ../models/smplx_v1_1
rm ../models/models_smplx_v1_1.zip
rmdir ../models/models

wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=smpl&sfile=SMPL_python_v.1.0.0.zip' -O '../models/SMPL_python_v.1.0.0.zip' --no-check-certificate --continue
unzip ../models/SMPL_python_v.1.0.0.zip -d ../models/
mv ../models/smpl/models ../models/smpl_v1_0_0
rm ../models/SMPL_python_v.1.0.0.zip
rm -rf ../models/smpl

wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=smplify&resume=1&sfile=mpips_smplify_public_v2.zip' -O '../models/mpips_smplify_public_v2.zip' --no-check-certificate --continue
unzip ../models/mpips_smplify_public_v2.zip -d ../models/
mv ../models/smplify_public/code/models/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl ../models/smpl_v1_0_0/
rm ../models/mpips_smplify_public_v2.zip
rm -rf ../models/smplify_public
rm -rf ../models/_MACOSX

mv ../models/smpl_v1_0_0/basicModel_f_lbs_10_207_0_v1.0.0.pkl ../models/smpl_v1_0_0/SMPL_FEMALE.pkl
mv ../models/smpl_v1_0_0/basicmodel_m_lbs_10_207_0_v1.0.0.pkl ../models/smpl_v1_0_0/SMPL_MALE.pkl
mv ../models/smpl_v1_0_0/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl ../models/smpl_v1_0_0/SMPL_NEUTRAL.pkl

wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=smpl&sfile=SMPL_python_v.1.1.0.zip' -O '../models/SMPL_python_v.1.1.0.zip' --no-check-certificate --continue
unzip ../models/SMPL_python_v.1.1.0.zip -d ../models/
mv ../models/SMPL_python_v.1.1.0/smpl/models ../models/smpl_v1_1_0
rm ../models/SMPL_python_v.1.1.0.zip
rm -rf ../models/SMPL_python_v.1.1.0

mv ../models/smpl_v1_1_0/basicmodel_f_lbs_10_207_0_v1.1.0.pkl ../models/smpl_v1_1_0/SMPL_FEMALE.pkl
mv ../models/smpl_v1_1_0/basicmodel_m_lbs_10_207_0_v1.1.0.pkl ../models/smpl_v1_1_0/SMPL_MALE.pkl
mv ../models/smpl_v1_1_0/basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl ../models/smpl_v1_1_0/SMPL_NEUTRAL.pkl

