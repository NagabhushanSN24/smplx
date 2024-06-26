# Converts SMPL-X to SMPL

import argparse
import datetime
import os
import pickle
import shutil
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import trimesh
import yaml

from smplx import SMPLXLayer

this_filepath = Path(__file__)
this_filename = this_filepath.stem


def execute_command(cmd: str):
    print(cmd)
    os.system(cmd)
    return


def clean_directory(dirpath: Path):
    if dirpath.exists():
        shutil.rmtree(dirpath)
    dirpath.mkdir(parents=True, exist_ok=True)
    return


def smplx2obj(gender: str, models_dirpath: Path, smplx_datapath: Path, output_filepath: Path):
    smplx_data = read_smplx_data(smplx_datapath)

    shape_params = smplx_data['betas'][0]  # (11, )
    global_pose_matrix = smplx_data['global_orient'][0]  # (1, 3, 3)
    body_pose_matrices = smplx_data['body_pose'][0]  # (21, 3, 3)
    pose_matrices = np.concatenate([global_pose_matrix, body_pose_matrices], axis=0)  # (22, 3, 3)
    num_shape_params = shape_params.shape[0]  # = 11

    shape_params_tr = torch.from_numpy(shape_params).float().unsqueeze(0)
    pose_matrices_tr = torch.from_numpy(pose_matrices).float().unsqueeze(0)

    smplx_model = SMPLXLayer(models_dirpath.as_posix(), num_betas=num_shape_params, gender=gender)
    smplx_output = smplx_model(betas=shape_params_tr, body_pose=pose_matrices_tr[:, 1:],
                               global_orient=pose_matrices_tr[:, :1], pose2rot=False)

    faces = smplx_model.faces
    vertices = smplx_output.vertices.detach().cpu().numpy().squeeze()

    vertex_colors = np.ones([vertices.shape[0], 4]) * [0.3, 0.3, 0.3, 0.8]
    # process=False to avoid creating a new mesh
    tri_mesh = trimesh.Trimesh(vertices, faces, vertex_colors=vertex_colors, process=False)
    tri_mesh.export(output_filepath)
    return


def save_configs(config_filepath: Path, meshes_dirpath: Path, gender, smpl_version: str, smpl_num_betas: int, output_dirpath: Path):
    configs_dict = {
        'datasets': {
            'mesh_folder': {
                'data_folder': meshes_dirpath.as_posix(),
            }
        },
        'deformation_transfer_path': 'transfer_data/smplx2smpl_deftrafo_setup.pkl',
        'mask_ids_fname': '',
        'summary_steps': 100,

        'edge_fitting': {
            'per_part': False
        },

        'optim': {
            'type': 'lbfgs',
            'maxiters': 200,
            'gtol': 1e-06,
            'ftol': 1e-10,
        },

        'body_model': {
            'model_type': 'smpl',
            'gender': gender,
            'ext': 'pkl',
            'folder': f'./models/smpl_{smpl_version}',
            'use_compressed': False,
            'use_face_contour': True,
            'smpl': {
                'betas': {
                    'num': smpl_num_betas
                }
            }
        },

        'output_folder': output_dirpath.as_posix()
    }
    with open(config_filepath.as_posix(), 'w') as configs_file:
        yaml.dump(configs_dict, configs_file, default_flow_style=False)
    return


def read_smplx_data(smplx_datapath: Path):
    with np.load(smplx_datapath, allow_pickle=True) as smplx_data:
        smplx_data = {key: smplx_data[key] for key in smplx_data.files}
    return smplx_data


def read_smpl_data(smpl_pkl_filepath: Path):
    with open(smpl_pkl_filepath, 'rb') as smpl_pkl_file:
        smpl_data = pickle.load(smpl_pkl_file)
    return smpl_data


def save_smpl_data(smpl_data, output_filepath: Path):
    smpl_dict = {}
    for key in smpl_data:
        if isinstance(smpl_data[key], torch.Tensor):
            smpl_dict[key] = smpl_data[key].detach().cpu().numpy()
        else:
            smpl_dict[key] = smpl_data[key]
    np.savez_compressed(output_filepath, **smpl_dict)
    return


def main():
    # gender = 'female'
    # models_dirpath = Path('./models/smplx_v1_1')
    # smplx_datapath = Path('../data/samples/IMG_0014_0000_smplx.npz')
    # output_filepath = Path('../data/samples/IMG_0014_0000_smplx.obj')
    # smplx2obj(gender, models_dirpath, smplx_datapath, output_filepath)
    args = parse_args()

    tmp_dirpath = Path('../tmp')
    clean_directory(tmp_dirpath)

    smplx_datapath = Path(args.smplx_datapath)
    # Save the SMPL-X mesh
    meshes_dirpath = tmp_dirpath / 'meshes'
    meshes_dirpath.mkdir(parents=True, exist_ok=True)
    obj_filepath = meshes_dirpath / f'{smplx_datapath.stem}.obj'
    smplx2obj(args.gender, Path(f'./models/smplx_{args.smplx_version}'), smplx_datapath, obj_filepath)

    # Save configs for SMPL-X to SMPL conversion
    config_filepath = tmp_dirpath / f'{smplx_datapath.stem}_config.yaml'
    smpl_pkl_filepath = tmp_dirpath / f'{smplx_datapath.stem}.pkl'
    save_configs(config_filepath, meshes_dirpath, args.gender, args.smpl_version, args.smpl_num_betas, smpl_pkl_filepath.parent)

    # Run the SMPL-X to SMPL conversion
    cmd = f'python -m transfer_model --exp-cfg {config_filepath.as_posix()}'
    execute_command(cmd)

    # Save the SMPL data in npz file
    smpl_data = read_smpl_data(smpl_pkl_filepath)
    save_smpl_data(smpl_data, Path(args.output_filepath))
    return


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smplx_datapath', type=str, help='path to SMPL-X data file')
    parser.add_argument('--gender', type=str, default='neutral', choices=['male', 'female', 'neutral'], help='gender of the SMPL-X model')
    parser.add_argument('--smplx_version', type=str, default='v1_1', choices=['v1_1'], help='version of the SMPL-X model')
    parser.add_argument('--smpl_version', type=str, default='v1_0_0', choices=['v1_0_0', 'v1_1_0'], help='version of the SMPL model')
    parser.add_argument('--smpl_num_betas', type=int, default=10, help='number of shape parameters in the SMPL model')
    parser.add_argument('--output_filepath', type=str, help='path to output SMPL data file')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    print('Program started at ' + datetime.datetime.now().strftime('%d/%m/%Y %I:%M:%S %p'))
    start_time = time.time()
    try:
        main()
        run_result = 'Program completed successfully!'
    except Exception as e:
        print(e)
        traceback.print_exc()
        run_result = 'Error: ' + str(e)
    end_time = time.time()
    print('Program ended at ' + datetime.datetime.now().strftime('%d/%m/%Y %I:%M:%S %p'))
    print('Execution time: ' + str(datetime.timedelta(seconds=end_time - start_time)))
