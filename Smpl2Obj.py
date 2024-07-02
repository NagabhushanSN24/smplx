# Saves the SMPL mesh as obj file. This can be used to feed transfer_model.

import datetime
import time
import traceback
import numpy as np

from pathlib import Path

import torch
import trimesh

from smplx import SMPLLayer

this_filepath = Path(__file__)
this_filename = this_filepath.stem


def smpl2obj(gender: str, models_dirpath: Path, smpl_datapath: Path, output_filepath: Path):
    smpl_data = read_smpl_data(smpl_datapath)

    shape_params = smpl_data['betas'][0]  # (10, )
    global_pose = smpl_data['global_orient'][0]  # (1, 3, 3)
    body_poses = smpl_data['body_pose'][0]  # (23, 3, 3)
    pose_matrices = np.concatenate([global_pose, body_poses], axis=0)  # (24, 3, 3)
    num_shape_params = shape_params.shape[0]  # = 10

    shape_params_tr = torch.from_numpy(shape_params).float().unsqueeze(0)
    pose_matrices_tr = torch.from_numpy(pose_matrices).float().unsqueeze(0)

    smpl_model = SMPLLayer(models_dirpath.as_posix(), num_betas=num_shape_params, gender=gender)
    smpl_output = smpl_model(betas=shape_params_tr, body_pose=pose_matrices_tr[:, 1:],
                             global_orient=pose_matrices_tr[:, :1], pose2rot=False)

    faces = smpl_model.faces
    vertices = smpl_output.vertices.detach().cpu().numpy().squeeze()
    joints = smpl_output.joints.detach().cpu().numpy().squeeze()

    vertex_colors = np.ones([vertices.shape[0], 4]) * [0.3, 0.3, 0.3, 0.8]
    # process=False to avoid creating a new mesh
    tri_mesh = trimesh.Trimesh(vertices, faces, vertex_colors=vertex_colors, process=False)
    tri_mesh.export(output_filepath)
    return


def read_smpl_data(smpl_datapath: Path):
    with np.load(smpl_datapath, allow_pickle=True) as smpl_data:
        smpl_data = {key: smpl_data[key] for key in smpl_data.files}
    return smpl_data


def demo1():
    gender = 'neutral'
    models_dirpath = Path('./models/smpl_v1_0_0')
    smpl_datapath = Path('../data/samples/IMG_0014_0000_smpl.npz')
    output_filepath = Path('../data/samples/IMG_0014_0000_smpl.obj')
    smpl2obj(gender, models_dirpath, smpl_datapath, output_filepath)
    return


def main():
    demo1()
    return


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
