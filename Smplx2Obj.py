# Saves the SMPL-X mesh as obj file. This can be used to feed transfer_model.
# https://github.com/vchoutas/smplx/issues/190
# https://github.com/XueYing126/smplx2smpl-shape/blob/c88c67d0e52229c3a854a16cf7eed18366e58de9/smplx_obj.py

import datetime
import time
import traceback
import numpy as np

from pathlib import Path

import torch
import trimesh

from smplx import SMPLXLayer

this_filepath = Path(__file__)
this_filename = this_filepath.stem


def smplx2obj(gender: str, models_dirpath: Path, smplx_datapath: Path, output_filepath: Path):
    smplx_data = read_smplx_data(smplx_datapath)

    shape_params = smplx_data['shape'][0]  # (11, )
    pose_matrices = smplx_data['body_pose'][0]  # (22, 3, 3)
    num_shape_params = shape_params.shape[0]  # = 11

    shape_params_tr = torch.from_numpy(shape_params).float().unsqueeze(0)
    pose_matrices_tr = torch.from_numpy(pose_matrices).float().unsqueeze(0)

    smplx_model = SMPLXLayer(models_dirpath.as_posix(), num_betas=num_shape_params, gender=gender)
    smplx_output = smplx_model(betas=shape_params_tr, body_pose=pose_matrices_tr[:, 1:],
                               global_orient=pose_matrices_tr[:, :1], pose2rot=False)

    faces = smplx_model.faces
    vertices = smplx_output.vertices.detach().cpu().numpy().squeeze()
    joints = smplx_output.joints.detach().cpu().numpy().squeeze()

    vertex_colors = np.ones([vertices.shape[0], 4]) * [0.3, 0.3, 0.3, 0.8]
    # process=False to avoid creating a new mesh
    tri_mesh = trimesh.Trimesh(vertices, faces, vertex_colors=vertex_colors, process=False)
    tri_mesh.export(output_filepath)
    return


def read_smplx_data(smplx_datapath: Path):
    with np.load(smplx_datapath) as smplx_data:
        smplx_data = {key: smplx_data[key] for key in smplx_data.files}
    return smplx_data


def demo1():
    gender = 'neutral'
    models_dirpath = Path('./models/smplx_v1_1')
    smplx_datapath = Path('../data/samples/IMG_0014_0000_smplx.npz')
    output_filepath = Path('../data/samples/IMG_0014_0000_smplx.obj')
    smplx2obj(gender, models_dirpath, smplx_datapath, output_filepath)
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
