import numpy as np

from . import bvh_helper
from . import math3d


class NTURGBDSkeleton(object):
    def __init__(self):
        self.root = 'SpineBase'
        self.keypoint2index = {
            'SpineBase': 0,
            'SpineMiddle': 1,
            'Neck': 2,
            'Head': 3,
            'LeftShoulder': 4,
            'LeftElbow': 5,
            'LeftWrist': 6,
            'LeftHand': 7,
            'RightShoulder': 8,
            'RightElbow': 9,
            'RightWrist': 10,
            'RightHand': 11,
            'LeftHip': 12,
            'LeftKnee': 13,
            'LeftAnkle': 14,
            'LeftFoot': 15,
            'RightHip': 16,
            'RightKnee': 17,
            'RightAnkle': 18,
            'RightFoot': 19,
            'Spine': 20,
            'LeftHandTip': 21,
            'LeftThumb': 22,
            'RightHandTip': 23,
            'RightThumb': 24,
        }

        self.children = {
            'SpineBase': ['SpineMiddle', 'LeftHip', 'RightHip'],
            'SpineMiddle': ['Spine'],
            'Neck': ['Head'],
            'Head': [],
            'LeftShoulder': ['LeftElbow'],
            'LeftElbow': ['LeftWrist'],
            'LeftWrist': ['LeftHand'],
            'LeftHand': ['LeftHandTip', 'LeftThumb'],
            'RightShoulder': ['RightElbow'],
            'RightElbow': ['RightWrist'],
            'RightWrist': ['RightHand'],
            'RightHand': ['RightHandTip', 'RightThumb'],
            'LeftHip': ['LeftKnee'],
            'LeftKnee': ['LeftAnkle'],
            'LeftAnkle': ['LeftFoot'],
            'LeftFoot': [],
            'RightHip': ['RightKnee'],
            'RightKnee': ['RightAnkle'],
            'RightAnkle': ['RightFoot'],
            'RightFoot': [],
            'Spine': ['Neck', 'LeftShoulder', 'RightShoulder'],
            'LeftHandTip': [],
            'LeftThumb': [],
            'RightHandTip': [],
            'RightThumb': [],
        }
        self.parent = {self.root: None}
        for parent, children in self.children.items():
            for child in children:
                self.parent[child] = parent

        # T-pose
        self.initial_directions = {
            'SpineBase': [0, 0, 0],
            'SpineMiddle': [0, 0, 1],
            'Neck': [0, 0, 1],
            'Head': [0, 0, 1],
            'LeftShoulder': [1, 0, 0],
            'LeftElbow': [1, 0, 0],
            'LeftWrist': [1, 0, 0],
            'LeftHand': [1, 0, 0],
            'RightShoulder': [-1, 0, 0],
            'RightElbow': [-1, 0, 0],
            'RightWrist': [-1, 0, 0],
            'RightHand': [-1, 0, 0],
            'LeftHip': [1, 0, 0],
            'LeftKnee': [0, 0, -1],
            'LeftAnkle': [0, 0, -1],
            'LeftFoot': [0, -1, 0],
            'RightHip': [-1, 0, 0],
            'RightKnee': [0, 0, -1],
            'RightAnkle': [0, 0, -1],
            'RightFoot': [0, -1, 0],
            'Spine': [0, 0, 1],
            'LeftHandTip': [1, 0, 0],
            'LeftThumb': [1, 0, 0],
            'RightHandTip': [-1, 0, 0],
            'RightThumb': [-1, 0, 0],
        }

    def get_initial_offset(self, poses_3d):
        # TODO: RANSAC
        bone_lens = {self.root: [0]}
        stack = [self.root]
        while stack:
            parent = stack.pop()
            p_idx = self.keypoint2index[parent]
            p_name = parent
            while p_idx == -1:
                # find real parent
                p_name = self.parent[p_name]
                p_idx = self.keypoint2index[p_name]
            for child in self.children[parent]:
                stack.append(child)

                if self.keypoint2index[child] == -1:
                    bone_lens[child] = [0.1]
                else:
                    c_idx = self.keypoint2index[child]
                    bone_lens[child] = np.linalg.norm(
                        poses_3d[:, p_idx] - poses_3d[:, c_idx],
                        axis=1
                    )

        bone_len = {}
        for joint in self.keypoint2index:
            mean = np.mean(np.trim_zeros(bone_lens[joint]))
            if np.isnan(mean):
                bone_len[joint] = np.array([0])
            else:
                bone_len[joint] = mean

        initial_offset = {}
        for joint, direction in self.initial_directions.items():
            direction = np.array(direction) / max(np.linalg.norm(direction), 1e-12)
            initial_offset[joint] = direction * bone_len[joint]

        return initial_offset

    def get_bvh_header(self, poses_3d):
        initial_offset = self.get_initial_offset(poses_3d)

        nodes = {}
        for joint in self.keypoint2index:
            is_root = joint == self.root
            is_end_site = 'EndSite' in joint
            nodes[joint] = bvh_helper.BvhNode(
                name=joint,
                offset=initial_offset[joint],
                rotation_order='zxy' if not is_end_site else '',
                is_root=is_root,
                is_end_site=is_end_site,
            )
        for joint, children in self.children.items():
            nodes[joint].children = [nodes[child] for child in children]
            for child in children:
                nodes[child].parent = nodes[joint]

        header = bvh_helper.BvhHeader(root=nodes[self.root], nodes=nodes)
        return header

    def pose2euler(self, pose, header):
        channel = []
        quats = {}
        eulers = {}
        stack = [header.root]
        while stack:
            node = stack.pop()
            joint = node.name
            joint_idx = self.keypoint2index[joint]

            if node.is_root:
                channel.extend(pose[joint_idx])

            index = self.keypoint2index
            order = None
            x_dir = None
            y_dir = None
            z_dir = None
            if joint in ['SpineBase', 'SpineMiddle']:
                child_idx = self.keypoint2index[node.children[0].name]
                x_dir = pose[index['LeftHip']] - pose[index['RightHip']]
                z_dir = pose[child_idx] - pose[joint_idx]
                order = 'zyx'
            elif joint in ['RightHip', 'RightKnee', 'RightAnkle']:
                child_idx = self.keypoint2index[node.children[0].name]
                x_dir = pose[index['SpineBase']] - pose[index['RightHip']]
                z_dir = pose[joint_idx] - pose[child_idx]
                order = 'zyx'
            elif joint in ['LeftHip', 'LeftKnee', 'LeftAnkle']:
                child_idx = self.keypoint2index[node.children[0].name]
                x_dir = pose[index['LeftHip']] - pose[index['SpineBase']]
                z_dir = pose[joint_idx] - pose[child_idx]
                order = 'zyx'
            elif joint == 'Spine':
                x_dir = pose[index['LeftShoulder']] - pose[index['RightShoulder']]
                z_dir = pose[joint_idx] - pose[index['SpineMiddle']]
                order = 'zyx'
            elif joint == 'Neck':
                y_dir = pose[index['Spine']] - pose[joint_idx]
                z_dir = pose[index['Head']] - pose[index['Spine']]
                order = 'zxy'
            elif joint == 'LeftShoulder':
                child_idx = self.keypoint2index[node.children[0].name]
                x_dir = pose[child_idx] - pose[joint_idx]
                y_dir = pose[child_idx] - pose[index['LeftHand']]
                order = 'xzy'
            elif joint in ['LeftElbow', 'LeftWrist']:
                child_idx = self.keypoint2index[node.children[0].name]
                x_dir = pose[child_idx] - pose[joint_idx]
                y_dir = pose[joint_idx] - pose[index['LeftShoulder']]
                order = 'xzy'
            elif joint == 'RightShoulder':
                child_idx = self.keypoint2index[node.children[0].name]
                x_dir = pose[joint_idx] - pose[child_idx]
                y_dir = pose[child_idx] - pose[index['RightHand']]
                order = 'xzy'
            elif joint in ['RightElbow', 'RightWrist']:
                child_idx = self.keypoint2index[node.children[0].name]
                x_dir = pose[joint_idx] - pose[child_idx]
                y_dir = pose[joint_idx] - pose[index['RightShoulder']]
                order = 'xzy'

            if order:
                dcm = math3d.dcm_from_axis(x_dir, y_dir, z_dir, order)
                quats[joint] = math3d.dcm2quat(dcm)
            else:
                quats[joint] = quats[self.parent[joint]].copy()

            local_quat = quats[joint].copy()
            if node.parent:
                local_quat = math3d.quat_divide(
                    q=quats[joint], r=quats[node.parent.name]
                )

            euler = math3d.quat2euler(
                q=local_quat, order=node.rotation_order
            )
            euler = np.rad2deg(euler)
            eulers[joint] = euler
            channel.extend(euler)

            for child in node.children[::-1]:
                if not child.is_end_site:
                    stack.append(child)

        return channel

    def poses2bvh(self, poses_3d, header=None, output_file=None):
        if not header:
            header = self.get_bvh_header(poses_3d)

        channels = []

        for frame, pose in enumerate(poses_3d):
            channels.append(self.pose2euler(pose, header))

        if output_file:
            bvh_helper.write_bvh(output_file, header, channels)

        return channels, header
