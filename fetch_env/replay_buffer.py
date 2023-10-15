import copy
import re
import h5py
import torch
import numpy as np
import pickle
import os
from sklearn.cluster import KMeans
from diffusion_gymnasium_v1.utils1.d4rl_utils import sequence_dataset



class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim, max_size=int(1e6), trim_buffer_mode="kmeans"):
        self.state = None
        self.action = None
        self.reward = None
        self.terminal = None
        self.goal = None
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_size = max_size
        self.ptr = 0
        self.state_size = 0
        self.state_kmeans = None
        self.action_kmeans = None
        self.state_trimmed = None
        self.action_trimmed = None
        self.action_state_kmeans = None
        self.action_state_trimmed = None
        self.trim_buffer_mode = trim_buffer_mode
        self.data_path = None

        self.xmin = None
        self.xmax = None
        self.ymin = None
        self.ymax = None
        self.zmin = None
        self.zmax = None

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def sample_action_state_separate(self, batch_size, horizon):
        result_state = []
        for i in range(batch_size):
            ind = np.random.randint(0, self.state_size, size=horizon)
            state = torch.FloatTensor(self.state[ind]).to(self.device)
            result_state.append(state)
        result_state = tuple(result_state)
        result_state = torch.stack(result_state, 0)
        result_action = []
        for i in range(batch_size):
            ind = np.random.randint(0, self.state_size, size=horizon) # state_size is the same as action_size
            action = torch.FloatTensor(self.action[ind]).to(self.device)
            result_action.append(action)
        result_action = tuple(result_action)
        result_action = torch.stack(result_action, 0)
        return result_action, result_state

    def sample_action_state_joint(self, batch_size, horizon):
        result = []
        for i in range(batch_size):
            ind = np.random.randint(0, self.state_size, size=horizon)
            state = torch.FloatTensor(self.state[ind]).to(self.device)
            action = torch.FloatTensor(self.action[ind]).to(self.device)
            action_state = torch.cat((action, state), 1)
            result.append(action_state)
        result = tuple(result)
        result = torch.stack(result, 0)
        return result

    def get_k_means_centroid(self, k=100, file_path=None, joint_as=False):

        """Retrieve the k-means results from a file. Otherwise run k-means and save the results to a pickle file.

        Args:
            k (int, optional): the number of clusters.
            joint_as (bool, optional): whether to have action in the state vector.
        """
        if joint_as:
            if file_path is None:
                file_path = f"replay_buffer_kmeans/{self.data_path}_{self.state_dim}_{self.action_dim}_joint_k_" + str(k) + ".pkl"
            if os.path.exists(file_path):
                with open(file_path, 'rb') as f:
                    action_state = pickle.load(f)
                self.action_state_kmeans = action_state
            
            else:
                action_state = torch.cat((self.action, self.state), dim=1)
                kmeans = KMeans(n_clusters=k).fit(np.array(action_state))
                action_state_centroids = torch.tensor(kmeans.cluster_centers_)
                closest_action_state_points = [action_state[np.argmin(np.linalg.norm(action_state - centroid, axis=1))] for centroid in action_state_centroids]
                closest_action_state_tensors = [torch.tensor(point) for point in closest_action_state_points]
                self.action_state_kmeans = torch.stack(closest_action_state_tensors)

                with open(file_path, 'wb') as f:
                    pickle.dump(self.action_state_kmeans, f)
        else:
            if file_path is None:
                file_path = f"replay_buffer_kmeans/{self.data_path}_{self.state_dim}_{self.action_dim}_joint_k_" + str(k) + ".pkl"
            if os.path.exists(file_path):
                with open(file_path, 'rb') as f:
                    action_state = pickle.load(f)
                self.action_kmeans = action_state["action"]
                self.state_kmeans = action_state["state"]
            else:
                kmeans = KMeans(n_clusters=k).fit(np.array(self.action))
                action_centroids = torch.tensor(kmeans.cluster_centers_)
                kmeans = KMeans(n_clusters=k).fit(np.array(self.state))
                state_centroids = torch.tensor(kmeans.cluster_centers_)
                closest_action_points = [self.action[np.argmin(np.linalg.norm(self.action - centroid, axis=1))] for centroid in action_centroids]
                closest_state_points = [self.state[np.argmin(np.linalg.norm(self.state - centroid, axis=1))] for centroid in state_centroids]

                # Convert the closest points to PyTorch tensors and stack them
                closest_action_tensors = [torch.tensor(point) for point in closest_action_points]
                closest_state_tensors = [torch.tensor(point) for point in closest_state_points]
                self.action_kmeans = torch.stack(closest_action_tensors)
                self.state_kmeans = torch.stack(closest_state_tensors)
                action_state = {"action": self.action_kmeans, "state": self.state_kmeans}
                with open(file_path, 'wb') as f:
                    pickle.dump(action_state, f)
        print("kmeans done")


    def get_euclidean_buffer(self, d=0.1, file_path=None, joint_as=False):
        if joint_as:
            if file_path is None:
                file_path = "replay_buffer_euclidean/joint_d_" + str(d) + ".pkl"
            if os.path.exists(file_path):
                with open(file_path, 'rb') as f:
                    action_state = pickle.load(f)
                self.action_state_trimmed = action_state
            else:
                action_state = torch.cat((self.action, self.state), dim=1)
                self.action_state_trimmed = action_state[0].unsqueeze(0)
                for a_s in action_state:
                    distance = np.linalg.norm(self.action_state_trimmed - a_s, axis=1)
                    if all(x > d for x in distance):
                        self.action_state_trimmed = torch.cat((self.action_state_trimmed, a_s.unsqueeze(0)), dim=0)
                with open(file_path, 'wb') as f:
                    pickle.dump(self.action_state_trimmed, f)
        else:
            if file_path is None:
                file_path = "replay_buffer_euclidean/d_" + str(d) + ".pkl"
            if os.path.exists(file_path):
                with open(file_path, 'rb') as f:
                    action_state = pickle.load(f)
                self.action_trimmed = action_state["action"]
                self.state_trimmed = action_state["state"]
            else:
                self.state_trimmed = self.state[0].unsqueeze(0)
                for state in self.state:
                    distance = np.linalg.norm(self.state_trimmed - state, axis=1)
                    if all(x > d for x in distance):
                        self.state_trimmed = torch.cat((self.state_trimmed, state.unsqueeze(0)), dim=0)
                self.action_trimmed = self.action[0].unsqueeze(0)
                for action in self.action:
                    distance = np.linalg.norm(self.action_trimmed - action, axis=1)
                    if all(x > d for x in distance):
                        self.action_trimmed = torch.cat((self.action_trimmed, action.unsqueeze(0)), dim=0)
                action_state = {"action": self.action_trimmed, "state": self.state_trimmed}
                with open(file_path, 'wb') as f:
                    pickle.dump(action_state, f)
        print("done")

    def convert_dict(self, file_path=None, joint_as=False, k=200, d=0.5):
        if file_path is None:
            raise ValueError("Must provide --data-path")
        else:
            pattern = r"(?:/|^)([^/]+)\.hdf5$"
            self.data_path = re.search(pattern, file_path).group(1)
            
            with h5py.File(file_path, "r") as f:
                all_obs = f['obs'][()]
                all_actions = f['actions'][()]
                all_ag = f['achieved-goals'][()]

            self.state = torch.FloatTensor(np.reshape(all_obs, (-1, all_obs.shape[-1]))[:, :self.state_dim])
            self.action = torch.FloatTensor(np.reshape(all_actions, (-1, all_actions.shape[-1]))[:, :self.action_dim])
            self.reward = np.random.rand(len(self.state))
            self.terminal = np.zeros(len(self.state), dtype=bool)
 

        self.xmin = torch.min(self.state[:, 0])
        self.xmax = torch.max(self.state[:, 0])
        self.ymin = torch.min(self.state[:, 1])
        self.ymax = torch.max(self.state[:, 1])
        self.zmin = torch.min(self.state[:, 2])
        self.zmax = torch.max(self.state[:, 2])
                
        self.state_size = self.state.shape[0]
        if self.trim_buffer_mode == "kmeans":
            self.get_k_means_centroid(k=k, joint_as=joint_as)
        elif self.trim_buffer_mode == "euclidean":
            self.get_euclidean_buffer(d=d, joint_as=joint_as)

    def get_3d_range(self):
        return self.xmin, self.xmax, self.ymin, self.ymax, self.zmin, self.zmax