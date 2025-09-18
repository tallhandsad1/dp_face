"""
Copyright 2020, Hao Zhu, Haotian Yang, NJU.
Bilinear model.
"""

import numpy as np
from .mesh_obj import mesh_obj
from scipy.stats import norm # Import for Gaussian noise

class facescape_bm(object):
    def __init__(self, filename):
        bm_model = np.load(filename, allow_pickle=True)
        self.shape_bm_core = bm_model['shape_bm_core']  # shape core

        # Calculating the residual converts the shape core into the residual representation
        sub_tensor = np.stack((self.shape_bm_core[:, 0, :],) * self.shape_bm_core.shape[1], 1)
        res_tensor = self.shape_bm_core - sub_tensor
        res_tensor[:, 0, :] = self.shape_bm_core[:, 0, :]
        self.shape_bm_core = res_tensor

        self.color_bm_core = bm_model['color_bm_core'] # color core
        self.color_bm_mean = bm_model['color_bm_mean'] # color mean

        self.fv_indices = bm_model['fv_indices'] # face - vertex indices
        self.ft_indices = bm_model['ft_indices'] # face - texture_coordinate indices
        self.fv_indices_front = bm_model['fv_indices_front'] # frontal face-vertex indices
        self.ft_indices_front = bm_model['ft_indices_front'] # frontal face-texture_coordinate indices
        
        self.vc_dict_front = bm_model['vc_dict_front'] # frontal vertex color dictionary
        self.v_indices_front = bm_model['v_indices_front'] # frontal vertex indices
        
        self.vert_num = bm_model['vert_num'] # vertex number
        self.face_num = bm_model['face_num'] # face number
        self.frontal_vert_num = bm_model['frontal_vert_num'] # frontal vertex number
        self.frontal_face_num = bm_model['frontal_face_num'] # frontal face number
        
        self.texcoords = bm_model['texcoords'] # texture coordinates (constant)
        self.facial_mask = bm_model['facial_mask'] # UV facial mask
        self.sym_dict = bm_model['sym_dict'] # symmetry dictionary
        self.lm_list_v16 = bm_model['lm_list_v16'] # landmark indices
        
        self.vert_10to16_dict = bm_model['vert_10to16_dict'] # vertex indices dictionary (v1.0 to v1.6)
        self.vert_16to10_dict = bm_model['vert_16to10_dict'] # vertex indices dictionary (v1.6 to v1.0)
        
        if 'id_mean' in bm_model.files:
            if bm_model['id_mean'].shape[0] == self.shape_bm_core.shape[2]: # Corrected to 50
                self.id_mean = bm_model['id_mean'] # identity factors mean
            else:
                self.id_mean = np.zeros(self.shape_bm_core.shape[2]) # Corrected to 50
                print(f"Warning: 'id_mean' dimension mismatch. Expected {self.shape_bm_core.shape[2]}, got {bm_model['id_mean'].shape[0]}. Using zeros as default.")
        else:
            # Provide a default if id_mean is not found, e.g., zeros
            self.id_mean = np.zeros(self.shape_bm_core.shape[2]) # Corrected to 50
            print("Warning: 'id_mean' not found in model file. Using zeros as default.")

        if 'id_var' in bm_model.files:
            if bm_model['id_var'].shape[0] == self.shape_bm_core.shape[2]: # Corrected to 50
                self.id_var = bm_model['id_var'] # identity factors variance
            else:
                self.id_var = np.ones(self.shape_bm_core.shape[2]) # Corrected to 50
                print(f"Warning: 'id_var' dimension mismatch. Expected {self.shape_bm_core.shape[2]}, got {bm_model['id_var'].shape[0]}. Using ones as default.")
        else:
            # Provide a default if id_var is not found, e.g., ones
            self.id_var = np.ones(self.shape_bm_core.shape[2]) # Corrected to 50
            print("Warning: 'id_var' not found in model file. Using ones as default.")
        
        # make expression GaussianMixture model
        if 'exp_gmm_weights' in bm_model.files:
            self.exp_gmm_weights = bm_model['exp_gmm_weights']
        if 'exp_gmm_means' in bm_model.files:
            self.exp_gmm_means = bm_model['exp_gmm_means']
        if 'exp_gmm_covariances' in bm_model.files:
            self.exp_gmm_covariances = bm_model['exp_gmm_covariances']
        
        if 'contour_line_right' in bm_model.files:
            self.contour_line_right = bm_model['contour_line_right'].tolist() # contour line - right
        if 'contour_line_left' in bm_model.files:
            self.contour_line_left = bm_model['contour_line_left'].tolist() # contour line - left
        if 'bottom_cand' in bm_model.files:
            self.bottom_cand = bm_model['bottom_cand'].tolist() # bottom cand

    # generate full mesh
    def gen_full(self, id_vec, exp_vec):
        verts = self.shape_bm_core.dot(id_vec).dot(exp_vec).reshape((-1, 3))
        mesh = mesh_obj()
        mesh.create(vertices = verts,
                    texcoords = self.texcoords,
                    faces_v = self.fv_indices,
                    faces_vt = self.ft_indices)
        return mesh

    def _add_gaussian_noise(self, data, epsilon, delta, sensitivity):
        """
        Adds Gaussian noise to `data` to achieve (epsilon, delta)-differential privacy.
        Assumes `sensitivity` is the L2 sensitivity of the `data` vector.

        Args:
            data (np.ndarray): The data vector to which noise will be added.
            epsilon (float): The privacy budget epsilon. Must be > 0.
            delta (float): The privacy budget delta. Must be > 0 and typically very small (e.g., 1e-5 to 1e-9).
            sensitivity (float): The L2 sensitivity of the query/data. For a vector, this is
                                 max(||D1 - D2||_2) over adjacent datasets D1, D2.

        Returns:
            np.ndarray: The data vector with added Gaussian noise.
        """
        if not (epsilon > 0):
            raise ValueError("Epsilon must be positive.")
        if not (0 < delta < 1): # Delta must be positive and less than 1
            raise ValueError("Delta must be a small positive value (e.g., 1e-5 to 1e-9).")
        if not (sensitivity > 0):
            raise ValueError("Sensitivity must be positive.")

        # Calculate standard deviation for Gaussian mechanism.
        # This formula is commonly used for (epsilon, delta)-DP with L2 sensitivity.
        # For small delta, sqrt(2 * log(1.25 / delta)) is a widely adopted factor.
        sigma = (sensitivity / epsilon) * np.sqrt(2 * np.log(1.25 / delta))
        
        noise = norm.rvs(loc=0, scale=sigma, size=data.shape)
        return data + noise

    def gen_full_dp(self, id_vec_clean, exp_vec, epsilon, delta, id_vec_sensitivity=None):
        """
        Generates a full 3D mesh with differential privacy applied to the identity vector.
        Gaussian noise is added to the id_vec.

        Args:
            id_vec_clean (np.ndarray): The clean (original) identity vector.
            exp_vec (np.ndarray): The expression vector.
            epsilon (float): The privacy budget epsilon for the Gaussian mechanism.
            delta (float): The privacy budget delta for the Gaussian mechanism.
            id_vec_sensitivity (float, optional): The L2 sensitivity of the id_vec.
                                                  If None, it defaults to sqrt(id_vec.shape[0])
                                                  assuming a per-component sensitivity of 1.0.

        Returns:
            mesh_obj: The generated 3D mesh with noisy identity.
        """
        if id_vec_sensitivity is None:
            # Default L2 sensitivity for id_vec, assuming per-component sensitivity of 1
            # and id_vec has `d` dimensions.
            id_vec_sensitivity = np.sqrt(id_vec_clean.shape[0])
            print(f"Using default L2 sensitivity for id_vec: {id_vec_sensitivity}")

        # Add Gaussian noise to the identity vector
        id_vec_noisy = self._add_gaussian_noise(id_vec_clean, epsilon, delta, id_vec_sensitivity)

        # Generate vertices using the noisy identity vector and original expression vector
        verts = self.shape_bm_core.dot(id_vec_noisy).dot(exp_vec).reshape((-1, 3))
        mesh = mesh_obj()
        mesh.create(vertices=verts,
                    texcoords=self.texcoords,
                    faces_v=self.fv_indices,
                    faces_vt=self.ft_indices)
        return mesh
    
    # generate facial mesh
    def gen_face(self, id_vec, exp_vec):
        verts = self.shape_bm_core.dot(id_vec).dot(exp_vec).reshape((-1, 3))
        mesh = mesh_obj()
        mesh.create(vertices = verts[self.v_indices_front], 
                    texcoords = self.texcoords, 
                    faces_v = self.fv_indices_front, 
                    faces_vt = self.ft_indices_front)
        return mesh
    
    # generate facial mesh with vertex color
    def gen_face_color(self, id_vec, exp_vec, vc_vec):
        
        verts = self.shape_bm_core.dot(id_vec).dot(exp_vec).reshape((-1, 3))
        vert_colors = self.color_bm_mean + self.color_bm_core.dot(vc_vec)
        vert_colors = vert_colors.reshape((-1, 3)) / 255
        mesh = mesh_obj()
        
        new_vert_colors = vert_colors[self.vc_dict_front][:,[2,1,0]]
        new_vert_colors[(self.vc_dict_front == -1)] = np.array([0, 0, 0], dtype = np.float32)
        
        mesh.create(vertices = verts[self.v_indices_front], 
                    vert_colors = new_vert_colors,
                    texcoords = self.texcoords, 
                    faces_v = self.fv_indices_front, 
                    faces_vt = self.ft_indices_front)
        return mesh

