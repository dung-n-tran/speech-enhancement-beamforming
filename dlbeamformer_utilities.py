import numpy as np
from config import sound_speed
def compute_steering_vector_ULA(u, microphone_array):
    return np.exp(-1j*2*np.pi*microphone_array.geometry*u).reshape((microphone_array.n_mics, 1))

def compute_steering_vector_ULA_new(microphone_array, angle, frequency):
    u = np.cos(angle * np.pi/180)
    return np.exp(-1j*2*np.pi*(frequency/sound_speed)*microphone_array.geometry*u).reshape((microphone_array.n_mics, 1))

def compute_MVDR_weight(source_steering_vector, signals):
    snapshot = signals.shape[1]
    sample_covariance_matrix = signals.dot(signals.transpose().conjugate()) / snapshot
    inverse_sample_covariance_matrix = np.linalg.inv(sample_covariance_matrix)
    normalization_factor = (source_steering_vector.transpose().conjugate().dot(inverse_sample_covariance_matrix).dot(source_steering_vector))
    weight = inverse_sample_covariance_matrix.dot(source_steering_vector) / normalization_factor
    return weight

def check_distortless_constraint(weight, source_steering_vector):
    assert(np.abs(weight.transpose().conjugate().dot(source_steering_vector)) - 1 < 1e-9)

def uniform_linear_array(n_mics, spacing):
    return spacing*np.arange(-(n_mics-1)/2, (n_mics-1)/2+1).reshape(1, n_mics)

def generate_gaussian_samples(power, shape):
    return np.sqrt(power/2)*np.random.randn(shape[0], shape[1]) + 1j*np.sqrt(power/2)*np.random.randn(shape[0], shape[1]); # signal samples

def generate_gaussian_training_data(microphone_array, training_snapshots, n_interference_list, u_list, interference_power_list, n_training_samples, noise_sigma):
    import itertools
    training_noise_interference_data_various_snapshots = []
    for training_snapshot in training_snapshots:
        training_noise_interference_data = []
        for n_interferences in n_interference_list:
            interferences_params = []
            for i_interference in range(n_interferences):
                interference_params = list(itertools.product(*[u_list, interference_power_list]))
                interferences_params.append(interference_params)
            interferences_param_sets = list(itertools.product(*interferences_params))        

            # for param_set in interferences_param_sets:
            for param_set in interferences_param_sets:
                for i_training_sample in range(n_training_samples):
                    nv = np.zeros((microphone_array.n_mics, training_snapshot), dtype=complex)
                    for i_interference in range(len(param_set)):
                        u, interference_power = param_set[i_interference]
                        vi = compute_steering_vector_ULA(u, microphone_array)
                        sigma = 10**(interference_power/10)
                        ii = generate_gaussian_samples(power=sigma, shape=(1, training_snapshot))
                        nv += vi.dot(ii)
                    noise = generate_gaussian_samples(power=noise_sigma, shape=(microphone_array.n_mics, training_snapshot))
                    nv += noise
                    training_noise_interference_data.append(nv)
        training_noise_interference_data_various_snapshots.append(training_noise_interference_data)
    return training_noise_interference_data_various_snapshots