import numpy as np

def compute_pose_features(keypoints):
    """
    keypoints: (17 or 33 joints) array from YOLO pose
    returns: 128D feature vector
    """

    # If no keypoints detected → return zero vector
    if keypoints is None or len(keypoints) == 0:
        return np.zeros(128, dtype=np.float32)

    # Normalize keypoints relative to first joint (pelvis/hip)
    center = keypoints[0]
    norm = keypoints - center  # shape: (num_joints, 2)

    # Flatten normalized keypoints
    flat = norm.flatten()

    # Helper: distance between 2 joints
    def dist(a, b):
        return np.linalg.norm(norm[a] - norm[b])

    # Distances between major joints (14 values)
    distances = [
        dist(0, 5), dist(0, 6),     # hip to shoulders
        dist(5, 7), dist(6, 8),     # upper arms
        dist(7, 9), dist(8,10),     # lower arms
        dist(11,13), dist(12,14),   # thighs
        dist(13,15), dist(14,16)    # legs
    ]

    # Combine
    features = np.concatenate([flat, distances], axis=0)

    # Ensure feature length = 128 (trim or pad)
    if len(features) < 128:
        features = np.pad(features, (0, 128 - len(features)))
    else:
        features = features[:128]

    return features.astype(np.float32)
