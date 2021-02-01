import numpy as np


def quaterniont_to_aa(qua: np.ndarray):
    """
        qua: [w, x, y, z]
    """
    assert len(qua) == 4
    qua /= np.linalg.norm(qua)
    w, x, y, z = qua[0], qua[1], qua[2], qua[3]
    theta = np.arccos(w) * 2
    ax = x / np.sin(theta / 2)
    ay = y / np.sin(theta / 2)
    az = z / np.sin(theta / 2)

    aa = np.array([ax, ay, az]) * theta

    for i in aa:
        # print(np.isnan(i))
        assert np.isnan(i) == False, f"{aa}"
    # assert np.isnan(aa).any() is False, f"{aa}"

    return aa
