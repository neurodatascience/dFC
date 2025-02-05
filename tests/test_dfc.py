import numpy as np

from pydfc import DFC


def test_create_dFC_state_based():
    ## STATE BASED ##
    # create 5 FCSs with 3 regions representing 5 states
    FCSs = np.array(
        [
            [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]],
            [[0.2, 0.3, 0.4], [-0.5, -0.6, -0.7], [0.8, 0.9, 0.1]],
            [[0.3, 0.4, 0.5], [0.6, 0.7, 0.8], [-0.9, 0.1, -0.2]],
            [[0.4, 0.5, 0.6], [-0.7, 0.8, 0.9], [0.1, 0.2, 0.3]],
            [[0.5, 0.6, 0.7], [0.8, 0.9, 0.1], [0.2, 0.3, 0.4]],
        ]
    )
    # create a random FCS_idx with 10 time points and 5 states
    FCS_idx = np.array([0, 1, 2, 3, 4, 0, 1, 2, 3, 4])

    dfc = DFC()
    dfc.set_dFC(FCSs=FCSs, FCS_idx=FCS_idx)

    assert dfc.FCSs.keys() == {"FCS1", "FCS2", "FCS3", "FCS4", "FCS5"}
    assert np.all(
        dfc.FCSs["FCS1"] == np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]])
    )
    assert np.all(
        dfc.FCSs["FCS2"]
        == np.array([[0.2, 0.3, 0.4], [-0.5, -0.6, -0.7], [0.8, 0.9, 0.1]])
    )
    assert np.all(
        dfc.FCSs["FCS3"]
        == np.array([[0.3, 0.4, 0.5], [0.6, 0.7, 0.8], [-0.9, 0.1, -0.2]])
    )
    assert np.all(
        dfc.FCSs["FCS4"] == np.array([[0.4, 0.5, 0.6], [-0.7, 0.8, 0.9], [0.1, 0.2, 0.3]])
    )
    assert np.all(
        dfc.FCSs["FCS5"] == np.array([[0.5, 0.6, 0.7], [0.8, 0.9, 0.1], [0.2, 0.3, 0.4]])
    )

    assert dfc.FCS_idx.keys() == {
        "TR0",
        "TR1",
        "TR2",
        "TR3",
        "TR4",
        "TR5",
        "TR6",
        "TR7",
        "TR8",
        "TR9",
    }
    assert dfc.FCS_idx["TR0"] == "FCS1"
    assert dfc.FCS_idx["TR1"] == "FCS2"
    assert dfc.FCS_idx["TR2"] == "FCS3"
    assert dfc.FCS_idx["TR3"] == "FCS4"
    assert dfc.FCS_idx["TR4"] == "FCS5"
    assert dfc.FCS_idx["TR5"] == "FCS1"
    assert dfc.FCS_idx["TR6"] == "FCS2"
    assert dfc.FCS_idx["TR7"] == "FCS3"
    assert dfc.FCS_idx["TR8"] == "FCS4"
    assert dfc.FCS_idx["TR9"] == "FCS5"

    assert dfc.n_regions == 3
    assert dfc.n_time == 10
    assert np.all(dfc.state_TC() == np.array([1, 2, 3, 4, 5, 1, 2, 3, 4, 5]))

    dFC_mat = dfc.get_dFC_mat()
    assert dFC_mat.shape == (10, 3, 3)
    assert np.all(
        dFC_mat[0] == np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]])
    )
    assert np.all(
        dFC_mat[1] == np.array([[0.2, 0.3, 0.4], [-0.5, -0.6, -0.7], [0.8, 0.9, 0.1]])
    )
    assert np.all(
        dFC_mat[2] == np.array([[0.3, 0.4, 0.5], [0.6, 0.7, 0.8], [-0.9, 0.1, -0.2]])
    )
    assert np.all(
        dFC_mat[3] == np.array([[0.4, 0.5, 0.6], [-0.7, 0.8, 0.9], [0.1, 0.2, 0.3]])
    )
    assert np.all(
        dFC_mat[4] == np.array([[0.5, 0.6, 0.7], [0.8, 0.9, 0.1], [0.2, 0.3, 0.4]])
    )
    assert np.all(
        dFC_mat[5] == np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]])
    )
    assert np.all(
        dFC_mat[6] == np.array([[0.2, 0.3, 0.4], [-0.5, -0.6, -0.7], [0.8, 0.9, 0.1]])
    )
    assert np.all(
        dFC_mat[7] == np.array([[0.3, 0.4, 0.5], [0.6, 0.7, 0.8], [-0.9, 0.1, -0.2]])
    )
    assert np.all(
        dFC_mat[8] == np.array([[0.4, 0.5, 0.6], [-0.7, 0.8, 0.9], [0.1, 0.2, 0.3]])
    )
    assert np.all(
        dFC_mat[9] == np.array([[0.5, 0.6, 0.7], [0.8, 0.9, 0.1], [0.2, 0.3, 0.4]])
    )

    dFC_mat = dfc.get_dFC_mat(TRs=[1, 3, 8])
    assert dFC_mat.shape == (3, 3, 3)
    assert np.all(
        dFC_mat[0] == np.array([[0.2, 0.3, 0.4], [-0.5, -0.6, -0.7], [0.8, 0.9, 0.1]])
    )
    assert np.all(
        dFC_mat[1] == np.array([[0.4, 0.5, 0.6], [-0.7, 0.8, 0.9], [0.1, 0.2, 0.3]])
    )
    assert np.all(
        dFC_mat[2] == np.array([[0.4, 0.5, 0.6], [-0.7, 0.8, 0.9], [0.1, 0.2, 0.3]])
    )

    dFC_dict = dfc.dFC2dict(TRs=[1, 3, 8])
    assert dFC_dict.keys() == {"TR1", "TR3", "TR8"}
    assert np.all(
        dFC_dict["TR1"]
        == np.array([[0.2, 0.3, 0.4], [-0.5, -0.6, -0.7], [0.8, 0.9, 0.1]])
    )
    assert np.all(
        dFC_dict["TR3"] == np.array([[0.4, 0.5, 0.6], [-0.7, 0.8, 0.9], [0.1, 0.2, 0.3]])
    )
    assert np.all(
        dFC_dict["TR8"] == np.array([[0.4, 0.5, 0.6], [-0.7, 0.8, 0.9], [0.1, 0.2, 0.3]])
    )


def test_create_dFC_state_free():
    ## STATE FREE ##
    # create 5 FCSs with 3 regions corresponding to 5 time points
    FCSs = np.array(
        [
            [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]],
            [[0.2, 0.3, 0.4], [-0.5, -0.6, -0.7], [0.8, 0.9, 0.1]],
            [[0.3, 0.4, 0.5], [0.6, 0.7, 0.8], [-0.9, 0.1, -0.2]],
            [[0.4, 0.5, 0.6], [-0.7, 0.8, 0.9], [0.1, 0.2, 0.3]],
            [[0.5, 0.6, 0.7], [0.8, 0.9, 0.1], [0.2, 0.3, 0.4]],
        ]
    )

    dfc = DFC()
    dfc.set_dFC(FCSs=FCSs)

    assert dfc.FCSs.keys() == {"FCS1", "FCS2", "FCS3", "FCS4", "FCS5"}
    assert np.all(
        dfc.FCSs["FCS1"] == np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]])
    )
    assert np.all(
        dfc.FCSs["FCS2"]
        == np.array([[0.2, 0.3, 0.4], [-0.5, -0.6, -0.7], [0.8, 0.9, 0.1]])
    )
    assert np.all(
        dfc.FCSs["FCS3"]
        == np.array([[0.3, 0.4, 0.5], [0.6, 0.7, 0.8], [-0.9, 0.1, -0.2]])
    )
    assert np.all(
        dfc.FCSs["FCS4"] == np.array([[0.4, 0.5, 0.6], [-0.7, 0.8, 0.9], [0.1, 0.2, 0.3]])
    )
    assert np.all(
        dfc.FCSs["FCS5"] == np.array([[0.5, 0.6, 0.7], [0.8, 0.9, 0.1], [0.2, 0.3, 0.4]])
    )

    assert dfc.FCS_idx.keys() == {"TR0", "TR1", "TR2", "TR3", "TR4"}
    assert dfc.FCS_idx["TR0"] == "FCS1"
    assert dfc.FCS_idx["TR1"] == "FCS2"
    assert dfc.FCS_idx["TR2"] == "FCS3"
    assert dfc.FCS_idx["TR3"] == "FCS4"
    assert dfc.FCS_idx["TR4"] == "FCS5"

    assert dfc.n_regions == 3
    assert dfc.n_time == 5

    dFC_mat = dfc.get_dFC_mat()
    assert dFC_mat.shape == (5, 3, 3)
    assert np.all(
        dFC_mat[0] == np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]])
    )
    assert np.all(
        dFC_mat[1] == np.array([[0.2, 0.3, 0.4], [-0.5, -0.6, -0.7], [0.8, 0.9, 0.1]])
    )
    assert np.all(
        dFC_mat[2] == np.array([[0.3, 0.4, 0.5], [0.6, 0.7, 0.8], [-0.9, 0.1, -0.2]])
    )
    assert np.all(
        dFC_mat[3] == np.array([[0.4, 0.5, 0.6], [-0.7, 0.8, 0.9], [0.1, 0.2, 0.3]])
    )
    assert np.all(
        dFC_mat[4] == np.array([[0.5, 0.6, 0.7], [0.8, 0.9, 0.1], [0.2, 0.3, 0.4]])
    )

    dFC_mat = dfc.get_dFC_mat(TRs=[1, 3, 4])
    assert dFC_mat.shape == (3, 3, 3)
    assert np.all(
        dFC_mat[0] == np.array([[0.2, 0.3, 0.4], [-0.5, -0.6, -0.7], [0.8, 0.9, 0.1]])
    )
    assert np.all(
        dFC_mat[1] == np.array([[0.4, 0.5, 0.6], [-0.7, 0.8, 0.9], [0.1, 0.2, 0.3]])
    )
    assert np.all(
        dFC_mat[2] == np.array([[0.5, 0.6, 0.7], [0.8, 0.9, 0.1], [0.2, 0.3, 0.4]])
    )

    dFC_dict = dfc.dFC2dict(TRs=[1, 3, 4])
    assert dFC_dict.keys() == {"TR1", "TR3", "TR4"}
    assert np.all(
        dFC_dict["TR1"]
        == np.array([[0.2, 0.3, 0.4], [-0.5, -0.6, -0.7], [0.8, 0.9, 0.1]])
    )
    assert np.all(
        dFC_dict["TR3"] == np.array([[0.4, 0.5, 0.6], [-0.7, 0.8, 0.9], [0.1, 0.2, 0.3]])
    )
    assert np.all(
        dFC_dict["TR4"] == np.array([[0.5, 0.6, 0.7], [0.8, 0.9, 0.1], [0.2, 0.3, 0.4]])
    )
