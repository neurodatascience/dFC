import numpy as np
import pytest

from pydfc import TIME_SERIES


def test_create_time_series():
    # create a random data with 100 regions and 1000 time points
    data = np.random.rand(100, 1000)
    # create a random locs with 100 regions and 3 coordinates
    locs = np.random.rand(100, 3)
    # create a random node_labels list with 100 regions
    node_labels = [f"Region {i}" for i in range(100)]
    time_series = TIME_SERIES(
        data=data,
        subj_id="sub-0001",
        Fs=2,
        locs=locs,
        node_labels=node_labels,
    )

    assert time_series.data.shape == (100, 1000)
    assert time_series.subj_id_lst == ["sub-0001"]
    assert time_series.Fs == 2
    assert time_series.n_time == 1000
    assert time_series.n_regions == 100
    assert np.all(time_series.nodes_lst == np.arange(0, time_series.n_regions, dtype=int))
    assert np.all(time_series.time == 1 / 2 + np.arange(0, 1000 / 2, 1 / 2))
    assert time_series.data_dict.keys() == {"sub-0001"}
    assert np.all(time_series.data_dict["sub-0001"]["data"] == data)


def test_append_ts():
    # create a random data with 100 regions and 1000 time points
    data_1 = np.random.rand(100, 1000)
    # create a random locs with 100 regions and 3 coordinates
    locs = np.random.rand(100, 3)
    # create a random node_labels list with 100 regions
    node_labels = [f"Region {i}" for i in range(100)]
    time_series = TIME_SERIES(
        data=data_1,
        subj_id="sub-0001",
        Fs=2,
        locs=locs,
        node_labels=node_labels,
    )

    # create a random data with 100 regions and 1000 time points
    data_2 = np.random.rand(100, 1000)
    time_series.append_ts(
        new_time_series=data_2,
        subj_id="sub-0002",
    )

    assert time_series.data.shape == (100, 2000)
    assert time_series.subj_id_lst == ["sub-0001", "sub-0002"]
    assert time_series.Fs == 2
    assert time_series.n_time is None
    assert time_series.n_regions == 100
    assert np.all(time_series.nodes_lst == np.arange(0, time_series.n_regions, dtype=int))
    assert time_series.time is None
    assert time_series.data_dict.keys() == {"sub-0001", "sub-0002"}
    assert np.all(time_series.data_dict["sub-0001"]["data"] == data_1)
    assert np.all(time_series.data_dict["sub-0002"]["data"] == data_2)
    assert np.all(
        time_series.data_dict["sub-0001"]["time_array"]
        == 1 / 2 + np.arange(0, 1000 / 2, 1 / 2)
    )
    assert np.all(
        time_series.data_dict["sub-0002"]["time_array"]
        == 1 / 2 + np.arange(0, 1000 / 2, 1 / 2)
    )

    # check if visualization will raise warning correctly
    with pytest.warns(
        UserWarning, match="Multiple subjects are not supported in visualization."
    ):
        time_series.visualize()


def test_concat_ts():
    # create a random data with 100 regions and 1000 time points
    data_1 = np.random.rand(100, 1000)
    # create a random locs with 100 regions and 3 coordinates
    locs = np.random.rand(100, 3)
    # create a random node_labels list with 100 regions
    node_labels = [f"Region {i}" for i in range(100)]
    time_series = TIME_SERIES(
        data=data_1,
        subj_id="sub-0001",
        Fs=2,
        locs=locs,
        node_labels=node_labels,
    )

    # create a random data with 100 regions and 1000 time points
    data_2 = np.random.rand(100, 1000)
    locs_2 = np.random.rand(100, 3)
    node_labels_2 = [f"Parcel {i}" for i in range(100)]

    time_series_2 = TIME_SERIES(
        data=data_2,
        subj_id="sub-0002",
        Fs=1,
        locs=locs,
        node_labels=node_labels,
    )

    # check Fs mismatch assert error
    with pytest.raises(AssertionError, match="Fs mismatch!"):
        time_series.concat_ts(time_series_2)

    time_series_2 = TIME_SERIES(
        data=data_2,
        subj_id="sub-0002",
        Fs=2,
        locs=locs_2,
        node_labels=node_labels,
    )
    # check locs mismatch assert error
    with pytest.raises(AssertionError, match="locs mismatch!"):
        time_series.concat_ts(time_series_2)

    time_series_2 = TIME_SERIES(
        data=data_2,
        subj_id="sub-0002",
        Fs=2,
        locs=locs,
        node_labels=node_labels_2,
    )
    # check node_labels mismatch assert error
    with pytest.raises(AssertionError, match="node_labels mismatch!"):
        time_series.concat_ts(time_series_2)

    time_series_2 = TIME_SERIES(
        data=data_2,
        subj_id="sub-0002",
        Fs=2,
        locs=locs,
        node_labels=node_labels,
    )
    time_series.concat_ts(time_series_2)

    assert time_series.data.shape == (100, 2000)
    assert time_series.subj_id_lst == ["sub-0001", "sub-0002"]
    assert time_series.Fs == 2
    assert time_series.n_time is None
    assert time_series.n_regions == 100
    assert np.all(time_series.nodes_lst == np.arange(0, 100, dtype=int))
    assert time_series.time is None
    assert time_series.data_dict.keys() == {"sub-0001", "sub-0002"}
    assert np.all(time_series.data_dict["sub-0001"]["data"] == data_1)
    assert np.all(time_series.data_dict["sub-0002"]["data"] == data_2)
    assert np.all(
        time_series.data_dict["sub-0001"]["time_array"]
        == 1 / 2 + np.arange(0, 1000 / 2, 1 / 2)
    )
    assert np.all(
        time_series.data_dict["sub-0002"]["time_array"]
        == 1 / 2 + np.arange(0, 1000 / 2, 1 / 2)
    )


def test_get_subj_ts():
    # create a random data with 100 regions and 1000 time points
    data_1 = np.random.rand(100, 1000)
    # create a random locs with 100 regions and 3 coordinates
    locs = np.random.rand(100, 3)
    # create a random node_labels list with 100 regions
    node_labels = [f"Region {i}" for i in range(100)]
    time_series = TIME_SERIES(
        data=data_1,
        subj_id="sub-0001",
        Fs=2,
        locs=locs,
        node_labels=node_labels,
    )
    # create a random data with 100 regions and 1000 time points
    data_2 = np.random.rand(100, 1000)
    time_series_2 = TIME_SERIES(
        data=data_2,
        subj_id="sub-0002",
        Fs=2,
        locs=locs,
        node_labels=node_labels,
    )
    time_series.concat_ts(time_series_2)

    subj_ts = time_series.get_subj_ts(subjs_id="sub-0001")
    assert subj_ts.data.shape == (100, 1000)
    assert subj_ts.subj_id_lst == ["sub-0001"]
    assert subj_ts.Fs == 2
    assert subj_ts.n_time == 1000
    assert subj_ts.n_regions == 100
    assert np.all(subj_ts.nodes_lst == np.arange(0, 100, dtype=int))
    assert np.all(subj_ts.time == 1 / 2 + np.arange(0, 1000 / 2, 1 / 2))
    assert subj_ts.data_dict.keys() == {"sub-0001"}
    assert np.all(subj_ts.data_dict["sub-0001"]["data"] == data_1)

    subj_ts = time_series.get_subj_ts(subjs_id="sub-0002")
    assert subj_ts.data.shape == (100, 1000)
    assert subj_ts.subj_id_lst == ["sub-0002"]
    assert subj_ts.Fs == 2
    assert subj_ts.n_time == 1000
    assert subj_ts.n_regions == 100
    assert np.all(subj_ts.nodes_lst == np.arange(0, 100, dtype=int))
    assert np.all(subj_ts.time == 1 / 2 + np.arange(0, 1000 / 2, 1 / 2))
    assert subj_ts.data_dict.keys() == {"sub-0002"}
    assert np.all(subj_ts.data_dict["sub-0002"]["data"] == data_2)

    # check that the original time_series is not changed
    assert time_series.data.shape == (100, 2000)
    assert time_series.subj_id_lst == ["sub-0001", "sub-0002"]
    assert time_series.Fs == 2
    assert time_series.n_time is None
    assert time_series.n_regions == 100
    assert np.all(time_series.nodes_lst == np.arange(0, 100, dtype=int))
    assert time_series.time is None
    assert time_series.data_dict.keys() == {"sub-0001", "sub-0002"}
    assert np.all(time_series.data_dict["sub-0001"]["data"] == data_1)
    assert np.all(time_series.data_dict["sub-0002"]["data"] == data_2)
    assert np.all(
        time_series.data_dict["sub-0001"]["time_array"]
        == 1 / 2 + np.arange(0, 1000 / 2, 1 / 2)
    )
    assert np.all(
        time_series.data_dict["sub-0002"]["time_array"]
        == 1 / 2 + np.arange(0, 1000 / 2, 1 / 2)
    )


def test_select_nodes():
    # create a random data with 100 regions and 1000 time points
    data = np.random.rand(100, 1000)
    # create a random locs with 100 regions and 3 coordinates
    locs = np.random.rand(100, 3)
    # create a random node_labels list with 100 regions
    node_labels = [f"Region {i}" for i in range(100)]
    time_series = TIME_SERIES(
        data=data,
        subj_id="sub-0001",
        Fs=2,
        locs=locs,
        node_labels=node_labels,
    )

    # select 5 nodes
    nodes_idx = np.array([0, 1, 23, 43, 87])
    time_series.select_nodes(nodes_idx=nodes_idx)
    assert time_series.data.shape == (5, 1000)
    assert np.all(time_series.data == data[nodes_idx, :])
    assert time_series.n_regions == 5
    assert np.all(time_series.nodes_lst == nodes_idx)
    assert np.all(time_series.locs == locs[nodes_idx])
    assert np.all(
        time_series.node_labels
        == ["Region 0", "Region 1", "Region 23", "Region 43", "Region 87"]
    )
    assert np.all(time_series.nodes_selection_ == nodes_idx)
