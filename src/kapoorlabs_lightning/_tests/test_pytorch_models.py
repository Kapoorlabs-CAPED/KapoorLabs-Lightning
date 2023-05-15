import numpy as np
import pytest
import torch
from pytorch_dataset_loaders.pytorch_models import (
    BLT_net_64,
    BLTConvLayer,
    BLTDenseLayer,
)


@pytest.mark.parametrize("b_input_shape", [(1, 3, 128, 128)])
@pytest.mark.parametrize("l_input_shape", [(1, 3, 128, 128)])
@pytest.mark.parametrize("t_input_shape", [(1, 3, 128, 128)])
def test_blt_conv_layer(b_input_shape, l_input_shape, t_input_shape):
    b_input = np.random.rand(*b_input_shape).astype(np.float32)
    l_input = np.random.rand(*l_input_shape).astype(np.float32)
    t_input = np.random.rand(*t_input_shape).astype(np.float32)

    in_channels = 3
    # create a BLTConvLayer
    blt_conv_layer = BLTConvLayer(in_channels, 3, 3, "test", True)

    # run the layer
    blt_conv_layer(
        torch.from_numpy(b_input),
        torch.from_numpy(l_input),
        torch.from_numpy(t_input),
    )


@pytest.mark.parametrize("b_input_shape", [(1, 128, 128)])
@pytest.mark.parametrize("l_input_shape", [(1, 128, 128)])
@pytest.mark.parametrize("t_input_shape", [(1, 128, 128)])
@pytest.mark.parametrize("use_bias", [True, False])
@pytest.mark.parametrize("readout", [True, False])
def test_blt_dense_layer(
    b_input_shape, l_input_shape, t_input_shape, use_bias, readout
):
    b_input = np.random.rand(*b_input_shape).astype(np.float32)
    l_input = np.random.rand(*l_input_shape).astype(np.float32)
    t_input = np.random.rand(*t_input_shape).astype(np.float32)

    in_features = 128
    out_features = 10
    # create a BLTDenseLayer
    blt_dense_layer = BLTDenseLayer(
        in_features, out_features, use_bias, readout
    )

    # run the layer
    blt_dense_layer(
        torch.from_numpy(b_input),
        torch.from_numpy(l_input),
        torch.from_numpy(t_input),
    )


@pytest.mark.parametrize("input_shape", [(2, 3, 64, 64)])
@pytest.mark.parametrize("timesteps", [2])
@pytest.mark.parametrize("lateral_connections", [True, False])
@pytest.mark.parametrize("topdown_connections", [True, False])
@pytest.mark.parametrize("LT_interaction", ["additive"])
@pytest.mark.parametrize("LT_position", ["all"])
@pytest.mark.parametrize("classifier_bias", [True, False])
@pytest.mark.parametrize("norm_type", ["LN", "None"])
def test_blt_net(
    input_shape,
    timesteps,
    lateral_connections,
    topdown_connections,
    LT_interaction,
    LT_position,
    classifier_bias,
    norm_type,
):
    # create a BLT_net
    blt_net = BLT_net_64(
        lateral_connections=lateral_connections,
        topdown_connections=topdown_connections,
        LT_interaction=LT_interaction,
        LT_position=LT_position,
        classifier_bias=classifier_bias,
        timesteps=timesteps,
        norm_type=norm_type,
    )
    inputs = np.random.rand(*input_shape).astype(np.float32)
    blt_net(torch.from_numpy(inputs))


if __name__ == "__main__":
    test_blt_net(
        input_shape=(4, 3, 64, 64),
        timesteps=2,
        lateral_connections=True,
        topdown_connections=True,
        LT_interaction="additive",
        LT_position="all",
        classifier_bias=True,
        norm_type="LN",
    )
