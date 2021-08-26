{
    "input": [
        "llvm -keys=cpu -link-params=0",
        "conv2d_NCHWc.x86",
        [
            ["TENSOR", [1, 3, 224, 224], "float32"],
            ["TENSOR", [64, 3, 7, 7], "float32"],
            [2, 2],
            [3, 3, 3, 3],
            [1, 1],
            "NCHW",
            "NCHW",
            "float32",
        ],
        {},
    ],

    "config": {
        "index": 228,
        "code_hash": null,
        "entity": [
            ["tile_ic", "sp", [-1, 1]],
            ["tile_oc", "sp", [-1, 4]],
            ["tile_ow", "sp", [-1, 7]],
            ["unroll_kw", "ot", false],
        ],
    },

    "result": [[0.00066373275, 0.0006597502499999999, 0.000640892], 0, 0, 0],
    "version": 0.2,
    "tvm_version": "0.8.dev0",
}
