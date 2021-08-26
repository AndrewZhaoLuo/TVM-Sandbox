{
    "i": [

        # Task
        [
            # workload key
            '["7da4b9353f31499138ec976c527728b7", 1, 1, 56, 56, 64, 2, 1, 3, 3, 64, 32, 1, 2, 1, 1, 32, 1, 2, 56, 56, 32]',
            # target
            "llvm -keys=cpu -link-params=0",
            # hardware params
            [12, 64, 64, 0, 0, 0, 0, 0],
            # host
            "",
            # data layout option (cast as int)
            2,
            # task input names (?)
            [],
        ],

        # State
        [
            # Stages
            [],
            # Transform steps
            [
                ["CI", 5],
                ["SP", 3, 0, 1, [1, 1, 1], 1],
                ["SP", 3, 4, 2, [1, 1, 1], 1],
                ["SP", 3, 8, 56, [2, 1, 2], 1],
                ["SP", 3, 12, 56, [4, 1, 2], 1],
                ["SP", 3, 16, 32, [4, 1, 8], 1],
                ["SP", 3, 20, 64, [16], 1],
                ["SP", 3, 22, 3, [3], 1],
                ["SP", 3, 24, 3, [3], 1],
                [
                    "RE",
                    3,
                    [
                        0,
                        4,
                        8,
                        12,
                        16,
                        1,
                        5,
                        9,
                        13,
                        17,
                        20,
                        22,
                        24,
                        2,
                        6,
                        10,
                        14,
                        18,
                        21,
                        23,
                        25,
                        3,
                        7,
                        11,
                        15,
                        19,
                    ],
                ],
                ["CR", 6],
                ["CA", 1, 3, 4],
                ["FU", 3, [0, 1, 2, 3]],
                ["AN", 3, 0, 3],
                ["FU", 6, [0, 1, 2, 3]],
                ["AN", 6, 0, 3],
                ["PR", 3, 0, "auto_unroll_max_step$64"],
                ["AN", 3, 22, 2],
            ],
        ],
    ],
    
    # Results
    "r": [
        [
            0.000520474,
            0.000509143,
            0.000523144,
            0.000497664,
            0.000495014,
            0.000493263,
            0.000496833,
            0.000494003,
            0.000493613,
            0.000493744,
        ],
        0,
        0.871013,
        1629911544,
    ],
    "v": "v0.6",
}
