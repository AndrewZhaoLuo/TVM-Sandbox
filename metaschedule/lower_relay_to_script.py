"""Example of lowering relay to metaschedule modules (no tuning)."""
from typing import *

import tvm
from tvm import relay
from tvm.meta_schedule import database as ms_database
from tvm.script import tir as T
from tvm.utils import roofline


@T.prim_func
def fast_erf(
    p0: T.Buffer[(1, 1000, 1000), "float32"],
    T_fast_erf: T.Buffer[(1, 1000, 1000), "float32"],
):
    # function attr dict
    T.func_attr(
        {
            "hash": "9d9d7da1920f9bcd",
            "target": T.target(
                {
                    "kind": "llvm",
                    "tag": "",
                    "keys": ["cpu"],
                    "host": T.target({"kind": "llvm", "tag": "", "keys": ["cpu"]}),
                }
            ),
            "tir.noalias": True,
            "global_symbol": "tvmgen_default_fused_fast_erf",
            "from_legacy_te_schedule": True,
            "tir.is_entry_func": True,
        }
    )
    # buffer definition
    T_fast_erf_1 = T.buffer_decl([1000000], dtype="float32", data=T_fast_erf.data)
    p0_1 = T.buffer_decl([1000000], dtype="float32", data=p0.data)
    # body
    for ax0_ax1_fused in T.parallel(1000):
        for ax2_outer, ax2_inner_s in T.grid(63, 16):
            if ax2_outer * 2 + ax2_inner_s // 8 < 125:
                cse_var_1: T.int32 = ax0_ax1_fused * 1000 + ax2_outer * 16 + ax2_inner_s
                T_fast_erf_1[cse_var_1] = (
                    T.max(T.min(p0_1[cse_var_1], T.float32(4)), T.float32(-4))
                    * (
                        T.max(T.min(p0_1[cse_var_1], T.float32(4)), T.float32(-4))
                        * T.max(T.min(p0_1[cse_var_1], T.float32(4)), T.float32(-4))
                        * (
                            T.max(T.min(p0_1[cse_var_1], T.float32(4)), T.float32(-4))
                            * T.max(T.min(p0_1[cse_var_1], T.float32(4)), T.float32(-4))
                            * (
                                T.max(
                                    T.min(p0_1[cse_var_1], T.float32(4)), T.float32(-4)
                                )
                                * T.max(
                                    T.min(p0_1[cse_var_1], T.float32(4)), T.float32(-4)
                                )
                                * (
                                    T.max(
                                        T.min(p0_1[cse_var_1], T.float32(4)),
                                        T.float32(-4),
                                    )
                                    * T.max(
                                        T.min(p0_1[cse_var_1], T.float32(4)),
                                        T.float32(-4),
                                    )
                                    * (
                                        T.max(
                                            T.min(p0_1[cse_var_1], T.float32(4)),
                                            T.float32(-4),
                                        )
                                        * T.max(
                                            T.min(p0_1[cse_var_1], T.float32(4)),
                                            T.float32(-4),
                                        )
                                        * (
                                            T.max(
                                                T.min(p0_1[cse_var_1], T.float32(4)),
                                                T.float32(-4),
                                            )
                                            * T.max(
                                                T.min(p0_1[cse_var_1], T.float32(4)),
                                                T.float32(-4),
                                            )
                                            * T.float32(-2.7261423674040941e-10)
                                            + T.float32(2.7706814620387377e-08)
                                        )
                                        + T.float32(-2.101023937939317e-06)
                                    )
                                    + T.float32(-5.6925062381196767e-05)
                                )
                                + T.float32(-0.00073499063728377223)
                            )
                            + T.float32(-0.0029545999132096767)
                        )
                        + T.float32(-0.016096033155918121)
                    )
                    / (
                        T.max(T.min(p0_1[cse_var_1], T.float32(4)), T.float32(-4))
                        * T.max(T.min(p0_1[cse_var_1], T.float32(4)), T.float32(-4))
                        * (
                            T.max(T.min(p0_1[cse_var_1], T.float32(4)), T.float32(-4))
                            * T.max(T.min(p0_1[cse_var_1], T.float32(4)), T.float32(-4))
                            * (
                                T.max(
                                    T.min(p0_1[cse_var_1], T.float32(4)), T.float32(-4)
                                )
                                * T.max(
                                    T.min(p0_1[cse_var_1], T.float32(4)), T.float32(-4)
                                )
                                * (
                                    T.max(
                                        T.min(p0_1[cse_var_1], T.float32(4)),
                                        T.float32(-4),
                                    )
                                    * T.max(
                                        T.min(p0_1[cse_var_1], T.float32(4)),
                                        T.float32(-4),
                                    )
                                    * T.float32(-1.4566071513399947e-05)
                                    + T.float32(-0.00021337404905352741)
                                )
                                + T.float32(-0.001682827016338706)
                            )
                            + T.float32(-0.0073733292520046234)
                        )
                        + T.float32(-0.014264739118516445)
                    )
                )


if __name__ == "__main__":

    saved_tir = roofline.SaveLoweredTIR()

    relay_mod = tvm.IRModule.from_expr(
        relay.erf(relay.var("input", dtype="float32", shape=()))
    )

    database = ms_database.JSONDatabase(work_dir=".")
    relay_mod = relay.transform.FastMath()(relay_mod)
    with database, tvm.transform.PassContext(
        config={
            "relay.backend.use_meta_schedule": True,
            "relay.backend.use_meta_schedule_dispatch": False,
            "relay.FuseOps.max_depth": 30,
        },
        instruments=[saved_tir],
        opt_level=3,
    ):
        mod = relay.build(relay_mod, target="llvm")

    for name, primfunc in saved_tir.functions.items():
        print(name)
        print(primfunc.script())
