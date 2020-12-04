licenses(["restricted"])

package(default_visibility = ["//visibility:public"])

load("//tensorflow/compiler/xla/tests:build_defs.bzl", "xla_test")
load("//tensorflow:tensorflow.bzl", "tf_cc_binary", "tf_cc_test")
load("//tensorflow/core/platform:build_config_root.bzl", "if_static")

cc_library(
    name = "vsi_platform_id",
    srcs = ["vsi_platform_id.cc"],
    hdrs = ["vsi_platform_id.h"],
    deps = ["//tensorflow/core:stream_executor_headers_lib"] + if_static(
        ["@com_google_protobuf//:protobuf"],
        ["@com_google_protobuf//:protobuf_headers"],
    ),
)

cc_library(
    name = "vsi_executor",
    srcs = ["vsi_executor.cc"],
    hdrs = ["vsi_executor.h",
            "vsi_utils.h"
    ],
    deps = [
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/core:lib",
        "//tensorflow/core:stream_executor_headers_lib",
        "//tensorflow/stream_executor/host:host_stream",
        "//tensorflow/stream_executor/host:host_timer",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "vsi_executable",
    srcs = ["vsi_executable.cc"],
    hdrs = ["vsi_executable.h"],
    deps = [
        ":vsi_executor",
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/service:executable",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/compiler/xla/service:hlo_cost_analysis",
        "//tensorflow/compiler/xla/service:hlo_evaluator",
        "//tensorflow/compiler/xla/service:hlo_execution_profile",
        "//tensorflow/compiler/xla/service:hlo_module_config",
        "//tensorflow/compiler/xla/service:maybe_owning_device_memory",
        "//tensorflow/compiler/xla/service:shaped_buffer",
        "//tensorflow/compiler/xla/service:transfer_manager",
        "//tensorflow/core:lib",
        "//tensorflow/core:stream_executor_no_cuda",
        "//tensorflow/core/platform:macros",
        "//tensorflow/core/platform:mutex",
        "//tensorflow/core/platform:types",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "vsi_platform",
    srcs = ["vsi_platform.cc"],
    hdrs = ["vsi_platform.h",
            "vsi_utils.h"
    ],
    deps = [
        ":vsi_executor",
        ":vsi_platform_id",
        "//tensorflow/core:stream_executor_headers_lib",
    ],
    alwayslink = True,  # Registers itself with the MultiPlatformManager.
)

cc_library(
    name = "vsi_plugin",
    deps = [
        ":vsi_executable",
        ":vsi_platform",
        "//tensorflow/core:stream_executor_no_cuda",
    ],
)