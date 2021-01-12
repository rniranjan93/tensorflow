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
        "//tensorflow/compiler/plugin/vsi/driver/TIM-VX:tim-vx_interface",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "vsi_executable",
    srcs = ["vsi_executable.cc"],
    hdrs = ["vsi_executable.h"],
    deps = [
        ":vsi_executor",
        ":visitor_base",
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
    name = "visitor_base",
    srcs = [
        "visitors/visitor_base.cc",
    ],
    hdrs = ["visitors/visitor_base.h"],
    deps = [
        "//tensorflow/compiler/xla/service/cpu:runtime_single_threaded_matmul",
        "//tensorflow/compiler/xla/service:dynamic_dimension_inference",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/compiler/xla/service:hlo_evaluator",
        "//tensorflow/compiler/xla/service:hlo_casting_utils",
        "//tensorflow/compiler/xla/service:hlo_query",
        "//tensorflow/compiler/xla/service:hlo_value",
        "//tensorflow/compiler/xla/service:hlo_buffer",
        "//tensorflow/compiler/xla/service:shape_inference",
        "//tensorflow/compiler/xla/service:buffer_assignment",
        "//tensorflow/compiler/xla:array2d",
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla:literal_util",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:window_util",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/core:lib",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/container:node_hash_map",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/meta:type_traits",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:span",
    ],
)
cc_library(
    name = "vsi_plugin",
    deps = [
        ":vsi_executable",
        ":vsi_platform",
        "//tensorflow/core:stream_executor_no_cuda",
    ],
)


