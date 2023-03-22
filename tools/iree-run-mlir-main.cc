// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// IREE source.mlir -> execution output test runner.
// This is meant to be called from LIT for FileCheck tests, and tries to match
// the interface of mlir-opt (featuring -split-input-file, etc) so it's easier
// to work with there. If you want a more generalized runner for standalone
// precompiled IREE modules use iree-run-module.
//
// By default all exported functions in the module will be run in order.
// All input values, provided via -function-inputs, will be passed to the
// functions (this means all input signatures must match). Results from the
// executed functions will be printed to stdout for checking.
//
// Example input:
// // RUN: iree-run-mlir %s | FileCheck %s
// // CHECK-LABEL: @foo
// // CHECK: 1xf32: 2
// func.func @foo() -> tensor<f32> {
//   %0 = arith.constant dense<2.0> : tensor<f32>
//   return %0 : tensor<f32>
// }
//
// Command line arguments are handled by LLVM's parser by default but -- can be
// used to separate the compiler flags from the runtime flags, such as:
//   iree-run-mlir --iree-hal-target-backends=vulkan-spirv -- --logtostderr

#include <cstdio>
#include <cstring>
#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include "iree/base/api.h"
#include "iree/base/internal/flags.h"
#include "iree/base/tracing.h"
#include "iree/compiler/embedding_api.h"
#include "iree/hal/api.h"
#include "iree/modules/hal/types.h"
#include "iree/tooling/context_util.h"
#include "iree/tooling/device_util.h"
#include "iree/tooling/vm_util.h"
#include "iree/vm/api.h"
#include "iree/vm/bytecode/module.h"

// static llvm::cl::opt<std::string> input_file_flag{
//     llvm::cl::Positional,
//     llvm::cl::desc("<input .mlir file>"),
//     llvm::cl::init("-"),
// };

// static llvm::cl::opt<bool> split_input_file_flag{
//     "split-input-file",
//     llvm::cl::desc("Split the input file into multiple modules"),
//     llvm::cl::init(true),
// };

// static llvm::cl::opt<bool> verify_passes_flag(
//     "verify-each",
//     llvm::cl::desc("Run the verifier after each transformation pass"),
//     llvm::cl::init(true));

// static llvm::cl::opt<bool> print_mlir_flag{
//     "print-mlir",
//     llvm::cl::desc("Prints MLIR IR after translation"),
//     llvm::cl::init(false),
// };

// static llvm::cl::opt<bool> print_annotated_mlir_flag{
//     "print-annotated-mlir",
//     llvm::cl::desc("Prints MLIR IR with final serialization annotations"),
//     llvm::cl::init(false),
// };

// static llvm::cl::opt<bool> print_flatbuffer_flag{
//     "print-flatbuffer",
//     llvm::cl::desc("Prints Flatbuffer text after serialization"),
//     llvm::cl::init(false),
// };

// static llvm::cl::opt<std::string> output_file_flag{
//     "o",
//     llvm::cl::desc("File path in which to write the compiled module file"),
//     llvm::cl::init(""),
// };

// static llvm::cl::opt<bool> run_flag{
//     "run",
//     llvm::cl::desc("Runs the module (vs. just compiling and verifying)"),
//     llvm::cl::init(true),
// };

// static llvm::cl::list<std::string> run_args_flag{
//     "run-arg",
//     llvm::cl::desc("Argument passed to the execution flag parser"),
//     llvm::cl::ConsumeAfter,
// };

IREE_FLAG_LIST(
    string, input,
    "An input (a) value or (b) buffer of the format:\n"
    "  (a) scalar value\n"
    "     value\n"
    "     e.g.: --input=\"3.14\"\n"
    "  (b) buffer:\n"
    "     [shape]xtype=[value]\n"
    "     e.g.: --input=\"2x2xi32=1 2 3 4\"\n"
    "Optionally, brackets may be used to separate the element values:\n"
    "  2x2xi32=[[1 2][3 4]]\n"
    "Raw binary files can be read to provide buffer contents:\n"
    "  2x2xi32=@some/file.bin\n"
    "\n"
    "Numpy npy files from numpy.save can be read to provide 1+ values:\n"
    "  @some.npy\n"
    "\n"
    "Each occurrence of the flag indicates an input in the order they were\n"
    "specified on the command line.");

IREE_FLAG_LIST(
    string, output,
    "Specifies how to handle an output from the invocation:\n"
    "  `` (empty): ignore output\n"
    "     e.g.: --output=\n"
    "  `-`: print textual form to stdout\n"
    "     e.g.: --output=-\n"
    "  `@file.npy`: create/overwrite a numpy npy file and write buffer view\n"
    "     e.g.: --output=@file.npy\n"
    "  `+file.npy`: create/append a numpy npy file and write buffer view\n"
    "     e.g.: --output=+file.npy\n"
    "\n"
    "Numpy npy files can be read in Python using numpy.load, for example an\n"
    "invocation producing two outputs can be concatenated as:\n"
    "    --output=@file.npy --output=+file.npy\n"
    "And then loaded in Python by reading from the same file:\n"
    "  with open('file.npy', 'rb') as f:\n"
    "    print(numpy.load(f))\n"
    "    print(numpy.load(f))\n"
    "\n"
    "Each occurrence of the flag indicates an output in the order they were\n"
    "specified on the command line.");

IREE_FLAG(int32_t, output_max_element_count, 1024,
          "Prints up to the maximum number of elements of output tensors, "
          "eliding the remainder.");

namespace iree {
namespace {

bool starts_with(std::string_view prefix, std::string_view in_str) {
  return in_str.size() >= prefix.size() &&
         in_str.compare(0, prefix.size(), prefix) == 0;
}

// Tries to guess a default device name from the backend, where possible.
// Users are still able to override this by passing in --device= flags.
std::string InferDefaultDeviceFromBackend(const std::string& backend) {
  if (backend == "vmvx" || backend == "llvm-cpu") {
    return "local-task";
  } else if (backend == "vmvx-inline") {
    return "";
  }
  size_t dash = backend.find('-');
  if (dash == std::string::npos) {
    return backend;
  } else {
    return backend.substr(0, dash);
  }
}

// Returns a list of target compiler backends to use for file evaluation.
Status GetTargetBackends(std::vector<std::string>* out_target_backends) {
  IREE_TRACE_SCOPE();
  // out_target_backends->clear();
  // auto target_backends =
  //     mlir::iree_compiler::IREE::HAL::TargetOptions::FromFlags::get().targets;
  // if (target_backends.empty()) {
  //   iree_allocator_t host_allocator = iree_allocator_system();
  //   iree_host_size_t driver_info_count = 0;
  //   iree_hal_driver_info_t* driver_infos = NULL;
  //   IREE_RETURN_IF_ERROR(iree_hal_driver_registry_enumerate(
  //       iree_hal_available_driver_registry(), host_allocator,
  //       &driver_info_count, &driver_infos));
  //   for (iree_host_size_t i = 0; i < driver_info_count; ++i) {
  //     target_backends.push_back(std::string(driver_infos[i].driver_name.data,
  //                                           driver_infos[i].driver_name.size));
  //   }
  //   iree_allocator_free(host_allocator, driver_infos);
  // }
  // *out_target_backends = std::move(target_backends);
  return OkStatus();
}

// Evaluates a single function in its own fiber, printing the results to stdout.
Status EvaluateFunction(iree_vm_context_t* context, iree_hal_device_t* device,
                        iree_hal_allocator_t* device_allocator,
                        iree_vm_function_t function,
                        iree_string_view_t function_name) {
  IREE_TRACE_SCOPE();
  iree_allocator_t host_allocator = iree_allocator_system();

  printf("EXEC @%.*s\n", (int)function_name.size, function_name.data);

  // Parse input values from the flags.
  vm::ref<iree_vm_list_t> inputs;
  IREE_RETURN_IF_ERROR(iree_tooling_parse_to_variant_list(
      device_allocator, FLAG_input_list().values, FLAG_input_list().count,
      host_allocator, &inputs));

  // If the function is async add fences so we can invoke it synchronously.
  vm::ref<iree_hal_fence_t> finish_fence;
  IREE_RETURN_IF_ERROR(iree_tooling_append_async_fence_inputs(
      inputs.get(), &function, device, /*wait_fence=*/NULL, &finish_fence));

  // Prepare outputs list to accept the results from the invocation.
  vm::ref<iree_vm_list_t> outputs;
  IREE_RETURN_IF_ERROR(iree_vm_list_create(iree_vm_make_undefined_type_def(),
                                           16, host_allocator, &outputs));

  // Synchronously invoke the function.
  IREE_RETURN_IF_ERROR(iree_vm_invoke(
      context, function, IREE_VM_INVOCATION_FLAG_NONE,
      /*policy=*/nullptr, inputs.get(), outputs.get(), host_allocator));

  // If the function is async we need to wait for it to complete.
  if (finish_fence) {
    IREE_RETURN_IF_ERROR(
        iree_hal_fence_wait(finish_fence.get(), iree_infinite_timeout()));
  }

  // Print outputs.
  if (FLAG_output_list().count == 0) {
    IREE_RETURN_IF_ERROR(
        iree_tooling_variant_list_fprint(
            IREE_SV("result"), outputs.get(),
            (iree_host_size_t)FLAG_output_max_element_count, stdout),
        "printing results");
  } else {
    IREE_RETURN_IF_ERROR(
        iree_tooling_output_variant_list(
            outputs.get(), FLAG_output_list().values, FLAG_output_list().count,
            (iree_host_size_t)FLAG_output_max_element_count, stdout),
        "outputting results");
  }

  return OkStatus();
}

// Evaluates all exported functions within given module.
Status EvaluateFunctions(iree_vm_instance_t* instance,
                         const std::string& default_device_uri,
                         void* binary_contents, uint64_t binary_size) {
  IREE_TRACE_SCOPE0("EvaluateFunctions");

  // Load any custom modules the user may have explicitly specified and then
  // append the module we compiled.
  iree_tooling_module_list_t module_list;
  iree_tooling_module_list_initialize(&module_list);
  IREE_RETURN_IF_ERROR(iree_tooling_load_modules_from_flags(
      instance, iree_allocator_system(), &module_list));

  // Load the bytecode module from the flatbuffer data.
  // We do this first so that if we fail validation we know prior to dealing
  // with devices.
  vm::ref<iree_vm_module_t> main_module;
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_module_create(
      instance, iree_make_const_byte_span(binary_contents, binary_size),
      iree_allocator_null(), iree_allocator_system(), &main_module));
  IREE_RETURN_IF_ERROR(
      iree_tooling_module_list_push_back(&module_list, main_module.get()));

  if (!run_flag) {
    // Just wanted verification; return without running.
    main_module.reset();
    iree_tooling_module_list_reset(&module_list);
    return OkStatus();
  }

  // Evaluate all exported functions.
  auto run_function = [&](int ordinal) -> Status {
    iree_vm_function_t function;
    IREE_RETURN_IF_ERROR(iree_vm_module_lookup_function_by_ordinal(
                             main_module.get(), IREE_VM_FUNCTION_LINKAGE_EXPORT,
                             ordinal, &function),
                         "looking up function export %d", ordinal);
    iree_string_view_t function_name = iree_vm_function_name(&function);
    if (iree_string_view_starts_with(function_name,
                                     iree_make_cstring_view("__")) ||
        iree_string_view_find_char(function_name, '$', 0) !=
            IREE_STRING_VIEW_NPOS) {
      // Skip internal or special functions.
      return OkStatus();
    }

    // Create the context we'll use for this (ensuring that we can't interfere
    // with other running evaluations, such as when in a multithreaded test
    // runner).
    vm::ref<iree_vm_context_t> context;
    vm::ref<iree_hal_device_t> device;
    vm::ref<iree_hal_allocator_t> device_allocator;
    IREE_RETURN_IF_ERROR(iree_tooling_create_context_from_flags(
        instance, module_list.count, module_list.values,
        iree_make_string_view(default_device_uri.data(),
                              default_device_uri.size()),
        iree_allocator_system(), &context, &device, &device_allocator));

    IREE_RETURN_IF_ERROR(iree_hal_begin_profiling_from_flags(device.get()));

    // Invoke the function and print results.
    IREE_RETURN_IF_ERROR(
        EvaluateFunction(context.get(), device.get(), device_allocator.get(),
                         function, function_name),
        "evaluating export function %d", ordinal);

    IREE_RETURN_IF_ERROR(iree_hal_end_profiling_from_flags(device.get()));

    context.reset();
    device_allocator.reset();
    device.reset();
    return OkStatus();
  };

  Status evaluate_status = OkStatus();
  auto module_signature = iree_vm_module_signature(main_module.get());
  for (iree_host_size_t i = 0; i < module_signature.export_function_count;
       ++i) {
    evaluate_status = run_function(i);
    if (!evaluate_status.ok()) {
      break;
    }
  }

  main_module.reset();
  iree_tooling_module_list_reset(&module_list);

  return evaluate_status;
}

Status EvaluateFile(void* binary_contents, uint64_t binary_size) {
  IREE_TRACE_SCOPE0("EvaluateFile");

  vm::ref<iree_vm_instance_t> instance;
  IREE_RETURN_IF_ERROR(
      iree_tooling_create_instance(iree_allocator_system(), &instance),
      "Creating instance");

  // TODO: Get the target backend inference back.
  std::vector<std::string> target_backends = {"vmvx"};
  IREE_RETURN_IF_ERROR(GetTargetBackends(&target_backends));
  for (const auto& target_backend : target_backends) {
    // Prepare the module for execution and evaluate it.
    IREE_TRACE_FRAME_MARK();
    // TODO: Get device_uri inference back.
    // std::string default_device_uri =
    //     InferDefaultDeviceFromBackend(target_backend);
    std::string default_device_uri = "local-task";
    (void)target_backend;
    IREE_RETURN_IF_ERROR(EvaluateFunctions(instance.get(), default_device_uri,
                                           binary_contents, binary_size),
                         "Evaluating functions");
  }

  instance.reset();
  return OkStatus();
}

// Runs the given .mlir file based on the current flags.
Status RunFile(const char* mlir_filename, iree_compiler_session_t* session) {
  IREE_TRACE_SCOPE0("RunFile");

  // Query the session for the hal-target-backends flag.
  // TODO: Not doing anything with this yet. Just showing the incantation.
  std::string target_backends_flag;
  ireeCompilerSessionGetFlags(
      session, false,
      [](const char* flag, size_t length, void* userdata) {
        std::string_view prefix = "--iree-hal-target-backends=";
        if (starts_with(prefix, std::string_view(flag, length))) {
          std::string* result = static_cast<std::string*>(userdata);
          *result = std::string(flag, length);
          fprintf(stderr, "TARGET BACKENDS: %s\n", result->c_str());
        }
      },
      static_cast<void*>(&target_backends_flag));

  struct MainState {
    iree_compiler_source_t* source = nullptr;
    std::vector<iree_compiler_source_t*> splitSources;
    ~MainState() {
      for (auto* splitSource : splitSources) {
        ireeCompilerSourceDestroy(splitSource);
      }
      if (source) {
        ireeCompilerSourceDestroy(source);
      }
    }
    void handleError(iree_compiler_error_t* error) {
      const char* msg = ireeCompilerErrorGetMessage(error);
      fprintf(stderr, "error compiling input file: %s\n", msg);
      ireeCompilerErrorDestroy(error);
    }
  };
  MainState s;
  if (auto error =
          ireeCompilerSourceOpenFile(session, mlir_filename, &s.source)) {
    s.handleError(error);
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION);
  }

  auto processBuffer = [&](iree_compiler_source_t* source) -> Status {
    IREE_TRACE_FRAME_MARK();
    // Stash per-invocation state in an RAII instance.
    struct InvState {
      InvState(MainState& s, iree_compiler_session_t* session) {
        inv = ireeCompilerInvocationCreate(session);
      }
      ~InvState() {
        ireeCompilerInvocationDestroy(inv);
        if (output) ireeCompilerOutputDestroy(output);
      }
      iree_compiler_invocation_t* inv;
      iree_compiler_output_t* output = nullptr;
    };
    InvState r(s, session);
    if (auto error = ireeCompilerOutputOpenMembuffer(&r.output)) {
      s.handleError(error);
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "failed to open output buffer");
    }

    ireeCompilerInvocationEnableConsoleDiagnostics(r.inv);
    if (!ireeCompilerInvocationParseSource(r.inv, source))
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "failed to parse input file");
    if (!ireeCompilerInvocationPipeline(r.inv, IREE_COMPILER_PIPELINE_STD)) {
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "failed to invoke main compiler pipeline");
    }
    if (auto error = ireeCompilerInvocationOutputVMBytecode(r.inv, r.output)) {
      s.handleError(error);
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "failed to emit output binary");
    }

    void* binary_data;
    uint64_t binary_size;
    if (auto error =
            ireeCompilerOutputMapMemory(r.output, &binary_data, &binary_size)) {
      s.handleError(error);
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "failed to access compiled memory buffer");
    }
    return EvaluateFile(binary_data, binary_size);
  };

  bool split_input_file_flag = false;  // TODO: get from flag
  if (split_input_file_flag) {
    if (auto error = ireeCompilerSourceSplit(
            s.source,
            [](iree_compiler_source_t* source, void* userData) {
              MainState* userState = static_cast<MainState*>(userData);
              userState->splitSources.push_back(source);
            },
            static_cast<void*>(&s))) {
      s.handleError(error);
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION);
    }
    for (auto* splitSource : s.splitSources) {
      IREE_RETURN_IF_ERROR(processBuffer(splitSource));
    }
    return OkStatus();
  } else {
    return processBuffer(s.source);
  }

  return OkStatus();
}

}  // namespace

extern "C" int main(int argc_raw, char** argv_raw) {
  IREE_TRACE_SCOPE0("iree-run-mlir");
  ireeCompilerGlobalInitialize();
  // Pre-process the arguments with the compiler's argument parser since it
  // has super-powers on Windows and must work on the default main arguments.
  ireeCompilerGetProcessCLArgs(&argc_raw, const_cast<const char***>(&argv_raw));

  // Do some light pre-processing:
  // Everything after "--" goes to the compiler. Also any "-Xcompiler"
  // args.
  std::vector<std::unique_ptr<std::string>> temp_strings;
  std::vector<char*> runtime_args = {argv_raw[0]};
  std::vector<char*> compiler_args = {argv_raw[0]};
  bool parsing_runtime_args = true;
  std::string_view xcompilerPrefix = "-Xcompiler,";
  for (int i = 1; i < argc_raw; ++i) {
    char* current_arg = argv_raw[i];
    // Always ok because argv is null terminated.
    char* next_arg = argv_raw[i + 1];
    if (!parsing_runtime_args) {
      compiler_args.push_back(current_arg);
      continue;
    }
    std::string_view check_arg(current_arg);

    if (check_arg == "--") {
      // All else to compiler.
      parsing_runtime_args = false;
    } else if (check_arg == "-Xcompiler") {
      // Skip and next goes to compiler.
      if (!next_arg || strcmp(next_arg, "--") == 0) {
        fprintf(stderr,
                "Syntax error: -Xcompiler must be followed by an argument to "
                "pass to the compiler but got none\n");
        return 1;
      }
      compiler_args.push_back(next_arg);
      i++;
    } else if (starts_with(xcompilerPrefix, check_arg)) {
      // Split by comma into compiler args.
      std::string_view sub_arg = check_arg.substr(xcompilerPrefix.size());
      for (;;) {
        size_t commaPos = sub_arg.find_first_of(',');
        if (commaPos == std::string_view::npos) break;
        temp_strings.push_back(
            std::make_unique<std::string>(sub_arg.substr(0, commaPos)));
        compiler_args.push_back(temp_strings.back()->data());
        sub_arg = sub_arg.substr(commaPos + 1);
      }
      temp_strings.push_back(std::make_unique<std::string>(sub_arg));
      compiler_args.push_back(temp_strings.back()->data());
    } else {
      runtime_args.push_back(current_arg);
    }
  }

  // Add nullptrs to end to match real argv behavior.
  compiler_args.push_back(nullptr);
  runtime_args.push_back(nullptr);

  ireeCompilerSetupGlobalCL(compiler_args.size() - 1,
                            const_cast<const char**>(compiler_args.data()),
                            "iree-run-mlir",
                            /*installSignalHandlers=*/true);
  int runtime_argc = runtime_args.size() - 1;
  char** runtime_argv = runtime_args.data();
  // Note that positional args are left in runtime_argv.
  iree_flags_parse_checked(IREE_FLAGS_PARSE_MODE_DEFAULT, &runtime_argc,
                           &runtime_argv);

  // Loop over each file and compile/run it.
  int rc = 0;
  iree_compiler_session_t* session = ireeCompilerSessionCreate();
  for (int i = 1; i < runtime_argc; ++i) {
    auto status = RunFile(runtime_argv[i], session);
    if (!status.ok()) {
      rc = 2;
      break;
    }
  }

  ireeCompilerSessionDestroy(session);
  ireeCompilerGlobalShutdown();
  return rc;
}

}  // namespace iree
