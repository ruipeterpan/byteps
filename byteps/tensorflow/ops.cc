// Copyright 2016 The TensorFlow Authors. All Rights Reserved.
// Modifications copyright (C) 2018 Uber Technologies, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================

#include <memory>
#include <queue>
#include <chrono>
#include <thread>
#include <unordered_map>
#include <sstream>
#include <fstream>
#include <iostream>

#include <cuda_runtime.h>
#include <cuda.h>

#include "ops.h"
#include "../common/logging.h"

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"

#include "tensorflow/compiler/tf2xla/shape_util.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/service/custom_call_target_registry.h"
#include "tensorflow/compiler/tf2xla/type_util.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/types.h"

#include "tensorflow/compiler/tf2xla/kernels/tensor_list_utils.h"

#include "tensorflow/core/common_runtime/gpu/gpu_id.h"
#include "tensorflow/core/common_runtime/gpu/gpu_init.h"
#include "tensorflow/core/common_runtime/gpu/gpu_id_utils.h"
#include "tensorflow/core/common_runtime/gpu/gpu_bfc_allocator.h"
#include "tensorflow/core/common_runtime/gpu/gpu_cudamalloc_allocator.h"

using namespace byteps;

namespace byteps {
namespace tensorflow {

std::unordered_map<std::string, std::shared_ptr<Xla_done_cb_args>> _name_to_done_args;
std::mutex _name_to_done_args_mtx;
std::condition_variable _name_to_done_args_cv;

namespace {

int printMatOnGPU(std::string &name, const void *buffer, int num_elem) {
  std::this_thread::sleep_for(std::chrono::seconds(2));
  std::ofstream outfile;
  int my_rank = common::byteps_rank();
  outfile.open("output-" + std::to_string(my_rank), std::ios_base::out | std::ios_base::app);
  outfile << "tensor_name: " << name << std::endl;

  float *host_buf = nullptr;
  host_buf = (float *)malloc(num_elem * sizeof(float));
  cudaMemcpy((void *) host_buf, buffer, sizeof(float) * num_elem, cudaMemcpyDeviceToHost);
  for (int i = 0; i < num_elem; i++) {
      outfile << *(host_buf + i) << ", ";
  }
  outfile << std::endl;
  free(host_buf);

  return 0;
}

::tensorflow::Status ConvertStatus(const common::Status& status) {
  switch (status.type()) {
    case common::OK:
      return ::tensorflow::Status::OK();
    case common::UNKNOWN_ERROR:
      return ::tensorflow::errors::Unknown(status.reason());
    case common::PRECONDITION_ERROR:
      return ::tensorflow::errors::FailedPrecondition(status.reason());
    case common::ABORTED:
      return ::tensorflow::errors::Aborted(status.reason());
    case common::INVALID_ARGUMENT:
      return ::tensorflow::errors::InvalidArgument(status.reason());
    default:
      return ::tensorflow::errors::Unknown("Unknown error.");
  }
}

int GetDeviceID(::tensorflow::OpKernelContext* context) {
  int device = CPU_DEVICE_ID;
  if (context->device() != nullptr &&
      context->device()->tensorflow_gpu_device_info() != nullptr) {
    device = context->device()->tensorflow_gpu_device_info()->gpu_id;
  }
  return device;
}

// Define all types for TensorUtil.
const common::DataType ConvertDType(int dtype) {
  switch (dtype) {
    case ::tensorflow::DT_UINT8:
      return common::BYTEPS_UINT8;
    case ::tensorflow::DT_INT8:
      return common::BYTEPS_INT8;
    // case ::tensorflow::DT_UINT16:
    //   return common::BYTEPS_UINT16;
    // case ::tensorflow::DT_INT16:
    //   return common::BYTEPS_INT16;
    case ::tensorflow::DT_INT32:
      return common::BYTEPS_INT32;
    case ::tensorflow::DT_INT64:
      return common::BYTEPS_INT64;
    case ::tensorflow::DT_HALF:
      return common::BYTEPS_FLOAT16;
    case ::tensorflow::DT_FLOAT:
      return common::BYTEPS_FLOAT32;
    case ::tensorflow::DT_DOUBLE:
      return common::BYTEPS_FLOAT64;
    // case ::tensorflow::DT_BOOL:
    //   return common::BYTEPS_BOOL;
    default:
      throw std::logic_error("Invalid tensor type.");
  }
}

}  // namespace

TFReadyEvent::TFReadyEvent(::tensorflow::DeviceContext* device_context) {
  auto executor = device_context->stream()->parent();
  auto ready_event = new perftools::gputools::Event(executor);
  ready_event->Init();
  device_context->stream()->ThenRecordEvent(ready_event);
  event_ = std::shared_ptr<perftools::gputools::Event>(ready_event);
}

bool TFReadyEvent::Ready() const {
  return event_->PollForStatus() !=
         perftools::gputools::Event::Status::kPending;
}

XlaReadyEvent::XlaReadyEvent(cudaStream_t stream) {
  cudaEventCreateWithFlags(
          &cuda_event_, cudaEventBlockingSync | cudaEventDisableTiming);
  cudaEventRecord(cuda_event_, stream);
}

bool XlaReadyEvent::Ready() const {
  auto status = cudaEventQuery(cuda_event_);
  if (status == cudaErrorNotReady) {
    return false;
  }
  return true;
}

TFTensor::TFTensor(::tensorflow::Tensor& tensor) : tensor_(tensor) {}

const common::DataType TFTensor::dtype() const {
  return ConvertDType(tensor_.dtype());
}

const common::TensorShape TFTensor::shape() const {
  common::TensorShape shape;
  for (auto dim : tensor_.shape()) {
    shape.AddDim(dim.size);
  }
  return shape;
}

const void* TFTensor::data() const {
  return (const void*)tensor_.tensor_data().data();
}

int64_t TFTensor::size() const { return (int64_t)tensor_.tensor_data().size(); }

XlaTensor::XlaTensor(void *data, int64_t num_elem,
                         ::tensorflow::DataType tf_dtype, int64_t size) {
  _data = data;
  _num_elem = num_elem;
  _tf_dtype = tf_dtype;
  _size = size;
}

const common::DataType XlaTensor::dtype() const {
  return ConvertDType(_tf_dtype);
}

const common::TensorShape XlaTensor::shape() const {
  common::TensorShape shape;
  shape.AddDim(_num_elem);
  return shape;
}

const void* XlaTensor::data() const {
  return (const void*)_data;
}

int64_t XlaTensor::size() const { return _size; }


// On GPU this event will signal that data is ready, and tensors are
// allocated.
common::ReadyEvent* RecordReadyEvent(::tensorflow::OpKernelContext* context) {
  auto device_context = context->op_device_context();
  if (device_context != nullptr) {
    return new TFReadyEvent(device_context);
  }
  return nullptr;
}

std::shared_ptr<common::ReadyEvent> RecordReadyEvent(cudaStream_t stream) {
  return std::make_shared<XlaReadyEvent>(stream);
}

extern "C" void byteps_tensorflow_declare_tensor(char* name) {
  std::string tensor_name(name);
  common::IsTensorDeclared(tensor_name);
  return;
}

void StartTask(::tensorflow::OpKernelContext* context,
               ::tensorflow::AsyncOpKernel::DoneCallback done,
               std::string node_name, std::shared_ptr<TFTensor> byteps_input,
               std::shared_ptr<TFTensor> byteps_output,
               std::shared_ptr<common::ReadyEvent> ready_event) {
  auto& byteps_context = common::GetContextFromName(node_name);
  auto device = GetDeviceID(context);
  auto size = byteps_input->size();
  auto dtype = byteps_input->dtype();
  void* cpubuff = (device == CPU_DEVICE_ID)
                      ? const_cast<void*>(byteps_input->data())
                      : nullptr;
  common::InitTensor(byteps_context, size, dtype, cpubuff);

  auto queue_list = common::GetPushQueueList(device);
  auto queue_list_pull = common::GetPullQueueList(device);
  queue_list->insert(queue_list->end(), queue_list_pull->begin(),
                     queue_list_pull->end());

  int my_rank =  common::byteps_rank();
  std::string name_key(node_name);
  std::replace(name_key.begin(), name_key.end(), '/', '_');
  BPS_LOG(DEBUG, my_rank) << " x2682 enqueuing name_key " << name_key <<
    " data_ptr: " << byteps_input->data() << " data_size: " << size << std::endl;

  // TODO: assign priority based on topological sort
  auto enqueue_result =
      EnqueueTensor(byteps_context, byteps_input, byteps_output, ready_event,
                    device, -byteps_context.declared_key, 0,
                    [context, done](const common::Status& status) {
                      context->SetStatus(ConvertStatus(status));
                      done();
                    },
                    queue_list);
  OP_REQUIRES_OK_ASYNC(context, ConvertStatus(enqueue_result), done);
}

class BytePSPushPullOp : public ::tensorflow::AsyncOpKernel {
  private:
     std::string input_tensor_name;
 public:
  explicit BytePSPushPullOp(::tensorflow::OpKernelConstruction* context)
      : AsyncOpKernel(context) {
          context->GetAttr("input_name", &input_tensor_name);
      }

  void ComputeAsync(::tensorflow::OpKernelContext* context,
                    DoneCallback done) override {
    OP_REQUIRES_OK_ASYNC(context, ConvertStatus(common::CheckInitialized()),
                         done);

    auto tensor = context->input(0);
    ::tensorflow::Tensor* output;
    OP_REQUIRES_OK_ASYNC(
        context, context->allocate_output(0, tensor.shape(), &output), done);
    // ReadyEvent makes sure input tensor is ready, and output is allocated.
    auto ready_event =
        std::shared_ptr<common::ReadyEvent>(RecordReadyEvent(context));
    auto bps_input = std::make_shared<TFTensor>(tensor);
    auto bps_output = std::make_shared<TFTensor>(*output);
    auto node_name = name();
    std::string tmp_name;
    if (input_tensor_name == "default_tensor_name") {
        tmp_name = node_name;
    } else {
        tmp_name = input_tensor_name;
    }
    auto& bps_context = common::GetContextFromName(tmp_name);
    if (bps_context.initialized) {
      StartTask(context, done, tmp_name, bps_input, bps_output, ready_event);
    } else {
      std::thread t(StartTask, context, done, tmp_name, bps_input, bps_output,
                    ready_event);
      t.detach();
    }
  }
};

REGISTER_KERNEL_BUILDER(Name("BytepsPushPull").Device(::tensorflow::DEVICE_CPU),
                        BytePSPushPullOp);
REGISTER_KERNEL_BUILDER(Name("BytepsPushPull").Device(::tensorflow::DEVICE_GPU),
                        BytePSPushPullOp);

REGISTER_OP("BytepsPushPull")
    .Attr("T: {int32, int64, float16, float32, float64}")
    .Attr("input_name: string = 'default_tensor_name'")
    .Input("tensor: T")
    .Output("sum: T")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return ::tensorflow::Status::OK();
    })
    .Doc(R"doc(
Perform an PushPull on a tensor. All other processes that do a reduction
on a tensor with the same name must have the same dimension for that tensor.
Tensors are reduced with other tensors that have the same node name for the
push_pull.
Arguments
    tensor:     A tensor to reduce.
Output
    sum:    A tensor with the same shape as `tensor`, summed across all processes.
)doc");

REGISTER_OP("BytepsPushPullXla")
  .Attr("T: {int32, int64, float16, float32, float64}")
  // .Attr("TT: {int32, int64, float16, float32, float64}")
  .Attr("input_name: string = 'default_tensor_name'")
  .Input("tensor: T")
  .Output("sum: T")
  .Output("out_handle: int32")
  .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    c->set_output(0, c->input(0));
    return ::tensorflow::Status::OK();
  });


class BytepsPushPullXlaOp : public ::tensorflow::XlaOpKernel {
  public:
    explicit BytepsPushPullXlaOp(::tensorflow::OpKernelConstruction* context) : ::tensorflow::XlaOpKernel(context) {
      context->GetAttr("input_name", &input_tensor_name);
    }
    ~BytepsPushPullXlaOp() override = default;

    void Compile(::tensorflow::XlaOpKernelContext* context) override {
      int my_rank = common::byteps_rank();
      OP_REQUIRES_OK(context, ConvertStatus(common::CheckInitialized()));
      BPS_LOG(DEBUG, my_rank) << " x2682 inside " << __func__ << std::endl;

      xla::XlaOp input_tensor = context->Input(0);
      const ::tensorflow::TensorShape input_tensor_shape = context->InputShape(0);
      auto input_tensor_xla_shape_or =
          TensorShapeToXLAShape(context->input_xla_type(0), input_tensor_shape);
      // xla::Shape output_tensor_shape = input_tensor_xla_shape_or.ValueOrDie();
      xla::Shape output_tensor_shape = input_tensor_xla_shape_or;

      auto node_name = name();
      std::string tmp_name;
      if (input_tensor_name == "default_tensor_name") {
        tmp_name = node_name;
      } else {
        tmp_name = input_tensor_name;
      }

      std::stringstream ss;
      ss << tmp_name;
      ss << " " << context->input_type(0);
      ss << " " << xla::ShapeUtil::ByteSizeOfPrimitiveType(output_tensor_shape.element_type());
      ss << " " << output_tensor_shape.rank();
      for (int i = 0; i < output_tensor_shape.rank(); i++) {
        ss << " " << output_tensor_shape.dimensions(i) ;
      }
      ss << std::endl;
      auto output_shape = xla::ShapeUtil::MakeShape(xla::S32, {2});
      context->SetOutput(
        1, xla::CustomCall(context->builder(),
          /*call_target_name=*/"StartTaskWrapper",
          {input_tensor}, output_shape, ss.str()));

      context->op_kernel_context()->set_output(0,
        context->op_kernel_context()->input(0));
    }

  private:
     std::string input_tensor_name;
};

REGISTER_XLA_OP(Name("BytepsPushPullXla").Device(::tensorflow::DEVICE_GPU_XLA_JIT),
    BytepsPushPullXlaOp);

////////////////////////////////////////////////////////////////////////////////
class BytepsPushPullKickoffXlaOp : public ::tensorflow::XlaOpKernel {
  public:
    explicit BytepsPushPullKickoffXlaOp(::tensorflow::OpKernelConstruction* context) : ::tensorflow::XlaOpKernel(context) {
      context->GetAttr("input_name", &input_tensor_name);
    }
    ~BytepsPushPullKickoffXlaOp() override = default;

    void Compile(::tensorflow::XlaOpKernelContext* context) override {
      int my_rank = common::byteps_rank();
      OP_REQUIRES_OK(context, ConvertStatus(common::CheckInitialized()));
      BPS_LOG(DEBUG, my_rank) << " x2682 inside " << __func__ << std::endl;

      xla::XlaOp input_tensor = context->Input(0);
      xla::XlaOp dummy_tensor = context->Input(1);
      const ::tensorflow::TensorShape input_tensor_shape = context->InputShape(0);
      auto input_tensor_xla_shape_or =
          TensorShapeToXLAShape(context->input_xla_type(0), input_tensor_shape);
      // xla::Shape output_tensor_shape = input_tensor_xla_shape_or.ValueOrDie();
      xla::Shape output_tensor_shape = input_tensor_xla_shape_or;

      auto node_name = name();
      std::string tmp_name;
      if (input_tensor_name == "default_tensor_name") {
        tmp_name = node_name;
        BPS_LOG(DEBUG, my_rank) << " x2682 inside " << __func__
                                << " tmp_name: " << tmp_name << " node_name: "
                                << node_name <<std::endl;

      } else {
        tmp_name = input_tensor_name;
        BPS_LOG(DEBUG, my_rank) << " x2682 inside " << __func__
                                << " tmp_name: " << tmp_name << " node_name: "
                                << node_name <<std::endl;
      }

      std::stringstream ss;
      ss << tmp_name;
      ss << " " << context->input_type(0);
      ss << " " << xla::ShapeUtil::ByteSizeOfPrimitiveType(output_tensor_shape.element_type());
      ss << " " << output_tensor_shape.rank();
      for (int i = 0; i < output_tensor_shape.rank(); i++) {
        ss << " " << output_tensor_shape.dimensions(i) ;
      }
      ss << std::endl;
      auto output_shape = xla::ShapeUtil::MakeShape(xla::S32, {2});
      context->SetOutput(
        0, xla::CustomCall(context->builder(),
          /*call_target_name=*/"customBytepsPushPullKickoffXlaOp",
          {input_tensor}, output_tensor_shape, ss.str()));

      // context->op_kernel_context()->set_output(0,
      //   context->op_kernel_context()->input(0));
    }
    bool IsExpensive() override { return true; }

  private:
     std::string input_tensor_name;
};

void PushPullKickoff(::tensorflow::OpKernelContext* context,
               std::string node_name, std::shared_ptr<common::Tensor> byteps_input,
               std::shared_ptr<common::Tensor> byteps_output,
               std::shared_ptr<common::ReadyEvent> ready_event) {
  int my_rank =  common::byteps_rank();
  BPS_LOG(DEBUG, my_rank) << " x2682 entered " << __PRETTY_FUNCTION__ << " "
    << node_name  << node_name << std::endl;

  auto& byteps_context = common::GetContextFromName(node_name);
  int device;
  CUDA_CALL(cudaGetDevice(&device));
  auto size = byteps_input->size();
  auto dtype = byteps_input->dtype();
  void* cpubuff = nullptr;
  common::InitTensor(byteps_context, size, dtype, cpubuff);

  auto queue_list = common::GetPushQueueList(device);
  auto queue_list_pull = common::GetPullQueueList(device);
  queue_list->insert(queue_list->end(), queue_list_pull->begin(),
                     queue_list_pull->end());

  std::mutex mtx;
  std::condition_variable cv;
  // TODO: assign priority based on topological sort

  std::string name_key(node_name);
  std::replace(name_key.begin(), name_key.end(), '/', '_');

  std::shared_ptr<Xla_done_cb_args> new_args(new Xla_done_cb_args);
  new_args->is_done = false;
  new_args->bps_out_buf = const_cast<void *>(byteps_output->data());
  new_args->bps_in_buf = const_cast<void *>(byteps_input->data());
  new_args->bps_buf_size = size;
  new_args->num_waiting = 1;

  BPS_LOG(DEBUG, my_rank) << " x2682 inserting name_key " << name_key <<
    " data_ptr: " << new_args->bps_in_buf << " data_size: " << size <<
    " out_data_ptr: " << new_args->bps_out_buf << std::endl;
  std::unique_lock<std::mutex> my_lk(_name_to_done_args_mtx);
  auto it = _name_to_done_args.find(name_key);
  // ASSERTF(it == _name_to_done_args.end(),
  //         std::string("duplicate tensor_name ") + std::string(name_key));
  if (it != _name_to_done_args.end()) {
    it->second->num_waiting++;
    if (it->second->num_waiting > 1) {
      BPS_LOG(DEBUG, my_rank) << " x2682 seen name_key " << name_key << std::endl;
      my_lk.unlock();
      return;
    }
  }
  _name_to_done_args[name_key] = new_args;
  my_lk.unlock();
  _name_to_done_args_cv.notify_all();
  /////////////
  return;
  /////////////
  auto enqueue_result =
      EnqueueTensor(byteps_context, byteps_input, byteps_output, ready_event,
                    device, -byteps_context.declared_key, 0,
                    [name_key, my_rank](const common::Status& status) {
                      std::unique_lock<std::mutex> my_lk(_name_to_done_args_mtx);
                      auto it = _name_to_done_args.find(name_key);
                      ASSERTF(it != _name_to_done_args.end(), "YOU SHOULD NOT SEE ME");
                      auto args = _name_to_done_args[name_key];
                      my_lk.unlock();
                      {
                        std::lock_guard<std::mutex> lk(args->mtx);
                        args->is_done = true;
                        BPS_LOG(DEBUG, my_rank) << " x2682 push_pull done name_key " << name_key << std::endl;
                      }

                      args->cv.notify_one();
                    },
                    queue_list);
  if (ConvertStatus(enqueue_result) != ::tensorflow::Status::OK()) {
    assert(0);
  }
}

void customBytepsPushPullKickoffXlaOp(CUstream stream, void** buffers,
                      const char* opaque, size_t opaque_len) {
  int my_rank = common::byteps_rank();
  BPS_LOG(DEBUG, my_rank) << " x2682 enter " << __func__ << std::endl;
  std::stringstream ss(opaque);
  std::string tmp_name;
  ::tensorflow::OpKernelContext* context = nullptr;

  ss >> tmp_name;
  ::tensorflow::DataType dt_type;
  int tmp_dt_type;
  ss >> std::dec >> tmp_dt_type;
  dt_type = static_cast<::tensorflow::DataType>(tmp_dt_type);
  size_t elem_size;
  ss >> elem_size;
  int ndim = 0;
  ss >> std::dec >> ndim;
  size_t buffer_size = 0;
  size_t num_elem = 1;
  for (int i = 0; i < ndim; i++) {
    size_t dim;
    ss >> std::dec >> dim;
    num_elem *= dim;
  }

  buffer_size = elem_size * num_elem;

  auto bps_input = std::make_shared<XlaTensor>(buffers[1], num_elem, dt_type, buffer_size);
  auto bps_output = std::make_shared<XlaTensor>(buffers[1], num_elem, dt_type, buffer_size);

  cudaMemcpyAsync(buffers[1], buffers[0], buffer_size, cudaMemcpyDeviceToDevice, stream);
  auto ready_event =
    std::shared_ptr<common::ReadyEvent>(RecordReadyEvent(stream));

  std::thread t(PushPullKickoff, context, tmp_name, bps_input, bps_input, ready_event);
  t.detach();
  BPS_LOG(DEBUG, my_rank) << " x2682 exit " << __func__ << std::endl;
}

REGISTER_OP("BytepsPushPullKickoffXla")
  .Attr("T: {int32, int64, float16, float32, float64}")
  // .Attr("TT: {int32, int64, float16, float32, float64}")
  .Attr("input_name: string = 'default_tensor_name'")
  .Input("input_tensor: T")
  .Input("input_dummy: int32")
  .Output("output_tensor: T")
  .SetIsStateful()
  .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    c->set_output(0, c->input(0));
    return ::tensorflow::Status::OK();
  });
REGISTER_XLA_OP(Name("BytepsPushPullKickoffXla").Device(::tensorflow::DEVICE_GPU_XLA_JIT),
    BytepsPushPullKickoffXlaOp);
XLA_REGISTER_CUSTOM_CALL_TARGET(customBytepsPushPullKickoffXlaOp, "CUDA");

//////
void WaitForTensor(std::string tensor_name,
              ::tensorflow::AsyncOpKernel::DoneCallback done) {
  int my_rank =  common::byteps_rank();

  std::unique_lock<std::mutex> my_big_lk(_name_to_done_args_mtx);
  _name_to_done_args_cv.wait(my_big_lk,
    [&tensor_name]{
      std::this_thread::yield();
      return _name_to_done_args.find(tensor_name) != _name_to_done_args.end();
    });

  auto it = _name_to_done_args.find(tensor_name);
  ASSERTF(it != _name_to_done_args.end(), "post 2");
  auto args = _name_to_done_args[tensor_name];
  my_big_lk.unlock();

  {
    std::unique_lock<std::mutex> lk(args->mtx);
    args->cv.wait(lk, [args]{
      std::this_thread::yield();
      return args->is_done;});
    lk.unlock();
  }
  BPS_LOG(DEBUG, my_rank) << " x2682 wait done on name_key " << tensor_name << std::endl;
  done();
}

class BytePSPushPullWaitOp : public ::tensorflow::AsyncOpKernel {
  private:
     std::string input_tensor_name;
 public:
  explicit BytePSPushPullWaitOp(::tensorflow::OpKernelConstruction* context)
      : AsyncOpKernel(context) {
          context->GetAttr("input_name", &input_tensor_name);
      }

  void ComputeAsync(::tensorflow::OpKernelContext* context,
                    DoneCallback done) override {
    int my_rank =  common::byteps_rank();
    OP_REQUIRES_OK_ASYNC(context, ConvertStatus(common::CheckInitialized()),
                         done);
    BPS_LOG(DEBUG, my_rank) << " x2682 about to wait on name_key "
      << input_tensor_name
      << " data_ptr: " << (void *) context->input(0).tensor_data().data() << std::endl;

    context->set_output(0, context->input(0));
    // std::thread t(WaitForTensor, input_tensor_name, done);
    // t.detach();
    done();
  }
};

REGISTER_OP("BytepsPushPullWait")
  .Attr("T: {int32, int64, float16, float32, float64}")
  // .Attr("TT: {int32, int64, float16, float32, float64}")
  .Attr("input_name: string = 'default_tensor_name'")
  .Input("input_tensor: T")
  .Output("output_tensor: T")
  .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    c->set_output(0, c->input(0));
    return ::tensorflow::Status::OK();
  });
REGISTER_KERNEL_BUILDER(Name("BytepsPushPullWait").Device(::tensorflow::DEVICE_GPU),
                        BytePSPushPullWaitOp);
////////////////////////////////////////////////////////////////////////////////
class BarrierHandleOutXlaOp : public ::tensorflow::XlaOpKernel {
 public:
  explicit BarrierHandleOutXlaOp(::tensorflow::OpKernelConstruction* ctx) : ::tensorflow::XlaOpKernel(ctx) { }
  void Compile(::tensorflow::XlaOpKernelContext* ctx) override {
    std::vector<xla::XlaOp> values;
    std::vector<::tensorflow::TensorShape> shapes;
    OP_REQUIRES_OK(ctx, ctx->InputList("values", &values, &shapes));

    std::cout << "my_name is " << name() << std::endl;
    // S32 is int32
    auto out_shapes = xla::ShapeUtil::MakeTupleShape({
      xla::ShapeUtil::MakeShape(xla::S32, {2})
    });
    auto results = xla::CustomCall(ctx->builder(),
                           /*call_target_name=*/"customBarrierHandleOutXlaOp",
                           values, out_shapes);
    auto out_0 = xla::GetTupleElement(results, 0);
    ctx->SetOutput(0, out_0);
  }
 private:
  std::vector<std::string> token_input_nodes_;
};

void
customBarrierHandleOutXlaOp(CUstream stream, void** buffers,
                  const char* opaque, size_t opaque_len) {
  int* out_buf = reinterpret_cast<int*>(buffers[1]);
  const int* in_buf = reinterpret_cast<const int*>(buffers[0]);
  std::cout<< "xxxxxxxxxxxxxxxxxxx side effect in " << __PRETTY_FUNCTION__ << std::endl;
}

REGISTER_OP("MyBarrierHandleOut")
    .Attr("T: {int32, int64, float16, float32, float64}")
    .Attr("N: int >= 1")
    .Input("values: N * T")
    .Output("output: int32");

REGISTER_XLA_OP(Name("MyBarrierHandleOut").Device(::tensorflow::DEVICE_GPU_XLA_JIT),
                BarrierHandleOutXlaOp);
XLA_REGISTER_CUSTOM_CALL_TARGET(customBarrierHandleOutXlaOp, "CUDA");

////////////////////////////////////////////////////////////////////////////////

class BytepsSyncTensorHandleOutXlaOp : public ::tensorflow::XlaOpKernel {
  public:
    explicit BytepsSyncTensorHandleOutXlaOp(::tensorflow::OpKernelConstruction* context) : ::tensorflow::XlaOpKernel(context) {
      context->GetAttr("tensor_name", &input_tensor_name);
    }
    ~BytepsSyncTensorHandleOutXlaOp() override = default;

    void Compile(::tensorflow::XlaOpKernelContext* context) override {
      OP_REQUIRES_OK(context, ConvertStatus(common::CheckInitialized()));
      xla::XlaOp input_handle = context->Input(0);
      xla::XlaOp input_tensor = context->Input(1);
      const ::tensorflow::TensorShape input_handle_shape = context->InputShape(0);
      auto input_handle_xla_shape_or =
          TensorShapeToXLAShape(context->input_xla_type(0), input_handle_shape);

      auto node_name = name();
      std::string tmp_name;
      if (input_tensor_name == "default_tensor_name") {
        tmp_name = node_name;
      } else {
        tmp_name = input_tensor_name;
      }

      std::stringstream ss;
      ss << tmp_name;
      ss << std::endl;
      context->SetOutput(
        0, xla::CustomCall(context->builder(),
          /*call_target_name=*/"SyncTensorHandleOutCustomOp",
          {input_handle, input_tensor}, input_handle_xla_shape_or, ss.str()));
    }

  private:
     std::string input_tensor_name;
};

void SyncTensorHandleOutCustomOp(CUstream stream, void** buffers,
                      const char* opaque, size_t opaque_len) {
  int my_rank = common::byteps_rank();
  std::string tmp_name;
  std::stringstream ss(opaque);
  int* handle_buf = reinterpret_cast<int*>(buffers[2]);

  /* set handle[0] = 0, handle[1] = 0x01010101 which is 16843009 in decimal */
  cudaMemset(handle_buf, 0, sizeof(int));
  cudaMemset(handle_buf + 1, 1, sizeof(int));

  ss >> tmp_name;
  BPS_LOG(DEBUG, my_rank) << " x2682 inside " << __func__ <<
      " waiting for name_key: " << tmp_name << std::endl;

  std::unique_lock<std::mutex> my_big_lk(_name_to_done_args_mtx);
  _name_to_done_args_cv.wait(my_big_lk,
    [&tmp_name]{
      std::this_thread::yield();
      return _name_to_done_args.find(tmp_name) != _name_to_done_args.end();
    });

  auto it = _name_to_done_args.find(tmp_name);
  ASSERTF(it != _name_to_done_args.end(), "post 2");
  auto args = _name_to_done_args[tmp_name];
  my_big_lk.unlock();
  BPS_LOG(DEBUG, my_rank) << " x2682 " << __func__ <<
      " got name_key: " << tmp_name << std::endl;
  {
    std::unique_lock<std::mutex> lk(args->mtx);
    args->cv.wait(lk, [args]{
      std::this_thread::yield();
      return args->is_done;});
    args->num_waiting--;
    if (args->num_waiting == 0) {
      args->is_done = false;
    } else {
      BPS_LOG(DEBUG, my_rank) << " x2682 exiting, i am not the last sync on " <<
      " name_key: " << tmp_name << std::endl;
    }
    lk.unlock();
  }
  std::unique_lock<std::mutex> my_lk(_name_to_done_args_mtx);
  _name_to_done_args.erase(tmp_name);
  my_lk.unlock();
  BPS_LOG(DEBUG, my_rank) << " x2682 sync done name_key " << tmp_name << std::endl;
}

REGISTER_OP("BytepsSyncTensorHandleOut")
    .Attr("T: {int32, int64, float16, float32, float64}")
    .Attr("tensor_name: string = 'default_tensor_name'")
    .Input("in_handle: int32")
    .Input("in_tensor: T")
    .Output("out_handle: int32")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return ::tensorflow::Status::OK();
    });
REGISTER_XLA_OP(Name("BytepsSyncTensorHandleOut").Device(::tensorflow::DEVICE_GPU_XLA_JIT),
  BytepsSyncTensorHandleOutXlaOp);
XLA_REGISTER_CUSTOM_CALL_TARGET(SyncTensorHandleOutCustomOp, "CUDA");
////////////////////////////////////////////////////////////////////////////////

class BytepsSyncTensorHandleOutV2XlaOp : public ::tensorflow::XlaOpKernel {
  public:
    explicit BytepsSyncTensorHandleOutV2XlaOp(::tensorflow::OpKernelConstruction* context) : ::tensorflow::XlaOpKernel(context) {
      context->GetAttr("tensor_name", &input_tensor_name);
    }
    ~BytepsSyncTensorHandleOutV2XlaOp() override = default;

    void Compile(::tensorflow::XlaOpKernelContext* context) override {
      OP_REQUIRES_OK(context, ConvertStatus(common::CheckInitialized()));
      xla::XlaOp input_handle = context->Input(0);
      xla::XlaOp input_tensor = context->Input(1);
      const ::tensorflow::TensorShape input_handle_shape = context->InputShape(0);
      auto input_handle_xla_shape_or =
          TensorShapeToXLAShape(context->input_xla_type(0), input_handle_shape);

      auto node_name = name();
      std::string tmp_name;
      if (input_tensor_name == "default_tensor_name") {
        tmp_name = node_name;
      } else {
        tmp_name = input_tensor_name;
      }

      std::stringstream ss;
      ss << tmp_name;
      ss << std::endl;
      context->SetOutput(
        0, xla::CustomCall(context->builder(),
          /*call_target_name=*/"SyncTensorHandleOutV2CustomOp",
          {input_handle, input_tensor}, input_handle_xla_shape_or, ss.str()));
    }

  private:
     std::string input_tensor_name;
};

void SyncTensorHandleOutV2CustomOp(CUstream stream, void** buffers,
                      const char* opaque, size_t opaque_len) {
  int my_rank = common::byteps_rank();
  std::string tmp_name;
  std::stringstream ss(opaque);
  int* handle_buf = reinterpret_cast<int*>(buffers[2]);

  /* set handle[0] = 0, handle[1] = 0x01010101 which is 16843009 in decimal */
  cudaMemset(handle_buf, 0, sizeof(int));
  cudaMemset(handle_buf + 1, 1, sizeof(int));

  ss >> tmp_name;
  BPS_LOG(DEBUG, my_rank) << " x2682 inside " << __func__ <<
      " waiting for name_key: " << tmp_name << std::endl;

  std::unique_lock<std::mutex> my_big_lk(_name_to_done_args_mtx);
  _name_to_done_args_cv.wait(my_big_lk,
    [&tmp_name]{
      std::this_thread::yield();
      return _name_to_done_args.find(tmp_name) != _name_to_done_args.end();
    });

  auto it = _name_to_done_args.find(tmp_name);
  ASSERTF(it != _name_to_done_args.end(), "post 2");
  auto args = _name_to_done_args[tmp_name];
  my_big_lk.unlock();
  BPS_LOG(DEBUG, my_rank) << " x2682 " << __func__ <<
      " got name_key: " << tmp_name << std::endl;
  {
    std::unique_lock<std::mutex> lk(args->mtx);
    args->cv.wait(lk, [args]{
      std::this_thread::yield();
      return args->is_done;});
    args->num_waiting--;
    if (args->num_waiting == 0) {
      args->is_done = false;
    } else {
      BPS_LOG(DEBUG, my_rank) << " x2682 exiting, i am not the last sync on " <<
      " name_key: " << tmp_name << std::endl;
    }
    lk.unlock();
  }
  std::unique_lock<std::mutex> my_lk(_name_to_done_args_mtx);
  _name_to_done_args.erase(tmp_name);
  my_lk.unlock();
  BPS_LOG(DEBUG, my_rank) << " x2682 sync done name_key " << tmp_name << std::endl;
}

REGISTER_OP("BytepsSyncTensorHandleOutV2")
    .Attr("T: {int32, int64, float16, float32, float64}")
    .Attr("tensor_name: string = 'default_tensor_name'")
    .Input("in_handle: int32")
    .Input("in_tensor: T")
    .Output("out_handle: int32")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return ::tensorflow::Status::OK();
    });
REGISTER_XLA_OP(Name("BytepsSyncTensorHandleOutV2").Device(::tensorflow::DEVICE_GPU_XLA_JIT),
  BytepsSyncTensorHandleOutV2XlaOp);
XLA_REGISTER_CUSTOM_CALL_TARGET(SyncTensorHandleOutV2CustomOp, "CUDA");
////////////////////////////////////////////////////////////////////////////////

class BytepsPushPullBlockingXlaOp : public ::tensorflow::XlaOpKernel {
  public:
    explicit BytepsPushPullBlockingXlaOp(::tensorflow::OpKernelConstruction* context) : ::tensorflow::XlaOpKernel(context) {
      context->GetAttr("input_name", &input_tensor_name);
    }
    ~BytepsPushPullBlockingXlaOp() override = default;

    void Compile(::tensorflow::XlaOpKernelContext* context) override {
      int my_rank = common::byteps_rank();
      OP_REQUIRES_OK(context, ConvertStatus(common::CheckInitialized()));

      xla::XlaOp input_tensor = context->Input(0);
      const ::tensorflow::TensorShape input_tensor_shape = context->InputShape(0);
      auto input_tensor_xla_shape_or =
          TensorShapeToXLAShape(context->input_xla_type(0), input_tensor_shape);
      // xla::Shape output_tensor_shape = input_tensor_xla_shape_or.ValueOrDie();
      xla::Shape output_tensor_shape = input_tensor_xla_shape_or;

      auto node_name = name();
      std::string tmp_name;
      if (input_tensor_name == "default_tensor_name") {
        tmp_name = node_name;
      } else {
        tmp_name = input_tensor_name;
      }

      std::stringstream ss;
      ss << tmp_name;
      ss << " " << context->input_type(0);
      ss << " " << xla::ShapeUtil::ByteSizeOfPrimitiveType(output_tensor_shape.element_type());
      ss << " " << output_tensor_shape.rank();
      for (int i = 0; i < output_tensor_shape.rank(); i++) {
        ss << " " << output_tensor_shape.dimensions(i) ;
      }
      ss << std::endl;
      context->SetOutput(
        0, xla::CustomCall(context->builder(),
          /*call_target_name=*/"StartTaskBlockingWrapper",
          {input_tensor}, input_tensor_xla_shape_or, ss.str()));
    }
  private:
     std::string input_tensor_name;
};

void StartTaskBlockingXla(::tensorflow::OpKernelContext* context,
               std::string node_name, std::shared_ptr<common::Tensor> byteps_input,
               std::shared_ptr<common::Tensor> byteps_output,
               std::shared_ptr<common::ReadyEvent> ready_event) {
  int my_rank = common::byteps_rank();
  auto& byteps_context = common::GetContextFromName(node_name);
  int device;
  CUDA_CALL(cudaGetDevice(&device));
  int myrank =  common::byteps_rank();
  auto size = byteps_input->size();
  auto dtype = byteps_input->dtype();
  void* cpubuff = nullptr;
  common::InitTensor(byteps_context, size, dtype, cpubuff);

  auto queue_list = common::GetPushQueueList(device);
  auto queue_list_pull = common::GetPullQueueList(device);
  queue_list->insert(queue_list->end(), queue_list_pull->begin(),
                     queue_list_pull->end());

  // TODO: assign priority based on topological sort

  std::string name_key(node_name);
  std::replace(name_key.begin(), name_key.end(), '/', '_');

  std::shared_ptr<Xla_done_cb_args> new_args(new Xla_done_cb_args);
  new_args->is_done = false;
  new_args->bps_out_buf = const_cast<void *>(byteps_output->data());
  new_args->bps_in_buf = const_cast<void *>(byteps_input->data());
  new_args->bps_buf_size = size;

  std::unique_lock<std::mutex> my_lk(_name_to_done_args_mtx);
  auto it = _name_to_done_args.find(name_key);
  ASSERTF(it == _name_to_done_args.end(), "duplicate tensor_name");
  _name_to_done_args[name_key] = new_args;
  my_lk.unlock();

  std::cout << "enter queue " << node_name << " rank " << my_rank << std::endl;
  auto enqueue_result =
      EnqueueTensor(byteps_context, byteps_input, byteps_output, ready_event,
                    device, -byteps_context.declared_key, 0,
                    [name_key](const common::Status& status) {
                      std::unique_lock<std::mutex> my_lk(_name_to_done_args_mtx);
                      auto it = _name_to_done_args.find(name_key);
                      ASSERTF(it != _name_to_done_args.end(), "YOU SHOULD NOT SEE ME");
                      auto args = _name_to_done_args[name_key];
                      my_lk.unlock();
                      {
                        std::unique_lock<std::mutex> lk(args->mtx);
                        args->is_done = true;
                      }
                      args->cv.notify_one();
                    },
                    queue_list);
  {
    std::unique_lock<std::mutex> my_big_lk(_name_to_done_args_mtx);
    auto args = _name_to_done_args[name_key];
    my_big_lk.unlock();
    std::unique_lock<std::mutex> lk(args->mtx);
    args->cv.wait(lk, [args]{
      std::this_thread::yield();
      return args->is_done;});
    lk.unlock();
  }
  std::cout << "exit queue " << node_name << " rank " << my_rank << std::endl;
}

void StartTaskBlockingWrapper(CUstream stream, void** buffers,
                      const char* opaque, size_t opaque_len) {
    std::stringstream ss(opaque);
    std::string tmp_name;
    ::tensorflow::OpKernelContext* context = nullptr;

    ss >> tmp_name;
    ::tensorflow::DataType dt_type;
    int tmp_dt_type;
    ss >> std::dec >> tmp_dt_type;
    dt_type = static_cast<::tensorflow::DataType>(tmp_dt_type);
    size_t elem_size;
    ss >> elem_size;
    int ndim = 0;
    ss >> std::dec >> ndim;
    size_t buffer_size = 0;
    size_t num_elem = 1;
    for (int i = 0; i < ndim; i++) {
      size_t dim;
      ss >> std::dec >> dim;
      num_elem *= dim;
    }

    buffer_size = elem_size * num_elem;
    ::tensorflow::PlatformGpuId platform_gpu_id(0);

    auto bps_input = std::make_shared<XlaTensor>(buffers[0], num_elem, dt_type, buffer_size);
    auto ready_event =
        std::shared_ptr<common::ReadyEvent>(RecordReadyEvent(stream));

    ::tensorflow::Tensor outputTensor(dt_type, ::tensorflow::TensorShape({num_elem}));
    auto bps_output = std::make_shared<XlaTensor>(buffers[1], num_elem, dt_type, buffer_size);

    StartTaskBlockingXla(context, tmp_name, bps_input, bps_output, ready_event);
}

XLA_REGISTER_CUSTOM_CALL_TARGET(StartTaskBlockingWrapper, "CUDA");

REGISTER_XLA_OP(Name("BytepsPushPullBlocking"), BytepsPushPullBlockingXlaOp);
REGISTER_KERNEL_BUILDER(Name("BytepsPushPullBlocking").Device(::tensorflow::DEVICE_CPU),
                        BytePSPushPullOp);
REGISTER_KERNEL_BUILDER(Name("BytepsPushPullBlocking").Device(::tensorflow::DEVICE_GPU),
                        BytePSPushPullOp);

REGISTER_OP("BytepsPushPullBlocking")
    .Attr("T: {int32, int64, float16, float32, float64}")
    .Attr("input_name: string = 'default_tensor_name'")
    .Input("tensor: T")
    .Output("sum: T")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return ::tensorflow::Status::OK();
    });

void StartTaskXla(::tensorflow::OpKernelContext* context,
               std::string node_name, std::shared_ptr<common::Tensor> byteps_input,
               std::shared_ptr<common::Tensor> byteps_output,
               std::shared_ptr<common::ReadyEvent> ready_event) {
  int my_rank =  common::byteps_rank();
  BPS_LOG(DEBUG, my_rank) << " x2682 inserting node_name " << node_name << std::endl;

  auto& byteps_context = common::GetContextFromName(node_name);
  int device;
  CUDA_CALL(cudaGetDevice(&device));
  auto size = byteps_input->size();
  auto dtype = byteps_input->dtype();
  void* cpubuff = nullptr;
  common::InitTensor(byteps_context, size, dtype, cpubuff);

  auto queue_list = common::GetPushQueueList(device);
  auto queue_list_pull = common::GetPullQueueList(device);
  queue_list->insert(queue_list->end(), queue_list_pull->begin(),
                     queue_list_pull->end());

  std::mutex mtx;
  std::condition_variable cv;
  // TODO: assign priority based on topological sort

  std::string name_key(node_name);
  std::replace(name_key.begin(), name_key.end(), '/', '_');

  std::shared_ptr<Xla_done_cb_args> new_args(new Xla_done_cb_args);
  new_args->is_done = false;
  new_args->bps_out_buf = const_cast<void *>(byteps_output->data());
  new_args->bps_in_buf = const_cast<void *>(byteps_input->data());
  new_args->bps_buf_size = size;
  new_args->num_waiting = 1;

  BPS_LOG(DEBUG, my_rank) << " x2682 inserting name_key " << name_key <<
    " data_ptr: " << new_args->bps_in_buf << std::endl;
  std::unique_lock<std::mutex> my_lk(_name_to_done_args_mtx);
  auto it = _name_to_done_args.find(name_key);
  ASSERTF(it == _name_to_done_args.end(), std::string("duplicate tensor_name ") +
    std::string(name_key));
  if (it != _name_to_done_args.end()) {
    it->second->num_waiting++;
    if (it->second->num_waiting > 1) {
      BPS_LOG(DEBUG, my_rank) << " x2682 seen name_key " << name_key << std::endl;
      my_lk.unlock();
      return;
    }
  }
  _name_to_done_args[name_key] = new_args;
  my_lk.unlock();
  _name_to_done_args_cv.notify_all();
  auto enqueue_result =
      EnqueueTensor(byteps_context, byteps_input, byteps_output, ready_event,
                    device, -byteps_context.declared_key, 0,
                    [name_key, my_rank](const common::Status& status) {
                      std::unique_lock<std::mutex> my_lk(_name_to_done_args_mtx);
                      auto it = _name_to_done_args.find(name_key);
                      ASSERTF(it != _name_to_done_args.end(), "YOU SHOULD NOT SEE ME");
                      auto args = _name_to_done_args[name_key];
                      my_lk.unlock();
                      {
                        std::lock_guard<std::mutex> lk(args->mtx);
                        args->is_done = true;
                        BPS_LOG(DEBUG, my_rank) << " x2682 done name_key " << name_key << std::endl;
                      }

                      args->cv.notify_one();
                    },
                    queue_list);
  if (ConvertStatus(enqueue_result) != ::tensorflow::Status::OK()) {
    assert(0);
  }
}

void StartTaskWrapper(CUstream stream, void** buffers,
                      const char* opaque, size_t opaque_len) {
  int my_rank = common::byteps_rank();
  BPS_LOG(DEBUG, my_rank) << " x2682 enter " << __func__ << std::endl;
  std::stringstream ss(opaque);
  std::string tmp_name;
  ::tensorflow::OpKernelContext* context = nullptr;

  ss >> tmp_name;
  ::tensorflow::DataType dt_type;
  int tmp_dt_type;
  ss >> std::dec >> tmp_dt_type;
  dt_type = static_cast<::tensorflow::DataType>(tmp_dt_type);
  size_t elem_size;
  ss >> elem_size;
  int ndim = 0;
  ss >> std::dec >> ndim;
  size_t buffer_size = 0;
  size_t num_elem = 1;
  for (int i = 0; i < ndim; i++) {
    size_t dim;
    ss >> std::dec >> dim;
    num_elem *= dim;
  }

  buffer_size = elem_size * num_elem;

  auto bps_input = std::make_shared<XlaTensor>(buffers[0], num_elem, dt_type, buffer_size);
  auto ready_event =
    std::shared_ptr<common::ReadyEvent>(RecordReadyEvent(stream));

  // auto bps_output = std::make_shared<XlaTensor>(buffers[1], num_elem, dt_type, buffer_size);
  auto bps_output = std::make_shared<XlaTensor>(buffers[0], num_elem, dt_type, buffer_size);

  std::thread t(StartTaskXla, context, tmp_name, bps_input, bps_input, ready_event);
  t.detach();
  BPS_LOG(DEBUG, my_rank) << " x2682 exit " << __func__ << std::endl;
}

XLA_REGISTER_CUSTOM_CALL_TARGET(StartTaskWrapper, "CUDA");

void SyncTensorCustomOp(CUstream stream, void** buffers,
                      const char* opaque, size_t opaque_len) {
  std::string tmp_name;
  std::stringstream ss(opaque);

  ss >> tmp_name;

  auto it = _name_to_done_args.find(tmp_name);
  ASSERTF(it != _name_to_done_args.end(), "post 2");
  auto args = it->second;
  {
    std::unique_lock<std::mutex> lk(args->mtx);
    args->cv.wait(lk, [args]{
      std::this_thread::yield();
      return args->is_done;});
    lk.unlock();
  }
  _name_to_done_args.erase(it);
}

class BytepsSyncTensorXlaOp : public ::tensorflow::XlaOpKernel {
  public:
    explicit BytepsSyncTensorXlaOp(::tensorflow::OpKernelConstruction* context) : ::tensorflow::XlaOpKernel(context) {
      context->GetAttr("input_name", &input_tensor_name);
    }
    ~BytepsSyncTensorXlaOp() override = default;

    void Compile(::tensorflow::XlaOpKernelContext* context) override {
      OP_REQUIRES_OK(context, ConvertStatus(common::CheckInitialized()));
      xla::XlaOp input_tensor = context->Input(0);
      const ::tensorflow::TensorShape input_tensor_shape = context->InputShape(0);
      auto input_tensor_xla_shape_or =
          TensorShapeToXLAShape(context->input_xla_type(0), input_tensor_shape);

      auto node_name = name();
      std::string tmp_name;
      if (input_tensor_name == "default_tensor_name") {
        tmp_name = node_name;
      } else {
        tmp_name = input_tensor_name;
      }

      std::stringstream ss;
      ss << tmp_name;
      ss << std::endl;
      context->SetOutput(
        0, xla::CustomCall(context->builder(),
          /*call_target_name=*/"SyncTensorCustomOp",
          {input_tensor}, input_tensor_xla_shape_or, ss.str()));

    }

  private:
     std::string input_tensor_name;
};

class BytePSSyncTensorOp : public ::tensorflow::OpKernel {
  public:
    explicit BytePSSyncTensorOp(::tensorflow::OpKernelConstruction* context) : ::tensorflow::OpKernel(context) {
      context->GetAttr("input_name", &input_tensor_name);
    }
    ~BytePSSyncTensorOp() override = default;

    void Compute(::tensorflow::OpKernelContext* context) override {
      OP_REQUIRES_OK(context, ConvertStatus(common::CheckInitialized()));
      auto input_tensor = context->input(0);

      auto node_name = name();
      std::string tmp_name;
      if (input_tensor_name == "default_tensor_name") {
        tmp_name = node_name;
      } else {
        tmp_name = input_tensor_name;
      }

      auto it = _name_to_done_args.find(tmp_name);
      ASSERTF(it != _name_to_done_args.end(), "pos 3");
      auto args = it->second;
      {
        std::unique_lock<std::mutex> lk(args->mtx);
        args->cv.wait(lk, [args]{
          std::this_thread::yield();
          return args->is_done;});
        lk.unlock();
      }
      _name_to_done_args.erase(it);
    }

  private:
     std::string input_tensor_name;
};

REGISTER_KERNEL_BUILDER(Name("BytepsSyncTensor").Device(::tensorflow::DEVICE_GPU),
                        BytePSSyncTensorOp);
REGISTER_OP("BytepsSyncTensor")
    .Attr("T: {int32, int64, float16, float32, float64}")
    .Attr("input_name: string = 'default_tensor_name'")
    .Input("tensor: T")
    .Output("sum: T")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return ::tensorflow::Status::OK();
    });
REGISTER_XLA_OP(Name("BytepsSyncTensor"), BytepsSyncTensorXlaOp);
XLA_REGISTER_CUSTOM_CALL_TARGET(SyncTensorCustomOp, "CUDA");

void SyncAllTensorsCustomOp(CUstream stream, void** buffers,
  const char* opaque, size_t opaque_len) {
  int num;
  int seen_count = 0;
  int num_synced = 0;
  std::vector<int> buf_sizes;
  std::string tmp_name;
  std::stringstream ss(opaque);
  int my_rank = common::byteps_rank();
  const float** in_buf = reinterpret_cast<const float**>(buffers[0]);

  ////////////////////
  std::stringstream tmp_ss(opaque);
  tmp_ss >> num;
  while (tmp_ss >> tmp_name) {
    int buf_size;
    tmp_ss >> buf_size;
    if (tmp_name == "throwaway_dummy") {
      continue;
    }
    auto args = _name_to_done_args[tmp_name];
    auto expecting = args->bps_in_buf;
    auto got_ptr = buffers[num_synced];
    if (got_ptr != expecting) {
      BPS_LOG(DEBUG, my_rank) << " x2682_error expecting ptr: " << expecting
        << " but got ptr: " << got_ptr << " name_key: " << tmp_name
        << std::endl;
    } else {
      BPS_LOG(DEBUG, my_rank) << " x2682_correct expecting ptr: " << expecting
        << " got ptr: " << got_ptr << " name_key: " << tmp_name
        << std::endl;
    }
    num_synced++;
  }
  num_synced = 0;
  ////////////////////

  ss >> num;
  while (ss >> tmp_name) {
    int buf_size;
    ss >> buf_size;
    if (tmp_name == "throwaway_dummy") {
      seen_count++;
      continue;
    }

    std::unique_lock<std::mutex> my_big_lk(_name_to_done_args_mtx);
    _name_to_done_args_cv.wait(my_big_lk,
      [&tmp_name]{
        std::this_thread::yield();
        return _name_to_done_args.find(tmp_name) != _name_to_done_args.end();
      });

    auto it = _name_to_done_args.find(tmp_name);
    ASSERTF(it != _name_to_done_args.end(), "pos 4");
    auto args = _name_to_done_args[tmp_name];
    my_big_lk.unlock();
    {
      int test_var = 0;
      std::unique_lock<std::mutex> lk(args->mtx);
      test_var = 1;
      lk.unlock();
      ASSERTF(test_var == 1, "test_var not set");
    }
    {
      std::unique_lock<std::mutex> lk(args->mtx);

      auto expecting = args->bps_in_buf;
      auto got_ptr = buffers[num_synced];
      if (got_ptr != expecting) {
        BPS_LOG(DEBUG, my_rank) << " x2682_error expecting ptr: " << expecting
          << " but got ptr: " << got_ptr << " name_key: " << tmp_name
          << std::endl;
      } else {
        BPS_LOG(DEBUG, my_rank) << " x2682_correct expecting ptr: " << expecting
          << "  got ptr: " << got_ptr << " name_key: " << tmp_name
          << std::endl;
      }
      num_synced++;

      while (!args->is_done) {
        args->cv.wait(lk);
      }
      args->num_waiting--;
      if (args->num_waiting == 0) {
        args->is_done = false;
      }
      lk.unlock();
    }
    std::unique_lock<std::mutex> my_lk(_name_to_done_args_mtx);
    _name_to_done_args.erase(tmp_name);
    my_lk.unlock();
    // cudaStreamSynchronize(stream);
    seen_count++;
  }
  ASSERTF(num == seen_count, "pos 5");
  BPS_LOG(DEBUG, my_rank) << " x2682 exitting " << __func__ << std::endl;
}

/**
 * get the buffer size of the i-th input tensor
 */
int get_buf_size(::tensorflow::XlaOpKernelContext* context, int index) {
    const ::tensorflow::TensorShape input_tensor_shape = context->InputShape(index);
    auto xla_tensor_shape_or =
          TensorShapeToXLAShape(context->input_xla_type(index), input_tensor_shape);
    xla::Shape tf_tensor_shape = xla_tensor_shape_or;
    int ret;

    ret = xla::ShapeUtil::ByteSizeOfPrimitiveType(tf_tensor_shape.element_type());
    for (int i = 0; i < tf_tensor_shape.rank(); i++) {
        ret *= tf_tensor_shape.dimensions(i) ;
    }

    return ret;
}

class BytePSSyncAllTensorsXlaOp : public ::tensorflow::XlaOpKernel {
  public:
    explicit BytePSSyncAllTensorsXlaOp(::tensorflow::OpKernelConstruction* context) : ::tensorflow::XlaOpKernel(context) {
      context->GetAttr("tensor_names", &tensor_names_to_sync);
    }
    ~BytePSSyncAllTensorsXlaOp() override = default;

    void Compile(::tensorflow::XlaOpKernelContext* ctx) override {
      int my_rank = common::byteps_rank();
      std::vector<xla::XlaOp> values;
      std::vector<xla::XlaOp> valid_values;
      std::vector<::tensorflow::TensorShape> shapes;
      OP_REQUIRES_OK(ctx, ctx->InputList("values", &values, &shapes));

      const int N = values.size();
      std::vector<xla::Shape> tmp_output_shapes;
      tmp_output_shapes.push_back(xla::ShapeUtil::MakeShape(xla::F32, {2}));

      auto output_shapes = xla::ShapeUtil::MakeTupleShape(tmp_output_shapes);

      std::stringstream ss;

      ss << N;
      for (int i = 0; i < N; i++) {
          auto& tmp_name = tensor_names_to_sync[i];
          if (tmp_name == "throwaway_dummy" ||
            tmp_name.length() == 0) {
            ss << " " << "throwaway_dummy";
          } else {
            ss << " " << tmp_name;
          }

          int tmp_size = get_buf_size(ctx, i);
          ss << " " << tmp_size;
      }
      ss << std::endl;
      xla::XlaOp results = xla::CustomCall(ctx->builder(),
        /*call_target_name=*/"SyncAllTensorsCustomOp",
        values, output_shapes, ss.str());

      for (int i = 0; i < N / 2; i++) {
        ctx->op_kernel_context()->set_output(i,
          ctx->op_kernel_context()->input(i));
      }
      xla::XlaOp tmp_tensor = xla::GetTupleElement(results, 0);
        ctx->SetOutput(N / 2, tmp_tensor);
    }

  private:
    std::vector<std::string> tensor_names_to_sync;
};

REGISTER_OP("BytepsSyncAllTensors")
    .Input("values: N * T")
    .Output("sum: M * T")
    .Attr("T: {int32, int64, float16, float32, float64}")
    .Attr("N: int >= 1")
    .Attr("M: int >= 1")
    .Attr("tensor_names: list(string)")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return ::tensorflow::Status::OK();
      // const int n = c->num_inputs();
      // for (int i = 0; i < n; i++) {
      //   c->set_output(i, c->input(i));
      // }
      // return ::tensorflow::Status::OK();
    });
REGISTER_XLA_OP(Name("BytepsSyncAllTensors").Device(::tensorflow::DEVICE_GPU_XLA_JIT),
  BytePSSyncAllTensorsXlaOp);
XLA_REGISTER_CUSTOM_CALL_TARGET(SyncAllTensorsCustomOp, "CUDA");
////////////////////////////////////////////////////////////////////////////
//for debugging

void PrintTensorsCustomOp(CUstream stream, void** buffers,
  const char* opaque, size_t opaque_len)
{
  int num;
  int count = 0;
  std::vector<int> buf_sizes;
  std::string tmp_name;
  std::stringstream ss(opaque);
  int my_rank = common::byteps_rank();

  ss >> num;
  while (ss >> tmp_name) {
    int buf_size;
    ss >> buf_size;
    cudaMemcpyAsync(buffers[count + num], buffers[count], buf_size, cudaMemcpyDeviceToDevice, stream);
    // cudaStreamSynchronize(stream);
    // printMatOnGPU(tmp_name, buffers[count], buf_size/4);
    count++;
  }
}

class BytePSPrintTensorsXlaOp : public ::tensorflow::XlaOpKernel
{
  public:
    explicit BytePSPrintTensorsXlaOp(::tensorflow::OpKernelConstruction* context) : ::tensorflow::XlaOpKernel(context) {
      context->GetAttr("tensor_names", &tensor_names_to_print);
    }
    ~BytePSPrintTensorsXlaOp() override = default;

    void Compile(::tensorflow::XlaOpKernelContext* ctx) override {
      std::vector<xla::XlaOp> values;
      std::vector<xla::XlaOp> valid_values;
      std::vector<::tensorflow::TensorShape> shapes;
      OP_REQUIRES_OK(ctx, ctx->InputList("values", &values, &shapes));
      const int N = values.size();
      std::vector<xla::Shape> tmp_output_shapes;
      // for (auto operand : values) {
      for (int i = 0; i < N; i++) {
        const xla::Shape* shape = (ctx->builder()->GetShapePtr(values[i])).ValueOrDie();
        tmp_output_shapes.push_back(*shape);
      }

      auto output_shapes = xla::ShapeUtil::MakeTupleShape(tmp_output_shapes);

      std::stringstream ss;
      ss << N;
      for (int i = 0; i < N; i++) {
          auto& tmp_name = tensor_names_to_print[i];
          int tmp_size = get_buf_size(ctx, i);
          ss << " " << tmp_name;
          ss << " " << tmp_size;
      }
      ss << std::endl;
      xla::XlaOp results = xla::CustomCall(ctx->builder(),
        /*call_target_name=*/"PrintTensorsCustomOp",
        valid_values, output_shapes, ss.str());

      for (int i = 0; i < N; i++) {
        xla::XlaOp tmp_tensor = xla::GetTupleElement(results, i);
        ctx->SetOutput(i, tmp_tensor);
      }
    }

  private:
    std::vector<std::string> tensor_names_to_print;
};

REGISTER_OP("BytepsPrintTensors")
    .Input("values: N * T")
    .Output("sum: N * T")
    .Attr("T: {int32, int64, float16, float32, float64}")
    .Attr("N: int >= 1")
    .Attr("tensor_names: list(string)")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return ::tensorflow::Status::OK();
    });
REGISTER_XLA_OP(Name("BytepsPrintTensors"), BytePSPrintTensorsXlaOp);
XLA_REGISTER_CUSTOM_CALL_TARGET(PrintTensorsCustomOp, "CUDA");

}  // namespace tensorflow
}  // namespace byteps
