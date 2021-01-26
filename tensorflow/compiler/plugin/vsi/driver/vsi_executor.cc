/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/compiler/plugin/vsi/driver/vsi_executor.h"

#include <memory.h>

#include "tensorflow/compiler/plugin/vsi/driver/vsi_utils.h"

namespace xla{
namespace vsiplugin{

VsiExecutor::VsiExecutor(std::shared_ptr<tim::vx::Context>vsiCtx, const int device_ordinal, se::PluginConfig pluginConfig)
 : kVsiContext(vsiCtx), ordinal_(device_ordinal), plugConfig_(pluginConfig) {
     kVsiGraphContainer[ordinal_] = kVsiContext->CreateGraph();
 }

VsiExecutor::~VsiExecutor() {}

se::DeviceMemoryBase VsiExecutor::Allocate(uint64 size, int64 memory_space){
    LOG(FATAL) << "Not Implemented";
    return se::DeviceMemoryBase();
}

void *VsiExecutor::GetSubBuffer(se::DeviceMemoryBase *parent, uint64 offset, uint64 size) {
     LOG(FATAL) << "Not Implemented";
     return nullptr;
}

void VsiExecutor::Deallocate(se::DeviceMemoryBase *mem) {
    LOG(FATAL) << "Not Implemented";
}

void *VsiExecutor::HostMemoryAllocate(uint64 size){
    void *ptr = malloc(size);
    return ptr;
}
void VsiExecutor::HostMemoryDeallocate(void *mem){
    free(mem);
}
bool VsiExecutor::HostMemoryRegister(void *mem, uint64 size){
    LOG(FATAL) << "Not Implemented";
}
bool VsiExecutor::HostMemoryUnregister(void *mem){
    LOG(FATAL) << "Not Implemented";
}
bool VsiExecutor::SynchronizeAllActivity(){
    LOG(FATAL) << "Not Implemented";
}

port::Status VsiExecutor::SynchronousMemZero(se::DeviceMemoryBase *location,
                                uint64 size){
        return port::InternalError("Not implemented");
    }
port::Status VsiExecutor::SynchronousMemSet(se::DeviceMemoryBase *location, int value,
                                uint64 size){
        return port::InternalError("Not implemented");
    }
port::Status VsiExecutor::SynchronousMemcpy(se::DeviceMemoryBase *gpu_dst,
                                const void *host_src, uint64 size){
        return port::InternalError("Not implemented");
    }
port::Status VsiExecutor::SynchronousMemcpy(void *host_dst,
                                const se::DeviceMemoryBase &gpu_src,
                                uint64 size){
        return port::InternalError("Not implemented");
    }
port::Status VsiExecutor::SynchronousMemcpyDeviceToDevice(
    se::DeviceMemoryBase *gpu_dst, const se::DeviceMemoryBase &gpu_src,
    uint64 size){
        return port::InternalError("Not implemented");
     }
port::Status VsiExecutor::MemZero(se::Stream *stream, se::DeviceMemoryBase *location,
                        uint64 size){
        return port::InternalError("Not implemented");
    }
port::Status VsiExecutor::Memset32(se::Stream *stream, se::DeviceMemoryBase *location,
                          uint32 pattern, uint64 size){
        return port::InternalError("Not implemented");
    }

bool VsiExecutor::Memcpy(se::Stream *stream, void *host_dst,
            const se::DeviceMemoryBase &gpu_src, uint64 size){
    LOG(FATAL) << "Not Implemented";
}
bool VsiExecutor::Memcpy(se::Stream *stream, se::DeviceMemoryBase *gpu_dst,
            const void *host_src, uint64 size){
    LOG(FATAL) << "Not Implemented";
}
bool VsiExecutor::MemcpyDeviceToDevice(se::Stream *stream, se::DeviceMemoryBase *gpu_dst,
                            const se::DeviceMemoryBase &gpu_src,
                            uint64 size){
    LOG(FATAL) << "Not Implemented";
}
bool VsiExecutor::HostCallback(se::Stream *stream, std::function<void()> callback){
    LOG(FATAL) << "Not Implemented";
}
bool VsiExecutor::HostCallback(se::Stream *stream,
                    std::function<port::Status()> callback){
    LOG(FATAL) << "Not Implemented";
}
port::Status VsiExecutor::AllocateEvent(se::Event *event){
    return port::InternalError("Not implemented");
}
port::Status VsiExecutor::DeallocateEvent(se::Event *event){
    return port::InternalError("Not implemented");
}
port::Status VsiExecutor::RecordEvent(se::Stream *stream, se::Event *event){
    return port::InternalError("Not implemented");
}
port::Status VsiExecutor::WaitForEvent(se::Stream *stream, se::Event *event){
    return port::InternalError("Not implemented");
}
se::Event::Status VsiExecutor::PollForEventStatus(se::Event *event){
    return se::Event::Status::kError;
}

bool VsiExecutor::AllocateStream(se::Stream *stream) {
    LOG(FATAL) << "Not Implemented";
}
void VsiExecutor::DeallocateStream(se::Stream *stream) {
    LOG(FATAL) << "Not Implemented";
}
bool VsiExecutor::CreateStreamDependency(se::Stream *dependent, se::Stream *other) {
    LOG(FATAL) << "Not Implemented";
}
bool VsiExecutor::AllocateTimer(Timer *timer) {
    LOG(FATAL) << "Not Implemented";
}
void VsiExecutor::DeallocateTimer(Timer *timer) {
    LOG(FATAL) << "Not Implemented";
}
bool VsiExecutor::StartTimer(se::Stream *stream, Timer *timer) {
    LOG(FATAL) << "Not Implemented";
}
bool VsiExecutor::StopTimer(se::Stream *stream, Timer *timer) {
    LOG(FATAL) << "Not Implemented";
}

port::Status VsiExecutor::BlockHostUntilDone(se::Stream *stream){
    return port::InternalError("Not implemented");
}
int VsiExecutor::PlatformDeviceCount(){
    LOG(FATAL) << "Not Implemented";
}
port::Status VsiExecutor::EnablePeerAccessTo(StreamExecutorInterface *other){
    return port::InternalError("Not implemented");
}
bool VsiExecutor::CanEnablePeerAccessTo(StreamExecutorInterface *other){
    LOG(FATAL) << "Not Implemented";
}
se::SharedMemoryConfig VsiExecutor::GetDeviceSharedMemoryConfig(){
    LOG(FATAL) << "Not Implemented";
}
port::Status VsiExecutor::SetDeviceSharedMemoryConfig(
    se::SharedMemoryConfig config){
    return port::InternalError("Not implemented");
}
port::StatusOr<std::unique_ptr<se::DeviceDescription>>
    VsiExecutor::CreateDeviceDescription() const {

        se::internal::DeviceDescriptionBuilder builder;

        builder.set_device_address_bits(64);

        builder.set_name("vsi-npu");
        builder.set_device_memory_size(static_cast<uint64>(4) * 1024 * 1024 * 1024);

        return builder.Build();
    }

    std::unique_ptr<se::internal::EventInterface> VsiExecutor::CreateEventImplementation(){
        LOG(FATAL) << "Not Implemented";
        return nullptr;
    }
    std::unique_ptr<se::internal::KernelInterface> VsiExecutor::CreateKernelImplementation(){
        LOG(FATAL) << "Not Implemented";
        return nullptr;
    }
    std::unique_ptr<se::internal::StreamInterface> VsiExecutor::GetStreamImplementation(){
        LOG(FATAL) << "Not Implemented";
        return nullptr;
    }
    std::unique_ptr<se::internal::TimerInterface> VsiExecutor::GetTimerImplementation(){
        LOG(FATAL) << "Not Implemented";
        return nullptr;
    }
 
} // namespace vsiplugin
} // namespace xla