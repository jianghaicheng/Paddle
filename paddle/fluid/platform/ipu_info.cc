/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/platform/ipu_info.h"

namespace paddle {
namespace platform {

std::vector<std::shared_ptr<popart::DeviceInfo>> GetIPUDevices() {
  char *p;
  bool use_cpu_model = false;
  if ((p = getenv("TF_POPLAR_FLAGS"))) {
    if (strcmp(p, "--use_cpu_model") == 0) {
      use_cpu_model = true;
    }
  }
  std::vector<std::shared_ptr<popart::DeviceInfo>> devices;
  if (use_cpu_model) {
    auto dm = popart::DeviceManager::createDeviceManager();
    devices = dm.enumerateDevices();
  } else {
    std::map<std::string, std::string> deviceOpts{{"numIPUs", "1"}};
    auto ipuModelDevice =
        popart::DeviceManager::createDeviceManager().createIpuModelDevice(
            deviceOpts);
    devices.push_back(ipuModelDevice);
  }
  return devices;
}
//! Get a list of device ids from environment variable or use all.
std::vector<int> GetSelectedIPUDevices() {
  std::vector<int> devices_ids;
  auto devices = GetIPUDevices();
  for (auto dev : devices) {
    devices_ids.push_back(dev->getId());
  }
  return devices_ids;
}

//! Get the total number of IPU devices in system.
int GetIPUDeviceCount() {
  auto devices = GetIPUDevices();
  if (devices.empty()) {
    LOG(ERROR)
        << "\nNo IPU detected in the system: are you sure the gc-driver is "
           "enabled ?";
    return 0;
  }
  return devices.size();
}
}  // namespace platform
}  // namespace paddle
