#!/usr/bin/env python3
#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# 
# Usage: python ./PopParse.py --popart_dir /paddle/poplar_sdk-ubuntu_18_04-2.1.0+617-6bb5f5b742/popart-ubuntu_18_04-2.1.0+145366-ce995e299d/include/
import enum
import argparse
import logging
import os
import re
import sys
import clang.cindex
import onnx
# from utils import _utils

parser = argparse.ArgumentParser()
# parser.add_argument("-c",
#                     "--clang",
#                     type=str,
#                     help="Manually set path to clang headers")
parser.add_argument("--popart_dir", default="default", help="Popart home")

args = parser.parse_args()
onnx.init(args.popart_dir, None)
print("popparse", onnx.popart_include_dir)
jsonOutput = onnx.parse()

# logging_level = logging.DEBUG if args.debug else logging.INFO
# logging.basicConfig(level=logging_level)
UnsupportedOps = ["abort", "ctcloss", "gru"]

CXXTypeToTypeClass = {
    # Scalar integers
    "int64_t": "INT",
    "int": "INT",
    "bool": "INT",
    "unsigned int": "INT",
    "popart::ReductionType": "INT",
    "popart::ScatterReduction": "INT",
    "nonstd::optional<int64_t>": "INT",
    "nonstd::optional<int>": "INT",
    "Attributes::Int": "INT",

    # Floats
    "float": "FLOAT",
    "nonstd::optional<float>": "FLOAT",

    # Non-scalar floats
    "std::vector<float>": "FLOAT_VEC",

    # Non-scalar integers.
    "std::vector<int64_t>": "INT_VEC",
    "nonstd::optional<std::vector<int64_t> >": "INT_VEC",
    "Attributes::Ints": "INT_VEC",

    # String
    "std::string": "STRING",
    "std::vector<std::string>": "STRING_VEC"
}


# Cleans up raw C++ type to remove reference or const qualifiers
def clean(cxxType):
    return cxxType.replace("&", "").replace("const", "").strip().rstrip()


# Convert the raw C++ type parsed from the header into the macro type.
def toType(cxxType):

    cleaned = clean(cxxType)

    if cleaned in CXXTypeToTypeClass:
        return CXXTypeToTypeClass[cleaned]

    print("toType: Unknown cxxType={0} / cleaned={1}".format(cxxType, cleaned))

    # Soft fail as it isn't unexpected for some popart functions to be unsupported right now.
    return "UNKNOWN"


def sources_dir():
    # ./scripts/utils/../../:
    return os.path.dirname(os.path.realpath(__file__))


macroFile = ""

classes = []
for classname in jsonOutput:
    classes.append(classname)
classes.reverse()

for opset in classes:
    macroFile += "// Ops from %s\n" % opset
    for name in jsonOutput[opset]:
        if name in UnsupportedOps:
            continue

        print("Generating code for {0}::{1}".format(opset, name))
        # Generate the macro
        opDecl = "OP_DECL("

        # funcName = name.capitalize()
        # opDecl += "popart, " + name + ", " + name
        opDecl += "popart_" + name + ","

        #if opset.startswith("AiOnnxOpset"):
        #    opDecl += "aiOnnxOpset." + name
        #else:
        #    opDecl += opset + "." + name
        if opset.startswith("AiOnnxOpset"):
            opDecl += " aiOnnxOpset." + name
        else:
            continue

        argVector = ""
        # bodyArgVector = ""

        earlyExit = True
        args = jsonOutput[opset][name]["args"]
        for arg in args:
            # Skip the first args and also the "name" arg.
            if arg["name"] == "args":
                # Guarantee we are working with an op which takes in popart tensors as 0th argument.
                earlyExit = False
                continue

            macroType = toType(arg["type"])

            if macroType == "UNKNOWN":
                print("Skipping OP: {0} due to parse failure on {1}".format(
                    name, str(arg)))
                earlyExit = True
                break

            argVector += "ARG(" + macroType + "," + arg["name"] + ") "

        if earlyExit:
            continue

        if argVector == "":
            argVector = "NONE"

        # if bodyArgVector == "":
        #     bodyArgVector = "NONE"

        opDecl += ", " + argVector
        # opDecl += ", " + bodyArgVector

        macroFile += opDecl + ")\n"
autoComment = """// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
"""
with open(os.path.join(sources_dir(), 'supported_ops_autogen.h'), 'w') as f:
    print(autoComment, file=f)
    print(macroFile, file=f)
# print(" _utils.sources_dir() ",sources_dir())
