#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

# Usage: python ./pop_parse.py --popart_dir /path_to/poplar_sdk*/popart-*/include/

import argparse
import os
import parse_onnx as onnx

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

// clang-format off

#pragma once"""

CXXTypeToTypeClass = {
    # Scalar integers
    "int64_t": "INT",
    "int": "INT",
    "bool": "INT",
    "unsigned int": "INT",
    "popart::ReductionType": "INT",
    "popart::ScatterReduction": "INT",
    "nonstd::optional<int64_t>": "OPT_INT",
    "nonstd::optional<int>": "OPT_INT",
    "Attributes::Int": "INT",

    # Floats
    "float": "FLOAT",
    "nonstd::optional<float>": "OPT_FLOAT",

    # Non-scalar floats
    "std::vector<float>": "FLOAT_VEC",

    # Non-scalar integers.
    "std::vector<int64_t>": "INT_VEC",
    "nonstd::optional<std::vector<int64_t> >": "OPT_INT_VEC",
    "Attributes::Ints": "INT_VEC",

    # String
    "std::string": "STRING",
    "std::vector<std::string>": "STRING_VEC",

    # popart
    "popart::TensorId": "STRING",
}

UnsupportedOps = [
    "abort", "ctcloss", "gru", "printtensor", "l1loss", "nllloss",
    "identityloss", "_ctcloss", "scatterreduce", "tfidfvectorizer", "rnn",
    "lstm"
]


# Cleans up raw C++ type to remove reference or const qualifiers
def clean(cxxType):
    return cxxType.replace("&", "").replace("const", "").strip().rstrip()


# Convert the raw C++ type parsed from the header into the macro type.
def toType(cxxType):
    cleaned = clean(cxxType)
    if cleaned in CXXTypeToTypeClass:
        return CXXTypeToTypeClass[cleaned]
    else:
        print(f"toType: Unknown cxxType={cxxType} / cleaned={cleaned}")
        return "UNKNOWN"


def sources_dir():
    return os.path.dirname(os.path.realpath(__file__))


def main(args):
    onnx.init(args.popart_dir, args.clang)
    jsonOutput = onnx.parse()

    classes = []
    for classname in jsonOutput:
        classes.append(classname)
    classes.reverse()

    macroFile = ""
    for opset in classes:
        macroFile += f"\n// Ops from {opset}"
        for name in jsonOutput[opset]:
            if name in UnsupportedOps:
                continue
            print(f"Generating code for {opset}::{name}")

            if 'logical_xor' in name:
                print()

            # Generate the macro
            opDecl = "\n"
            opDecl += "OP_DECL("
            if opset.startswith("AiOnnxOpset"):
                opDecl += "popart_" + name + ","
                opDecl += " aiOnnxOpset." + name
            elif opset.startswith("AiGraphcoreOpset"):
                opDecl += "popart_" + name + "_v2" + ","
                opDecl += " aiGraphcoreOpset." + name
            else:
                continue

            argVector = ""
            earlyExit = True
            args = jsonOutput[opset][name]["args"]
            for arg in args:
                # Skip the first args and also the "name" arg.
                if arg["name"] == "args":
                    earlyExit = False
                    continue

                macroType = toType(arg["type"])
                if macroType == "UNKNOWN":
                    print(f"Skip OP: {name} due to parse failure on {str(arg)}")
                    earlyExit = True
                    break
                elif macroType.startswith("OPT_"):
                    macroType = macroType.replace('OPT_', '')
                    argVector += "OPT_ARG(" + macroType + "," + arg[
                        "name"] + ") "
                else:
                    argVector += "ARG(" + macroType + "," + arg["name"] + ") "

            if earlyExit:
                continue

            if argVector == "":
                argVector = "NONE"

            opDecl += ", " + argVector
            macroFile += opDecl + ") // NOLINT"

    with open(os.path.join(sources_dir(), 'supported_ops_autogen.h'), 'w') as f:
        print(autoComment, file=f)
        print(macroFile, file=f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--clang",
        type=str,
        default=None,
        help="Manually set path to clang headers")
    parser.add_argument("--popart_dir", required=True, help="Popart home")
    args = parser.parse_args()
    print(args)

    # TODO(alleng) support signatures

    main(args)
