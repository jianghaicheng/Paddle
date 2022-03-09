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

import logging
import os
import re
from ctypes.util import find_library
import clang.cindex

popart_include_dir = None

popart_files = ["builder.hpp", "builder.h.gen"]
nodeBlacklist = {"DomainOpSet", "Builder", "getOpsetVersion"}


def init(popart_path=None, clang_path=None):
    builder_path = os.path.isfile(
        os.path.join(popart_path, "popart", "builder.hpp"))
    assert builder_path, ("Unable to locate popART's popart/builder.hpp "
                          "in " + popart_path)
    global popart_include_dir
    popart_include_dir = popart_path

    print("Will pick up popART headers from: {0}".format(popart_include_dir))
    for (i, fname) in enumerate(popart_files):
        popart_files[i] = os.path.join(popart_include_dir, "popart", fname)

    if clang.cindex.Config.loaded:
        # Already initialised
        return

    if clang_path is None:
        for version in [11, 9, 8, 7, 6]:
            logging.debug('Trying to find: clang-%s', str(version))
            clang_path = find_library('clang-' + str(version))
            if clang_path is not None:
                break

    assert clang_path is not None, 'Could not find clang'
    logging.info('Will use clang: %s', clang_path)
    clang.cindex.Config.set_library_file(clang_path)


def find_functions(jsonOutput, node, namespace=""):
    # If this is not the file path provided on the comand line, skip.
    if node.location.file is not None and str(
            node.location.file) not in popart_files:
        return
    if node.spelling in nodeBlacklist:
        return

    if node.kind == clang.cindex.CursorKind.CLASS_DECL:
        namespace = node.spelling

    if node.kind != clang.cindex.CursorKind.CXX_METHOD:
        for child in node.get_children():
            find_functions(jsonOutput, child, namespace)
        return

    functionName = node.spelling
    returnType = str(node.type.spelling).split("(")[0]
    operation = dict()
    operation["type"] = returnType
    operation["args"] = []

    if node.access_specifier != clang.cindex.AccessSpecifier.PUBLIC:
        return

    argNum = 0
    for child in node.get_children():
        argument = {}
        if child.kind != clang.cindex.CursorKind.PARM_DECL:
            continue

        argument["type"] = child.type.spelling
        argument["name"] = child.spelling

        # skip 'name' argument
        if argument['name'] == 'name':
            continue

        # skip DebugContext argument
        if re.search('DebugContext', argument['type']):
            continue

        argument["num"] = argNum
        operation["args"].append(argument)
        argNum += 1

    if namespace not in jsonOutput:
        jsonOutput[namespace] = {}

    jsonOutput[namespace][functionName] = operation


# Parse popART header files and extract onnx operator information
# Returns Map of operators, return types and arguments
def parse():
    index = clang.cindex.Index.create()
    print(" popart_include_dir ", popart_include_dir)
    path = os.path.realpath(
        os.path.join(popart_include_dir, "popart", "builder.hpp"))
    logging.info('Parsing: %s', path)
    tu = index.parse(
        path,
        args=[
            "-std=c++14", "-I" + popart_include_dir, "-DONNX_NAMESPACE=onnx"
        ])

    for diag in tu.diagnostics:
        logging.warning(diag)

    json = dict()
    find_functions(json, tu.cursor)

    classes_onnx = []
    for name in json:
        if name.startswith("AiOnnx"):
            classes_onnx.append(name)
        elif name.startswith("AiGraphcore"):
            pass
        else:
            del json[name]
    classes_onnx.reverse()

    added_functions = set()
    for opset in classes_onnx:
        to_remove = []

        for name in json[opset]:
            if name in added_functions:
                to_remove.append(name)
            else:
                added_functions.add(name)

        for name in to_remove:
            json[opset].pop(name)

    return json
