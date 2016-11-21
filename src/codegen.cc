/*!
 *  Copyright (c) 2016 by Contributors
 * \file codegen.cc
 * \brief implementation of code generation
 */
#include <nnvm-fusion/base.h>
#include <nnvm-fusion/ast.h>
#include <nnvm/pass.h>
#include <nnvm/tuple.h>
#include <iostream>
#include <fstream>
#include "./internal.h"

namespace nnvm {
namespace fusion {
namespace {

using ASTPtrIter = std::vector<ASTPtr>::const_iterator;


uint32_t GetVariableNum(const InternalNodePtr internal) {
  uint32_t num = 0;
  for (auto it = internal->inputs.begin(); it != internal->inputs.end(); ++it) {
    if (it->node->is_variable()) {
      ++num;
    } else {
      num += GetVariableNum(it->node);
    }
  }
  return num;
}


ASTPtr GenASTPtr(const InternalNodePtr internal, ASTPtrIter begin, ASTPtrIter end) {
  static const OpMap<FCodeGen>& gen_map = Op::GetAttr<FCodeGen>("FCodeGen");
  FCodeGen gen = gen_map[internal->op()];
  std::vector<ASTPtr> cur_inputs;
  auto it = begin;
  for (auto jt = internal->inputs.begin(); jt != internal->inputs.end(); ++jt) {
    if (it >= end) {
      LOG(FATAL) << "CodeGen inputs string short";
    }
    if (jt->node->is_variable()) {
      cur_inputs.push_back(*it);
      ++it;
    } else {
      uint32_t num = GetVariableNum(jt->node);
      cur_inputs.push_back(GenASTPtr(jt->node, it, it + num));
      it += num;
    }
  }
  return gen(internal, cur_inputs)[0];
}


void PrintInternalNode(const InternalNodePtr internal, int indent=0) {
  std::string s = "";
  for (int i = 0; i < indent; ++i) {
    s += "  ";
  }
  std::cout << s;
  if (indent == 0) {
  } else {
    std::cout << "| ";
  }
  std::cout << internal->attrs.name << std::endl;

  for (auto jt = internal->inputs.begin(); jt != internal->inputs.end(); ++jt) {
    if (jt->node == nullptr) {
      std::cout << s << "  | nullptr" << std::endl;
    } else {
      PrintInternalNode(jt->node, indent + 1);
    }
  }
}


bool NeedBroadCast(const InternalNode* var) {
  TShape s = nnvm::get<TShape>(var->attrs.parsed);
  if (s.ndim() == 1 && s[0] == 1) {
    return true;
  }
  return false;
}


Kernel KernelCodeGen(const std::string& kernel_name, InternalGraph internal_graph) {
  const std::string type_str = "float";
  CHECK_EQ(internal_graph.outputs.size(), 1)
    << "For now, we assume that internal graph should have only one output.";
  InternalNodePtr internal = internal_graph.outputs[0].node;
  // PrintInternalNode(internal);
  const IndexedGraph &idx = internal_graph.indexed_graph();
  const std::vector<uint32_t> &variable_nodes = idx.input_nodes();

  std::string arg_str = "(";
  for (uint32_t i = 0; i < variable_nodes.size(); ++i) {
    arg_str += type_str + " *x" + std::to_string(i) + ", ";
  }
  arg_str += type_str + " *y, ";
  arg_str += "const unsigned int num_elements)";

  std::vector<ASTPtr> decl_exprs;
  std::vector<ASTPtr> assign_exprs;
  std::vector<ASTPtr> inputs;

  ASTPtr output_var     = ASTPtr(new VariableAST("y"));
  for (uint32_t i = 0; i < variable_nodes.size(); ++i) {
    ASTPtr x = ASTPtr(new VariableAST("x" + std::to_string(i)));
    ASTPtr index = nullptr;
    if (NeedBroadCast(idx[variable_nodes[i]].source)) {
      index = ASTPtr(new IntAST(0));
    } else {
      index = ASTPtr(new VariableAST("global_idx"));
    }
    ASTPtr elem_val     = ASTPtr(new ArraySubscriptAST(x, index));
    ASTPtr elem_var     = ASTPtr(new VariableAST("x" + std::to_string(i) + "e"));
    ASTPtr elem_decl    = ASTPtr(new DeclFloatAST(elem_var));
    ASTPtr elem_assign  = ASTPtr(new AssignAST(elem_var, elem_val));

    decl_exprs.push_back(elem_decl);
    assign_exprs.push_back(elem_assign);
    inputs.push_back(elem_var);
  }
  std::string exp_str = GenASTPtr(internal, inputs.begin(), inputs.end())->CodeGen();

  std::string kernel_str =
    "extern \"C\" __global__ void " + kernel_name + arg_str + " {\n" +
    "  unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;\n";
  for (uint32_t i = 0; i < decl_exprs.size(); ++i) {
    kernel_str += "  " + decl_exprs[i]->CodeGen()   + "\n";
    kernel_str += "  " + assign_exprs[i]->CodeGen() + "\n";
  }
  kernel_str = kernel_str +
    "  if (global_idx < num_elements) {\n" +
    "    y[global_idx] = " + exp_str + ";\n" +
    "  }\n"
    "}";
  return Kernel(kernel_name, kernel_str);
}


Graph CodeGen(Graph ret) {
  std::unordered_map<uint32_t, Kernel>  m_kernel;
  const std::unordered_map<const Node*, InternalGraph>* m_internal_graph =
    &(ret.GetAttr<std::unordered_map<const Node*, InternalGraph>>("internal_graph"));

  const IndexedGraph& idx = ret.indexed_graph();
  for (uint32_t nid = 0; nid < idx.num_nodes(); ++nid) {
    const Node* node = idx[nid].source;
    if (node->op() != nullptr && m_internal_graph->count(node) != 0) {
      m_kernel[nid] = KernelCodeGen(node->attrs.name, m_internal_graph->at(node));
    }
  }
  ret.attrs["kernel"] = std::make_shared<any>(std::move(m_kernel));
  return ret;
}


// register pass
NNVM_REGISTER_PASS(CodeGen)
.describe("generate CUDA code from subgraph")
.set_body(CodeGen)
.set_change_graph(true)
.depend_graph_attr("internal_graph")
.provide_graph_attr("kernel");

}  // namespace
}  // namespace fusion
}  // namespace nnvm
