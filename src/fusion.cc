/*!
 *  Copyright (c) 2016 by Contributors
 * \file operator_fusion.cc
 * \brief
 */
#include <nnvm-fusion/base.h>
#include <nnvm/pass.h>
#include <nnvm/symbolic.h>
#include <nnvm/tuple.h>
#include <nnvm/op_attr_types.h>
#include "./internal.h"

namespace nnvm {
namespace fusion {
namespace {

InternalNodePtr CreateNode(NodePtr n) {
  InternalNodePtr ret = InternalNode::Create();
  ret->attrs.op       = n->op();
  ret->attrs.name     = n->attrs.name;
  ret->attrs.dict     = n->attrs.dict;
  ret->attrs.parsed   = n->attrs.parsed;
  return ret;
}


InternalNodePtr CreateVariableNode(Graph& g, NodePtr n) {
  const std::vector<TShape>* shape_map =
    &(g.GetAttr<std::vector<TShape>>("shape"));
  const IndexedGraph &idx = g.indexed_graph();

  NodePtr var = Node::Create();
  var->attrs.op = nullptr;
  var->attrs.name = "var";
  TShape s = (*shape_map)[idx.node_id(n.get())];
  var->attrs.parsed = s;
  return var;
}


bool IsFusible(Graph& g, NodePtr n1, NodePtr n2) {
  static const OpMap<bool>& ewise_map = Op::GetAttr<bool>("IsElementWise");
  static const OpMap<FCodeGen>& gen_map = Op::GetAttr<FCodeGen>("FCodeGen");
  const std::unordered_map<const Node*, uint32_t>* m_times =
    &(g.GetAttr<std::unordered_map<const Node*, uint32_t>>("times"));

  if (n1->op() != nullptr         &&
      n2->op() != nullptr         &&
      ewise_map.count(n1->op())   &&
      ewise_map.count(n2->op())   &&
      gen_map.count(n1->op())     &&
      gen_map.count(n2->op())     &&
      n2->num_outputs() == 1      &&
      m_times->at(n2.get()) == 1) {
    return true;
  }
  return false;
}


bool SetupFusion(Graph& g, NodePtr node,
                 std::unordered_set<Node*>& tobe_merged,
                 std::unordered_map<const Node*, InternalNodePtr>&  m_internal) {
  NodePtr fusion_node{nullptr};
  std::vector<NodeEntry>& inputs = node->inputs;

  bool need_fusion = false;

  InternalNodePtr internal;
  if (m_internal.count(node.get()) == 0) {
    internal = CreateNode(node);
    for (auto &item : inputs) {
      InternalNodePtr var = CreateVariableNode(g, item.node);
      internal->inputs.push_back(NodeEntry{var, 0, 0});
    }
    m_internal[node.get()]  = internal;
  } else {
    internal = m_internal.at(node.get());
  }

  for (auto it = inputs.begin(); it != inputs.end(); ++it) {
    if (IsFusible(g, node, it->node)) {
      // LOG(INFO) << "  Merge Node: " << it->node->attrs.name;
      need_fusion = true;
      tobe_merged.insert(it->node.get());
      InternalNodePtr internal_inode = CreateNode(it->node);
      for (auto &item : it->node->inputs) {
        InternalNodePtr var = CreateVariableNode(g, item.node);
        internal_inode->inputs.push_back(NodeEntry{var, 0, 0});
      }
      m_internal[it->node.get()] = internal_inode;
      internal->inputs[it - inputs.begin()] =
        NodeEntry{internal_inode, 0, it->version+1};
    }
  }

  return need_fusion;
}


NodePtr GetFusionNode(Graph& g, NodePtr node,
                      std::unordered_map<const Node*, InternalNodePtr>& m_internal,
                      std::unordered_map<const Node*, NodePtr>&         m_mirror,
                      std::unordered_map<const Node*, InternalGraph>&   m_internal_graph) {
  InternalNodePtr internal = m_internal.at(node.get());

  NodePtr fnode;
  if (m_mirror.count(node.get()) == 0) {
    // create a new fusion node
    fnode = Node::Create();
    fnode->attrs.op = Op::Get("fusion_op");
    static int count = 0;
    fnode->attrs.name = std::move("fusion" + std::to_string(count++));
    fnode->inputs = node->inputs;
    // LOG(INFO) << "Create Fusion Node: " << fnode->attrs.name;
    InternalGraph internal_graph;
    internal_graph.outputs.emplace_back(NodeEntry{internal, 0, 0});
    m_internal_graph[fnode.get()] = std::move(internal_graph);
  } else {
    fnode = m_mirror.at(node.get());
    // LOG(INFO) << "Use Old Fusion Node: " << fnode->attrs.name;
  }

  return fnode;
}


void MergeIntoInput(NodePtr fnode,
                    std::unordered_set<Node*>& tobe_merged) {
  std::vector<NodeEntry>& inputs = fnode->inputs;
  for (auto it = inputs.begin(); it != inputs.end(); ++it) {
    if (tobe_merged.count(it->node.get()) != 0) {
      NodePtr key = it->node;
      it = inputs.erase(it);
      int index = it - inputs.begin();
      inputs.insert(it, key->inputs.begin(), key->inputs.end());
      it = inputs.begin() + index;
    }
  }
}


void Remap(Graph& g, std::unordered_map<const Node*, NodePtr>& m_mirror) {
  // remap old node with new node
  auto remap = [&](const NodePtr& n) {
    // for those are not in mirror_map, if need_map,
    // create a new node and add it to mirror_map
    if (m_mirror.count(n.get()) == 0) {
      bool need_map = false;
      NodePtr new_node = CreateNode(n);
      // rebuild inputs and control_deps of new node
      for (const NodeEntry& e : n->inputs) {
        if (m_mirror.count(e.node.get()) != 0) {
          need_map = true;
          new_node->inputs.emplace_back(
            NodeEntry{m_mirror.at(e.node.get()), e.index, e.version+1});
        }
        else {
          new_node->inputs.push_back(e);
        }
      }
      for (const NodePtr& e : n->control_deps) {
        if (m_mirror.count(e.get()) != 0) {
          need_map = true;
          new_node->control_deps.push_back(m_mirror.at(e.get()));
        }
        else {
          new_node->control_deps.push_back(e);
        }
      }
      if (need_map) {
        m_mirror[n.get()] = std::move(new_node);
      }
    }
  };
  DFSVisit(g.outputs, remap);
}


void Update(std::unordered_map<const Node*, NodePtr>& m_mirror) {
  // update inputs and control deps of nodes which
  // are in mirror_map already, like fusion nodes
  for (auto kv : m_mirror) {
    for (auto it = kv.second->inputs.begin(); it != kv.second->inputs.end(); ++it) {
      if (m_mirror.count(it->node.get())) {
        *it = NodeEntry{m_mirror.at(it->node.get()), it->index, it->version+1};
      }
    }
    for (auto it = kv.second->control_deps.begin();
         it != kv.second->control_deps.end(); ++it) {
      if (m_mirror.count(it->get())) {
        *it = m_mirror.at(it->get());
      }
    }
  }
}


Graph Fusion(Graph&& src) {
  std::unordered_map<const Node*, InternalGraph>   m_internal_graph;

  std::unordered_map<const Node*, NodePtr>         m_mirror;
  std::unordered_map<const Node*, InternalNodePtr> m_internal;
  std::unordered_map<const Node*, uint32_t>        m_times;

  // build topo order and times map
  std::vector<NodePtr> topo_order;
  DFSVisit(src.outputs, [&](const NodePtr& node) {
    topo_order.push_back(node);
    for (const auto& input: node->inputs) {
      if (m_times.count(input.node.get())) {
          m_times.at(input.node.get())++;
        } else {
          m_times[input.node.get()] = 1;
        }
      }
  });
  src.attrs["times"] = std::make_shared<any>(std::move(m_times));


  for (auto rit = topo_order.rbegin(); rit != topo_order.rend(); ++rit) {
    // LOG(INFO) << "Current Node: " << (*rit)->attrs.name;
    std::unordered_set<Node*> tobe_merged;

    bool need_fusion = SetupFusion(src, *rit, tobe_merged, m_internal);
    if (need_fusion) {
      NodePtr fnode = GetFusionNode(src, *rit, m_internal, m_mirror, m_internal_graph);
      MergeIntoInput(fnode, tobe_merged);

      m_mirror[rit->get()] = fnode;
      for (auto it = tobe_merged.begin(); it != tobe_merged.end(); ++it) {
        if (!(*it)->is_variable()) {
          m_mirror[*it] = fnode;
        }
      }
    }
  }

  Remap(src, m_mirror);
  Update(m_mirror);

  // rebuild return graph
  Graph ret;
  for (const NodeEntry& e: src.outputs) {
    auto it = m_mirror.find(e.node.get());
    if (it != m_mirror.end()) {
      ret.outputs.emplace_back(NodeEntry{it->second, e.index, e.version+1});
    } else {
      ret.outputs.push_back(e);
    }
  }
  ret.attrs["internal_graph"] = std::make_shared<any>(std::move(m_internal_graph));
  return ret;
}


#define SHAPE_ASSIGN(lhs, rhs)                              \
  if ((lhs).ndim() == 0) (lhs) = (rhs);                     \
  else                                                      \
    CHECK_EQ(lhs, rhs) << "shape inference inconsistant";   \

// simply return the shape as same
inline bool FusionShape(const NodeAttrs& attrs,
                        std::vector<TShape> *ishape,
                        std::vector<TShape> *oshape) {
  TShape def_v;
  for (TShape& pshape : *oshape) {
    if (pshape.ndim() != 0) {
      def_v = pshape; break;
    }
  }
  if (def_v.ndim() == 0) {
    for (TShape& pshape : *ishape) {
      if (pshape.ndim() != 0) {
        def_v = pshape;
        if (pshape.ndim() != 1 || pshape[0] != 1) {
          break;
        }
      }
    }
  }
  if (def_v.ndim() == 0) return false;

  for (TShape& pshape : *oshape) {
    SHAPE_ASSIGN(pshape, def_v);
  }
  for (TShape& pshape : *ishape) {
    if (pshape.ndim() == 1 && pshape[0] == 1) {
      continue;
    }
    SHAPE_ASSIGN(pshape, def_v);
  }
  return true;
}

// register pass
NNVM_REGISTER_PASS(Fusion)
.describe("fuse multiple kernels into one")
.set_body(Fusion)
.set_change_graph(true)
.depend_graph_attr("shape")
.provide_graph_attr("internal_graph");

NNVM_REGISTER_OP(fusion_op)
.describe("fusion op")
.set_attr<bool>("IsElementWise", true)
.set_attr<FInferShape>("FInferShape", FusionShape);

}  // namespace
}  // namespace pass
}  // namespace nnvm
